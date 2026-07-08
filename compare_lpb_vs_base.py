"""
Compare LPB vs Base Policy — generate two visualization videos.

Video 1: base policy drives the env, LPB's eef trajectory overlaid.
Video 2: LPB drives the env, base policy's eef trajectory overlaid.

See docs/superpowers/specs/2026-07-06-lpb-vs-base-comparison-design.md

Usage:
    python compare_lpb_vs_base.py --config-name=compare_transport
    python compare_lpb_vs_base.py --config-name=compare_transport compare_seed=100001
"""
import sys
import os
import json
import pathlib
from typing import Dict, List, Optional

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import dill

from diffusion_policy.workspace.base_workspace import BaseWorkspace

# Line-buffered output for real-time progress in nohup logs
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


def _load_policy_payload(policy_checkpoint: str):
    """Load the .ckpt payload and return (payload, cfg_task). Shared by base + LPB loaders."""
    print(f"[loader] Loading policy checkpoint: {policy_checkpoint}")
    with open(policy_checkpoint, 'rb') as f:
        payload = torch.load(f, pickle_module=dill)
    cfg_task = payload['cfg']
    return payload, cfg_task


def _apply_common_overrides(cfg_task, cfg: DictConfig):
    """Override stale .ckpt paths and inject eval-time fields (CLAUDE.md §6.1).

    Mirrors the inline overrides in eval_base_policy.py and
    eval_test_time_optimization.py. Mutates `cfg_task` in place and returns it.
    """
    cfg_task.n_action_steps = 15
    cfg_task.policy.n_action_steps = 15
    cfg_task.task.env_runner.n_action_steps = 15
    cfg_task.task.env_runner.dataset_path = cfg.demo_dataset_path
    cfg_task.task.dataset.dataset_path = cfg.demo_dataset_path
    return cfg_task


def _load_base_policy(cfg: DictConfig):
    """Load base policy (NO planner). Mirrors eval_base_policy.py."""
    payload, cfg_task = _load_policy_payload(cfg.policy_checkpoint)
    cfg_task = _apply_common_overrides(cfg_task, cfg)

    cls = hydra.utils.get_class(cfg_task._target_)
    workspace = cls(cfg_task, output_dir=cfg.output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg_task.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(cfg.device)
    policy.to(device)
    policy.eval()

    normalizer_dir = os.path.dirname(os.path.dirname(cfg.policy_checkpoint))
    normalizer_path = os.path.join(normalizer_dir, 'normalizer.pth')
    policy.normalizer.load_state_dict(torch.load(normalizer_path))
    policy.normalizer.to(device)

    # Deliberately do NOT call policy.initialize_planner(...)
    print("[loader] Base policy loaded (no planner)")
    return policy, cfg_task


def _load_lpb_policy(cfg: DictConfig):
    """Load LPB policy (WITH planner / dynamics model). Mirrors eval_test_time_optimization.py.

    NOTE: the actual `initialize_planner` signature is
        initialize_planner(self, planner_target, demo_dataset_config,
                           dynamics_model_ckpt, action_step, output_dir,
                           guidance_start_timestep, guidance_scale,
                           threshold, demo_dataset_path=None)
    so the kwargs here are kept in lock-step with that signature and with
    eval_test_time_optimization.py:73-83.
    """
    payload, cfg_task = _load_policy_payload(cfg.policy_checkpoint)
    cfg_task = _apply_common_overrides(cfg_task, cfg)

    cls = hydra.utils.get_class(cfg_task._target_)
    workspace = cls(cfg_task, output_dir=cfg.output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg_task.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(cfg.device)
    policy.to(device)
    policy.eval()

    normalizer_dir = os.path.dirname(os.path.dirname(cfg.policy_checkpoint))
    normalizer_path = os.path.join(normalizer_dir, 'normalizer.pth')
    policy.normalizer.load_state_dict(torch.load(normalizer_path))
    policy.normalizer.to(device)

    # Inject planner + dynamics model. demo_dataset_config MUST be the already-
    # overridden cfg_task.task.dataset (CLAUDE.md §6.1 gotcha #2: planner.py
    # reads dataset_path off this config object during __init__).
    policy.initialize_planner(
        planner_target=cfg.planner_target,
        demo_dataset_config=cfg_task.task.dataset,
        dynamics_model_ckpt=cfg.dynamics_model_checkpoint,
        action_step=cfg_task.n_action_steps,
        output_dir=cfg.output_dir,
        guidance_start_timestep=cfg.guidance_start_timestep,
        guidance_scale=cfg.guidance_scale,
        threshold=cfg.threshold,
        demo_dataset_path=cfg.get('demo_dataset_path', None),
    )
    print("[loader] LPB policy loaded (with planner)")
    return policy, cfg_task


def _build_env(cfg_task, cfg: DictConfig, seed: int,
               render_obs_key: str = 'shouldercamera0_image'):
    """
    Build a SINGLE RobomimicImageWrapper-based env, seed it, and reset.

    Mirrors `create_env` + `_initialize_env` from
    `diffusion_policy/env_runner/robomimic_image_sequential_runner.py`
    (NOT the parallel `RobomimicImageRunner`, which builds N envs via
    `AsyncVectorEnv`). We need a single env because the comparison script
    drives it manually step-by-step and reads eef pose / images at every
    step.

    Construction layering (outside-in):
        MultiStepWrapper                      # n_obs_steps / n_action_steps / max_steps
          > VideoRecordingWrapper             # H.264 recorder (file_path set later per episode)
              > RobomimicImageWrapper         # gym.Env wrapper around robomimic EnvRobosuite
                  > EnvRobosuite (robomimic)  # actual robosuite env

    Required cfg_task.task.env_runner fields used:
        dataset_path, shape_meta, n_obs_steps, n_action_steps, max_steps,
        render_obs_key, fps, crf, abs_action
    """
    import collections
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
    from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
    from diffusion_policy.model.common.rotation_transformer import RotationTransformer

    env_runner_cfg = cfg_task.task.env_runner
    dataset_path = os.path.expanduser(env_runner_cfg.dataset_path)
    shape_meta = OmegaConf.to_container(env_runner_cfg.shape_meta, resolve=True)
    n_obs_steps = env_runner_cfg.n_obs_steps
    n_action_steps = env_runner_cfg.n_action_steps
    max_steps = getattr(env_runner_cfg, 'max_steps', cfg.get('max_steps', 400))
    fps = getattr(env_runner_cfg, 'fps', 10)
    crf = getattr(env_runner_cfg, 'crf', 22)
    abs_action = getattr(env_runner_cfg, 'abs_action', True)

    # ---- 1. Read env_meta from dataset and apply abs_action controller tweak ----
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta['env_kwargs']['use_object_obs'] = False
    if abs_action:
        # Mirrors sequential runner __init__: switch controller to absolute mode
        # so the env consumes rotation_6d actions directly.
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    # ---- 2. Initialize obs modality mapping (required by robomimic) ----
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    # ---- 3. Build the robomimic EnvRobosuite ----
    robomimic_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )
    # Robosuite hard reset causes excessive memory consumption; disable
    # (same as both runners).
    robomimic_env.env.hard_reset = False

    # ---- 4. Wrap: RobomimicImageWrapper -> VideoRecordingWrapper -> MultiStepWrapper ----
    robomimic_wrapper = RobomimicImageWrapper(
        env=robomimic_env,
        shape_meta=shape_meta,
        init_state=None,                # test mode: seed-driven reset
        render_obs_key=render_obs_key,
    )
    video_recorder = VideoRecorder.create_h264(
        fps=fps,
        codec='h264',
        input_pix_fmt='rgb24',
        crf=crf,
        thread_type='FRAME',
        thread_count=1,
    )
    video_wrapper = VideoRecordingWrapper(
        env=robomimic_wrapper,
        video_recoder=video_recorder,
        file_path=None,                 # set per-episode by caller
        steps_per_render=max(20 // fps, 1),
    )
    env = MultiStepWrapper(
        env=video_wrapper,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
    )

    # ---- 5. Seed + reset ----
    # MultiStepWrapper forwards .seed() to its wrapped env (gym.Wrapper behavior);
    # RobomimicImageWrapper.seed() sets np.random.seed + self._seed, and the
    # next reset() consumes _seed to produce a deterministic initial state.
    env.seed(seed)
    obs = env.reset()
    print(f"[env] Built env, seed={seed}, reset OK. obs keys: {sorted(obs.keys())}")
    return env, obs


def _run_rollout(policy, env, cfg, use_guidance: bool, label: str) -> Dict:
    """
    Run a full rollout with `policy` on `env` and sample eef poses / frames
    every `cfg.eef_sample_interval` env chunks.

    `env` MUST already be built. We re-seed with cfg.compare_seed here so the
    same env instance can be reused for both rollouts (the seed_state_map
    cache in RobomimicImageWrapper makes the second reset cheap).

    The env is a MultiStepWrapper, so one env.step() call consumes
    `n_action_steps` underlying robosuite steps with one action chunk
    `(n_action_steps, action_dim)`; we treat that as one "chunk" / decision
    step. cfg.max_steps caps the number of chunks.

    Returns a dict with keys:
        eef_traj      (T_sample, 2, 3) float  - sampled robot0/robot1 eef xyz
        frames        {view: (T_sample, H, W, 3) uint8}
        actions       (T_sample, action_dim_env) float - env-space action chunks
        success       bool
        success_step  int  (-1 if never succeeded)
        n_steps       int  - total env.step() chunks executed
        n_samples     int  - == eef_traj.shape[0]

    Verified against the actual policy/env API:
      - predict_action[_dyn_guided] expects obs_dict {key: (B, To, ...)} torch tensors
        (see diffusion_unet_hybrid_image_policy.py:273-353 and :355-436).
      - The returned 'action' is shape (B, n_action_steps, action_dim) and is
        ALREADY unnormalized (lines 342 and 425 call normalizer['action'].unnormalize).
        So we do NOT re-unnormalize in this function.
      - MultiStepWrapper.step(action) expects action of shape
        (n_action_steps, action_dim) and iterates internally
        (see gym_util/multistep_wrapper.py:101-124).
      - For abs_action envs (Transport: control_delta=False), the policy outputs
        rotation_6d (20-dim dual-arm) but the underlying robosuite controller
        wants axis_angle (14-dim). Both runners call undo_transform_action()
        before env.step(); we mirror that here.
      - get_success_label() lives on RobomimicImageWrapper (innermost env).
        Wrapper stack: MultiStepWrapper -> VideoRecordingWrapper -> RobomimicImageWrapper,
        so we access it via env.env.env.get_success_label()
        (matches robomimic_image_sequential_runner.py:243).
      - obs from env.reset()/env.step() is a dict with values shaped
        (n_obs_steps, ...) (MultiStepWrapper stacks them); we index [-1] for the
        latest observation when sampling eef/frames.
      - RobomimicImageWrapper.get_observation() copies raw robomimic obs through
        unchanged (env/robomimic/robomimic_image_wrapper.py:71-80), so image
        keys arrive as uint8 (H, W, 3) arrays.
    """
    from diffusion_policy.model.common.rotation_transformer import RotationTransformer

    # ---- abs_action / rotation transformer (mirrors both runners) ----
    # Transport uses abs_action=True; rotation_6d<->axis_angle conversion is
    # required before env.step. If this script is ever extended to other tasks,
    # make `abs_action` a cfg field instead of hard-coding True.
    abs_action = True
    rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d') if abs_action else None

    def undo_transform_action(action: np.ndarray) -> np.ndarray:
        """rotation_6d (...,20) -> axis_angle (...,14). Mirrors both runners."""
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(*raw_shape[:-1], 2, 10)
        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3:3 + d_rot]
        gripper = action[..., [-1]]
        rot = rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)
        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)
        return uaction

    # ---- seed + reset (deterministic re-roll) ----
    env.seed(cfg.compare_seed)
    obs = env.reset()
    policy.reset()

    eef_list: List[np.ndarray] = []
    frame_dict: Dict[str, List[np.ndarray]] = {v: [] for v in cfg.render_views}
    actions_list: List[np.ndarray] = []

    success = False
    success_step = -1
    n_chunks = 0  # number of env.step() calls (each consumes n_action_steps)

    print(f"[rollout:{label}] starting. use_guidance={use_guidance}, "
          f"max_chunks={cfg.max_steps}, sample_interval={cfg.eef_sample_interval}")

    for chunk_idx in range(cfg.max_steps):
        # ---- build obs dict (B=1, To, ...) ----
        # Each obs value is (n_obs_steps, ...); add batch dim -> (1, n_obs_steps, ...).
        np_obs_dict = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
        obs_dict = {k: torch.from_numpy(v).to(policy.device) for k, v in np_obs_dict.items()}

        # ---- policy forward ----
        # predict_action_dyn_guided uses grad internally (for classifier guidance),
        # so we do NOT wrap it in torch.no_grad(). predict_action is pure DDPM
        # sampling and benefits from no_grad.
        if use_guidance:
            action_dict = policy.predict_action_dyn_guided(obs_dict)
        else:
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

        # action shape: (1, n_action_steps, action_dim_policy); squeeze batch.
        action = action_dict['action'][0].detach().to('cpu').numpy()
        if not np.all(np.isfinite(action)):
            raise RuntimeError(
                f"[rollout:{label}] NaN/Inf in policy action at chunk {chunk_idx}")

        # ---- env step ----
        # MultiStepWrapper.step expects (n_action_steps, action_dim).
        # For abs_action envs the policy outputs rotation_6d (20-dim dual-arm);
        # convert to axis_angle (14-dim) before stepping.
        env_action = undo_transform_action(action) if abs_action else action

        obs, reward, done, info = env.step(env_action)
        n_chunks += 1

        # ---- sample eef + frames at the configured cadence ----
        if chunk_idx % cfg.eef_sample_interval == 0:
            # obs[*] is stacked (n_obs_steps, ...); index [-1] for the latest.
            eef = np.stack([
                obs['robot0_eef_pos'][-1].copy(),
                obs['robot1_eef_pos'][-1].copy(),
            ])  # (2, 3)
            eef_list.append(eef)

            for v in cfg.render_views:
                img = obs[v][-1]  # (H, W, 3)
                # RobomimicImageWrapper passes through robomimic's uint8
                # (H,W,3) arrays unchanged. Defensively coerce dtype.
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                frame_dict[v].append(img.copy())

            actions_list.append(env_action.copy())

        # ---- success / termination ----
        try:
            cur_success = bool(env.env.env.get_success_label())
        except (AttributeError, RuntimeError) as e:
            print(f"[rollout:{label}] WARN: could not read success label ({e}); treating as False")
            cur_success = False

        if cur_success and not success:
            success = True
            success_step = chunk_idx
            print(f"[rollout:{label}] SUCCESS at chunk {chunk_idx}")
            break

        if bool(np.all(done)):
            print(f"[rollout:{label}] env returned done=True at chunk {chunk_idx} (truncated/terminated)")
            break

    # ---- pack outputs ----
    eef_traj = np.stack(eef_list) if eef_list else np.zeros((0, 2, 3), dtype=np.float32)
    frames = {
        v: (np.stack(frame_dict[v]) if frame_dict[v]
            else np.zeros((0, 140, 140, 3), dtype=np.uint8))
        for v in cfg.render_views
    }
    if actions_list:
        actions = np.stack(actions_list)
    else:
        # Fall back to env action dim (axis_angle 14 for Transport after undo_transform).
        actions = np.zeros((0, 14), dtype=np.float32)

    print(f"[rollout:{label}] done. n_chunks={n_chunks}, sampled_points={len(eef_list)}, "
          f"success={success}, success_step={success_step}")
    return {
        'eef_traj': eef_traj,
        'frames': frames,
        'actions': actions,
        'success': success,
        'success_step': success_step,
        'n_steps': n_chunks,
        'n_samples': len(eef_list),
    }


def _get_camera_params(env, camera_name: str) -> Dict:
    """
    Extract camera position (3,), orientation (3, 3), fovy (scalar) from the
    MuJoCo sim underlying the wrapper stack.

    Wrapper chain (verified against _build_env + robomimic_image_wrapper.py:87):
        MultiStepWrapper → VideoRecordingWrapper → RobomimicImageWrapper
        → EnvRobosuite (robomimic) → robosuite env → MuJoCo sim
    RobomimicImageWrapper.get_flattened_state uses `self.env.env.sim`, where
    self.env is EnvRobosuite and EnvRobosuite.env is the robosuite env. So
    from the outermost MultiStepWrapper, sim lives at env.env.env.env.env.sim.

    Tries multiple APIs because robosuite / mujoco versions differ:
      (1) sim.model.camera_name2id + sim.model.cam_fovy + sim.data.cam_xpos/xmat
          (standard MuJoCo sim API).
      (2) sim._render_context.offscreen.cameras[cam_name] (mujoco viewer API;
          sometimes the only populated source in headless EGL contexts).

    Camera-name convention: robosuite camera names typically lack the
    `_image` suffix that the obs key carries (e.g. obs key
    'shouldercamera0_image' → MuJoCo camera 'shouldercamera0'), so we strip
    it.
    """
    # Walk down .env until we find an object exposing .sim (robosuite env) or
    # .env.env.sim (EnvRobosuite -> robosuite env). Defensive against callers
    # passing any layer of the wrapper stack.
    node = env
    sim = None
    for _ in range(8):
        if hasattr(node, 'sim') and node.sim is not None:
            sim = node.sim
            break
        if not hasattr(node, 'env'):
            break
        node = node.env
    if sim is None:
        raise RuntimeError(
            f"Could not locate MuJoCo `sim` on the env wrapper stack for "
            f"camera '{camera_name}'.")

    cam_name = camera_name.replace('_image', '')

    # ---- API (1): standard MuJoCo sim model/data arrays ----
    try:
        cam_id = sim.model.camera_name2id(cam_name)
        cam_pos = sim.data.cam_xpos[cam_id].copy()
        cam_mat = sim.data.cam_xmat[cam_id].copy().reshape(3, 3)
        fovy = float(sim.model.cam_fovy[cam_id])
        return {'pos': cam_pos, 'mat': cam_mat, 'fovy': fovy}
    except (AttributeError, KeyError, IndexError) as e:
        print(f"[proj] sim.model camera API failed for '{cam_name}': {e}; "
              f"trying offscreen viewers...")

    # ---- API (2): mujoco offscreen render context cameras ----
    try:
        viewer = sim._render_context.offscreen
        cam = viewer.cameras[cam_name]
        cam_pos = np.array(cam.pos)
        cam_mat = np.array(cam.mat).reshape(3, 3)
        fovy = float(cam.fovy)
        return {'pos': cam_pos, 'mat': cam_mat, 'fovy': fovy}
    except (AttributeError, KeyError, TypeError) as e:
        raise RuntimeError(
            f"Could not get camera params for '{cam_name}'. "
            f"Tried sim.model.camera_name2id and sim._render_context.offscreen.cameras. "
            f"Original error: {e}")


def _project_to_camera(points_world: np.ndarray, env, camera_name: str,
                       img_size: int = 140) -> np.ndarray:
    """
    Project 3D world-coords (N, 3) to 2D pixel coords (N, 2) for `camera_name`.

    Uses pinhole perspective; assumes a square image (img_size x img_size).
    For Transport the rendered obs images are 140x140 uint8 (verified in
    scripts/verify_rollout_data.py:129-136 and CLAUDE.md §7), so img_size=140
    is the correct default.

    Convention notes:
      - MuJoCo `cam_xmat` is the camera-to-world rotation stored as a
        row-major flat 9-array; reshape(3,3) yields the cam→world matrix and
        mat.T is world→cam.
      - MuJoCo `cam_fovy` is the vertical field-of-view in degrees.
      - Pixel coordinates follow image convention: +x right, +y down, origin
        top-left. Camera frame is OpenGL-style (-y forward, +z up) so we
        negate the y term when dividing by z.
    """
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"points_world must have shape (N, 3); got {pts.shape}.")

    params = _get_camera_params(env, camera_name)
    pos = np.asarray(params['pos'], dtype=np.float64).reshape(3)
    mat = np.asarray(params['mat'], dtype=np.float64).reshape(3, 3)
    fovy = float(params['fovy'])

    # World → camera frame (mat is cam→world, so mat.T is world→cam).
    p_rel = pts - pos[None, :]
    p_cam = p_rel @ mat.T  # (N, 3)

    # =====================================================================
    # [DIAGNOSTIC] Projection sanity — print camera params and the first
    # input point's p_cam under BOTH transpose conventions, so we can
    # decide whether the fix should be `p_rel @ mat` (no transpose) and
    # whether MuJoCo uses OpenGL (-Z forward, visible points have z<0) or
    # +Z forward. Prints only once per process; remove after the fix lands.
    # =====================================================================
    if not getattr(_project_to_camera, '_diag_printed', False):
        _project_to_camera._diag_printed = True
        print("=" * 70)
        print(f"[diag] camera_name={camera_name}, img_size={img_size}")
        print(f"[diag] cam_pos (world)  = {pos}")
        print(f"[diag] cam_fovy (deg)   = {fovy}")
        print(f"[diag] cam_mat (cam→world; columns = cam axes in world) =\n{mat}")
        print(f"[diag] mat[:,0] cam_X in world (should be 'right')  = {mat[:, 0]}")
        print(f"[diag] mat[:,1] cam_Y in world (should be 'up')     = {mat[:, 1]}")
        print(f"[diag] mat[:,2] cam_Z in world (should be 'back',   = {mat[:, 2]}")
        print(f"[diag]         opposite the viewing direction)")
        # Compare both transpose conventions on the first input point.
        p0_world = pts[0]
        p0_rel   = p0_world - pos
        p_cam_bug = p0_rel @ mat.T    # current code
        p_cam_fix = p0_rel @ mat      # candidate fix
        f_diag = (img_size / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        print(f"[diag] focal length f = {f_diag:.4f} px")
        print(f"[diag] point[0] world  = {p0_world}")
        print(f"[diag] point[0] p_rel  = {p0_rel}")
        print(f"[diag] p_cam[0] BUGGY (p_rel @ mat.T) = {p_cam_bug}  "
              f"z={p_cam_bug[2]:+.4f}")
        print(f"[diag] p_cam[0] FIXED (p_rel @ mat)   = {p_cam_fix}  "
              f"z={p_cam_fix[2]:+.4f}")
        print(f"[diag] MuJoCo OpenGL (-Z forward): FIXED z should be "
              f"NEGATIVE if the point is actually visible.")
        print("=" * 70)

    # Vertical focal length from fovy (degrees). Square image so f_x = f_y.
    f = (img_size / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)

    z = p_cam[:, 2]
    valid = z > 1e-6
    px = np.full_like(z, -1.0)
    py = np.full_like(z, -1.0)
    px[valid] = f * p_cam[valid, 0] / z[valid] + img_size / 2.0
    py[valid] = -f * p_cam[valid, 1] / z[valid] + img_size / 2.0

    return np.stack([px, py], axis=1)


def _render_video_frame(
    bg_frame: np.ndarray,
    overlay_eef_world: np.ndarray,
    env,
    camera_name: str,
    current_step: int,
    total_overlay_steps: int,
    scale: int,
    overlay_method: str,  # 'lpb' or 'base'
    img_size: int = 140,
) -> np.ndarray:
    """Compose one video frame: background image + overlay eef trajectory.

    Background `bg_frame` is a uint8 (img_size, img_size, 3) RGB render from
    `camera_name`. The `overlay_eef_world` array is shape (T, 2, 3) — T timesteps,
    two robot arms, world xyz. We project every step to `camera_name` pixel
    coords via `_project_to_camera`, then split into:
        * "past"   = overlay[:current_step+1]   drawn solid, full alpha
        * "future" = overlay[current_step+1:]   drawn dashed, alpha=0.35
    Each arm gets a distinct marker ('o' for arm 0, 's' for arm 1) at the
    `current_step` position so the two arms are visually distinguishable.

    Color encodes time along the overlay trajectory (cmap depends on
    `overlay_method`: 'lpb' -> plasma, 'base' -> viridis). No colorbar is
    drawn (kept simple; the title annotates step index).

    `scale` controls supersampling: output pixel size = img_size*scale.
    E.g. img_size=140, scale=2 -> (280, 280, 3) uint8.

    matplotlib API notes (server env has matplotlib=3.6.1 per
    conda_environment.yaml):
      - `FigureCanvasToBase.tostring_rgb()` is NOT deprecated in 3.6; it
        was deprecated in 3.10. We use the more future-proof
        `buffer_rgba()` + np.frombuffer + [..., :3] slice, which works on
        3.6+ and avoids the deprecation entirely.
      - LineCollection `linestyles=` accepts a single string ('-' or '--')
        applied to ALL segments. The draft originally passed a 1-tuple
        like `('-',)` which matplotlib interprets as a DashPattern tuple
        and errors out. Fixed here by passing a plain string.
      - `extent=[0, W, H, 0]` + `set_ylim(H, 0)` flips the imshow so the
        pixel (0,0) is top-left, matching the projection convention in
        `_project_to_camera` (px right, py down, origin top-left).

    All matplotlib imports are inside the function because (a) we must set
    the backend to 'Agg' before pyplot is first imported, and (b) it keeps
    the top of this module importable on Windows (no matplotlib needed for
    the static syntax check). The backend-set call is guarded so it only
    runs once per process.
    """
    import matplotlib
    matplotlib.use('Agg')  # headless; safe to call repeatedly
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    if overlay_method == 'lpb':
        cmap_name = 'plasma'
    elif overlay_method == 'base':
        cmap_name = 'viridis'
    else:
        raise ValueError(
            f"overlay_method must be 'lpb' or 'base'; got {overlay_method!r}")
    cmap = plt.get_cmap(cmap_name)

    # ---- project the entire overlay trajectory to this camera's pixels ----
    # overlay_eef_world: (T, 2, 3) -> (T*2, 3) flat -> project -> (T, 2, 2)
    n_total = int(overlay_eef_world.shape[0])
    flat_pts = overlay_eef_world.reshape(n_total * 2, 3)
    flat_px = _project_to_camera(flat_pts, env, camera_name, img_size=img_size)
    all_px = flat_px.reshape(n_total, 2, 2)  # (T, 2 arms, 2 px)

    # ---- figure setup: img_size*scale pixels ----
    fig, ax = plt.subplots(
        figsize=(img_size * scale / 100.0, img_size * scale / 100.0),
        dpi=100,
    )
    # imshow with extent flips y so (0,0) is top-left, matching projection.
    ax.imshow(bg_frame, extent=[0, img_size, img_size, 0])
    ax.set_xlim(0, img_size)
    ax.set_ylim(img_size, 0)  # reversed: top-left origin
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    past_end = min(current_step + 1, n_total)
    future_start = past_end
    future_end = n_total

    norm = Normalize(vmin=0, vmax=max(n_total - 1, 1))

    def _draw_arm_segments(points_px: np.ndarray, base_indices: np.ndarray,
                           linestyle: str, alpha: float):
        """Draw both arms from `points_px` (T_seg, 2, 2).

        Arm 0 uses linestyle as-is; arm 1 ALSO uses linestyle as-is. Earlier
        draft tried to override to '--' for arm 1 but that conflicted with
        the caller's linestyle arg. Both arms use the same linestyle here;
        they are distinguishable by marker shape at `current_step`.

        `base_indices` are the global step indices (for color mapping) that
        correspond to rows of `points_px`.
        """
        for arm_idx in range(2):
            pts = points_px[:, arm_idx]  # (T_seg, 2)
            # skip points that failed projection (px < 0 sentinel)
            valid = (pts[:, 0] >= 0) & (pts[:, 1] >= 0)
            if valid.sum() < 2:
                continue
            seg_pts = pts[valid]
            seg_idx = base_indices[valid]
            seg_colors = cmap(norm(seg_idx))
            seg_colors[:, 3] = alpha
            segments = np.stack([seg_pts[:-1], seg_pts[1:]], axis=1)
            # `linestyles` accepts a single string applied to all segments
            # in matplotlib 3.6 (verified against the LineCollection
            # docstring: "linestyles : linestyle or list of linestyles").
            lc = LineCollection(
                segments,
                colors=seg_colors[:-1],
                linewidths=2,
                linestyles=linestyle,
            )
            ax.add_collection(lc)

    # ---- past (solid, full alpha) ----
    if past_end > 0:
        idx = np.arange(past_end)
        _draw_arm_segments(all_px[:past_end], idx, linestyle='-', alpha=1.0)

    # ---- future (dashed, faded) ----
    if future_end > future_start:
        idx = np.arange(future_start, future_end)
        _draw_arm_segments(
            all_px[future_start:future_end], idx, linestyle='--', alpha=0.35)

    # ---- current position markers ----
    if 0 <= current_step < n_total:
        for arm_idx, marker in enumerate(['o', 's']):
            pt = all_px[current_step, arm_idx]
            if pt[0] >= 0 and pt[1] >= 0:
                ax.plot(
                    pt[0], pt[1], marker,
                    color=cmap(norm(current_step)),
                    markersize=10,
                    markeredgecolor='white',
                    markeredgewidth=1.5,
                )

    ax.set_title(
        f'{camera_name}\n{overlay_method} traj @ step {current_step}/{n_total}',
        fontsize=10,
    )

    # ---- rasterize figure to uint8 (H, W, 3) ----
    fig.canvas.draw()
    # buffer_rgba() is the non-deprecated path on mpl 3.6+; returns a
    # memoryview of the renderer's RGBA buffer.
    rgba = np.asarray(fig.canvas.buffer_rgba())
    out = rgba[:, :, :3].copy()  # drop alpha; copy to make C-contiguous
    plt.close(fig)
    return out


def _make_video(
    driver_result: Dict,
    overlay_result: Dict,
    overlay_method: str,
    env,
    view_names: List[str],
    cfg: DictConfig,
    output_path: str,
):
    """Generate one MP4: driver's frames as background, overlay's eef trajectory projected.

    Iterates over `driver_result['frames'][view][t]` (background images from
    the driver rollout) and composes a side-by-side panel of all `view_names`
    cameras, with the `overlay_result['eef_traj']` drawn on top via
    `_render_video_frame`.

    Frame dimensions:
        Single panel: img_size * cfg.video_frame_scale on each side
        (e.g. 140 * 2 = 280). For len(view_names)=2 cameras, the final frame
        is (280, 560, 3) uint8. `macro_block_size=1` is passed to imageio so
        ffmpeg does not force the width/height to multiples of 16 (560 and
        280 are not multiples of 16).

    Length mismatch handling: the driver rollout may have more or fewer
    sampled steps than the overlay. We iterate `t in range(n_frames)` (driver
    length) and cap the overlay step at `overlay_n - 1`. If
    `overlay_n == 0` we skip the video entirely (nothing to draw).

    imageio API (verified for imageio 2.22.0 + imageio-ffmpeg 0.4.7 in
    conda_environment.yaml):
        `imageio.v2.get_writer(path, fps=..., codec='libx264', quality=N,
        macro_block_size=1)` is the supported v2 API on this version.
        `quality` is a 0-10 scale (lower = higher quality); we use 8 per the
        design spec. `macro_block_size=1` disables the default multiple-of-16
        padding.

    Memory: imageio writes frames incrementally via ffmpeg, so we never hold
    the full video in memory — only one composed frame at a time (~470 KB for
    280x560x3 uint8).

    Cleanup: writer.close() runs in a finally block so a mid-loop exception
    still flushes the partial file (or closes the ffmpeg subprocess cleanly).
    """
    import imageio.v2 as imageio  # v2 API for backward compat

    n_frames = int(driver_result['n_samples'])
    overlay_n = int(overlay_result['n_samples'])

    if n_frames == 0:
        print(f"[video] SKIP: driver has 0 frames ({output_path})")
        return
    if overlay_n == 0:
        print(f"[video] SKIP: overlay has 0 samples, nothing to draw ({output_path})")
        return

    # img_size from driver frame H (frames are (T, H, W, 3) per _run_rollout).
    # All views share the same H (Transport: 140), so we read from view 0.
    img_size = int(driver_result['frames'][view_names[0]].shape[1])
    out_h = img_size * cfg.video_frame_scale
    out_w = img_size * cfg.video_frame_scale * len(view_names)

    print(f"[video] writing {output_path}: {n_frames} frames, "
          f"{len(view_names)} views side-by-side, {out_h}x{out_w}px")

    writer = imageio.get_writer(
        output_path,
        fps=cfg.video_fps,
        codec='libx264',
        quality=8,
        macro_block_size=1,
    )

    try:
        for t in range(n_frames):
            # Cap overlay step at last available index (handles driver outlasting overlay).
            overlay_step = min(t, overlay_n - 1)
            panels = []
            for view in view_names:
                bg = driver_result['frames'][view][t]
                panel = _render_video_frame(
                    bg_frame=bg,
                    overlay_eef_world=overlay_result['eef_traj'],
                    env=env,
                    camera_name=view,
                    current_step=overlay_step,
                    total_overlay_steps=overlay_n,
                    scale=cfg.video_frame_scale,
                    overlay_method=overlay_method,
                )
                panels.append(panel)
            # Horizontal stack: (out_h, out_w, 3). All panels are identical-shaped
            # (out_h, out_h, 3) per _render_video_frame's contract.
            frame = np.concatenate(panels, axis=1)
            writer.append_data(frame)

            if t % 20 == 0:
                print(f"  [video] frame {t}/{n_frames}")
    finally:
        # Ensure the ffmpeg subprocess is closed even on exception so we don't
        # leak a hung writer or leave the file half-flushed without close().
        writer.close()

    print(f"[video] wrote {output_path} ({n_frames} frames)")


@hydra.main(config_path="dyn_model/conf/planner", config_name="compare_transport")
def main(cfg: DictConfig):
    print("=" * 60)
    print("LPB vs Base Policy Comparison")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    pathlib.Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    # Save config
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, 'compare_config.yaml'))

    # ---- Phase 1: base rollout ----
    print("\n[Phase 1] Loading base policy + running base rollout...")
    base_policy, cfg_task = _load_base_policy(cfg)
    env, _ = _build_env(cfg_task, cfg, seed=cfg.compare_seed)
    base_result = _run_rollout(base_policy, env, cfg, use_guidance=False, label='base')
    del base_policy
    torch.cuda.empty_cache()

    # ---- Phase 2: LPB rollout ----
    print("\n[Phase 2] Loading LPB policy + running LPB rollout...")
    lpb_policy, _ = _load_lpb_policy(cfg)
    env2, _ = _build_env(cfg_task, cfg, seed=cfg.compare_seed)
    lpb_result = _run_rollout(lpb_policy, env2, cfg, use_guidance=True, label='lpb')
    del lpb_policy
    torch.cuda.empty_cache()

    # ---- Phase 3: sanity check projection ----
    print("\n[Phase 3] Projection sanity check...")
    cam = cfg.render_views[0]
    eef0 = base_result['eef_traj'][0, 0]
    px = _project_to_camera(eef0.reshape(1, 3), env, cam)[0]
    assert 0 <= px[0] < 140 and 0 <= px[1] < 140, \
        f"Projection sanity check failed: {px}"
    print(f"  OK: world {eef0} -> px {px}")

    # ---- Phase 4: save rollout data ----
    print("\n[Phase 4] Saving rollout data...")
    np.savez_compressed(
        os.path.join(cfg.output_dir, 'rollout_data.npz'),
        base_eef=base_result['eef_traj'],
        base_actions=base_result['actions'],
        base_success=base_result['success'],
        base_success_step=base_result['success_step'],
        lpb_eef=lpb_result['eef_traj'],
        lpb_actions=lpb_result['actions'],
        lpb_success=lpb_result['success'],
        lpb_success_step=lpb_result['success_step'],
    )

    # ---- Phase 5: render videos ----
    print("\n[Phase 5a] Video 1: base driving, LPB overlay...")
    _make_video(
        driver_result=base_result,
        overlay_result=lpb_result,
        overlay_method='lpb',
        env=env,
        view_names=list(cfg.render_views),
        cfg=cfg,
        output_path=os.path.join(cfg.output_dir, 'base_driving_lpb_overlay.mp4'),
    )

    print("\n[Phase 5b] Video 2: LPB driving, base overlay...")
    _make_video(
        driver_result=lpb_result,
        overlay_result=base_result,
        overlay_method='base',
        env=env2,
        view_names=list(cfg.render_views),
        cfg=cfg,
        output_path=os.path.join(cfg.output_dir, 'lpb_driving_base_overlay.mp4'),
    )

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Output dir: {cfg.output_dir}")
    print(f"  - base_driving_lpb_overlay.mp4 ({base_result['n_samples']} frames)")
    print(f"  - lpb_driving_base_overlay.mp4 ({lpb_result['n_samples']} frames)")
    print(f"  - rollout_data.npz")
    print(f"  Base success: {base_result['success']} @ step {base_result['success_step']}")
    print(f"  LPB  success: {lpb_result['success']} @ step {lpb_result['success_step']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
