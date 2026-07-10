"""
compare_viz — visualize LPB vs base policy divergence.

Main output: ONE composited video where LPB (base + dynamics + classifier
guidance) drives the env from a seed-initialized state (recorded at full env
fps via the standard `VideoRecordingWrapper`), with the BASE policy's eef
trajectory overlaid as a colored polyline (past = solid, future = dashed).

Also saved as raw outputs (no overlay):
  - base_driver*.mp4 — the base policy's own rollout video
  - lpb_driver*.mp4  — the LPB policy's own rollout video
so you can inspect each policy's actual behavior independently.

Two modes:
  - Default (find_divergent_seed=false): use cfg.compare_seed directly.
  - Divergent search (find_divergent_seed=true): try seeds starting from
    cfg.compare_seed, increment by 1 until LPB succeeds AND base fails, then
    visualize that trajectory. Bases that succeed are skipped (no LPB run);
    this minimizes the expensive LPB rolloffs.

Cross-env compatibility: works for any task (transport, square, tool_hang,
libero, ...) given a matching compare_<env>.yaml. The env-specific fields
are read from the policy checkpoint's task config or the compare yaml:
  - render_obs_key  <- cfg.task.env_runner.render_obs_key
  - abs_action      <- cfg.task.env_runner.abs_action
  - n_action_steps  <- cfg.n_action_steps (top-level, in compare_<env>.yaml)

Usage:
    python scripts/compare_viz.py --config-name=compare_transport
    python scripts/compare_viz.py --config-name=compare_transport find_divergent_seed=true
    python scripts/compare_viz.py --config-name=compare_transport \\
        find_divergent_seed=true max_seed_search=100
"""
import os
import sys
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import dill

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env_runner.robomimic_image_sequential_runner import create_env

# Line-buffered output for real-time progress in nohup logs
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

# Hydra config path is anchored to the repo root, regardless of where the
# script is invoked from. This file lives at <root>/scripts/compare_viz.py.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "dyn_model", "conf", "planner")


# ===========================================================================
# Extra camera angles — record ADDITIONAL videos from these viewpoints,
# alongside the main render_obs_key video. Edit here, no need to touch yaml.
# ===========================================================================
# Each camera produces its own base_driver_<cam>.mp4, lpb_driver_<cam>.mp4,
# and lpb_driving_base_overlay_<cam>.mp4 in the output dir, in ADDITION to
# the main (shouldercamera0) set. So with EXTRA_RENDER_KEYS=('agentview_image',)
# you get 6 videos total: 3 from shouldercamera0 + 3 from agentview.
#
# Cameras must exist in the env XML (transport_model_file_1.4.xml for
# Transport) and be FIXED in world (not eye-in-hand, which translates with
# the arm and would make the world-coord overlay projection incoherent).
#
# Transport options: frontview_image, birdview_image, agentview_image,
# sideview_image, shouldercamera1_image (shouldercamera0_image is the main
# view; listing it here would just duplicate).
#
# IMPORTANT: this ONLY affects the rendered videos. The policy's obs inputs
# are unchanged (still the 4 cameras it was trained on). We bypass
# RobomimicImageWrapper and call MuJoCo's offscreen render directly, so the
# extra cameras don't need to be in shape_meta.
#
# Note: extra videos are rendered at ONE FRAME PER CHUNK (vs the main video's
# every-few-env-steps via VideoRecordingWrapper). So they play ~7-15x faster
# than real time. Fine for trajectory visualization; would need a custom
# wrapper to match the main video's fps.
EXTRA_RENDER_KEYS = ('agentview_image',)


# ===========================================================================
# Policy loaders
# ===========================================================================

def _load_policy_payload(policy_checkpoint: str):
    """Load the .ckpt payload and return (payload, cfg_task). Shared by base + LPB loaders."""
    print(f"[loader] Loading policy checkpoint: {policy_checkpoint}")
    with open(policy_checkpoint, 'rb') as f:
        payload = torch.load(f, pickle_module=dill)
    cfg_task = payload['cfg']
    return payload, cfg_task


def _apply_common_overrides(cfg_task, cfg: DictConfig):
    """Override stale .ckpt paths and inject eval-time fields (CLAUDE.md §6.1).

    n_action_steps is read from cfg (top-level field in compare_<env>.yaml),
    NOT hardcoded — different envs use different chunk sizes (transport=15,
    square=8, etc.).
    """
    n_action_steps = cfg.get('n_action_steps', 8)
    cfg_task.n_action_steps = n_action_steps
    cfg_task.policy.n_action_steps = n_action_steps
    cfg_task.task.env_runner.n_action_steps = n_action_steps
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

    print("[loader] Base policy loaded (no planner)")
    return policy, cfg_task


def _load_lpb_policy(cfg: DictConfig):
    """Load LPB policy (WITH planner / dynamics model). Mirrors eval_test_time_optimization.py."""
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

    # demo_dataset_config MUST be the already-overridden cfg_task.task.dataset
    # (CLAUDE.md §6.1 gotcha #2: planner.py reads dataset_path off this config
    # object during __init__).
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


# ===========================================================================
# Env construction — thin wrapper around the standard runner's create_env
# ===========================================================================

def _build_env(cfg_task, cfg: DictConfig):
    """Build env via the standard `create_env` factory.

    Does NOT seed or reset — the caller does that AFTER setting
    `env.env.file_path` so VideoRecordingWrapper records.

    Returns (env, render_obs_key, n_action_steps, steps_per_render). Wrapper
    stack (outside-in): MultiStepWrapper → VideoRecordingWrapper
    → RobomimicImageWrapper → EnvRobosuite. `env.env` is the
    VideoRecordingWrapper (where `file_path` lives).
    """
    import robomimic.utils.file_utils as FileUtils

    env_runner_cfg = cfg_task.task.env_runner
    dataset_path = os.path.expanduser(env_runner_cfg.dataset_path)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta['env_kwargs']['use_object_obs'] = False
    abs_action = getattr(env_runner_cfg, 'abs_action', False)
    if abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    shape_meta = OmegaConf.to_container(env_runner_cfg.shape_meta, resolve=True)
    render_obs_key = env_runner_cfg.render_obs_key
    n_obs_steps = env_runner_cfg.n_obs_steps
    n_action_steps = env_runner_cfg.n_action_steps
    max_steps = getattr(env_runner_cfg, 'max_steps', cfg.get('max_steps', 400))
    fps = getattr(env_runner_cfg, 'fps', 10)
    crf = getattr(env_runner_cfg, 'crf', 22)

    env = create_env(
        env_meta=env_meta,
        shape_meta=shape_meta,
        enable_render=True,
        render_obs_key=render_obs_key,
        fps=fps,
        crf=crf,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_steps=max_steps,
    )

    steps_per_render = max(20 // fps, 1)

    print(f"[env] Built env via create_env. render_obs_key={render_obs_key}, "
          f"abs_action={abs_action}, n_action_steps={n_action_steps}, "
          f"fps={fps}, steps_per_render={steps_per_render}, max_steps={max_steps}")
    return env, render_obs_key, n_action_steps, steps_per_render, abs_action


# ===========================================================================
# Rollout — mirrors SequentialRobomimicImageRunner.run's inner loop
# ===========================================================================

def _run_rollout(policy, env, cfg, use_guidance: bool, label: str,
                 abs_action: bool, video_path: Optional[str] = None,
                 extra_render_keys: Tuple[str, ...] = ()) -> Dict:
    """Run a rollout. If `video_path` is set, env records at full fps to that
    MP4 via VideoRecordingWrapper (file_path must be assigned BEFORE reset,
    since reset() stops any prior recorder).

    `env` MUST already be built. We re-seed with cfg.compare_seed here so the
    same env instance can be reused for both rollouts.

    One env.step() consumes `n_action_steps` underlying robosuite steps; we
    treat that as one "chunk" / decision step. cfg.max_steps caps chunks.

    `extra_render_keys` (optional): for each camera obs-key in this tuple
    (e.g. 'agentview_image'), render ONE frame per chunk via MuJoCo's
    offscreen context (bypassing RobomimicImageWrapper, so the policy's obs
    inputs are completely unaffected). Frames are returned in `extra_frames`.

    Returns a dict with keys:
        eef_traj      (T, 2, 3) float  - robot0/robot1 eef xyz per chunk
        actions       (T, action_dim_env) float
        success       bool
        success_step  int  (-1 if never succeeded)
        n_steps       int  - chunks executed
        n_samples     int  - == eef_traj.shape[0]
        video_path    Optional[str]
        extra_frames  Dict[str, List[np.ndarray]] - {cam_key: [(H,W,3) uint8]}
    """
    from diffusion_policy.model.common.rotation_transformer import RotationTransformer

    rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d') if abs_action else None

    def undo_transform_action(action: np.ndarray) -> np.ndarray:
        """rotation_6d (...,20 or 10) -> axis_angle (...,14 or 7). Mirrors runner."""
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
            uaction = uaction.reshape(*raw_shape[:-1], 14)
        return uaction

    # ---- set video file_path BEFORE reset (env.env = VideoRecordingWrapper) ----
    env.env.file_path = video_path
    if video_path is not None:
        print(f"[rollout:{label}] env will record to {video_path}")

    # ---- seed + reset ----
    env.seed(cfg.compare_seed)
    obs = env.reset()
    policy.reset()

    eef_list: List[np.ndarray] = []
    actions_list: List[np.ndarray] = []
    extra_frames: Dict[str, List[np.ndarray]] = {k: [] for k in extra_render_keys}
    success = False
    success_step = -1
    n_chunks = 0

    print(f"[rollout:{label}] starting. use_guidance={use_guidance}, "
          f"abs_action={abs_action}, max_chunks={cfg.max_steps}, seed={cfg.compare_seed}")

    for chunk_idx in range(cfg.max_steps):
        np_obs_dict = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
        obs_dict = {k: torch.from_numpy(v).to(policy.device) for k, v in np_obs_dict.items()}

        if use_guidance:
            action_dict = policy.predict_action_dyn_guided(obs_dict)
        else:
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

        action = action_dict['action'][0].detach().to('cpu').numpy()
        if not np.all(np.isfinite(action)):
            raise RuntimeError(
                f"[rollout:{label}] NaN/Inf in policy action at chunk {chunk_idx}")

        env_action = undo_transform_action(action) if abs_action else action
        obs, reward, done, info = env.step(env_action)
        n_chunks += 1

        eef = np.stack([
            obs['robot0_eef_pos'][-1].copy(),
            obs['robot1_eef_pos'][-1].copy(),
        ])  # (2, 3)
        eef_list.append(eef)
        actions_list.append(env_action.copy())

        # ---- render extra cameras at this chunk's end-state ----
        # One frame per chunk per camera. Sim state is current (env.step just
        # returned), so we can render any camera in the XML. Bypasses
        # RobomimicImageWrapper entirely; policy's obs inputs are untouched.
        for cam_key in extra_render_keys:
            try:
                frame = _render_extra_camera(env, cam_key)
                extra_frames[cam_key].append(frame)
            except Exception as e:
                # Don't fail the whole rollout if one extra render errors
                # (e.g., camera name typo, EGL hiccup). Just log and skip.
                print(f"[rollout:{label}] WARN: extra render '{cam_key}' failed at "
                      f"chunk {chunk_idx}: {e}")
                extra_frames[cam_key].append(None)

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
            print(f"[rollout:{label}] env done=True at chunk {chunk_idx}")
            break

    # env.render() stops the recorder (flushing the MP4) and returns file_path.
    final_video_path = env.render() if video_path is not None else None

    eef_traj = np.stack(eef_list) if eef_list else np.zeros((0, 2, 3), dtype=np.float32)
    actions = np.stack(actions_list) if actions_list else np.zeros((0, 14), dtype=np.float32)

    print(f"[rollout:{label}] done. n_chunks={n_chunks}, success={success}, "
          f"success_step={success_step}, extra_frames="
          f"{ {k: len([f for f in v if f is not None]) for k, v in extra_frames.items()} }")
    return {
        'eef_traj': eef_traj,
        'actions': actions,
        'success': success,
        'success_step': success_step,
        'n_steps': n_chunks,
        'n_samples': len(eef_list),
        'video_path': final_video_path,
        'extra_frames': extra_frames,
    }


# ===========================================================================
# Extra-camera helpers — render from any MuJoCo camera without touching obs
# ===========================================================================

def _render_extra_camera(env, camera_obs_key: str, img_size: int = 140) -> np.ndarray:
    """Render the current sim state from a specific MuJoCo camera, bypassing
    RobomimicImageWrapper.

    Why bypass the wrapper: `RobomimicImageWrapper.render()` returns
    `render_cache`, which is populated from `raw_obs[render_obs_key]` during
    the last `get_observation()` call. The raw_obs only contains cameras that
    are in shape_meta (4 for Transport). agentview / birdview etc. aren't in
    shape_meta, so they're not in raw_obs and the wrapper can't render them.

    This helper walks down to the underlying mujoco sim and calls
    `sim._render_context.offscreen.render(W, H, cam_id)` directly. The sim
    knows about ALL cameras in the env XML, regardless of shape_meta.

    Args:
        env: wrapped env (MultiStepWrapper → ... → robosuite env).
        camera_obs_key: obs key like 'agentview_image' (will strip '_image'
            to get the MuJoCo camera name).
        img_size: render output is img_size x img_size.

    Returns:
        np.ndarray (img_size, img_size, 3) uint8.
    """
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
            f"Could not find `sim` on the env wrapper stack to render "
            f"extra camera '{camera_obs_key}'.")

    cam_name = camera_obs_key.replace('_image', '')
    cam_id = sim.model.camera_name2id(cam_name)

    # mujoco-py 2.x: offscreen.render(W, H, cam_id) -> (H, W, 3) uint8
    # (some versions return (3, H, W); we normalize below).
    img = sim._render_context.offscreen.render(img_size, img_size, cam_id)
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        # channels-first (3, H, W) -> channels-last (H, W, 3)
        img = np.moveaxis(img, 0, -1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _save_frames_to_mp4(frames: List[np.ndarray], output_path: str, fps: int):
    """Write a list of (H, W, 3) uint8 frames to an MP4 via imageio.

    Skips None entries (failed extra renders) silently so a single bad frame
    doesn't abort the whole video.
    """
    import imageio.v2 as imageio
    n_valid = sum(1 for f in frames if f is not None)
    if n_valid == 0:
        print(f"[video] SKIP: no valid frames to write to {output_path}")
        return
    writer = imageio.get_writer(
        output_path, fps=fps, codec='libx264',
        quality=8, macro_block_size=1,
    )
    try:
        for frame in frames:
            if frame is None:
                continue
            writer.append_data(frame)
    finally:
        writer.close()
    print(f"[video] wrote {output_path} ({n_valid} frames)")


# ===========================================================================
# Camera projection (3D world → 2D pixel)
# ===========================================================================

def _get_camera_params(env, camera_name: str) -> Dict:
    """Extract camera pos (3,), mat (3,3), fovy (scalar) from the MuJoCo sim."""
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

    try:
        cam_id = sim.model.camera_name2id(cam_name)
        cam_pos = sim.data.cam_xpos[cam_id].copy()
        cam_mat = sim.data.cam_xmat[cam_id].copy().reshape(3, 3)
        fovy = float(sim.model.cam_fovy[cam_id])
        return {'pos': cam_pos, 'mat': cam_mat, 'fovy': fovy}
    except (AttributeError, KeyError, IndexError) as e:
        print(f"[proj] sim.model camera API failed for '{cam_name}': {e}; "
              f"trying offscreen viewers...")

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
                       img_size: int) -> np.ndarray:
    """Project 3D world-coords (N, 3) to 2D pixel coords (N, 2).

    MuJoCo cameras use OpenGL convention (looks along -Z); visible points have
    z_cam < 0, so we project with `depth = -z_cam`. For NumPy row vectors,
    right-multiplying by `mat` applies `mat.T` (= world→cam).
    """
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N, 3); got {pts.shape}.")

    params = _get_camera_params(env, camera_name)
    pos = np.asarray(params['pos'], dtype=np.float64).reshape(3)
    mat = np.asarray(params['mat'], dtype=np.float64).reshape(3, 3)
    fovy = float(params['fovy'])

    p_rel = pts - pos[None, :]
    p_cam = p_rel @ mat  # world→cam for row vectors

    f = (img_size / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)

    z = p_cam[:, 2]
    depth = -z
    valid = depth > 1e-6
    px = np.full_like(z, -1.0)
    py = np.full_like(z, -1.0)
    px[valid] =  f * p_cam[valid, 0] / depth[valid] + img_size / 2.0
    py[valid] = -f * p_cam[valid, 1] / depth[valid] + img_size / 2.0

    return np.stack([px, py], axis=1)


# ===========================================================================
# Frame renderer — draws overlay trajectory on a single background image
# ===========================================================================

def _render_video_frame(
    bg_frame: np.ndarray,
    overlay_eef_world: np.ndarray,
    env,
    camera_name: str,
    current_step: int,
    total_overlay_steps: int,
    scale: int,
    overlay_method: str,  # 'lpb' or 'base'
    img_size: int,
) -> np.ndarray:
    """Compose one video frame: background image + overlay eef trajectory.

    Past = overlay[:current_step+1]   solid, full alpha.
    Future = overlay[current_step+1:] dashed, alpha=0.35.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    cmap_name = 'plasma' if overlay_method == 'lpb' else 'viridis'
    cmap = plt.get_cmap(cmap_name)

    n_total = int(overlay_eef_world.shape[0])
    flat_pts = overlay_eef_world.reshape(n_total * 2, 3)
    flat_px = _project_to_camera(flat_pts, env, camera_name, img_size=img_size)
    all_px = flat_px.reshape(n_total, 2, 2)

    fig, ax = plt.subplots(
        figsize=(img_size * scale / 100.0, img_size * scale / 100.0),
        dpi=100,
    )
    ax.imshow(bg_frame, extent=[0, img_size, img_size, 0])
    ax.set_xlim(0, img_size)
    ax.set_ylim(img_size, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    past_end = min(current_step + 1, n_total)
    future_start = past_end
    future_end = n_total

    norm = Normalize(vmin=0, vmax=max(n_total - 1, 1))

    def _draw_arm_segments(points_px: np.ndarray, base_indices: np.ndarray,
                           linestyle: str, alpha: float):
        for arm_idx in range(2):
            pts = points_px[:, arm_idx]
            valid = (pts[:, 0] >= 0) & (pts[:, 1] >= 0)
            if valid.sum() < 2:
                continue
            seg_pts = pts[valid]
            seg_idx = base_indices[valid]
            seg_colors = cmap(norm(seg_idx))
            seg_colors[:, 3] = alpha
            segments = np.stack([seg_pts[:-1], seg_pts[1:]], axis=1)
            lc = LineCollection(
                segments,
                colors=seg_colors[:-1],
                linewidths=2,
                linestyles=linestyle,
            )
            ax.add_collection(lc)

    if past_end > 0:
        idx = np.arange(past_end)
        _draw_arm_segments(all_px[:past_end], idx, linestyle='-', alpha=1.0)

    if future_end > future_start:
        idx = np.arange(future_start, future_end)
        _draw_arm_segments(
            all_px[future_start:future_end], idx, linestyle='--', alpha=0.35)

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

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    out = rgba[:, :, :3].copy()
    plt.close(fig)
    return out


# ===========================================================================
# Video composition — read LPB's MP4 + overlay base eef trajectory
# ===========================================================================

def _make_video(
    lpb_video_path: str,
    lpb_n_chunks: int,
    base_eef_traj: np.ndarray,
    env,
    render_obs_key: str,
    cfg: DictConfig,
    output_path: str,
):
    """Read LPB's MP4 frame-by-frame, overlay BASE policy's eef trajectory,
    write composited MP4.

    Frame ↔ chunk: frames_per_chunk = n_video_frames / lpb_n_chunks.
    """
    import imageio.v2 as imageio

    if not os.path.exists(lpb_video_path):
        print(f"[video] SKIP: LPB video not found at {lpb_video_path}")
        return

    reader = imageio.get_reader(lpb_video_path)
    n_video_frames = reader.count_frames()
    base_n = int(base_eef_traj.shape[0])

    if n_video_frames == 0:
        print(f"[video] SKIP: LPB video has 0 frames")
        reader.close()
        return
    if base_n == 0:
        print(f"[video] SKIP: base eef_traj has 0 samples")
        reader.close()
        return

    first_frame = reader.get_data(0)
    img_size = int(first_frame.shape[0])

    frames_per_chunk = n_video_frames / max(lpb_n_chunks, 1)
    print(f"[video] LPB video: {n_video_frames} frames, {img_size}x{img_size}px, "
          f"fps={cfg.video_fps}")
    print(f"[video] LPB ran {lpb_n_chunks} chunks; frames_per_chunk≈{frames_per_chunk:.2f}")
    print(f"[video] base has {base_n} eef samples; overlaying as '{output_path}'")

    writer = imageio.get_writer(
        output_path,
        fps=cfg.video_fps,
        codec='libx264',
        quality=8,
        macro_block_size=1,
    )

    try:
        for f in range(n_video_frames):
            bg = reader.get_data(f)
            lpb_chunk_idx = min(int(f / frames_per_chunk), lpb_n_chunks - 1)
            base_chunk_idx = min(lpb_chunk_idx, base_n - 1)

            panel = _render_video_frame(
                bg_frame=bg,
                overlay_eef_world=base_eef_traj,
                env=env,
                camera_name=render_obs_key,
                current_step=base_chunk_idx,
                total_overlay_steps=base_n,
                scale=cfg.video_frame_scale,
                overlay_method='base',
                img_size=img_size,
            )
            writer.append_data(panel)

            if f % 50 == 0:
                print(f"  [video] frame {f}/{n_video_frames} "
                      f"(lpb_chunk={lpb_chunk_idx}, base_chunk={base_chunk_idx})")
    finally:
        writer.close()
        reader.close()

    print(f"[video] wrote {output_path} ({n_video_frames} frames)")


# ===========================================================================
# Divergent seed search — find a seed where LPB succeeds AND base fails
# ===========================================================================

def _try_one_seed(cfg, cfg_task, base_policy, lpb_policy, abs_action: bool,
                  output_dir: str, seed: int) -> Optional[Dict]:
    """Try one seed. Returns dict with everything needed for visualization if
    LPB succeeded AND base failed; returns None otherwise.

    Strategy: run base first (cheaper, no guidance). If base succeeded, skip
    (we need base to fail). Only run LPB if base failed.

    Both rollouts record videos so the user can inspect the actual trajectories
    for the chosen seed. On non-divergent seeds, both video files are deleted
    to keep the output dir clean.

    On success, the LPB env is RETAINED (caller needs it for camera projection
    during Phase 5). On failure, envs + videos are cleaned up.
    """
    cfg.compare_seed = seed  # _run_rollout reads this

    # ---- base rollout (with video) ----
    base_env, _, _, _, _ = _build_env(cfg_task, cfg)
    base_video_path = os.path.join(output_dir, f'base_driver_s{seed}.mp4')
    base_result = _run_rollout(
        base_policy, base_env, cfg, use_guidance=False,
        label=f'base_s{seed}', abs_action=abs_action, video_path=base_video_path,
        extra_render_keys=EXTRA_RENDER_KEYS,
    )
    del base_env
    torch.cuda.empty_cache()

    if base_result['success']:
        print(f"[search:seed={seed}] base SUCCEEDED -> skip (need base to fail)")
        if os.path.exists(base_video_path):
            os.remove(base_video_path)
        return None

    # ---- base failed; try LPB (with video) ----
    lpb_env, _, _, _, _ = _build_env(cfg_task, cfg)
    lpb_video_path = os.path.join(output_dir, f'lpb_driver_s{seed}.mp4')
    lpb_result = _run_rollout(
        lpb_policy, lpb_env, cfg, use_guidance=True,
        label=f'lpb_s{seed}', abs_action=abs_action, video_path=lpb_video_path,
        extra_render_keys=EXTRA_RENDER_KEYS,
    )

    if lpb_result['success']:
        print(f"[search:seed={seed}] LPB SUCCEEDED + base FAILED -> DIVERGENT!")
        return {
            'seed': seed,
            'base_result': base_result,
            'lpb_result': lpb_result,
            'lpb_env': lpb_env,           # retained for Phase 5 projection
            'lpb_video_path': lpb_video_path,
            'base_video_path': base_video_path,
        }

    # both failed; cleanup both videos
    print(f"[search:seed={seed}] both failed -> continue searching")
    del lpb_env
    torch.cuda.empty_cache()
    for p in (lpb_video_path, base_video_path):
        if os.path.exists(p):
            os.remove(p)
    return None


# ===========================================================================
# Main
# ===========================================================================

@hydra.main(config_path=_CONFIG_DIR, config_name="compare_transport")
def main(cfg: DictConfig):
    print("=" * 60)
    print("compare_viz — LPB vs Base Policy")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    pathlib.Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, 'compare_config.yaml'))

    # Always need base policy + cfg_task to discover env flags
    print("\n[init] Loading base policy...")
    base_policy, cfg_task = _load_base_policy(cfg)

    find_divergent = bool(cfg.get('find_divergent_seed', False))

    if find_divergent:
        # ---- divergent-seed search mode ----
        print("\n[init] Loading LPB policy (for search)...")
        lpb_policy, _ = _load_lpb_policy(cfg)

        # Probe one env build to discover abs_action etc. cheaply (we don't
        # actually run a rollout here; we just read cfg_task flags).
        abs_action = getattr(cfg_task.task.env_runner, 'abs_action', False)
        render_obs_key = cfg_task.task.env_runner.render_obs_key
        print(f"[init] render_obs_key={render_obs_key}, abs_action={abs_action}")

        max_search = int(cfg.get('max_seed_search', 50))
        print(f"\n[search] find_divergent_seed=True, start_seed={cfg.compare_seed}, "
              f"max_attempts={max_search}")

        found = None
        for offset in range(max_search):
            seed = cfg.compare_seed + offset
            print(f"\n[search] === attempt {offset+1}/{max_search}, seed={seed} ===")
            found = _try_one_seed(
                cfg, cfg_task, base_policy, lpb_policy,
                abs_action=abs_action, output_dir=cfg.output_dir, seed=seed,
            )
            if found is not None:
                break

        # Free both policies — visualization only needs env + results.
        del base_policy, lpb_policy
        torch.cuda.empty_cache()

        if found is None:
            print(f"\n[search] FAILED: no divergent seed found in {max_search} attempts.")
            print(f"[search] Try raising max_seed_search or compare_seed.")
            return

        seed_used = found['seed']
        base_result = found['base_result']
        lpb_result = found['lpb_result']
        lpb_env = found['lpb_env']
        print(f"\n[search] === FOUND divergent seed: {seed_used} ===")

    else:
        # ---- direct mode: use cfg.compare_seed as-is ----
        abs_action = getattr(cfg_task.task.env_runner, 'abs_action', False)
        render_obs_key = cfg_task.task.env_runner.render_obs_key
        seed_used = cfg.compare_seed
        print(f"\n[run] find_divergent_seed=False; using seed={seed_used} directly")
        print(f"[run] render_obs_key={render_obs_key}, abs_action={abs_action}")

        base_env, _, _, _, _ = _build_env(cfg_task, cfg)
        base_video_path = os.path.join(cfg.output_dir, 'base_driver.mp4')
        base_result = _run_rollout(
            base_policy, base_env, cfg, use_guidance=False,
            label='base', abs_action=abs_action, video_path=base_video_path,
            extra_render_keys=EXTRA_RENDER_KEYS,
        )
        del base_policy, base_env
        torch.cuda.empty_cache()

        print("\n[run] Loading LPB policy...")
        lpb_policy, _ = _load_lpb_policy(cfg)
        lpb_env, _, _, _, _ = _build_env(cfg_task, cfg)
        lpb_video_path = os.path.join(cfg.output_dir, 'lpb_driver.mp4')
        lpb_result = _run_rollout(
            lpb_policy, lpb_env, cfg, use_guidance=True,
            label='lpb', abs_action=abs_action, video_path=lpb_video_path,
            extra_render_keys=EXTRA_RENDER_KEYS,
        )
        del lpb_policy
        torch.cuda.empty_cache()

    # ===========================================================================
    # Phase 3: sanity check projection
    # ===========================================================================
    print("\n[Phase 3] Projection sanity check...")
    if base_result['n_samples'] > 0:
        # Probe img_size from LPB video if available, otherwise assume 140.
        img_size_probe = 140
        eef0 = base_result['eef_traj'][0, 0]
        px = _project_to_camera(eef0.reshape(1, 3), lpb_env, render_obs_key,
                                img_size=img_size_probe)[0]
        print(f"  world {eef0} -> px {px} (camera={render_obs_key}, assumed {img_size_probe}px)")
        if not (0 <= px[0] < img_size_probe and 0 <= px[1] < img_size_probe):
            print(f"  WARN: eef0 projects out of frame; overlay will be sparse.")
        else:
            print(f"  OK")
    else:
        print("  SKIP: base rollout produced 0 samples")

    # ===========================================================================
    # Phase 4: save rollout data
    # ===========================================================================
    print("\n[Phase 4] Saving rollout data...")
    np.savez_compressed(
        os.path.join(cfg.output_dir, 'rollout_data.npz'),
        seed_used=seed_used,
        find_divergent_seed=find_divergent,
        base_eef=base_result['eef_traj'],
        base_actions=base_result['actions'],
        base_success=base_result['success'],
        base_success_step=base_result['success_step'],
        lpb_eef=lpb_result['eef_traj'],
        lpb_actions=lpb_result['actions'],
        lpb_success=lpb_result['success'],
        lpb_success_step=lpb_result['success_step'],
    )

    # ===========================================================================
    # Phase 5: compose video (LPB driving + base overlay) — main render_obs_key
    # ===========================================================================
    print("\n[Phase 5] Composing video: LPB driving + base overlay...")
    lpb_video = lpb_result.get('video_path')
    if not lpb_video or not os.path.exists(lpb_video):
        print(f"[Phase 5] SKIP: LPB did not produce a video (video_path={lpb_video})")
    else:
        _make_video(
            lpb_video_path=lpb_video,
            lpb_n_chunks=lpb_result['n_steps'],
            base_eef_traj=base_result['eef_traj'],
            env=lpb_env,
            render_obs_key=render_obs_key,
            cfg=cfg,
            output_path=os.path.join(cfg.output_dir, 'lpb_driving_base_overlay.mp4'),
        )

    # ===========================================================================
    # Phase 5b: extra-camera videos + overlays (one set per EXTRA_RENDER_KEYS)
    # ===========================================================================
    # For each extra camera, render 3 files:
    #   base_driver_<cam>.mp4                  — raw base rollout from <cam>
    #   lpb_driver_<cam>.mp4                   — raw LPB rollout from <cam>
    #   lpb_driving_base_overlay_<cam>.mp4     — LPB <cam> video + base eef overlay
    # (mirrors the 3-file main-video output, just from a different angle).
    # Naming follows the main set: divergent-search mode keeps the _s{seed}
    # suffix on the main driver files; extra files always use the plain name
    # since the seed is already recorded in rollout_data.npz.
    for cam_key in EXTRA_RENDER_KEYS:
        cam_short = cam_key.replace('_image', '')
        print(f"\n[Phase 5b:{cam_short}] saving extra-camera videos...")

        base_frames = base_result.get('extra_frames', {}).get(cam_key, [])
        lpb_frames = lpb_result.get('extra_frames', {}).get(cam_key, [])

        if base_frames:
            _save_frames_to_mp4(
                base_frames,
                os.path.join(cfg.output_dir, f'base_driver_{cam_short}.mp4'),
                cfg.video_fps,
            )
        else:
            print(f"[Phase 5b:{cam_short}] no base frames, skipping base_driver_{cam_short}.mp4")

        if lpb_frames:
            lpb_cam_path = os.path.join(cfg.output_dir, f'lpb_driver_{cam_short}.mp4')
            _save_frames_to_mp4(lpb_frames, lpb_cam_path, cfg.video_fps)

            # Overlay: same _make_video call as Phase 5, but with this camera's
            # video as background and projection using this camera's params.
            _make_video(
                lpb_video_path=lpb_cam_path,
                lpb_n_chunks=lpb_result['n_steps'],
                base_eef_traj=base_result['eef_traj'],
                env=lpb_env,
                render_obs_key=cam_key,
                cfg=cfg,
                output_path=os.path.join(cfg.output_dir,
                                         f'lpb_driving_base_overlay_{cam_short}.mp4'),
            )
        else:
            print(f"[Phase 5b:{cam_short}] no LPB frames, skipping "
                  f"lpb_driver_{cam_short}.mp4 and overlay")

    # ===========================================================================
    # Summary
    # ===========================================================================
    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Output dir: {cfg.output_dir}")
    print(f"  Seed used: {seed_used}  (find_divergent_seed={find_divergent})")
    print(f"  - base_driver*.mp4 (raw base rollout, {base_result['n_steps']} chunks)")
    print(f"  - lpb_driver*.mp4 (raw LPB rollout, {lpb_result['n_steps']} chunks)")
    print(f"  - lpb_driving_base_overlay.mp4 (LPB video + base eef overlay)")
    for cam_key in EXTRA_RENDER_KEYS:
        cam_short = cam_key.replace('_image', '')
        print(f"  - base_driver_{cam_short}.mp4 / lpb_driver_{cam_short}.mp4 / "
              f"lpb_driving_base_overlay_{cam_short}.mp4  (extra angle)")
    print(f"  - rollout_data.npz")
    print(f"  LPB  success: {lpb_result['success']} @ step {lpb_result['success_step']}")
    print(f"  Base success: {base_result['success']} @ step {base_result['success_step']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
