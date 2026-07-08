"""
Compare LPB vs Base Policy — generate ONE visualization video.

The video shows LPB (base + dynamics model + classifier guidance) driving the
env from a seed-initialized state, recorded at full env fps via the standard
`VideoRecordingWrapper` (same path as `eval_test_time_optimization.py`). The
BASE policy's eef trajectory — obtained from a second rollout started from the
SAME seed but run WITHOUT guidance — is projected onto the video as a colored
polyline (past = solid, future = dashed).

Pipeline:
    1. Build env via the standard `create_env` factory.
    2. LPB rollout:  set file_path -> env records MP4 at full fps; sample eef
                     every chunk for diagnostics.
    3. Base rollout: separate env instance, same seed, no video recording,
                     sample eef every chunk.
    4. Sanity-check the camera projection.
    5. Read LPB's MP4 frame-by-frame, overlay base's eef trajectory, write MP4.

See docs/superpowers/specs/2026-07-06-lpb-vs-base-comparison-design.md

Usage:
    python compare_lpb_vs_base.py --config-name=compare_transport
    python compare_lpb_vs_base.py --config-name=compare_transport compare_seed=100001
"""
import sys
import os
import json
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


# ===========================================================================
# Policy loaders (unchanged from previous version)
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

def _build_env(cfg_task, cfg: DictConfig) -> Tuple[object, str, int, int]:
    """Build env via the standard `create_env` factory from
    `robomimic_image_sequential_runner`.

    Does NOT seed or reset — the caller does that AFTER setting
    `env.env.file_path` so that VideoRecordingWrapper records.

    Returns (env, render_obs_key, n_action_steps, steps_per_render).
    Wrapper stack (outside-in): MultiStepWrapper → VideoRecordingWrapper
    → RobomimicImageWrapper → EnvRobosuite. So `env.env` IS the
    VideoRecordingWrapper (where `file_path` lives).
    """
    import robomimic.utils.file_utils as FileUtils

    env_runner_cfg = cfg_task.task.env_runner
    dataset_path = os.path.expanduser(env_runner_cfg.dataset_path)

    # Read env_meta and apply abs_action controller tweak (same as sequential
    # runner __init__: control_delta=False makes the env consume rotation_6d
    # actions directly).
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta['env_kwargs']['use_object_obs'] = False
    abs_action = getattr(env_runner_cfg, 'abs_action', True)
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

    # Robosuite hard reset causes excessive memory consumption; disable
    # (same as sequential runner __init__).
    # create_env already does env.env.hard_reset=False on the EnvRobosuite,
    # so this is just informational.
    steps_per_render = max(20 // fps, 1)

    print(f"[env] Built env via create_env. render_obs_key={render_obs_key}, "
          f"n_action_steps={n_action_steps}, fps={fps}, steps_per_render={steps_per_render}, "
          f"max_steps={max_steps}")
    return env, render_obs_key, n_action_steps, steps_per_render


# ===========================================================================
# Rollout — mirrors SequentialRobomimicImageRunner.run's inner loop
# ===========================================================================

def _run_rollout(policy, env, cfg, use_guidance: bool, label: str,
                 video_path: Optional[str] = None) -> Dict:
    """Run a rollout. If `video_path` is set, the env records at full fps to
    that MP4 via VideoRecordingWrapper (file_path must be assigned BEFORE
    reset, since reset() stops any prior recorder).

    `env` MUST already be built. We re-seed with cfg.compare_seed here so the
    same env instance can be reused for both rollouts.

    The env is a MultiStepWrapper, so one env.step() call consumes
    `n_action_steps` underlying robosuite steps with one action chunk
    `(n_action_steps, action_dim)`; we treat that as one "chunk" / decision
    step. cfg.max_steps caps the number of chunks.

    Returns a dict with keys:
        eef_traj      (T, 2, 3) float  - sampled robot0/robot1 eef xyz per chunk
        actions       (T, action_dim_env) float - env-space action chunks per chunk
        success       bool
        success_step  int  (-1 if never succeeded)
        n_steps       int  - total env.step() chunks executed (== T unless break)
        n_samples     int  - == eef_traj.shape[0] == T
        video_path    Optional[str] - the recorded MP4 path (None if not recording)
    """
    from diffusion_policy.model.common.rotation_transformer import RotationTransformer

    # ---- abs_action / rotation transformer (mirrors sequential runner) ----
    abs_action = True
    rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d') if abs_action else None

    def undo_transform_action(action: np.ndarray) -> np.ndarray:
        """rotation_6d (...,20) -> axis_angle (...,14). Mirrors sequential runner."""
        raw_shape = action.shape
        if raw_shape[-1] == 20:
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

    # ---- set video file_path BEFORE reset ----
    # env.env is VideoRecordingWrapper (MultiStepWrapper.env).
    # If file_path is None, no frames are recorded.
    env.env.file_path = video_path
    if video_path is not None:
        print(f"[rollout:{label}] env will record to {video_path}")

    # ---- seed + reset (deterministic re-roll) ----
    env.seed(cfg.compare_seed)
    obs = env.reset()
    policy.reset()

    eef_list: List[np.ndarray] = []
    actions_list: List[np.ndarray] = []
    success = False
    success_step = -1
    n_chunks = 0

    print(f"[rollout:{label}] starting. use_guidance={use_guidance}, "
          f"max_chunks={cfg.max_steps}")

    for chunk_idx in range(cfg.max_steps):
        # ---- build obs dict (B=1, To, ...) ----
        np_obs_dict = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
        obs_dict = {k: torch.from_numpy(v).to(policy.device) for k, v in np_obs_dict.items()}

        # ---- policy forward ----
        # predict_action_dyn_guided uses grad internally (for classifier
        # guidance), so we do NOT wrap it in torch.no_grad(). predict_action
        # is pure DDPM sampling and benefits from no_grad.
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
        env_action = undo_transform_action(action) if abs_action else action
        obs, reward, done, info = env.step(env_action)
        n_chunks += 1

        # ---- sample eef + action EVERY chunk (smoothest overlay trajectory) ----
        eef = np.stack([
            obs['robot0_eef_pos'][-1].copy(),
            obs['robot1_eef_pos'][-1].copy(),
        ])  # (2, 3)
        eef_list.append(eef)
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

    # ---- finalize video ----
    # env.render() on MultiStepWrapper delegates down to VideoRecordingWrapper.render,
    # which stops the recorder (flushing the MP4) and returns the file_path.
    final_video_path = env.render() if video_path is not None else None

    # ---- pack outputs ----
    eef_traj = np.stack(eef_list) if eef_list else np.zeros((0, 2, 3), dtype=np.float32)
    actions = np.stack(actions_list) if actions_list else np.zeros((0, 14), dtype=np.float32)

    print(f"[rollout:{label}] done. n_chunks={n_chunks}, sampled_points={len(eef_list)}, "
          f"success={success}, success_step={success_step}")
    return {
        'eef_traj': eef_traj,
        'actions': actions,
        'success': success,
        'success_step': success_step,
        'n_steps': n_chunks,
        'n_samples': len(eef_list),
        'video_path': final_video_path,
    }


# ===========================================================================
# Camera projection (3D world → 2D pixel)
# ===========================================================================

def _get_camera_params(env, camera_name: str) -> Dict:
    """Extract camera position (3,), orientation (3, 3), fovy (scalar) from the
    MuJoCo sim underlying the wrapper stack."""
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
                       img_size: int = 140) -> np.ndarray:
    """Project 3D world-coords (N, 3) to 2D pixel coords (N, 2) for `camera_name`.

    Convention notes:
      - MuJoCo `cam_xmat` columns are the cam axes in world (cam→world rotation).
      - MuJoCo cameras use OpenGL convention (looks along -Z, +Y up, +X right);
        visible points have z_cam < 0, so we project with `depth = -z_cam`.
      - For NumPy row vectors, right-multiplying by `mat` applies `mat.T`
        (= world→cam) — NOT `mat.T` which would be wrong.
    """
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N, 3); got {pts.shape}.")

    params = _get_camera_params(env, camera_name)
    pos = np.asarray(params['pos'], dtype=np.float64).reshape(3)
    mat = np.asarray(params['mat'], dtype=np.float64).reshape(3, 3)
    fovy = float(params['fovy'])

    p_rel = pts - pos[None, :]
    p_cam = p_rel @ mat  # apply world→cam to each row

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
    img_size: int = 140,
) -> np.ndarray:
    """Compose one video frame: background image + overlay eef trajectory.

    Past trajectory = overlay[:current_step+1]   drawn solid, full alpha.
    Future = overlay[current_step+1:]            drawn dashed, alpha=0.35.
    Each arm gets a distinct marker at `current_step` ('o' arm0, 's' arm1).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    if overlay_method == 'lpb':
        cmap_name = 'plasma'
    elif overlay_method == 'base':
        cmap_name = 'viridis'
    else:
        raise ValueError(f"overlay_method must be 'lpb' or 'base'; got {overlay_method!r}")
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
    """Read LPB's recorded MP4 frame-by-frame, overlay BASE policy's eef
    trajectory (projected via `env`'s camera params), write composited MP4.

    Frame ↔ chunk mapping: LPB's MP4 has `n_video_frames` frames recorded at
    `steps_per_render` env-steps per frame. LPB ran `lpb_n_chunks` chunks of
    `n_action_steps` env-steps each. So:
        frames_per_chunk = n_video_frames / lpb_n_chunks
        lpb_chunk_idx(f) = int(f / frames_per_chunk)
    The overlay at frame f uses base_eef_traj[min(lpb_chunk_idx, base_n-1)]:
    "where was base's arm at the same moment in its own rollout".
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

    # Read first frame to get img_size (Transport: 140x140).
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
            # Map this video frame to LPB's chunk index.
            lpb_chunk_idx = min(int(f / frames_per_chunk), lpb_n_chunks - 1)
            # Base's chunk index at the same wall-clock moment: same chunk_idx,
            # capped at base_n - 1 (if base ran fewer chunks).
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
# Main
# ===========================================================================

@hydra.main(config_path="dyn_model/conf/planner", config_name="compare_transport")
def main(cfg: DictConfig):
    print("=" * 60)
    print("LPB vs Base Policy Comparison")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    pathlib.Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, 'compare_config.yaml'))

    # ---- Phase 1: LPB rollout (records MP4 via standard VideoRecordingWrapper) ----
    print("\n[Phase 1] Loading LPB policy + running LPB rollout (records video)...")
    lpb_policy, cfg_task = _load_lpb_policy(cfg)
    lpb_env, render_obs_key, n_action_steps, steps_per_render = _build_env(cfg_task, cfg)
    lpb_video_path = os.path.join(cfg.output_dir, 'lpb_driver.mp4')
    lpb_result = _run_rollout(
        lpb_policy, lpb_env, cfg, use_guidance=True, label='lpb',
        video_path=lpb_video_path,
    )
    del lpb_policy
    torch.cuda.empty_cache()

    # ---- Phase 2: base rollout (same seed, no video, only eef sampling) ----
    print("\n[Phase 2] Loading base policy + running base rollout (eef only)...")
    base_policy, _ = _load_base_policy(cfg)
    base_env, _, _, _ = _build_env(cfg_task, cfg)
    base_result = _run_rollout(
        base_policy, base_env, cfg, use_guidance=False, label='base',
        video_path=None,
    )
    del base_policy
    torch.cuda.empty_cache()

    # ---- Phase 3: sanity check projection ----
    # Project base's first eef sample to render_obs_key; should land in-frame.
    print("\n[Phase 3] Projection sanity check...")
    if base_result['n_samples'] > 0:
        eef0 = base_result['eef_traj'][0, 0]
        px = _project_to_camera(eef0.reshape(1, 3), lpb_env, render_obs_key)[0]
        print(f"  world {eef0} -> px {px} (camera={render_obs_key})")
        # Soft warning (not hard assert) — out-of-frame projection of eef0
        # doesn't necessarily break the video; the renderer skips invalid pts.
        if not (0 <= px[0] < 140 and 0 <= px[1] < 140):
            print(f"  WARN: eef0 projects out of frame; overlay will be sparse.")
        else:
            print(f"  OK")
    else:
        print("  SKIP: base rollout produced 0 samples")

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

    # ---- Phase 5: compose video (LPB driving + base overlay) ----
    print("\n[Phase 5] Composing video: LPB driving + base overlay...")
    if lpb_result['video_path'] is None or not os.path.exists(lpb_result['video_path']):
        print(f"[Phase 5] SKIP: LPB did not produce a video "
              f"(video_path={lpb_result['video_path']})")
    else:
        _make_video(
            lpb_video_path=lpb_result['video_path'],
            lpb_n_chunks=lpb_result['n_steps'],
            base_eef_traj=base_result['eef_traj'],
            env=lpb_env,  # use LPB's env for camera projection (fixed cameras)
            render_obs_key=render_obs_key,
            cfg=cfg,
            output_path=os.path.join(cfg.output_dir, 'lpb_driving_base_overlay.mp4'),
        )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Output dir: {cfg.output_dir}")
    print(f"  - lpb_driver.mp4 (raw LPB rollout, {lpb_result['n_steps']} chunks)")
    print(f"  - lpb_driving_base_overlay.mp4 (LPB video + base eef overlay)")
    print(f"  - rollout_data.npz")
    print(f"  LPB  success: {lpb_result['success']} @ step {lpb_result['success_step']}")
    print(f"  Base success: {base_result['success']} @ step {base_result['success_step']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
