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


@hydra.main(config_path="dyn_model/conf/planner", config_name="compare_transport")
def main(cfg: DictConfig):
    print("=" * 60)
    print("LPB vs Base Policy Comparison")
    print("=" * 60)
    # TODO: wire everything together in Task 9
    print("Config loaded:")
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    main()
