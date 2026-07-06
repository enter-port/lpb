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
