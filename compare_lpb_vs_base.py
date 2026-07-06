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
