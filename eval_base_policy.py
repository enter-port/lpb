"""
Evaluate the base diffusion policy WITHOUT the dynamics model / guidance.

This is a stripped-down version of eval_test_time_optimization.py:
  - Loads the base policy checkpoint
  - Does NOT initialize a planner (no OOD detection, no gradient guidance)
  - Uses the PARALLEL RobomimicImageRunner, which calls policy.predict_action()
    (the sequential runner calls predict_action_dyn_guided() and would crash
     without a planner)
  - Reports the test success rate as test/mean_score

The evaluation settings (n_test, test_start_seed, n_action_steps) default to
the same values as eval_transport.yaml so the result is directly comparable
with the LPB (test-time optimization) result.

Usage:
    python eval_base_policy.py --config-name=eval_base_transport

    # override from the command line:
    python eval_base_policy.py --config-name=eval_base_transport \
        policy_checkpoint=path/to/xxx.ckpt \
        output_dir=data/my_base_test n_test=50 n_envs=14
"""
import sys
import os
import pathlib

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import dill
import wandb
import json

from diffusion_policy.workspace.base_workspace import BaseWorkspace

# line-buffer stdout/stderr so progress shows up in real time
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


@hydra.main(config_path="dyn_model/conf/planner", config_name="eval_base_transport")
def main(cfg: DictConfig):
    output_dir = cfg.output_dir

    if os.path.exists(output_dir):
        confirm = input(f"Output path {output_dir} already exists! Overwrite? (y/N): ")
        if confirm.lower() != 'y':
            sys.exit(1)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # save the resolved config
    config_save_path = os.path.join(output_dir, 'eval_config.yaml')
    OmegaConf.save(config=cfg, f=config_save_path)
    print(f"Configuration saved to {config_save_path}")

    # ------------------------------------------------------------------
    #  Load policy checkpoint
    # ------------------------------------------------------------------
    print(f"Loading policy checkpoint: {cfg.policy_checkpoint}")
    with open(cfg.policy_checkpoint, 'rb') as f:
        payload = torch.load(f, pickle_module=dill)

    cfg_task = payload['cfg']

    # ---- override fields that must match this eval run ----
    cfg_task.n_action_steps = cfg.n_action_steps
    cfg_task.policy.n_action_steps = cfg.n_action_steps
    cfg_task.task.env_runner.n_action_steps = cfg.n_action_steps

    cfg_task.task.env_runner.n_test = cfg.n_test
    cfg_task.task.env_runner.n_test_vis = min(cfg.n_test, 6)  # cap rendered videos
    cfg_task.task.env_runner.n_train = 0
    cfg_task.task.env_runner.n_train_vis = 0
    cfg_task.task.env_runner.test_start_seed = cfg.test_start_seed

    # CRITICAL (CLAUDE.md §6.1): the .ckpt stores the ORIGINAL author's
    # absolute paths inside payload['cfg']. Override them with local paths
    # so the runner can find the dataset for env metadata.
    cfg_task.task.env_runner.dataset_path = cfg.demo_dataset_path
    cfg_task.task.dataset.dataset_path = cfg.demo_dataset_path

    # Use the parallel runner (calls predict_action, no planner needed).
    cfg_task.task.env_runner._target_ = cfg.env_runner_target
    # cap parallel envs to avoid GPU OOM during offscreen rendering
    OmegaConf.update(cfg_task, 'task.env_runner.n_envs',
                     cfg.n_envs, force_add=True)

    # ------------------------------------------------------------------
    #  Build workspace + load weights
    # ------------------------------------------------------------------
    cls = hydra.utils.get_class(cfg_task._target_)
    workspace = cls(cfg_task, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg_task.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(cfg.device)
    policy.to(device)
    policy.eval()

    # normalizer lives next to the checkpoints/ folder
    normalizer_dir = os.path.dirname(os.path.dirname(cfg.policy_checkpoint))
    normalizer_path = os.path.join(normalizer_dir, 'normalizer.pth')
    policy.normalizer.load_state_dict(torch.load(normalizer_path))
    policy.normalizer.to(device)

    # NOTE: deliberately do NOT call policy.initialize_planner(...).
    # Without a planner, predict_action() runs plain DDPM sampling.

    # ------------------------------------------------------------------
    #  Run evaluation
    # ------------------------------------------------------------------
    env_runner = hydra.utils.instantiate(
        cfg_task.task.env_runner,
        output_dir=output_dir,
    )

    runner_log = env_runner.run(policy)

    # ------------------------------------------------------------------
    #  Save results
    # ------------------------------------------------------------------
    results = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            results[key] = value._path
        else:
            results[key] = value

    results_path = os.path.join(output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True, default=str)

    # headline metric
    test_score = results.get('test/mean_score', None)
    print(f"\n===== Base Policy Evaluation (no dynamics model) =====")
    print(f"  Checkpoint     : {cfg.policy_checkpoint}")
    print(f"  n_test         : {cfg.n_test}")
    print(f"  test_start_seed: {cfg.test_start_seed}")
    print(f"  n_action_steps : {cfg.n_action_steps}")
    if test_score is not None:
        print(f"  test/mean_score (success rate): {test_score:.4f}")
    else:
        print(f"  (test/mean_score not found in runner log)")
        print(f"  available keys: {sorted(results.keys())}")
    print(f"  Full results   : {results_path}")
    print(f"=======================================================")


if __name__ == '__main__':
    main()
