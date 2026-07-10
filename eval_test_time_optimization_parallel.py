"""
Evaluate LPB (base + dynamics + classifier guidance) with PARALLEL env rollout.

This is a parallel-envs version of eval_test_time_optimization.py, following
the structural style of eval_base_policy.py. The algorithm is identical to
LPB; only the env runner is swapped from SequentialRobomimicImageRunner to
the parallel RobomimicImageRunner (AsyncVectorEnv).

Reuses the SAME config as the sequential version (eval_transport.yaml); the
runner-target override and the env count are baked into this file (see
N_ENVS below) so you can tune parallelism without editing the shared config.

Why an adapter is needed:
  - The parallel runner calls `policy.predict_action(obs_dict)` with obs
    batched across envs: shapes (n_envs, n_obs_steps, ...).
  - LPB's classifier-guidance path is `predict_action_dyn_guided`, which was
    written for batch=1 (per-env gradient guidance through the planner).
    Batching it across envs would tangle each env's guidance gradient with
    its neighbours'.
  - Solution: wrap the LPB policy in `ParallelGuidedPolicyAdapter`. The
    wrapper exposes the `predict_action(obs_dict)` API the parallel runner
    expects, and internally calls `predict_action_dyn_guided` once per env
    (loop), then concatenates the actions back into a (n_envs, T_a, A) batch.

Result: env STEPPING stays parallel (AsyncVectorEnv), only the guided
diffusion sampling within each chunk is sequential across envs. Net speedup
vs sequential depends on whether env stepping or policy compute dominates;
for DDPM (100 steps) policy compute usually dominates, so this is mainly
useful when you have many envs and want to overlap env steps with policy
compute of the next chunk.

Usage:
    python eval_test_time_optimization_parallel.py --config-name=eval_transport

    # override from CLI (any field of eval_transport.yaml):
    python eval_test_time_optimization_parallel.py --config-name=eval_transport \\
        policy_checkpoint=path/to/x.ckpt n_test=50 \\
        guidance_scale=0.2 threshold=2.8
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


# ===========================================================================
# Tunable parallel-eval knobs — edit here, no need to touch the yaml
# ===========================================================================

# Number of parallel envs (AsyncVectorEnv workers). Transport is heavy
# (4 image obs per step + offscreen EGL rendering); 7-14 is a reasonable
# range on a 24GB GPU. Lower this if you hit GPU OOM during rollout.
N_ENVS = 7

# Override the runner class. The original eval_transport.yaml points at the
# SEQUENTIAL runner; we force the PARALLEL one here so we can reuse that
# config unchanged.
PARALLEL_RUNNER_TARGET = 'diffusion_policy.env_runner.robomimic_image_runner.RobomimicImageRunner'


# ===========================================================================
# Adapter: make LPB's per-env guided sampling usable by the parallel runner
# ===========================================================================

class ParallelGuidedPolicyAdapter:
    """Wrap an LPB policy so `predict_action(batched_obs)` — which the parallel
    RobomimicImageRunner calls — routes to `predict_action_dyn_guided` once
    per env.

    Deliberately does NOT inherit from BaseImagePolicy / nn.Module: the
    parallel runner only uses `.device`, `.dtype`, `.reset()`, and
    `.predict_action(...)`, none of which require nn.Module machinery.
    Inheriting from nn.Module would force us to either call
    `super().__init__()` (and risk re-creating the obs encoder / noise
    predictor) or skip it (which breaks because nn.Module forbids assigning
    a submodule before its own __init__ — that's exactly the original error).
    A plain-Python adapter sidesteps both.

    Forwards `.device` / `.dtype` / `.reset()` to the wrapped LPB policy.

    Note on planner state: the planner increments `self.idx` / `self.timestep`
    across calls, but those are diagnostic counters (used only for "print
    once" guards) — they do NOT affect guidance math. So calling
    `predict_action_dyn_guided` multiple times in a row across envs is safe.
    """

    def __init__(self, lpb_policy):
        self.policy = lpb_policy

    # ---- delegated properties ----
    @property
    def device(self):
        return self.policy.device

    @property
    def dtype(self):
        return self.policy.dtype

    def reset(self):
        self.policy.reset()

    # ---- the main routing ----
    def predict_action(self, obs_dict: dict) -> dict:
        """obs_dict values: (n_envs, n_obs_steps, ...). Returns
        {'action': (n_envs, n_action_steps, action_dim)}.

        Internal loop is sequential across envs. Each call to
        `predict_action_dyn_guided` runs full DDPM + classifier guidance for
        one env's obs, so this is the per-chunk bottleneck.
        """
        # Find batch dim from any obs tensor (they're all batched the same).
        first_key = next(iter(obs_dict))
        n_envs = obs_dict[first_key].shape[0]

        actions = []
        for i in range(n_envs):
            # Keep batch dim of 1 — predict_action_dyn_guided expects (B, To, ...).
            obs_i = {k: v[i:i + 1] for k, v in obs_dict.items()}
            action_i = self.policy.predict_action_dyn_guided(obs_i)
            actions.append(action_i['action'])
        action = torch.cat(actions, dim=0)  # (n_envs, n_action_steps, action_dim)
        return {'action': action}

    # The parallel runner does NOT call predict_action_dyn_guided, so we don't
    # need to expose it on the wrapper. Keep it for completeness if anyone
    # wants to drive the wrapper sequentially.
    def predict_action_dyn_guided(self, obs_dict: dict) -> dict:
        return self.policy.predict_action_dyn_guided(obs_dict)


# ===========================================================================
# Main — parallel-env LPB eval (reuses eval_transport.yaml as config)
# ===========================================================================

@hydra.main(config_path="dyn_model/conf/planner", config_name="eval_transport")
def main(cfg: DictConfig):
    output_dir = cfg.output_dir

    if os.path.exists(output_dir):
        confirm = input(f"Output path {output_dir} already exists! Overwrite? (y/N): ")
        if confirm.lower() != 'y':
            sys.exit(1)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    config_save_path = os.path.join(output_dir, 'eval_config.yaml')
    OmegaConf.save(config=cfg, f=config_save_path)
    print(f"Configuration saved to {config_save_path}")
    print(f"[parallel] N_ENVS={N_ENVS}, runner={PARALLEL_RUNNER_TARGET}")

    # ------------------------------------------------------------------
    #  Load policy checkpoint
    # ------------------------------------------------------------------
    print(f"Loading policy checkpoint: {cfg.policy_checkpoint}")
    with open(cfg.policy_checkpoint, 'rb') as f:
        payload = torch.load(f, pickle_module=dill)

    cfg_task = payload['cfg']

    # ---- overrides that must match this eval run (mirror eval_base_policy) ----
    cfg_task.n_action_steps = cfg.n_action_steps
    cfg_task.policy.n_action_steps = cfg.n_action_steps
    cfg_task.task.env_runner.n_action_steps = cfg.n_action_steps

    cfg_task.task.env_runner.n_test = cfg.n_test
    cfg_task.task.env_runner.n_test_vis = min(cfg.n_test, 6)
    cfg_task.task.env_runner.n_train = 0
    cfg_task.task.env_runner.n_train_vis = 0
    cfg_task.task.env_runner.test_start_seed = cfg.test_start_seed

    # CRITICAL (CLAUDE.md §6.1): the .ckpt stores the ORIGINAL author's
    # absolute paths. Override them with local paths.
    cfg_task.task.env_runner.dataset_path = cfg.demo_dataset_path
    cfg_task.task.dataset.dataset_path = cfg.demo_dataset_path

    # ---- THE parallel-env change vs eval_test_time_optimization.py ----
    # Force the parallel runner (the shared yaml points at the sequential one)
    # and inject N_ENVS from this file's top-level constant.
    cfg_task.task.env_runner._target_ = PARALLEL_RUNNER_TARGET
    OmegaConf.update(cfg_task, 'task.env_runner.n_envs',
                     N_ENVS, force_add=True)

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

    normalizer_dir = os.path.dirname(os.path.dirname(cfg.policy_checkpoint))
    normalizer_path = os.path.join(normalizer_dir, 'normalizer.pth')
    policy.normalizer.load_state_dict(torch.load(normalizer_path))
    policy.normalizer.to(device)

    # ------------------------------------------------------------------
    #  Initialize planner (LPB step that eval_base_policy.py skips)
    # ------------------------------------------------------------------
    # demo_dataset_config MUST be the already-overridden cfg path (CLAUDE.md
    # §6.1 gotcha #2: planner.py reads dataset_path off this config object
    # during __init__).
    payload['cfg'].task.dataset.dataset_path = cfg.demo_dataset_path
    cfg_task.task.env_runner.dataset_path = cfg.demo_dataset_path

    policy.initialize_planner(
        planner_target=cfg.planner_target,
        demo_dataset_config=payload['cfg'].task.dataset,
        dynamics_model_ckpt=cfg.dynamics_model_checkpoint,
        action_step=cfg_task.n_action_steps,
        output_dir=cfg.output_dir,
        guidance_start_timestep=cfg.guidance_start_timestep,
        guidance_scale=cfg.guidance_scale,
        threshold=cfg.threshold,
        demo_dataset_path=cfg.get('demo_dataset_path', None),
    )
    print(f"[planner] initialized: guidance_scale={cfg.guidance_scale}, "
          f"threshold={cfg.threshold}, guidance_start_timestep={cfg.guidance_start_timestep}")

    # ------------------------------------------------------------------
    #  Wrap LPB policy so parallel runner routes predict_action -> per-env
    #  predict_action_dyn_guided. See ParallelGuidedPolicyAdapter docstring.
    # ------------------------------------------------------------------
    parallel_policy = ParallelGuidedPolicyAdapter(policy)

    # ------------------------------------------------------------------
    #  Run evaluation (parallel envs)
    # ------------------------------------------------------------------
    env_runner = hydra.utils.instantiate(
        cfg_task.task.env_runner,
        output_dir=output_dir,
    )

    runner_log = env_runner.run(parallel_policy)

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
    print(f"\n===== LPB Parallel Eval (base + dynamics + guidance) =====")
    print(f"  Checkpoint      : {cfg.policy_checkpoint}")
    print(f"  Dynamics model  : {cfg.dynamics_model_checkpoint}")
    print(f"  n_test          : {cfg.n_test}")
    print(f"  test_start_seed : {cfg.test_start_seed}")
    print(f"  n_action_steps  : {cfg.n_action_steps}")
    print(f"  n_envs          : {N_ENVS}")
    print(f"  guidance_scale  : {cfg.guidance_scale}")
    print(f"  threshold       : {cfg.threshold}")
    print(f"  env_runner      : {PARALLEL_RUNNER_TARGET}")
    if test_score is not None:
        print(f"  test/mean_score (success rate): {test_score:.4f}")
    else:
        print(f"  (test/mean_score not found in runner log)")
        print(f"  available keys: {sorted(results.keys())}")
    print(f"  Full results    : {results_path}")
    print(f"===========================================================")


if __name__ == '__main__':
    main()
