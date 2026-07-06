# LPB vs Base Policy Comparison Visualization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `compare_lpb_vs_base.py` that generates two MP4 videos comparing LPB and base policy rollouts from the same seed-initialized state.

**Architecture:** Run two separate full rollouts (LPB and base) from the same `env.seed(N); env.reset()`. For each video, drive the simulation with one method's frames and overlay the other method's eef trajectory (world coords) projected to the camera pixels. 2x1 subplot (shouldercamera0 | shouldercamera1) per frame, scaled 2x.

**Tech Stack:** PyTorch, MuJoCo/robosuite (via `RobomimicImageWrapper`), matplotlib (frame rendering), imageio-ffmpeg (video writing), Hydra/OmegaConf (config).

**Spec:** `docs/superpowers/specs/2026-07-06-lpb-vs-base-comparison-design.md`

**Running environment:** Linux server with GPU (`/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lpb/`). Windows is edit-only — sanity checks that need GPU/MuJoCo run on the server.

---

### Task 1: Create config file

**Files:**
- Create: `dyn_model/conf/planner/compare_transport.yaml`

- [ ] **Step 1: Write the config file**

```yaml
# Config for compare_lpb_vs_base.py — generates two comparison videos.
# See docs/superpowers/specs/2026-07-06-lpb-vs-base-comparison-design.md

hydra:
  run:
    dir: .

# ---- Reused from eval_transport.yaml ----
dynamics_model_checkpoint: '/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lpb/data/outputs/2026.06.16/11.15.55_transport/checkpoints/model_60.pth'
policy_checkpoint: '/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lpb/data/outputs/base_policy_6_15/checkpoints/270.ckpt'
guidance_start_timestep: 10
guidance_scale: 0.2
threshold: 2.8
demo_dataset_path: '/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lpb/data/transport/data/expert_demonstration/transport_ph_demo_v141_20_perc.hdf5'
device: "cuda"
planner_target: 'dyn_model.planner.Planner'

# ---- Script-specific ----
compare_seed: 100000              # initial state seed (matches eval test_start_seed format)
max_steps: 700                    # rollout cap (matches task.env_runner.max_steps)
eef_sample_interval: 5            # sample eef + frame every N steps (~140 points over 700 steps)
render_views:
  - shouldercamera0_image
  - shouldercamera1_image
video_fps: 20
video_frame_scale: 2              # upscale each camera image (140 -> 280)
output_dir: 'data/compare_lpb_vs_base'
```

- [ ] **Step 2: Verify the file parses as YAML**

Run on the server:
```bash
python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('dyn_model/conf/planner/compare_transport.yaml'); print(OmegaConf.to_yaml(cfg))"
```
Expected: full config printed without errors.

- [ ] **Step 3: Commit**

```bash
git add dyn_model/conf/planner/compare_transport.yaml
git commit -m "Add compare_lpb_vs_base config"
```

---

### Task 2: Create main script skeleton

**Files:**
- Create: `compare_lpb_vs_base.py`

- [ ] **Step 1: Write the skeleton**

```python
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


@hydra.main(config_path="dyn_model/conf/planner", config_name="compare_transport")
def main(cfg: DictConfig):
    print("=" * 60)
    print("LPB vs Base Policy Comparison")
    print("=" * 60)
    # TODO: wire everything together in Task 10
    print("Config loaded:")
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify Hydra loads the config**

Run on the server:
```bash
python compare_lpb_vs_base.py --config-name=compare_transport
```
Expected: prints "LPB vs Base Policy Comparison" + full config, then exits cleanly (no exceptions about missing keys).

- [ ] **Step 3: Commit**

```bash
git add compare_lpb_vs_base.py
git commit -m "Add compare_lpb_vs_base.py skeleton with Hydra entry"
```

---

### Task 3: Add policy loaders

**Files:**
- Modify: `compare_lpb_vs_base.py` (add `_load_policy_payload`, `_load_base_policy`, `_load_lpb_policy` before `main`)

This task adds the three loader functions. They duplicate the inline logic from `eval_base_policy.py` and `eval_test_time_optimization.py` (per spec §"加载代码的处理方式": duplicate not refactor).

- [ ] **Step 1: Add the three loader functions above `main()`**

Insert this code between the imports and `main()`:

```python
def _load_policy_payload(policy_checkpoint: str):
    """Load the .ckpt payload and return (payload, cfg_task). Shared by base + LPB loaders."""
    print(f"[loader] Loading policy checkpoint: {policy_checkpoint}")
    with open(policy_checkpoint, 'rb') as f:
        payload = torch.load(f, pickle_module=dill)
    cfg_task = payload['cfg']
    return payload, cfg_task


def _apply_common_overrides(cfg_task, cfg):
    """Override stale .ckpt paths and inject eval-time fields (CLAUDE.md §6.1)."""
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

    # Inject planner + dynamics model (CLAUDE.md §6.4)
    planner_target = hydra.utils.get_class(cfg.planner_target)
    policy.initialize_planner(
        planner=planner_target,
        dynamics_model_checkpoint=cfg.dynamics_model_checkpoint,
        guidance_start_timestep=cfg.guidance_start_timestep,
        guidance_scale=cfg.guidance_scale,
        threshold=cfg.threshold,
        demo_dataset_path=cfg.demo_dataset_path,
        policy_ckpt_path=cfg.policy_checkpoint,
        device=cfg.device,
    )
    print("[loader] LPB policy loaded (with planner)")
    return policy, cfg_task
```

**Note:** The exact `initialize_planner` keyword arguments must match `diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py`. Verify before finalizing — see Step 2.

- [ ] **Step 2: Verify `initialize_planner` signature matches**

Run on the server:
```bash
grep -A 20 "def initialize_planner" diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py
```
Compare the signature to the kwargs in `_load_lpb_policy`. Adjust the kwargs in the loader to match exactly. Common pitfall: parameter names like `dynamics_checkpoint` vs `dynamics_model_checkpoint`, or `planner_class` vs `planner`.

Also check `eval_test_time_optimization.py` to see how it calls `initialize_planner`:
```bash
grep -A 15 "initialize_planner" eval_test_time_optimization.py
```
Copy the exact call pattern.

- [ ] **Step 3: Sanity check — loaders run end-to-end**

Add a temporary test hook in `main()`:
```python
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # TEMP sanity check (delete after Task 10 wires up real flow)
    base_policy, _ = _load_base_policy(cfg)
    del base_policy
    torch.cuda.empty_cache()
    lpb_policy, _ = _load_lpb_policy(cfg)
    print("[sanity] both policies loaded OK")
```

Run:
```bash
python compare_lpb_vs_base.py --config-name=compare_transport
```
Expected: prints "[loader] Base policy loaded (no planner)", "[loader] LPB policy loaded (with planner)", "[sanity] both policies loaded OK". If `initialize_planner` fails with a missing arg, fix per Step 2.

- [ ] **Step 4: Commit**

```bash
git add compare_lpb_vs_base.py
git commit -m "Add base + LPB policy loaders"
```

---

### Task 4: Add env builder

**Files:**
- Modify: `compare_lpb_vs_base.py` (add `_build_env`)

- [ ] **Step 1: Add `_build_env` function (after the loaders, before `main`)**

```python
def _build_env(cfg_task, cfg, seed: int, render_obs_key: str = 'shouldercamera0_image'):
    """
    Build a single RobomimicImageWrapper env and reset to the given seed.
    Uses cfg_task.task.env_runner config as the template.
    """
    # Pull the envmaker + task config out of cfg_task
    # The env_runner has a `_target_` for the underlying env factory; we use
    # RobomimicImageWrapper directly since we just need one env, not a batch.
    import gym
    from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
    from diffusion_policy.env.robomimic.robomimic_image_wrapper import EnvRobosuite  # re-export

    env_runner_cfg = cfg_task.task.env_runner

    # Build robosuite env using the same factory the runner uses
    env_factory = hydra.utils.instantiate(env_runner_cfg.env_factory)
    env = env_factory(render_obs_key=render_obs_key)
    env.seed(seed)
    obs = env.reset()
    print(f"[env] Built env, seed={seed}, reset OK. obs keys: {sorted(obs.keys())}")
    return env, obs
```

**Note:** The exact `env_factory` pattern may differ in this codebase. Verify in Step 2.

- [ ] **Step 2: Verify how the env_runner builds envs**

Run:
```bash
grep -B 2 -A 30 "def __init__" diffusion_policy/env_runner/robomimic_image_runner.py | head -80
```
Look for how `env_fns` / `env_factory` is constructed. Common patterns:
- `env_factory = hydra.utils.instantiate(env_runner_cfg.env_factory)` then `env = env_factory(render_obs_key=...)`
- Direct construction: `env = RobomimicImageWrapper(env_robosuite_instance, shape_meta=..., render_obs_key=...)`

Mirror the pattern. If `env_factory` doesn't work, fall back to building `EnvRobosuite` directly from `cfg_task.task.env_runner.env` (look at `robomimic_image_runner.py:__init__` for the exact construction).

- [ ] **Step 3: Sanity check — env builds and resets**

Update the temporary hook in `main()`:
```python
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    base_policy, cfg_task = _load_base_policy(cfg)
    env, obs = _build_env(cfg_task, cfg, seed=cfg.compare_seed)
    print(f"[sanity] env.step working? trying one step...")
    action = np.zeros(cfg_task.task.shape_meta.action.shape, dtype=np.float32)
    obs2, reward, done, info = env.step(action)
    print(f"[sanity] step OK, reward={reward}, eef_pos robot0={obs2['robot0_eef_pos']}")
    del base_policy
    print("[sanity] all good")
```

Run:
```bash
python compare_lpb_vs_base.py --config-name=compare_transport
```
Expected: prints env obs keys including `robot0_eef_pos`, `shouldercamera0_image`, etc.; prints a real eef position (3 numbers, not zeros); "[sanity] all good".

- [ ] **Step 4: Commit**

```bash
git add compare_lpb_vs_base.py
git commit -m "Add _build_env helper"
```

---

### Task 5: Add rollout runner

**Files:**
- Modify: `compare_lpb_vs_base.py` (add `_run_rollout`)

- [ ] **Step 1: Add `_run_rollout` function (after `_build_env`, before `main`)**

```python
def _run_rollout(policy, env, cfg, use_guidance: bool, label: str) -> Dict:
    """
    Run a full rollout. Resets env to cfg.compare_seed first.
    Returns dict with eef_traj (T, 2, 3), frames {view: (T, H, W, 3)}, actions, success info.
    """
    # Reset to known initial state
    env.seed(cfg.compare_seed)
    obs = env.reset()

    eef_list = []     # list of (2, 3) arrays
    frame_dict = {v: [] for v in cfg.render_views}
    actions_list = []

    success = False
    success_step = -1

    n_steps = 0
    for t in range(cfg.max_steps):
        # Get action from policy
        with torch.no_grad():
            obs_dict_t = {k: torch.from_numpy(np.stack([v])).to(policy.device) for k, v in obs.items()}
            if use_guidance:
                action_dict = policy.predict_action_dyn_guided(obs_dict_t)
            else:
                action_dict = policy.predict_action(obs_dict_t)
        action = action_dict['action'][0].cpu().numpy()  # (action_dim,)

        # Execute one substep (n_action_steps handled internally by the wrapper)
        obs, reward, done, info = env.step(action)
        n_steps += 1

        # Sample every eef_sample_interval
        if t % cfg.eef_sample_interval == 0:
            eef = np.stack([
                obs['robot0_eef_pos'].copy(),
                obs['robot1_eef_pos'].copy(),
            ])  # (2, 3)
            eef_list.append(eef)
            for v in cfg.render_views:
                # Image obs is (C, H, W) float in [0, 1]; convert to (H, W, 3) uint8
                img = obs[v]
                if img.ndim == 3 and img.shape[0] == 3:  # (C, H, W)
                    img = np.transpose(img, (1, 2, 0))
                img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
                frame_dict[v].append(img.copy())
            actions_list.append(action.copy())

        # Check success (sequential-runner style early break)
        if env.get_success_label():
            success = True
            success_step = t
            print(f"[rollout:{label}] SUCCESS at step {t}")
            break

    eef_traj = np.stack(eef_list) if eef_list else np.zeros((0, 2, 3))
    frames = {v: (np.stack(frame_dict[v]) if frame_dict[v] else np.zeros((0, 140, 140, 3), dtype=np.uint8))
              for v in cfg.render_views}
    actions = np.stack(actions_list) if actions_list else np.zeros((0, env.action_space.shape[0]))

    print(f"[rollout:{label}] done. n_steps={n_steps}, sampled_points={len(eef_list)}, success={success}")
    return {
        'eef_traj': eef_traj,
        'frames': frames,
        'actions': actions,
        'success': success,
        'success_step': success_step,
        'n_steps': n_steps,
        'n_samples': len(eef_list),
    }
```

**Note on action chunking:** diffusion policy outputs 15-step chunks internally. `predict_action` caches the trajectory and returns the next chunk slice on each call. So calling `predict_action` every step and executing the returned action is correct (matches `robomimic_image_runner.run`).

- [ ] **Step 2: Sanity check — both rollouts run for ~20 steps**

Update the temporary hook in `main()`:
```python
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    cfg.max_steps = 20  # quick test
    cfg.eef_sample_interval = 1

    base_policy, cfg_task = _load_base_policy(cfg)
    env, _ = _build_env(cfg_task, cfg, seed=cfg.compare_seed)
    base_result = _run_rollout(base_policy, env, cfg, use_guidance=False, label='base')
    print(f"[sanity] base eef_traj shape: {base_result['eef_traj'].shape}")
    print(f"[sanity] base frame[view0] shape: {base_result['frames'][cfg.render_views[0]].shape}")

    del base_policy
    torch.cuda.empty_cache()

    lpb_policy, _ = _load_lpb_policy(cfg)
    env2, _ = _build_env(cfg_task, cfg, seed=cfg.compare_seed)
    lpb_result = _run_rollout(lpb_policy, env2, cfg, use_guidance=True, label='lpb')
    print(f"[sanity] lpb eef_traj shape: {lpb_result['eef_traj'].shape}")

    print("[sanity] both rollouts OK")
```

Run:
```bash
python compare_lpb_vs_base.py --config-name=compare_transport
```
Expected: both rollouts complete 20 steps each, eef_traj shape is `(~20, 2, 3)`, frame shape is `(~20, 140, 140, 3)`, no exceptions. **Reset `cfg.max_steps` back to 700 in the config file before committing** (the override `cfg.max_steps=20` is only in the test hook).

- [ ] **Step 3: Verify determinism — both rollouts start from the same eef**

Verify the first sampled eef is identical between base and lpb rollouts:
```python
print(f"[sanity] base eef[0]: {base_result['eef_traj'][0]}")
print(f"[sanity] lpb  eef[0]: {lpb_result['eef_traj'][0]}")
print(f"[sanity] match: {np.allclose(base_result['eef_traj'][0], lpb_result['eef_traj'][0])}")
```
Expected: `match: True` (same initial state → same first eef position). If False, the env reset is not deterministic; investigate `seed_state_map` caching.

- [ ] **Step 4: Commit**

```bash
git add compare_lpb_vs_base.py
git commit -m "Add _run_rollout with sampling + early termination"
```

---

### Task 6: Add 3D→2D projection

**Files:**
- Modify: `compare_lpb_vs_base.py` (add `_project_to_camera`)

This is the highest-risk task. The exact MuJoCo API for camera extrinsics may vary; the function includes a manual fallback.

- [ ] **Step 1: Add `_project_to_camera` and `_get_camera_params` (after `_run_rollout`)**

```python
def _get_camera_params(env, camera_name: str) -> Dict:
    """
    Extract camera position (3,), orientation (3, 3), fovy (scalar) from MuJoCo sim.
    Tries multiple known APIs since robosuite versions differ.
    """
    # Navigate to the underlying robosuite env / mujoco sim
    sim_env = env.env.env  # RobomimicImageWrapper.env.env = robosuite env
    sim = sim_env.sim

    # Camera name in robosuite usually matches the obs key (e.g. 'shouldercamera0_image')
    # but MuJoCo camera names might drop the '_image' suffix.
    cam_name = camera_name.replace('_image', '')

    # Try robosuite's viewer API first
    try:
        cam_id = sim.model.camera_name2id(cam_name)
        cam_pos = sim.data.cam_xpos[cam_id].copy()
        cam_mat = sim.data.cam_xmat[cam_id].copy().reshape(3, 3)
        fovy = sim.model.cam_fovy[cam_id]
        return {'pos': cam_pos, 'mat': cam_mat, 'fovy': float(fovy)}
    except (AttributeError, KeyError, IndexError) as e:
        print(f"[proj] sim.model camera API failed: {e}, trying offscreen viewers...")

    # Fallback: robosuite's offscreen render context
    try:
        viewer = sim_env.sim._render_context.offscreen
        cam = viewer.cameras[cam_name]  # dict-like
        cam_pos = np.array(cam.pos)
        cam_mat = np.array(cam.mat).reshape(3, 3)
        fovy = float(cam.fovy)
        return {'pos': cam_pos, 'mat': cam_mat, 'fovy': fovy}
    except (AttributeError, KeyError) as e:
        raise RuntimeError(
            f"Could not get camera params for '{cam_name}'. "
            f"Tried sim.model.camera_name2id and sim._render_context.offscreen.cameras. "
            f"Original error: {e}"
        )


def _project_to_camera(points_world: np.ndarray, env, camera_name: str, img_size: int = 140) -> np.ndarray:
    """
    Project 3D world-coords (N, 3) to 2D pixel coords (N, 2) for the given camera.
    Uses pinhole perspective; assumes square image (img_size x img_size).
    """
    params = _get_camera_params(env, camera_name)
    pos = params['pos']  # (3,)
    mat = params['mat']  # (3, 3) — camera-to-world rotation
    fovy = params['fovy']

    # World → camera frame
    p_rel = points_world - pos[None, :]              # (N, 3)
    p_cam = p_rel @ mat.T                             # (N, 3); mat is cam→world so mat.T is world→cam

    # Pinhole projection (vertical fovy given)
    f = (img_size / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)  # focal length in pixels

    # Avoid division by zero (points at or behind camera)
    z = p_cam[:, 2]
    valid = z > 1e-6
    px = np.full_like(z, -1.0)  # sentinel for invalid
    py = np.full_like(z, -1.0)
    px[valid] = f * p_cam[valid, 0] / z[valid] + img_size / 2.0
    py[valid] = -f * p_cam[valid, 1] / z[valid] + img_size / 2.0  # flip y

    return np.stack([px, py], axis=1)  # (N, 2)
```

- [ ] **Step 2: Sanity check — project a known point**

Update the hook in `main()`:
```python
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    base_policy, cfg_task = _load_base_policy(cfg)
    env, obs = _build_env(cfg_task, cfg, seed=cfg.compare_seed)

    # Project the current robot0_eef_pos and check it lands in-frame
    eef0 = obs['robot0_eef_pos']  # (3,)
    cam = cfg.render_views[0]
    px = _project_to_camera(eef0.reshape(1, 3), env, cam)[0]
    print(f"[sanity] robot0_eef world: {eef0}")
    print(f"[sanity] projected px ({cam}): {px}")
    assert 0 <= px[0] < 140 and 0 <= px[1] < 140, \
        f"Projection out of frame: {px}. Check camera params."

    # Save the first frame with a dot drawn on it for visual verification
    import matplotlib.pyplot as plt
    img = obs[cam]
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.plot(px[0], px[1], 'ro', markersize=10)
    ax.set_title(f'{cam}\nred dot = projected robot0_eef_pos')
    fig.savefig('data/projection_sanity.png', dpi=100, bbox_inches='tight')
    print(f"[sanity] saved data/projection_sanity.png — open it and check the red dot is on the robot arm")
```

Run:
```bash
mkdir -p data
python compare_lpb_vs_base.py --config-name=compare_transport
```

**Manually inspect `data/projection_sanity.png`**: the red dot should land on (or very near) the robot0 gripper in the image. If it lands in empty space:
- The camera matrix convention may be wrong. Try `p_cam = p_rel @ mat` (no transpose).
- The `fovy` interpretation may differ. Try `f = (img_size / 2.0) / np.tan(np.deg2rad(fovy))` (no /2).
- Iterate until the dot lands correctly. Update the projection code accordingly.

- [ ] **Step 3: Commit (only after projection is verified)**

```bash
git add compare_lpb_vs_base.py
git commit -m "Add 3D→2D camera projection with sanity check"
```

---

### Task 7: Add frame renderer

**Files:**
- Modify: `compare_lpb_vs_base.py` (add `_render_video_frame`)

- [ ] **Step 1: Add `_render_video_frame` function (after `_project_to_camera`)**

```python
def _render_video_frame(
    bg_frame: np.ndarray,           # (H, W, 3) uint8
    overlay_eef_world: np.ndarray,  # (T, 2, 3)
    env,
    camera_name: str,
    current_step: int,
    total_overlay_steps: int,
    scale: int,
    overlay_method: str,            # 'lpb' or 'base'
    img_size: int = 140,
) -> np.ndarray:
    """Compose one video frame: background + overlay trajectory (past solid + future dashed)."""
    import matplotlib
    matplotlib.use('Agg')  # headless
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    cmap = 'plasma' if overlay_method == 'lpb' else 'viridis'

    # Project overlay trajectory to this camera's pixels
    n_total = overlay_eef_world.shape[0]
    all_px = _project_to_camera(
        overlay_eef_world.reshape(-1, 3), env, camera_name, img_size=img_size
    ).reshape(n_total, 2, 2)  # (T, 2 arms, 2 px)

    # Build figure (no margins)
    fig, ax = plt.subplots(figsize=(img_size * scale / 100, img_size * scale / 100), dpi=100)
    ax.imshow(bg_frame, extent=[0, img_size, img_size, 0])  # flip y so origin top-left
    ax.set_xlim(0, img_size)
    ax.set_ylim(img_size, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')

    # Split past / future
    past_end = min(current_step + 1, n_total)
    future_start = past_end
    future_end = n_total

    norm = Normalize(vmin=0, vmax=max(n_total - 1, 1))

    def _draw_segment(points_px, linestyle, alpha, label_prefix):
        """Draw a 2-arm trajectory as 2 LineCollections."""
        for arm_idx, arm_ls in enumerate(['-', '--']):  # robot0 solid, robot1 dashed
            pts = points_px[:, arm_idx]  # (T_seg, 2)
            # Filter invalid points (z<=0 in projection) — break line there
            valid = (pts[:, 0] >= 0) & (pts[:, 1] >= 0)
            if valid.sum() < 2:
                continue
            # Build segments: pairs of consecutive valid points
            seg_pts = pts[valid]
            seg_colors = plt.get_cmap(cmap)(
                norm(np.arange(valid.sum()))
            )
            seg_colors[:, 3] = alpha  # set alpha
            if len(seg_pts) >= 2:
                segments = np.stack([seg_pts[:-1], seg_pts[1:]], axis=1)
                lc = LineCollection(segments, colors=seg_colors[:-1],
                                    linewidths=2, linestyles=(arm_ls if linestyle is None else linestyle))
                ax.add_collection(lc)

    if past_end > 0:
        _draw_segment(all_px[:past_end], linestyle=None, alpha=1.0, label_prefix='past')
    if future_end > future_start:
        _draw_segment(all_px[future_start:future_end], linestyle='--', alpha=0.35, label_prefix='future')

    # Current points (2 arms)
    if current_step < n_total:
        for arm_idx, marker in enumerate(['o', 's']):
            pt = all_px[current_step, arm_idx]
            if pt[0] >= 0 and pt[1] >= 0:
                ax.plot(pt[0], pt[1], marker, color=plt.get_cmap(cmap)(norm(current_step)),
                        markersize=10, markeredgecolor='white', markeredgewidth=1.5)

    # Title / status text
    ax.set_title(f'{camera_name}\n{overlay_method} traj @ step {current_step}/{n_total}',
                 fontsize=10)

    # Render to numpy
    fig.canvas.draw()
    out = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    out = out.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return out
```

**Note:** `_draw_segment` uses `linestyles=(arm_ls if linestyle is None else linestyle)`. For future (dashed), both arms will be dashed (no solid/dashed distinction between arms in the future preview). This is acceptable; past trajectory distinguishes arms by solid/dashed.

- [ ] **Step 2: Sanity check — render one frame and save**

Update hook in `main()`:
```python
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    base_policy, cfg_task = _load_base_policy(cfg)
    env, obs = _build_env(cfg_task, cfg, seed=cfg.compare_seed)
    base_result = _run_rollout(base_policy, env, cfg, use_guidance=False, label='base')

    # Test render: base's first frame + base's own trajectory as overlay
    cam = cfg.render_views[0]
    bg = base_result['frames'][cam][0]
    out = _render_video_frame(
        bg_frame=bg,
        overlay_eef_world=base_result['eef_traj'],
        env=env,
        camera_name=cam,
        current_step=0,
        total_overlay_steps=base_result['n_samples'],
        scale=cfg.video_frame_scale,
        overlay_method='base',
    )
    import imageio.v2 as imageio
    imageio.imwrite('data/frame_render_sanity.png', out)
    print(f"[sanity] saved data/frame_render_sanity.png (shape {out.shape})")
    print("[sanity] check: should show base's first frame + a dashed future trajectory")
```

Run:
```bash
python compare_lpb_vs_base.py --config-name=compare_transport
```
Inspect `data/frame_render_sanity.png`. Expected: the first frame of base's rollout, with a dashed viridis-colored line showing the future eef trajectory. Current step (0) marked with circles/squares.

- [ ] **Step 3: Commit**

```bash
git add compare_lpb_vs_base.py
git commit -m "Add frame renderer with past/future trajectory split"
```

---

### Task 8: Add video generator

**Files:**
- Modify: `compare_lpb_vs_base.py` (add `_make_video`)

- [ ] **Step 1: Add `_make_video` function (after `_render_video_frame`)**

```python
def _make_video(
    driver_result: Dict,
    overlay_result: Dict,
    overlay_method: str,
    env,
    view_names: List[str],
    cfg: DictConfig,
    output_path: str,
):
    """Generate one MP4: driver's frames as background, overlay's eef trajectory projected."""
    import imageio.v2 as imageio

    n_frames = driver_result['n_samples']  # how many sampled frames the driver produced
    overlay_n = overlay_result['n_samples']
    img_size = driver_result['frames'][view_names[0]].shape[1]  # 140
    out_h = img_size * cfg.video_frame_scale
    out_w = img_size * cfg.video_frame_scale * len(view_names)  # 2x1 side-by-side

    writer = imageio.get_writer(
        output_path,
        fps=cfg.video_fps,
        codec='libx264',
        quality=8,
        macro_block_size=1,  # avoid forcing multiples of 16
    )

    for t in range(n_frames):
        overlay_step = min(t, overlay_n - 1)
        # Render each view
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
        # Side-by-side concat (all panels are same shape)
        frame = np.concatenate(panels, axis=1)
        writer.append_data(frame)

        if t % 20 == 0:
            print(f"  [video] frame {t}/{n_frames}")

    writer.close()
    print(f"[video] wrote {output_path} ({n_frames} frames)")
```

- [ ] **Step 2: Sanity check — generate a tiny 5-frame video**

Update hook in `main()`:
```python
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    cfg.max_steps = 25
    cfg.eef_sample_interval = 5

    base_policy, cfg_task = _load_base_policy(cfg)
    env, _ = _build_env(cfg_task, cfg, seed=cfg.compare_seed)
    base_result = _run_rollout(base_policy, env, cfg, use_guidance=False, label='base')
    del base_policy; torch.cuda.empty_cache()

    lpb_policy, _ = _load_lpb_policy(cfg)
    env2, _ = _build_env(cfg_task, cfg, seed=cfg.compare_seed)
    lpb_result = _run_rollout(lpb_policy, env2, cfg, use_guidance=True, label='lpb')
    del lpb_policy; torch.cuda.empty_cache()

    os.makedirs(cfg.output_dir, exist_ok=True)
    _make_video(base_result, lpb_result, 'lpb', env, list(cfg.render_views), cfg,
                os.path.join(cfg.output_dir, 'sanity_base_driving.mp4'))
    print("[sanity] video written, open it to verify")
```

Run:
```bash
python compare_lpb_vs_base.py --config-name=compare_transport
```

Inspect `data/compare_lpb_vs_base/sanity_base_driving.mp4`. Expected: 5 frames, 2x1 side-by-side shoulder cameras, base's actual rollout with LPB's trajectory overlaid (plasma colormap, dashed for future).

- [ ] **Step 3: Commit**

```bash
git add compare_lpb_vs_base.py
git commit -m "Add video generator with side-by-side shoulder cameras"
```

---

### Task 9: Wire main() end-to-end

**Files:**
- Modify: `compare_lpb_vs_base.py` (replace the sanity hook with the real flow)

- [ ] **Step 1: Replace main() body with the real pipeline**

```python
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
    # Re-create env to make sure seed_state_map cache is fresh (or reuse — both fine)
    env2, _ = _build_env(cfg_task, cfg, seed=cfg.compare_seed)
    lpb_result = _run_rollout(lpb_policy, env2, cfg, use_guidance=True, label='lpb')
    del lpb_policy
    torch.cuda.empty_cache()

    # ---- Phase 3: sanity check projection ----
    print("\n[Phase 3] Projection sanity check...")
    cam = cfg.render_views[0]
    eef0 = base_result['eef_traj'][0, 0]  # robot0 at sample 0
    px = _project_to_camera(eef0.reshape(1, 3), env, cam)[0]
    assert 0 <= px[0] < 140 and 0 <= px[1] < 140, \
        f"Projection sanity check failed: {px}"
    print(f"  OK: world {eef0} → px {px}")

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
```

- [ ] **Step 2: Commit (before running full pipeline)**

```bash
git add compare_lpb_vs_base.py
git commit -m "Wire main() with full compare pipeline"
```

---

### Task 10: End-to-end smoke test (full max_steps run)

**Files:** None (this task runs the script and inspects output).

- [ ] **Step 1: Run the full pipeline with default seed**

```bash
python compare_lpb_vs_base.py --config-name=compare_transport
```

Expected output:
- Phase 1: base rollout finishes (700 steps OR early success)
- Phase 2: LPB rollout finishes (700 steps OR early success)
- Phase 3: projection sanity check passes
- Phase 5: both videos written
- Final summary printed with success flags

- [ ] **Step 2: Manually inspect both videos**

Open the two MP4 files:
- `data/compare_lpb_vs_base/base_driving_lpb_overlay.mp4` — base drives, LPB overlaid in plasma
- `data/compare_lpb_vs_base/lpb_driving_base_overlay.mp4` — LPB drives, base overlaid in viridis

Check:
- Side-by-side shoulder cameras render correctly
- Overlay trajectory starts near the actual gripper position at step 0
- Past trajectory is solid, future preview is dashed
- robot0 (solid line style) and robot1 (dashed) both visible
- Time gradient progresses over the video
- Video ends when the driving method's rollout ends

- [ ] **Step 3: If a second seed is interesting, run one more**

```bash
python compare_lpb_vs_base.py --config-name=compare_transport \
    compare_seed=100001 output_dir=data/compare_lpb_vs_base_seed100001
```

- [ ] **Step 4: Final commit (if any tweaks were made)**

```bash
git status
# if changes:
git add -A
git commit -m "Polish compare_lpb_vs_base.py after end-to-end test"
```

---

## Self-Review

### Spec coverage
- §1 Goal (two videos, side-by-side shoulder cameras, growing trail + future preview): Tasks 7-9 ✓
- §3 Non-goals (no batch, no real robot, no edit existing scripts): respected throughout ✓
- §4 Architecture (Hydra entry, two new files): Task 1 + 2 ✓
- §5.1 Config: Task 1 ✓
- §5.2 `load_lpb_policy` / `load_base_policy`: Task 3 ✓
- §5.2 `build_env`: Task 4 ✓
- §5.2 `run_rollout`: Task 5 ✓
- §5.2 `project_to_camera`: Task 6 ✓
- §5.2 `render_video_frame`: Task 7 ✓
- §5.2 `make_video`: Task 8 ✓
- §6 Output files (mp4 × 2, npz, compare_config.yaml): Task 9 ✓
- §7.1 Length mismatch (overlay_step = min(t, overlay_n-1)): Task 8 ✓
- §7.2 Driver success early break: Task 5 ✓
- §7.3 Out-of-frame points (filter via `valid`): Task 7 ✓
- §7.4 Missing ckpt path: would surface as `FileNotFoundError` in `_load_policy_payload` ✓
- §7.5 GPU OOM (serial loading + del + empty_cache): Tasks 9 ✓
- §8 Memory management (del + empty_cache): Tasks 9 ✓
- §9 Determinism (seed_state_map caching): Task 5 sanity check (Step 3) ✓
- §10 Risks (projection API, render_obs_key): Task 6 + Task 4 sanity checks ✓
- §11 Testing (sanity check + visual inspection): Tasks 6, 7, 8, 10 ✓

### Placeholder scan
- Task 3 Step 2 says "verify signature matches" — this is **legitimate runtime verification** (not a placeholder), because the spec acknowledged this depends on the exact `initialize_planner` signature in the codebase. Concrete command + fallback action provided.
- Task 4 Step 2 says "verify how env_runner builds envs" — same reasoning. Concrete commands + fallback provided.
- Task 6 Step 2 includes "iterate until the dot lands correctly" — this is a known-risk debugging step, not a placeholder. Concrete failure modes listed.

No "TBD" / "implement later" / "fill in details" anywhere.

### Type consistency
- `_run_rollout` returns dict with keys `eef_traj`, `frames`, `actions`, `success`, `success_step`, `n_steps`, `n_samples` — used consistently in Tasks 8 and 9 ✓
- `_project_to_camera` takes `(points_world, env, camera_name, img_size=140)` returns `(N, 2)` — called consistently in Tasks 6, 7 ✓
- `_render_video_frame` signature matches between Task 7 (definition) and Task 8 (call) ✓
- `_make_video` signature matches between Task 8 (definition) and Task 9 (calls) ✓
- `overlay_method` parameter: `'lpb'` or `'base'`, used to pick cmap — consistent ✓

No type drift detected.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-06-lpb-vs-base-comparison.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for catching issues early since each task gets reviewed before the next.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review. Faster if everything goes smoothly.

Which approach?
