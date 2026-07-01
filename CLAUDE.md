# CLAUDE.md — Latent Policy Barrier (LPB)

This document is a comprehensive guide to the LPB codebase for Claude Code sessions. It synthesizes the paper (arXiv:2508.05941) with the actual implementation.

---

## 1. What LPB Is

**Latent Policy Barrier (LPB)** is a method for robust visuomotor policy learning via behavior cloning. It tackles **covariate shift** — the phenomenon where small deviations from expert trajectories compound into failure.

**Core insight:** Decouple two conflicting objectives into separate modules:
- **Precise imitation** → base diffusion policy trained ONLY on clean expert demos
- **OOD recovery** → dynamics model trained on expert + suboptimal rollout data

At inference time, the dynamics model acts as a "barrier" (inspired by Control Barrier Functions) that detects when the agent drifts out of the expert distribution and steers it back via gradient guidance in latent space.

**Paper:** Sun & Song, "Latent Policy Barrier: Learning Robust Visuomotor Policies by Staying In-Distribution", arXiv:2508.05941, 2025.

---

## 2. Architecture Overview

### Two-module design

```
┌─────────────────────────────────────────────────────────┐
│  MODULE 1: Base Diffusion Policy (π_θ)                  │
│  - Trained ONLY on expert demonstrations               │
│  - ResNet-18 visual encoder h_θ (FROZEN after training)│
│  - ConditionalUnet1D noise predictor (DDPM, 100 steps) │
│  - FiLM conditioning on obs features + diffusion step   │
└─────────────────────────────────────────────────────────┘
                          │ shares h_θ (frozen)
                          ▼
┌─────────────────────────────────────────────────────────┐
│  MODULE 2: Visual Latent Dynamics Model (d_ϕ)           │
│  - Trained on expert + policy rollout data              │
│  - Frozen encoder h_θ (reused from base policy)         │
│  - Proprio encoder + Action encoder (MLP embeddings)    │
│  - ViT predictor f_ϕ (decoder-only transformer)         │
│  - Predicts: future latent = f_ϕ(h_θ(O_t), A_t)         │
│  - Loss: MSE in latent space                            │
└─────────────────────────────────────────────────────────┘
```

### Why share the frozen encoder?
During test-time optimization, the dynamics model predicts future latents in the SAME embedding space the policy uses to decide actions. Freezing ensures consistency and prevents representation collapse.

---

## 3. Three-Phase Pipeline

### Phase 1 — Base Policy Training
```bash
python train.py --config-dir=. --config-name=image_transport_diffusion_policy_cnn.yaml \
    training.seed=42 training.device=cuda:0
```
- Train diffusion policy on **20%** of expert demos (limited-data regime)
- Saves intermediate checkpoints at fixed intervals (e.g. every 50 epochs after warmup t₀=70)
- Entry point: `train.py` → `diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py`
- Single-GPU only (uses standard PyTorch training loop)

### Phase 2 — Rollout Collection
Run intermediate base-policy checkpoints in the eval environment to collect diverse (success + failure) trajectories.
- Script: `scripts/collect_rollout.py` (standalone, NOT in original repo — added for reproducibility)
- Uses ONLY the base policy (no dynamics model) for rollouts
- Saves all transitions regardless of success → broad state-action coverage
- Schedule (Transport): t₀=70, Δt=50, tfinal=270, N=30 episodes/ckpt → 150 total trajectories
- Combined with expert demos into `transport_rollout_and_demo.hdf5`

### Phase 3 — Dynamics Model Training
```bash
python dyn_model/train.py --config-name=train.yaml
```
- Multi-GPU via `accelerate`
- Config: `dyn_model/conf/train.yaml` + `dyn_model/conf/env/{transport,libero}.yaml`
- Encoder is FROZEN (`train_encoder: False`, `use_pretrained_encoder: True`)
- Only predictor + proprio/action encoders are trainable
- Saves to `checkpoints/model_{epoch}.pth` + `normalizer.pth` + `hydra.yaml`

### Phase 4 — Test-Time Optimization (Inference)
```bash
python eval_test_time_optimization.py --config-name=eval_transport
```
- Loads base policy ckpt + dynamics model ckpt
- Runs rollouts with gradient-guided denoising when OOD detected
- Outputs: `eval_results.json` + videos in `output_dir`

---

## 4. Test-Time Optimization Algorithm (Algorithm 1)

This is the heart of LPB. At each control timestep:

```
1. Observe O_t, encode z_t = h_θ(o_t)
2. Compute latent OOD score:
      δ(z_t) = min_{z' ∈ Z_expert} ||z_t - z'||²₂
   (chunked nearest-neighbor search, chunk_size=2048)
3. IF δ(z_t) > threshold τ:
     FOR k = K, ..., K - K_guide:  (last K_guide denoising steps)
       a. Sample noisy action A_t^k from base policy denoising
       b. Predict future latent: ẑ_{t+h} = d_ϕ(z_t, A_t^k)
       c. Compute gradient: ∇_{A_t^k} δ(ẑ_{t+h})
       d. Apply classifier guidance:
            trajectory += η * sqrt(1 - ᾱ_k) * (-∇δ)
4. Execute first T_a actions from refined trajectory
```

**Key implementation** (`diffusion_unet_hybrid_image_policy.py:212-271`):
- `guided_conditional_sample()` — modifies the DDPM denoising loop
- `Planner.compute_loss()` — predicts future latent, computes NN cost
- `Planner.compute_current_reward()` — checks current OOD score against threshold
- Guidance only triggered when `current_cost > self.threshold` AND `t < guidance_start_timestep`
- Gradient scaling: `grad_scale = guidance_scale * (1 - alphas_cumprod[t]).sqrt()`

---

## 5. Repository Structure

```
lpb/
├── train.py                              # Base policy training (Hydra entry)
├── eval_test_time_optimization.py        # Test-time optimization eval
├── analyze_rollout_data.py               # HDF5 dataset analysis utility
├── conda_environment.yaml                # Environment spec
├── README.md                             # Official README
│
├── diffusion_policy/                     # Base policy codebase
│   ├── config/
│   │   ├── task/                         # Task configs (transport_image, libero_image, ...)
│   │   └── image_transport_diffusion_policy_cnn.yaml  # Main training config
│   ├── policy/
│   │   └── diffusion_unet_hybrid_image_policy.py      # ★ Base policy + guided sampling
│   ├── model/
│   │   ├── diffusion/                    # ConditionalUnet1D, EMA, schedulers
│   │   ├── common/                       # LinearNormalizer, rotation_transformer
│   │   └── vision/                       # Crop randomizers
│   ├── dataset/
│   │   └── robomimic_replay_image_dataset.py           # Expert demo dataset
│   ├── env_runner/
│   │   ├── robomimic_image_runner.py                   # Parallel env runner
│   │   └── robomimic_image_sequential_runner.py        # Sequential runner (used in eval)
│   ├── env/                              # Robomimic env wrappers
│   ├── gym_util/                         # AsyncVectorEnv (spawn context for EGL)
│   ├── workspace/                        # Training workspaces
│   └── common/                           # pytorch_util, robomimic_config_util
│
├── dyn_model/                            # Dynamics model + planner codebase
│   ├── train.py                          # ★ Dynamics model training (accelerate)
│   ├── planner.py                        # ★ Planner: OOD detection + gradient guidance
│   ├── plan.py                           # Model loading utilities
│   ├── conf/
│   │   ├── train.yaml                    # Dynamics training top-level config
│   │   ├── env/
│   │   │   ├── transport.yaml            # Transport env config
│   │   │   └── libero.yaml               # Libero env config
│   │   ├── planner/
│   │   │   ├── eval_transport.yaml       # ★ Transport eval config
│   │   │   └── eval_libero.yaml          # Libero eval config
│   │   ├── predictor/vit.yaml            # ViT predictor hyperparams
│   │   └── action_encoder/proprio.yaml   # Action/proprio encoder hyperparams
│   ├── models/
│   │   ├── visual_dyn_model.py           # ★ VisualDynamicsModel (encode + predict)
│   │   ├── resnet_encoder.py             # ★ ResNetEncoder (loads from base policy ckpt)
│   │   ├── vit.py                        # ViT predictor (decoder-only transformer)
│   │   ├── proprio.py                    # ProprioceptiveEmbedding MLP
│   │   └── language_encoder.py           # CLIP-based language encoder (Libero)
│   └── datasets/
│       ├── robomimic_dset.py             # ★ RobomimicImageDynamicsModelDataset
│       ├── libero_dset.py                # Libero dataset variant
│       └── img_transforms.py             # Crop transforms (train/eval)
│
├── scripts/                              # Helper scripts
│   ├── collect_rollout.py                # Standalone rollout collection
│   ├── verify_rollout_data.py            # HDF5 format verification
│   ├── test_mujoco_render.py             # EGL rendering test
│   ├── error.txt                         # Historical error log
│   └── pipeline_summary.md               # Pipeline docs
│
├── ckpts/                                # Official pretrained checkpoints
│   └── official/
│       ├── base_policy/
│       │   ├── .hydra/config.yaml        # Training config (reference only)
│       │   ├── checkpoints/270.ckpt      # ★ Base policy checkpoint
│       │   └── normalizer.pth            # Action normalizer
│       └── dyn_model/
│           ├── hydra.yaml                # ★ Dynamics model config (READ by planner)
│           ├── checkpoints/model_60.pth  # ★ Dynamics model checkpoint
│           └── normalizer.pth            # Dynamics normalizer
│
└── data/                                 # Datasets (gitignored)
    └── transport/
        ├── data/
        │   ├── expert_demonstration/     # transport_ph_demo_v141_20_perc.hdf5
        │   └── rollout/                  # transport_rollout_and_demo.hdf5, transport_val.hdf5
        └── model_ckpt/
            ├── base_policy/              # Symlink or copy of ckpts/official/base_policy
            └── dyn_model/                # Symlink or copy of ckpts/official/dyn_model
```

★ = most important files to understand.

---

## 6. Key Implementation Details

### 6.1 The `.ckpt` binary config trap
**CRITICAL:** The `.ckpt` checkpoint files (torch.save with dill) contain `payload['cfg']` — the FULL training config with all original paths baked in. The `.hydra/config.yaml` text file next to a checkpoint is just a reference copy; **the code does NOT read it**.

When `eval_test_time_optimization.py:32-33` does:
```python
with open(cfg.policy_checkpoint, 'rb') as f:
    payload = torch.load(f, pickle_module=dill)
```
…then `payload['cfg'].task.dataset.dataset_path` contains the AUTHOR's original absolute paths (e.g. `/store/real/zhanyis/...`), not your local paths. This causes `FileNotFoundError` if you only edit the YAML files.

**Fix pattern** (must override in `eval_test_time_optimization.py` before calling `initialize_planner`):
```python
payload['cfg'].task.dataset.dataset_path = cfg.demo_dataset_path
cfg_task_env_runner.task.env_runner.dataset_path = cfg.demo_dataset_path
```

Note: `planner.py:69` reads `demo_dataset_config.dataset_path` BEFORE the override in `get_demo_latents()` (line 102-103) takes effect — this is a latent bug in LPB.

### 6.2 Encoder sharing & freezing
- `ResNetEncoder` (`dyn_model/models/resnet_encoder.py`) loads the obs_encoder backbone directly from the base policy checkpoint
- During dynamics training: `train_encoder: False` + `use_pretrained_encoder: True` → all encoder params frozen
- The SAME encoder instance is reused at inference for consistency

### 6.3 Latent OOD score computation
`Planner.compute_nn_reward()` (`planner.py:178-208`):
- Concatenates all demo visual latents into `demo_visual_latents` tensor
- Chunked nearest-neighbor via `torch.cdist` (chunk_size=2048 to fit in GPU memory)
- Returns `-min_distance` as "reward" (higher = closer to expert manifold)
- Environment-specific latent slicing:
  - ToolHang/Square: use `visual_latent[..., 512:]`
  - Transport: use `visual_latent[..., :1024]`

### 6.4 Frameskip action representation
The dynamics model concatenates `frameskip` (e.g. 15) consecutive actions into one super-step:
```python
# In robomimic_dset.py + planner.py
action_batch = rearrange(actions, 'b (h f) a -> b h (f a)', f=frameskip)
```
This means action_dim passed to the dynamics model = `original_action_dim * frameskip`.

### 6.5 Gradient guidance math (Eq. 4 in paper)
Modified noise prediction during guided denoising:
```
ε̂(A_t^k) = ε_θ(A_t^k) - η * sqrt(1 - ᾱ_k) * ∇_{A_t^k} δ(d_ϕ(z_t, A_t^k))
```
In code (`diffusion_unet_hybrid_image_policy.py:253-259`):
```python
trajectory0 = scheduler.step(model_output, t, trajectory).pred_original_sample
loss = self.planner.compute_loss(trajectory0, current_obs)
cond_grad = -torch.autograd.grad(loss, trajectory)[0]
grad_scale = guidance_scale * (1 - scheduler.alphas_cumprod[t]).sqrt()
trajectory = trajectory.detach() + grad_scale * cond_grad
```

### 6.6 Normalizer layering
Three normalizers coexist at inference:
1. `policy.normalizer` — for base policy obs/action normalization (loaded from base_policy/normalizer.pth)
2. `dyn_model_normalizer` — for dynamics model obs/action normalization (loaded from dyn_model/normalizer.pth)
3. `policy_action_normalizer` — bridge between policy output and dynamics input

The planner must un-normalize policy actions then re-normalize with dyn_model normalizer:
```python
# planner.py:211-213
init_actions_unnormalized = self.policy_action_normalizer.unnormalize(init_actions_normalized)
init_actions = self.dyn_model_normalizer['act'].normalize(init_actions_unnormalized)
```

---

## 7. Hyperparameters

### Transport task (representative)

| Parameter | Value | Location |
|-----------|-------|----------|
| Guidance scale η | 0.2 | `eval_transport.yaml` |
| OOD threshold τ | 2.8 | `eval_transport.yaml` |
| `guidance_start_timestep` (K_guide) | 10 | `eval_transport.yaml` |
| `n_action_steps` (T_a) | 15 | `eval_transport.yaml` |
| `n_obs_steps` (T_o) | 2 | base policy config |
| `horizon` | 32 | base policy config |
| Image size | 140 → crop 128 | env config |
| `frameskip` | 15 | `train.yaml` |
| Action dim | 20 (rotation_6d, dual arm) | env config |
| Proprio dim | 18 | env config |
| Views | 4 (2 eye-in-hand + 2 shoulder) | env config |

### Dynamics model (ViT predictor)

| Parameter | Value |
|-----------|-------|
| Depth | 6 |
| Heads | 16 |
| MLP dim | 2048 |
| Dropout | 0.1 |
| num_hist | 1 |
| num_pred | 1 (only 1 supported) |
| Action emb dim | 330 |
| Proprio emb dim | 32 |

### Rollout collection schedule (Transport)

| Parameter | Value |
|-----------|-------|
| t₀ (warmup epochs) | 70 |
| Δt (ckpt interval) | 50 |
| tfinal | 270 |
| N (episodes per ckpt) | 30 |
| Total trajectories | 150 |

### Test-time optimization per task (Table 9)

| Task | η | τ |
|------|---|---|
| Push-T | 0.05 | 3.2 |
| Square | 0.05 | 5.0 |
| Tool-Hang | 0.05 | 1.4 |
| Transport | 0.2 | 2.8 |
| Libero-10 | 0.2 | 1.1 |

---

## 8. Data Format

### Expert demos (robomimic HDF5)
```
data/
├── demo_0/
│   ├── obs/                    # Group
│   │   ├── robot0_eef_pos      (T, 3)
│   │   ├── robot0_eef_quat     (T, 4)
│   │   ├── robot0_gripper_qpos (T, 2)
│   │   ├── robot1_eef_pos      (T, 3)
│   │   ├── robot1_eef_quat     (T, 4)
│   │   ├── robot1_gripper_qpos (T, 2)
│   │   ├── robot0_eye_in_hand_image  (T, 140, 140, 3) uint8
│   │   ├── robot1_eye_in_hand_image  (T, 140, 140, 3) uint8
│   │   ├── shouldercamera0_image     (T, 140, 140, 3) uint8
│   │   └── shouldercamera1_image     (T, 140, 140, 3) uint8
│   ├── actions     (T, 14) axis_angle  # raw actions
│   ├── abs_actions (T, 20) rotation_6d # transformed (used by policy)
│   ├── states      (T, ...)
│   └── dones       (T,)
├── demo_1/
...
└── mask/  # optional train/val split
```

### Combined rollout+demo dataset
- Format: same robomimic HDF5
- Transport example: 245 train episodes (94.2%) / 15 val episodes (5.8%), ~130k train steps
- Rollout episodes have varying lengths (avg ~534 steps), reflecting success + failure mix
- `scripts/collect_rollout.py` produces this format from base policy rollouts

### Dynamics model checkpoint
```
checkpoints/model_{epoch}.pth contains:
  - encoder        (ResNet state dict, frozen)
  - predictor      (ViT state dict, trained)
  - proprio_encoder
  - action_encoder
  - epoch
```

---

## 9. Common Workflows

### 9.1 Train base policy from scratch
```bash
# 1. Place expert demos at data/transport/data/expert_demonstration/
# 2. Edit diffusion_policy/config/image_transport_diffusion_policy_cnn.yaml paths
python train.py --config-dir=. --config-name=image_transport_diffusion_policy_cnn.yaml \
    training.seed=42 training.device=cuda:0 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

### 9.2 Collect rollouts for dynamics training
Edit `scripts/collect_rollout.py` top section:
```python
START_CKPT_PATH = 'data/outputs/.../checkpoints/200.ckpt'
START_EPOCH = 200
DELTA_N = 50
END_EPOCH = 300
N_ROLLOUTS_PER_CKPT = 30
EXPERT_DATASET_PATH = 'data/transport/data/expert_demonstration/transport_ph_demo_v141_20_perc.hdf5'
OUTPUT_HDF5_PATH = 'data/transport/data/rollout/transport_rollout.hdf5'
```
Then: `PYTHONPATH=. python scripts/collect_rollout.py`

### 9.3 Train dynamics model
```bash
# 1. Edit dyn_model/conf/env/transport.yaml paths (train_data_path, val_data_path, policy_ckpt_path)
# 2. Edit dyn_model/conf/train.yaml defaults to use env: transport
python dyn_model/train.py --config-name=train.yaml
```
Multi-GPU: `accelerate launch dyn_model/train.py --config-name=train.yaml`

### 9.4 Run test-time optimization eval
```bash
# 1. Edit dyn_model/conf/planner/eval_transport.yaml with ckpt + dataset paths
# 2. IMPORTANT: paths inside .ckpt may need override (see §6.1)
python eval_test_time_optimization.py --config-name=eval_transport
```

### 9.5 Analyze dataset composition
```bash
python analyze_rollout_data.py
# Edit TRAIN_HDF5 / VAL_HDF5 at top of file first
```

---

## 10. Environment & Dependencies

### Conda setup
```bash
mamba env create -f conda_environment.yaml
conda activate lpb
```

### Headless rendering (EGL)
MuJoCo/robosuite needs EGL for offscreen rendering on headless servers:
```bash
export MUJOCO_GL=egl
# If EGL platform device unsupported, fallback:
export MUJOCO_GL=osmesa  # requires libosmesa6-dev
```
- `AsyncVectorEnv` uses `context='spawn'` (in `gym_util/async_vector_env.py`) to avoid EGL `EGL_BAD_ALLOC` from fork
- Test rendering: `python scripts/test_mujoco_render.py`

### Hardware
- Base policy training: single GPU, ~24-48h to converge
- Dynamics model training: 1-6 GPUs (via accelerate), ~24-36h
- Paper used NVIDIA L40S (46 GB VRAM)

---

## 11. Gotchas & Known Issues

1. **`.ckpt` internal paths**: Always override `payload['cfg'].task.dataset.dataset_path` and `payload['cfg'].task.env_runner.dataset_path` in `eval_test_time_optimization.py` — don't rely on the YAML files.

2. **`planner.py:69` reads dataset_path too early**: `FileUtils.get_env_metadata_from_dataset(dataset_path=...)` runs before `get_demo_latents()` override. If the demo_dataset_path override isn't applied to `demo_dataset_config` before planner init, this line fails.

3. **Latent slicing differs per env**: `compute_nn_reward()` hardcodes slicing for ToolHang/Square/Transport. Adding a new env requires editing this.

4. **`num_pred` only supports 1**: The dynamics model predictor only predicts one step ahead; longer horizons would need code changes.

5. **Encoder checkpoint loading**: `ResNetEncoder.__init__` requires the base policy `.ckpt` to exist and load successfully. A missing/mismatched policy ckpt breaks dynamics training and eval.

6. **Abs action rotation conversion**: `rotation_transformer` converts axis_angle (14-dim raw) → rotation_6d (20-dim). Both `actions` and `abs_actions` exist in HDF5; the policy uses `abs_actions`.

7. **Crop randomizer**: Base policy uses stochastic crop during training but `eval_fixed_crop: True` swaps in deterministic `CropRandomizer` at eval. The dynamics model uses `get_eval_crop_transform_resnet` (deterministic center crop).

8. **Sequential vs parallel runner**: `eval_transport.yaml` uses `robomimic_image_sequential_runner.SequentialRobomimicImageRunner` (not the parallel one) because gradient guidance requires sequential per-env processing.

---

## 12. Ablation Findings (from paper Appendix A)

- **Data source for dynamics**: Policy rollout (0.85) > Noisy demos (0.73) > Epsilon-greedy (0.71) on Transport
- **Action optimization**: Classifier guidance / LPB (0.85) > MPC (0.80) > Gradient descent (0.73)
- **Latent space**: Base policy encoder (0.85) > DINOv2 (0.79) > Reconstruction autoencoder (0.68)
- **Guidance scale η**: Sweet spot ~0.05-0.2; 0.3+ causes overshooting
- **K_guide (denoising steps with guidance)**: Robust from 10-35; last 10 steps sufficient
- **Rollout data quantity**: Performance saturates around 300 rollouts on Tool-Hang

---

## 13. Paper Results Summary (Table 1, 20% demos)

| Task | Expert BC | Mixed BC | Filtered BC | CCIL | CQL | **LPB** |
|------|-----------|----------|-------------|------|-----|---------|
| Square | 0.56 | 0.50 | 0.65 | 0.63 | 0.0 | **0.65** |
| Transport | 0.68 | 0.60 | 0.79 | 0.69 | 0.0 | **0.85** |
| Tool-Hang | 0.27 | 0.24 | 0.29 | 0.14 | 0.0 | **0.39** |
| Push-T | 0.51 | 0.47 | 0.59 | 0.48 | 0.29 | **0.65** |
| Libero-10 | 0.65 | 0.50 | 0.71 | 0.61 | 0.0 | **0.75** |

Key takeaways:
- LPB matches or beats all baselines
- Largest gains on long-horizon precision tasks (Tool-Hang, Transport)
- Filtered BC plateaus because even "successful" rollouts contain suboptimal actions
- CQL fails without dense reward

---

## 14. File-to-Concept Cross-Reference

| Paper concept | Code location |
|---------------|---------------|
| Base policy π_θ | `diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py` |
| Visual encoder h_θ | `dyn_model/models/resnet_encoder.py` (extracted from base policy) |
| Dynamics model d_ϕ | `dyn_model/models/visual_dyn_model.py` |
| Dynamics predictor f_ϕ | `dyn_model/models/vit.py` (ViTPredictor) |
| Latent OOD score δ | `dyn_model/planner.py:178` (`compute_nn_reward`) |
| Classifier guidance (Eq. 4) | `diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py:253-259` |
| Threshold τ check | `diffusion_unet_hybrid_image_policy.py:241` (`current_cost >= self.threshold`) |
| Rollout data collection | `scripts/collect_rollout.py` |
| Expert latent precomputation | `dyn_model/planner.py:98-168` (`get_demo_latents`) |
| Dynamics MSE loss (Eq. 2) | `dyn_model/models/visual_dyn_model.py:150-180` (`forward`) |
| Algorithm 1 (full inference) | `eval_test_time_optimization.py` + `planner.py` + `diffusion_unet_hybrid_image_policy.py:predict_action_dyn_guided` |
