"""
Collect rollout trajectories from base policy checkpoints for dynamics model training.

This script loads base diffusion policy checkpoints at specified intervals,
runs rollouts using ONLY the base policy (NO dynamics model guidance),
and saves all trajectories to a single HDF5 file in robomimic format.

Usage:
    PYTHONPATH=. python collect_rollout.py
"""

# ==================== CONFIGURATION ====================
# Starting checkpoint path (provides the directory and start epoch)
START_CKPT_PATH = '/path/to/data/outputs/.../checkpoints/200.ckpt'

# Starting epoch number (must match the filename in START_CKPT_PATH)
START_EPOCH = 200

# Checkpoint interval (loads START_EPOCH, START_EPOCH+DELTA_N, ...)
DELTA_N = 50

# Ending epoch (inclusive)
END_EPOCH = 300

# Number of rollout episodes per checkpoint
N_ROLLOUTS_PER_CKPT = 30

# Expert demonstration dataset (for environment configuration and metadata)
EXPERT_DATASET_PATH = '/path/to/transport_ph_demo_v141_20_perc.hdf5'

# Output HDF5 file (all rollouts from all checkpoints combined)
OUTPUT_HDF5_PATH = '/path/to/transport_rollout.hdf5'

# Starting seed for test environments (each rollout uses a unique seed)
TEST_START_SEED = 100000

# Device
DEVICE = 'cuda'
# ========================================================


import os
import sys
import pathlib
import collections
import numpy as np
import torch
import h5py
import tqdm
import dill

ROOT_DIR = str(pathlib.Path(__file__).parent.absolute())
sys.path.insert(0, ROOT_DIR)

from omegaconf import OmegaConf
import hydra

from diffusion_policy.dataset.robomimic_replay_image_dataset import undo_transform_action
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


# ---------------------------------------------------------------------------
#  Policy loading
# ---------------------------------------------------------------------------

def load_policy(ckpt_path, device):
    """
    Load base diffusion policy from a checkpoint.
    Uses ONLY the base policy — no dynamics model / planner is loaded.
    """
    print(f'  Loading checkpoint: {ckpt_path}')

    with open(ckpt_path, 'rb') as f:
        payload = torch.load(f, pickle_module=dill, map_location='cpu')

    cfg = payload['cfg']
    use_ema = cfg.training.get('use_ema', False)

    # Instantiate policy via Hydra (handles nested _target_ objects like
    # DDPMScheduler correctly).
    policy = hydra.utils.instantiate(cfg.policy)

    # Load model weights — prefer EMA if available (matches training-time
    # rollout behaviour in train_diffusion_unet_hybrid_workspace.py:217-218).
    if use_ema and 'ema_model' in payload['state_dicts']:
        policy.load_state_dict(payload['state_dicts']['ema_model'])
        print('  Using EMA model weights')
    else:
        policy.load_state_dict(payload['state_dicts']['model'])
        print('  Using regular model weights')

    # Also load normalizer from the separate normalizer.pth file if it exists
    # (same logic as eval_test_time_optimization.py).
    ckpt_dir = os.path.dirname(ckpt_path)
    output_dir = os.path.dirname(ckpt_dir)
    normalizer_path = os.path.join(output_dir, 'normalizer.pth')
    if os.path.exists(normalizer_path):
        normalizer_state_dict = torch.load(normalizer_path, map_location='cpu')
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(normalizer_state_dict)
        policy.set_normalizer(normalizer)
        print(f'  Loaded normalizer from {normalizer_path}')
    else:
        print('  Warning: normalizer.pth not found, '
              'relying on checkpoint normalizer')

    policy.to(device)
    policy.eval()

    return policy, cfg


# ---------------------------------------------------------------------------
#  Environment creation
# ---------------------------------------------------------------------------

def create_env_wrapper(env_meta, shape_meta, abs_action):
    """
    Create a robomimic environment wrapped in RobomimicImageWrapper.
    No MultiStepWrapper / VideoRecordingWrapper — we manage obs history
    and stepping manually so that we can record every (obs, action, state)
    triple for saving.
    """
    # Observation modality mapping (required by robomimic)
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    # Create the robomimic env
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )
    # Disable hard reset to save memory (same as the runners)
    env.env.hard_reset = False

    # Pick any image key as render_obs_key (only used by render(); we never
    # call render, but the wrapper needs a valid key).
    image_keys = [k for k, v in shape_meta['obs'].items()
                  if v.get('type') == 'rgb']
    render_obs_key = image_keys[0] if image_keys else \
        list(shape_meta['obs'].keys())[0]

    env_wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta,
        init_state=None,
        render_obs_key=render_obs_key,
    )
    return env_wrapper


# ---------------------------------------------------------------------------
#  Single rollout
# ---------------------------------------------------------------------------

def run_single_rollout(env_wrapper, policy, seed,
                       n_obs_steps, n_action_steps, max_steps,
                       abs_action, rotation_transformer, device):
    """
    Run one rollout episode using ONLY the base policy
    (policy.predict_action — no dynamics-model guidance).

    Returns
    -------
    obs_list    : list[dict]   length T+1  (includes terminal obs)
    action_list : list[np.ndarray]  length T, each (14,) axis_angle
    state_list  : list[np.ndarray]  length T+1
    """
    # ---- reset ----
    env_wrapper.seed(seed)
    obs = env_wrapper.reset()
    if hasattr(policy, 'reset'):
        policy.reset()

    obs_list    = [obs]
    state_list  = [env_wrapper.get_flattened_state()]
    action_list = []

    done = False
    step = 0

    while not done and step < max_steps:
        # ---- build (n_obs_steps,) stacked obs dict for policy ----
        n_recent = min(len(obs_list), n_obs_steps)
        recent = obs_list[-n_recent:]
        # pad at the front if the episode just started
        while len(recent) < n_obs_steps:
            recent.insert(0, obs_list[0])

        obs_dict = {}
        for key in recent[0]:
            obs_dict[key] = np.stack([o[key] for o in recent])  # (To, *)

        # add batch dim, move to device
        obs_tensor = dict_apply(
            obs_dict,
            lambda x: torch.from_numpy(np.expand_dims(x, axis=0)).to(device))

        # ---- predict action chunk with base policy only ----
        with torch.no_grad():
            action_dict = policy.predict_action(obs_tensor)

        # (n_action_steps, action_dim)  — action_dim = 20 for Transport
        action = action_dict['action'][0].cpu().numpy()

        if not np.all(np.isfinite(action)):
            print(f'    Warning: NaN/Inf action at step {step}, '
                  f'ending episode early')
            break

        # Convert abs_action (rotation_6d, 20-d) -> axis_angle (14-d)
        if abs_action:
            env_actions = undo_transform_action(action, rotation_transformer)
        else:
            env_actions = action

        # ---- execute the action chunk step by step ----
        for i in range(n_action_steps):
            if done or step >= max_steps:
                break

            obs, reward, done, info = env_wrapper.step(env_actions[i])

            obs_list.append(obs)
            state_list.append(env_wrapper.get_flattened_state())
            action_list.append(env_actions[i])
            step += 1

            if done:
                break

    return obs_list, action_list, state_list


# ---------------------------------------------------------------------------
#  HDF5 saving
# ---------------------------------------------------------------------------

def save_episode_to_hdf5(demo_grp, obs_list, action_list, state_list,
                         rgb_keys, lowdim_keys):
    """
    Save one rollout episode into an HDF5 demo group in robomimic format.

    HDF5 layout produced:
        demo_i/
            actions      (T, 14)  float32   axis_angle
            abs_actions  (T, 14)  float32   axis_angle (same as actions
                                               for control_delta=False)
            states       (T, D)   float32   sim states
            obs/
                <rgb_key>     (T, H, W, C)  uint8
                <lowdim_key>  (T, dim)      float32
    """
    T = len(action_list)
    if T == 0:
        return False

    # ---- actions ----
    actions = np.array(action_list, dtype=np.float32)       # (T, 14)
    demo_grp.create_dataset('actions', data=actions)
    # abs_actions is the same 14-d axis_angle for Transport
    # (control_delta=False).  _convert_robomimic_to_replay will convert
    # abs_actions -> rotation_6d (20-d) when loading for training.
    demo_grp.create_dataset('abs_actions', data=actions)

    # ---- sim states ----
    states = np.array(state_list[:T], dtype=np.float32)     # (T, D)
    demo_grp.create_dataset('states', data=states)

    # ---- observations ----
    obs_grp = demo_grp.create_group('obs')

    # Images: env returns (C, H, W) float [0,1].
    #         HDF5 / robomimic convention is (H, W, C) uint8 [0,255].
    for key in rgb_keys:
        imgs = np.stack([obs_list[t][key] for t in range(T)])  # (T,C,H,W)
        imgs = np.moveaxis(imgs, 1, -1)                        # (T,H,W,C)
        imgs = (imgs * 255).clip(0, 255).astype(np.uint8)
        obs_grp.create_dataset(key, data=imgs)

    # Low-dim observations
    for key in lowdim_keys:
        data = np.stack([obs_list[t][key] for t in range(T)])
        obs_grp.create_dataset(key, data=data.astype(np.float32))

    return True


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    # ---- validate paths ----
    ckpt_dir = os.path.dirname(START_CKPT_PATH)
    assert os.path.isdir(ckpt_dir), \
        f'Checkpoint directory not found: {ckpt_dir}'
    assert os.path.isfile(START_CKPT_PATH), \
        f'Start checkpoint not found: {START_CKPT_PATH}'
    assert os.path.isfile(EXPERT_DATASET_PATH), \
        f'Expert dataset not found: {EXPERT_DATASET_PATH}'

    out_parent = os.path.dirname(OUTPUT_HDF5_PATH)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    # ---- env metadata from expert dataset ----
    print(f'Loading env metadata from: {EXPERT_DATASET_PATH}')
    env_meta = FileUtils.get_env_metadata_from_dataset(EXPERT_DATASET_PATH)
    env_meta['env_kwargs']['use_object_obs'] = False

    # ---- load first checkpoint just to read cfg ----
    first_ckpt = os.path.join(ckpt_dir, f'{START_EPOCH}.ckpt')
    print(f'Reading config from: {first_ckpt}')
    with open(first_ckpt, 'rb') as f:
        first_payload = torch.load(f, pickle_module=dill,
                                   map_location='cpu')
    cfg = first_payload['cfg']
    del first_payload

    # ---- extract configuration from cfg ----
    abs_action  = cfg.task.env_runner.abs_action
    shape_meta  = OmegaConf.to_container(cfg.task.dataset.shape_meta,
                                         resolve=True)
    n_obs_steps    = cfg.policy.n_obs_steps
    n_action_steps = cfg.policy.n_action_steps
    max_steps      = cfg.task.env_runner.max_steps

    if abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d') \
        if abs_action else None

    # classify obs keys
    rgb_keys, lowdim_keys = [], []
    for key, attr in shape_meta['obs'].items():
        obs_type = attr.get('type', 'low_dim')
        if obs_type == 'rgb':
            rgb_keys.append(key)
        elif obs_type == 'low_dim':
            lowdim_keys.append(key)

    print('\n===== Configuration =====')
    print(f'  abs_action     : {abs_action}')
    print(f'  n_obs_steps    : {n_obs_steps}')
    print(f'  n_action_steps : {n_action_steps}')
    print(f'  max_steps      : {max_steps}')
    print(f'  rgb_keys       : {rgb_keys}')
    print(f'  lowdim_keys    : {lowdim_keys}')
    print(f'  epochs         : {list(range(START_EPOCH, END_EPOCH+1, DELTA_N))}')
    print('=========================\n')

    # ---- create env (reused across all rollouts) ----
    print('Creating environment ...')
    env_wrapper = create_env_wrapper(env_meta, shape_meta, abs_action)

    # ---- open output HDF5 ----
    with h5py.File(OUTPUT_HDF5_PATH, 'w') as out_f:
        data_grp = out_f.create_group('data')

        # copy env_args from expert dataset (needed for env recreation
        # during dynamics-model eval)
        try:
            with h5py.File(EXPERT_DATASET_PATH, 'r') as ef:
                if 'env_args' in ef['data'].attrs:
                    data_grp.attrs['env_args'] = ef['data'].attrs['env_args']
        except Exception:
            print('  Warning: could not copy env_args from expert dataset')

        demo_idx = 0

        # ---- iterate over checkpoints ----
        for epoch in range(START_EPOCH, END_EPOCH + 1, DELTA_N):
            ckpt_path = os.path.join(ckpt_dir, f'{epoch}.ckpt')
            if not os.path.isfile(ckpt_path):
                print(f'\n[epoch {epoch}] checkpoint not found, skipping')
                continue

            print(f'\n[epoch {epoch}] loading policy ...')
            policy, _ = load_policy(ckpt_path, DEVICE)

            print(f'[epoch {epoch}] running {N_ROLLOUTS_PER_CKPT} rollouts ...')
            for ri in tqdm.tqdm(range(N_ROLLOUTS_PER_CKPT),
                                desc=f'  ckpt {epoch}',
                                mininterval=5.0):
                # Reuse the same N_ROLLOUTS_PER_CKPT seeds for every
                # checkpoint so that env resets after the first ckpt use
                # the cached seed_state_map (much faster).
                seed = TEST_START_SEED + ri

                try:
                    obs_list, action_list, state_list = run_single_rollout(
                        env_wrapper, policy, seed,
                        n_obs_steps, n_action_steps, max_steps,
                        abs_action, rotation_transformer, DEVICE)
                except Exception as e:
                    print(f'    Error during rollout (seed={seed}): {e}')
                    continue

                if len(action_list) == 0:
                    print(f'    Empty rollout (seed={seed}), skipping')
                    continue

                demo_grp = data_grp.create_group(f'demo_{demo_idx}')
                ok = save_episode_to_hdf5(
                    demo_grp, obs_list, action_list, state_list,
                    rgb_keys, lowdim_keys)
                if ok:
                    demo_grp.attrs['ckpt_epoch'] = epoch
                    demo_grp.attrs['seed'] = seed
                    demo_idx += 1

            # free GPU memory before loading the next checkpoint
            del policy
            torch.cuda.empty_cache()

        data_grp.attrs['total'] = demo_idx

    # ---- summary ----
    print(f'\n===== Done =====')
    print(f'  Output         : {OUTPUT_HDF5_PATH}')
    print(f'  Total episodes : {demo_idx}')
    ep_lens = []
    with h5py.File(OUTPUT_HDF5_PATH, 'r') as f:
        for i in range(demo_idx):
            ep_lens.append(f[f'data/demo_{i}/actions'].shape[0])
    if ep_lens:
        ep_lens = np.array(ep_lens)
        print(f'  Episode length : mean={ep_lens.mean():.1f}  '
              f'min={ep_lens.min()}  max={ep_lens.max()}')
    print('================')


if __name__ == '__main__':
    main()
