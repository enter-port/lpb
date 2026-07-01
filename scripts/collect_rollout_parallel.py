"""
Parallel version of collect_rollout.py — runs N_ENVS environments
concurrently using AsyncVectorEnv, giving near-linear speedup.

Architecture:
    Each AsyncVectorEnv worker runs:
        MultiStepWrapper(
            TrajectoryRecordingWrapper(
                RobomimicImageWrapper(robomimic_env)
            )
        )
    The TrajectoryRecordingWrapper sits INSIDE MultiStepWrapper so it
    captures every individual (obs, action, state) transition.

Key detail: this repo's AsyncVectorEnv has auto-reset DISABLED
(async_vector_env.py line 580-581 — reset call is commented out).
So each env runs exactly one episode. The lockstep loop continues
until np.all(done); envs that finish early become no-ops (MultiStepWrapper
breaks on done[-1]==True) and wait for the slowest env.

Usage:
    PYTHONPATH=. python scripts/collect_rollout_parallel.py
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

# Number of parallel environments (match your GPU/CPU headroom;
# the official runner uses 28 for Transport)
N_ENVS = 14

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
import gym

ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT_DIR)

from omegaconf import OmegaConf
import hydra

from diffusion_policy.dataset.robomimic_replay_image_dataset import undo_transform_action
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


# ---------------------------------------------------------------------------
#  TrajectoryRecordingWrapper
# ---------------------------------------------------------------------------

class TrajectoryRecordingWrapper(gym.Wrapper):
    """
    Records every individual (obs, action, state) transition.

    MUST be placed INSIDE MultiStepWrapper so that MultiStepWrapper calls
    this wrapper's step() once per action (not once per action chunk).

    After rollout, retrieve data via get_recorded_trajectory().
    """

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._obs_list = [obs]
        self._action_list = []
        self._state_list = [self.env.get_flattened_state()]
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._action_list.append(action)
        self._obs_list.append(obs)
        self._state_list.append(self.env.get_flattened_state())
        return obs, reward, done, info

    def get_recorded_trajectory(self):
        """
        Returns dict with:
            obs_list:    list of T+1 obs dicts (initial + after each action)
            action_list: list of T action arrays
            state_list:  list of T+1 state arrays
        """
        T = len(self._action_list)
        return {
            'obs_list': self._obs_list[:T + 1],
            'action_list': self._action_list,
            'state_list': self._state_list[:T + 1],
        }


# ---------------------------------------------------------------------------
#  Policy loading (same as sequential)
# ---------------------------------------------------------------------------

def load_policy(ckpt_path, device):
    print(f'  Loading checkpoint: {ckpt_path}')

    with open(ckpt_path, 'rb') as f:
        payload = torch.load(f, pickle_module=dill, map_location='cpu')

    cfg = payload['cfg']
    use_ema = cfg.training.get('use_ema', False)

    policy = hydra.utils.instantiate(cfg.policy)

    if use_ema and 'ema_model' in payload['state_dicts']:
        policy.load_state_dict(payload['state_dicts']['ema_model'])
        print('  Using EMA model weights')
    else:
        policy.load_state_dict(payload['state_dicts']['model'])
        print('  Using regular model weights')

    ckpt_dir = os.path.dirname(ckpt_path)
    output_dir = os.path.dirname(ckpt_dir)
    normalizer_path = os.path.join(output_dir, 'normalizer.pth')
    if os.path.exists(normalizer_path):
        normalizer_state_dict = torch.load(normalizer_path, map_location='cpu')
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(normalizer_state_dict)
        policy.set_normalizer(normalizer)
        print(f'  Loaded normalizer from {normalizer_path}')

    policy.to(device)
    policy.eval()
    return policy, cfg


# ---------------------------------------------------------------------------
#  Environment creation
# ---------------------------------------------------------------------------

def create_robomimic_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    env.env.hard_reset = False
    return env


def make_env_fns(env_meta, shape_meta, n_obs_steps, n_action_steps,
                 max_steps, render_obs_key):
    """
    Create env_fn (with rendering) and dummy_env_fn (without rendering).
    The dummy is needed by AsyncVectorEnv to initialise observation/action
    spaces without creating an EGL context in the parent process.
    """

    def _build(robomimic_env):
        return MultiStepWrapper(
            TrajectoryRecordingWrapper(
                RobomimicImageWrapper(
                    env=robomimic_env,
                    shape_meta=shape_meta,
                    init_state=None,
                    render_obs_key=render_obs_key,
                )
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps,
        )

    def env_fn():
        return _build(create_robomimic_env(env_meta, shape_meta,
                                           enable_render=True))

    def dummy_env_fn():
        return _build(create_robomimic_env(env_meta, shape_meta,
                                            enable_render=False))

    return env_fn, dummy_env_fn


# ---------------------------------------------------------------------------
#  Parallel rollout — one batch of N_ENVS seeds
# ---------------------------------------------------------------------------

def run_parallel_batch(env, policy, batch_seeds, n_envs,
                       abs_action, rotation_transformer, device,
                       tqdm_desc=''):
    """
    Run one batch of rollouts in parallel.

    Parameters
    ----------
    env          : AsyncVectorEnv (already created, n_envs workers)
    policy       : loaded base policy on `device`
    batch_seeds  : list of seeds (len <= n_envs)

    Returns
    -------
    list of trajectory dicts (len == len(batch_seeds))
    """
    n_active = len(batch_seeds)

    # Pad to n_envs with the first seed (results from padded envs are discarded)
    padded_seeds = list(batch_seeds)
    if n_active < n_envs:
        padded_seeds += [batch_seeds[0]] * (n_envs - n_active)

    # ---- initialise each env with its seed ----
    def make_init_fn(seed):
        def init_fn(env):
            # env = MultiStepWrapper
            # env.env = TrajectoryRecordingWrapper
            # env.env.env = RobomimicImageWrapper
            env.env.env.init_state = None
            env.seed(seed)
        return dill.dumps(init_fn)

    init_fns = [make_init_fn(s) for s in padded_seeds]
    env.call_each('run_dill_function', args_list=[(f,) for f in init_fns])

    # ---- reset all envs (clears recordings) ----
    obs = env.reset()
    policy.reset()

    # ---- lockstep rollout until all envs done ----
    pbar = tqdm.tqdm(desc=tqdm_desc, total=700, leave=False,
                     mininterval=5.0)
    done = False
    while not done:
        obs_dict = dict(obs)
        obs_tensor = dict_apply(
            obs_dict, lambda x: torch.from_numpy(np.asarray(x)).to(device))

        with torch.no_grad():
            action_dict = policy.predict_action(obs_tensor)

        action = action_dict['action'].detach().to('cpu').numpy()
        # action shape: (n_envs, n_action_steps, action_dim)

        if not np.all(np.isfinite(action)):
            print(f'  Warning: NaN/Inf action, ending batch early')
            break

        # Convert abs_action (rotation_6d, 20-d) → axis_angle (14-d)
        if abs_action:
            env_action = undo_transform_action(action, rotation_transformer)
        else:
            env_action = action

        obs, reward, done, info = env.step(env_action)
        done = np.all(done)
        pbar.update(action.shape[1])

    pbar.close()

    # ---- collect recorded trajectories ----
    all_trajs = env.call('get_recorded_trajectory')
    return all_trajs[:n_active]


# ---------------------------------------------------------------------------
#  HDF5 saving (same format as sequential script)
# ---------------------------------------------------------------------------

def save_episode_to_hdf5(demo_grp, trajectory, rgb_keys, lowdim_keys):
    obs_list = trajectory['obs_list']
    action_list = trajectory['action_list']
    state_list = trajectory['state_list']

    T = len(action_list)
    if T == 0:
        return False

    # actions (axis_angle, 14-d for Transport dual-arm)
    actions = np.array(action_list, dtype=np.float32)
    demo_grp.create_dataset('actions', data=actions)
    demo_grp.create_dataset('abs_actions', data=actions)

    # states
    states = np.array(state_list[:T], dtype=np.float32)
    demo_grp.create_dataset('states', data=states)

    # observations
    obs_grp = demo_grp.create_group('obs')

    for key in rgb_keys:
        imgs = np.stack([obs_list[t][key] for t in range(T)])  # (T,C,H,W)
        imgs = np.moveaxis(imgs, 1, -1)                         # (T,H,W,C)
        imgs = (imgs * 255).clip(0, 255).astype(np.uint8)
        obs_grp.create_dataset(key, data=imgs)

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
    assert os.path.isdir(ckpt_dir), f'Checkpoint dir not found: {ckpt_dir}'
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

    # ---- read config from first checkpoint ----
    first_ckpt = os.path.join(ckpt_dir, f'{START_EPOCH}.ckpt')
    print(f'Reading config from: {first_ckpt}')
    with open(first_ckpt, 'rb') as f:
        first_payload = torch.load(f, pickle_module=dill, map_location='cpu')
    cfg = first_payload['cfg']
    del first_payload

    abs_action = cfg.task.env_runner.abs_action
    shape_meta = OmegaConf.to_container(cfg.task.dataset.shape_meta,
                                        resolve=True)
    n_obs_steps = cfg.policy.n_obs_steps
    n_action_steps = cfg.policy.n_action_steps
    max_steps = cfg.task.env_runner.max_steps

    if abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d') \
        if abs_action else None

    rgb_keys, lowdim_keys = [], []
    for key, attr in shape_meta['obs'].items():
        obs_type = attr.get('type', 'low_dim')
        if obs_type == 'rgb':
            rgb_keys.append(key)
        elif obs_type == 'low_dim':
            lowdim_keys.append(key)

    print('\n===== Configuration =====')
    print(f'  abs_action      : {abs_action}')
    print(f'  n_obs_steps     : {n_obs_steps}')
    print(f'  n_action_steps  : {n_action_steps}')
    print(f'  max_steps       : {max_steps}')
    print(f'  N_ENVS          : {N_ENVS}')
    print(f'  N_ROLLOUTS/ckpt : {N_ROLLOUTS_PER_CKPT}')
    print(f'  rgb_keys        : {rgb_keys}')
    print(f'  lowdim_keys     : {lowdim_keys}')
    print(f'  epochs          : {list(range(START_EPOCH, END_EPOCH+1, DELTA_N))}')
    print('=========================\n')

    # ---- create parallel envs (reused across all checkpoints) ----
    image_keys = [k for k, v in shape_meta['obs'].items()
                  if v.get('type') == 'rgb']
    render_obs_key = image_keys[0] if image_keys \
        else list(shape_meta['obs'].keys())[0]

    env_fn, dummy_env_fn = make_env_fns(
        env_meta, shape_meta, n_obs_steps, n_action_steps,
        max_steps, render_obs_key)

    print(f'Creating {N_ENVS} parallel environments ...')
    env = AsyncVectorEnv(
        [env_fn] * N_ENVS,
        dummy_env_fn=dummy_env_fn,
    )

    # ---- open output HDF5 ----
    with h5py.File(OUTPUT_HDF5_PATH, 'w') as out_f:
        data_grp = out_f.create_group('data')

        # copy env_args from expert dataset
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

            # seeds for this checkpoint
            base_seed = TEST_START_SEED + epoch  # unique per checkpoint
            all_seeds = [base_seed + ri for ri in range(N_ROLLOUTS_PER_CKPT)]

            n_collected = 0
            # ---- run in batches of N_ENVS ----
            for batch_start in range(0, N_ROLLOUTS_PER_CKPT, N_ENVS):
                batch_seeds = all_seeds[batch_start:batch_start + N_ENVS]
                batch_label = (f'  ckpt {epoch} '
                               f'[{batch_start}:{batch_start+len(batch_seeds)}]')

                try:
                    trajectories = run_parallel_batch(
                        env, policy, batch_seeds, N_ENVS,
                        abs_action, rotation_transformer, DEVICE,
                        tqdm_desc=batch_label)
                except Exception as e:
                    print(f'  Error in batch (seeds {batch_seeds}): {e}')
                    continue

                # ---- save trajectories ----
                for traj in trajectories:
                    T = len(traj['action_list'])
                    if T == 0:
                        continue

                    demo_grp = data_grp.create_group(f'demo_{demo_idx}')
                    ok = save_episode_to_hdf5(
                        demo_grp, traj, rgb_keys, lowdim_keys)
                    if ok:
                        demo_grp.attrs['ckpt_epoch'] = epoch
                        demo_idx += 1
                        n_collected += 1

                out_f.flush()  # persist to disk after each batch

            print(f'[epoch {epoch}] collected {n_collected} episodes')

            # free GPU memory before loading next checkpoint
            del policy
            torch.cuda.empty_cache()

        data_grp.attrs['total'] = demo_idx

    # ---- close envs ----
    env.close()

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
