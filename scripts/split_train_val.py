"""
Combine two HDF5 datasets, verify format, split into train/val by ratio,
and save as separate HDF5 files in robomimic format.

Pipeline:
  1. Verify format of both input files (structural + optional conversion test)
  2. Pool all episodes from both files, shuffle, split by TRAIN_RATIO
  3. Write train and val HDF5 files (renumbered demo_0, demo_1, ...)
  4. Verify format of both output files again

Usage:
    PYTHONPATH=. python scripts/split_train_val.py
"""

# ==================== CONFIGURATION ====================
# Input HDF5 files — episodes from BOTH files are pooled, then split
INPUT_HDF5_1 = '/path/to/transport_rollout.hdf5'
INPUT_HDF5_2 = '/path/to/transport_ph_demo_v141_20_perc.hdf5'

# Train fraction (e.g. 0.95 = 95% train, 5% val)
TRAIN_RATIO = 0.95

# Output files
OUTPUT_TRAIN_HDF5 = '/path/to/transport_rollout_and_demo.hdf5'
OUTPUT_VAL_HDF5 = '/path/to/transport_val.hdf5'

# Random seed for reproducible shuffling
SEED = 42

# Run full conversion test on output files.
# This loads through _convert_robomimic_to_replay (same pipeline as
# dynamics model training). Requires diffusion_policy on PYTHONPATH.
# Set to False for faster, structural-only verification.
RUN_CONVERSION_TEST = True

# abs_action flag (True for Transport / Robomimic abs_action tasks)
ABS_ACTION = True
# ========================================================

import os
import sys
import pathlib
import numpy as np
import h5py

ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT_DIR)


# ---------------------------------------------------------------------------
#  Format verification
# ---------------------------------------------------------------------------

def check_structure(hdf5_path, label=""):
    """
    Structural check of a robomimic-format HDF5 file.
    Returns (issues_list, info_dict).
    info_dict contains: n_demos, ep_lengths, action_dim, obs_keys, has_env_args
    """
    print(f"\n{'='*60}")
    print(f"  Structural check: {label}")
    print(f"  {hdf5_path}")
    print(f"{'='*60}")

    issues = []
    info = {}

    with h5py.File(hdf5_path, 'r') as f:
        if 'data' not in f:
            issues.append("ERROR: missing top-level 'data' group")
            return issues, info

        data_grp = f['data']
        demo_keys = sorted(data_grp.keys(), key=lambda x: int(x.split('_')[1]))
        n_demos = len(demo_keys)
        info['n_demos'] = n_demos
        print(f"  Num demos: {n_demos}")

        if n_demos == 0:
            issues.append("ERROR: no demos found")
            return issues, info

        # ---- env_args ----
        info['has_env_args'] = 'env_args' in data_grp.attrs
        if info['has_env_args']:
            print(f"  env_args: found")
        else:
            issues.append("WARNING: 'env_args' attribute missing "
                          "(needed for env recreation during eval)")

        # ---- first demo structure ----
        demo0 = data_grp[demo_keys[0]]

        required_keys = ['actions', 'abs_actions', 'states', 'obs']
        for key in required_keys:
            if key not in demo0:
                issues.append(f"ERROR: demo missing '{key}'")

        # ---- actions ----
        if 'actions' in demo0:
            actions = demo0['actions']
            info['action_dim'] = actions.shape[-1]
            print(f"  actions     shape: {actions.shape}  dtype: {actions.dtype}")
            if not np.all(np.isfinite(actions[:])):
                issues.append("ERROR: NaN/Inf in actions")

        # ---- abs_actions ----
        if 'abs_actions' in demo0:
            abs_actions = demo0['abs_actions']
            info['abs_action_dim'] = abs_actions.shape[-1]
            print(f"  abs_actions shape: {abs_actions.shape}  dtype: {abs_actions.dtype}")

        # ---- states ----
        if 'states' in demo0:
            states = demo0['states']
            print(f"  states      shape: {states.shape}  dtype: {states.dtype}")

        # ---- obs ----
        obs_keys = []
        if 'obs' in demo0:
            obs_grp = demo0['obs']
            obs_keys = list(obs_grp.keys())
            info['obs_keys'] = obs_keys
            print(f"  obs keys ({len(obs_keys)}):")
            for key in obs_keys:
                arr = obs_grp[key]
                is_image = ('image' in key or key.endswith('_image'))
                type_tag = 'rgb' if (is_image and arr.dtype == np.uint8) else 'low_dim'
                print(f"    {key:40s} shape: {str(arr.shape):20s} "
                      f"dtype: {arr.dtype}  ({type_tag})")

                if is_image:
                    if arr.dtype != np.uint8:
                        issues.append(f"ERROR: image '{key}' should be uint8, "
                                      f"got {arr.dtype}")
                    if len(arr.shape) != 4 or arr.shape[-1] != 3:
                        issues.append(f"ERROR: image '{key}' should be (T,H,W,3), "
                                      f"got {arr.shape}")

        # ---- consistency: all episodes same structure ----
        ref_action_shape = demo0['actions'].shape[1:] if 'actions' in demo0 else None
        ep_lens = []
        for dk in demo_keys:
            ep = data_grp[dk]
            ep_len = ep['actions'].shape[0]
            ep_lens.append(ep_len)

            if 'actions' in ep and ref_action_shape is not None:
                if ep['actions'].shape[1:] != ref_action_shape:
                    issues.append(f"ERROR: {dk} action dim mismatch: "
                                  f"{ep['actions'].shape[1:]} vs {ref_action_shape}")

            if 'obs' in ep:
                cur_obs_keys = set(ep['obs'].keys())
                if cur_obs_keys != set(obs_keys):
                    issues.append(f"ERROR: {dk} obs keys differ from demo_0: "
                                  f"{cur_obs_keys ^ set(obs_keys)}")

        ep_lens = np.array(ep_lens)
        info['ep_lengths'] = ep_lens
        print(f"\n  Episode lengths: min={ep_lens.min()}  max={ep_lens.max()}  "
              f"mean={ep_lens.mean():.1f}  total={ep_lens.sum()}")

    return issues, info


def check_conversion(hdf5_path, abs_action=True, action_dim=None):
    """
    Attempt to load through _convert_robomimic_to_replay (same pipeline
    as dynamics model training). Auto-detects shape_meta from the file.
    Returns True on success.
    """
    from diffusion_policy.dataset.robomimic_replay_image_dataset import \
        _convert_robomimic_to_replay
    from diffusion_policy.model.common.rotation_transformer import \
        RotationTransformer
    import zarr

    print(f"\n  Conversion test: {hdf5_path}")

    # ---- auto-detect shape_meta from HDF5 ----
    with h5py.File(hdf5_path, 'r') as f:
        demo_keys = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))
        demo0 = f['data'][demo_keys[0]]
        obs_meta = {}
        for key in demo0['obs'].keys():
            arr = demo0['obs'][key]
            shape = list(arr.shape[1:])  # drop T
            if len(shape) == 3 and arr.dtype == np.uint8:
                # (H, W, 3) -> (3, H, W)
                obs_meta[key] = {'shape': [shape[2], shape[0], shape[1]],
                                 'type': 'rgb'}
            else:
                obs_meta[key] = {'shape': shape, 'type': 'low_dim'}

        if action_dim is None:
            action_dim = demo0['abs_actions'].shape[-1] if abs_action \
                else demo0['actions'].shape[-1]

    shape_meta = {'obs': obs_meta, 'action': {'shape': [action_dim]}}
    rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

    try:
        replay_buffer = _convert_robomimic_to_replay(
            store=zarr.MemoryStore(),
            shape_meta=shape_meta,
            dataset_path=hdf5_path,
            abs_action=abs_action,
            rotation_transformer=rotation_transformer,
        )
        n_steps = replay_buffer.episode_ends[-1] if len(
            replay_buffer.episode_ends) > 0 else 0
        print(f"  Conversion succeeded: {len(replay_buffer.episode_ends)} eps, "
              f"{n_steps} steps")
        return True
    except Exception as e:
        print(f"  Conversion FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
#  Episode collection and splitting
# ---------------------------------------------------------------------------

def collect_episodes(file_path):
    """Return sorted list of episode keys from a file."""
    with h5py.File(file_path, 'r') as f:
        keys = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))
    return keys


def copy_episode(src_file, src_key, dest_data_grp, dest_key,
                 obs_keys_to_keep=None):
    """
    Copy one episode from src_file to dest_data_grp.

    If obs_keys_to_keep is given, only those obs keys are copied; any
    extra keys in the source (e.g. joint_pos, velocities in expert demo)
    are silently dropped so the output has a uniform obs schema across
    all episodes regardless of which input file they came from.
    """
    src_demo = src_file[f'data/{src_key}']
    dest_demo = dest_data_grp.create_group(dest_key)

    # ---- copy scalar datasets ----
    for key in ['actions', 'abs_actions', 'states']:
        if key in src_demo:
            dest_demo.create_dataset(key, data=src_demo[key][:])

    # ---- copy obs (filtered) ----
    src_obs = src_demo['obs']
    dest_obs = dest_demo.create_group('obs')
    if obs_keys_to_keep is None:
        obs_keys_to_keep = list(src_obs.keys())
    for key in obs_keys_to_keep:
        if key in src_obs:
            dest_obs.create_dataset(key, data=src_obs[key][:])


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    input_paths = [INPUT_HDF5_1, INPUT_HDF5_2]
    output_paths = [OUTPUT_TRAIN_HDF5, OUTPUT_VAL_HDF5]

    # ---- validate paths ----
    for p in input_paths:
        assert os.path.isfile(p), f'Input file not found: {p}'
    for p in output_paths:
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # ================================================================
    #  STEP 1: Verify input files
    # ================================================================
    print('\n' + '#'*60)
    print('#  STEP 1: Verify input files')
    print('#'*60)

    input_infos = []
    for path, tag in zip(input_paths, ['input_1', 'input_2']):
        issues, info = check_structure(path, tag)
        input_infos.append(info)
        if any('ERROR' in i for i in issues):
            print(f"\n  FATAL: {tag} has structural errors:")
            for i in issues:
                if 'ERROR' in i:
                    print(f"    {i}")
            sys.exit(1)
        elif issues:
            print(f"\n  {tag} has warnings:")
            for i in issues:
                print(f"    {i}")

    # ---- cross-file consistency check ----
    # Obs keys may differ between files (e.g. expert demo has extra
    # modalities like joint_pos, velocities, object state that rollout
    # data doesn't save). We take the INTERSECTION and only keep common
    # keys in the output. Action dim must still match exactly.
    print(f"\n  Cross-file consistency:")
    common_obs = set(input_infos[0].get('obs_keys', []))
    for idx, info in enumerate(input_infos[1:], 1):
        cur_obs = set(info.get('obs_keys', []))
        only_prev = common_obs - cur_obs
        only_cur  = cur_obs - common_obs
        if only_prev:
            print(f"    input_{idx+1} missing (dropped): {sorted(only_prev)}")
        if only_cur:
            print(f"    input_{idx+1} extra   (dropped): {sorted(only_cur)}")
        common_obs = common_obs & cur_obs
        cur_adim = info.get('action_dim')
        if cur_adim != input_infos[0].get('action_dim'):
            print(f"    ERROR: action dim differs: "
                  f"{input_infos[0].get('action_dim')} vs {cur_adim}")
            sys.exit(1)
    print(f"    Common obs keys ({len(common_obs)}): {sorted(common_obs)}")
    print(f"    action dim match ({input_infos[0].get('action_dim')})")
    if len(common_obs) == 0:
        print(f"    ERROR: no common obs keys between files!")
        sys.exit(1)

    # ================================================================
    #  STEP 2: Pool episodes and split
    # ================================================================
    print('\n' + '#'*60)
    print('#  STEP 2: Pool and split episodes')
    print('#'*60)

    all_episodes = []  # list of (file_path, demo_key)
    for path in input_paths:
        keys = collect_episodes(path)
        all_episodes.extend([(path, k) for k in keys])
        print(f"  {os.path.basename(path)}: {len(keys)} episodes")

    total = len(all_episodes)
    print(f"  Total pooled: {total} episodes")

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(total)
    n_train = int(round(total * TRAIN_RATIO))
    train_idx = sorted(perm[:n_train].tolist())
    val_idx = sorted(perm[n_train:].tolist())

    print(f"  Split (ratio={TRAIN_RATIO}): "
          f"train={len(train_idx)}, val={len(val_idx)}")
    print(f"  (seed={SEED})")

    # ================================================================
    #  STEP 3: Write output files
    # ================================================================
    print('\n' + '#'*60)
    print('#  STEP 3: Write output files')
    print('#'*60)

    # Use env_args from first input file that has it
    env_args_source = None
    for path in input_paths:
        with h5py.File(path, 'r') as f:
            if 'env_args' in f['data'].attrs:
                env_args_source = path
                break

    splits = [
        (OUTPUT_TRAIN_HDF5, train_idx, 'train'),
        (OUTPUT_VAL_HDF5, val_idx, 'val'),
    ]

    for out_path, split_indices, tag in splits:
        print(f"\n  Writing {tag}: {out_path}")
        print(f"    {len(split_indices)} episodes")

        # Open all unique source files + output file
        src_files = {p: h5py.File(p, 'r') for p in input_paths}

        with h5py.File(out_path, 'w') as out_f:
            data_grp = out_f.create_group('data')

            # Copy env_args
            if env_args_source is not None:
                data_grp.attrs['env_args'] = src_files[env_args_source][
                    'data'].attrs['env_args']

            for new_idx, global_idx in enumerate(split_indices):
                src_path, src_key = all_episodes[global_idx]
                dest_key = f'demo_{new_idx}'
                copy_episode(src_files[src_path], src_key, data_grp, dest_key,
                             obs_keys_to_keep=common_obs)

            data_grp.attrs['total'] = len(split_indices)

        for f in src_files.values():
            f.close()

        # Report
        with h5py.File(out_path, 'r') as f:
            ep_lens = []
            for k in sorted(f['data'].keys(),
                            key=lambda x: int(x.split('_')[1])):
                ep_lens.append(f[f'data/{k}/actions'].shape[0])
            ep_lens = np.array(ep_lens)
            print(f"    Written: {len(ep_lens)} eps, "
                  f"{ep_lens.sum()} steps, "
                  f"mean_len={ep_lens.mean():.1f}")

    # ================================================================
    #  STEP 4: Verify output files
    # ================================================================
    print('\n' + '#'*60)
    print('#  STEP 4: Verify output files')
    print('#'*60)

    all_ok = True
    for out_path, _, tag in splits:
        issues, info = check_structure(out_path, f"output {tag}")

        has_errors = any('ERROR' in i for i in issues)
        if has_errors:
            print(f"\n  FAILED: {tag} has structural errors:")
            for i in issues:
                if 'ERROR' in i:
                    print(f"    {i}")
            all_ok = False
        else:
            print(f"\n  {tag} structural check: OK")

        if RUN_CONVERSION_TEST and not has_errors:
            ok = check_conversion(out_path,
                                  abs_action=ABS_ACTION,
                                  action_dim=info.get('action_dim'))
            if not ok:
                all_ok = False
                print(f"  {tag} conversion test: FAILED")
            else:
                print(f"  {tag} conversion test: OK")

    # ================================================================
    #  Summary
    # ================================================================
    print('\n' + '='*60)
    if all_ok:
        print("  RESULT: All checks passed")
        print(f"    Train: {OUTPUT_TRAIN_HDF5}  ({len(train_idx)} eps)")
        print(f"    Val:   {OUTPUT_VAL_HDF5}  ({len(val_idx)} eps)")
    else:
        print("  RESULT: Some checks FAILED — review output above")
    print('='*60)


if __name__ == '__main__':
    main()
