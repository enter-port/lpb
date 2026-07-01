"""
Verify that collected rollout HDF5 data has the correct format
for dynamics model training.

Usage:
    PYTHONPATH=. python verify_rollout_data.py
"""

# ==================== CONFIGURATION ====================
ROLLOUT_HDF5_PATH = '/path/to/transport_rollout.hdf5'
EXPERT_HDF5_PATH  = '/path/to/transport_ph_demo_v141.hdf5'
ABS_ACTION        = True   # True for Transport
# ========================================================

import os
import sys
import pathlib
import numpy as np
import h5py
import collections

ROOT_DIR = str(pathlib.Path(__file__).parent.absolute())
sys.path.insert(0, ROOT_DIR)

from diffusion_policy.dataset.robomimic_replay_image_dataset import _convert_robomimic_to_replay
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
import zarr


def check_structure(hdf5_path, label=""):
    """Step 1: Check HDF5 structure — keys, shapes, dtypes."""
    print(f"\n{'='*60}")
    print(f"  Structural check: {label}")
    print(f"  {hdf5_path}")
    print(f"{'='*60}")

    issues = []

    with h5py.File(hdf5_path, 'r') as f:
        # ---- top-level ----
        if 'data' not in f:
            issues.append("ERROR: missing top-level 'data' group")
            return issues

        data_grp = f['data']
        demo_keys = sorted(data_grp.keys(), key=lambda x: int(x.split('_')[1]))
        n_demos = len(demo_keys)
        print(f"  Num demos: {n_demos}")

        if n_demos == 0:
            issues.append("ERROR: no demos found")
            return issues

        # ---- env_args attribute ----
        if 'env_args' in data_grp.attrs:
            print(f"  env_args: found ✓")
        else:
            issues.append("WARNING: 'env_args' attribute missing "
                          "(needed for env recreation during eval)")

        # ---- check first demo structure ----
        demo0 = data_grp[demo_keys[0]]

        required_keys = ['actions', 'abs_actions', 'states', 'obs']
        for key in required_keys:
            if key not in demo0:
                issues.append(f"ERROR: demo missing '{key}'")

        # ---- actions ----
        if 'actions' in demo0:
            actions = demo0['actions']
            print(f"  actions     shape: {actions.shape}  dtype: {actions.dtype}")
            if actions.shape[-1] != 14:
                issues.append(f"ERROR: actions last dim should be 14, got {actions.shape[-1]}")
            if not np.all(np.isfinite(actions[:])):
                issues.append("ERROR: NaN/Inf in actions")

        # ---- abs_actions ----
        if 'abs_actions' in demo0:
            abs_actions = demo0['abs_actions']
            print(f"  abs_actions shape: {abs_actions.shape}  dtype: {abs_actions.dtype}")
            if abs_actions.shape[-1] != 14:
                issues.append(f"ERROR: abs_actions last dim should be 14, got {abs_actions.shape[-1]}")

        # ---- states ----
        if 'states' in demo0:
            states = demo0['states']
            print(f"  states      shape: {states.shape}  dtype: {states.dtype}")

        # ---- obs ----
        if 'obs' in demo0:
            obs_grp = demo0['obs']
            obs_keys = list(obs_grp.keys())
            print(f"  obs keys: {obs_keys}")

            for key in obs_keys:
                arr = obs_grp[key]
                print(f"    {key:40s} shape: {str(arr.shape):20s} dtype: {arr.dtype}")

                # Check image format
                if 'image' in key or key.endswith('_image'):
                    if arr.dtype != np.uint8:
                        issues.append(f"ERROR: image '{key}' should be uint8, got {arr.dtype}")
                    if len(arr.shape) != 4 or arr.shape[-1] != 3:
                        issues.append(f"ERROR: image '{key}' should be (T,H,W,3), got {arr.shape}")

        # ---- episode lengths ----
        ep_lens = []
        for dk in demo_keys:
            ep_lens.append(data_grp[dk]['actions'].shape[0])
        ep_lens = np.array(ep_lens)
        print(f"\n  Episode lengths: min={ep_lens.min()}  max={ep_lens.max()}  "
              f"mean={ep_lens.mean():.1f}")

    return issues


def check_conversion(hdf5_path, abs_action=True):
    """Step 2: Try loading through _convert_robomimic_to_replay (same as dynamics model)."""
    print(f"\n{'='*60}")
    print(f"  Conversion test (same pipeline as dynamics model training)")
    print(f"{'='*60}")

    # Build shape_meta matching Transport task
    shape_meta = {
        'obs': {
            'robot0_eef_pos':           {'shape': [3], 'type': 'low_dim'},
            'robot0_eef_quat':          {'shape': [4], 'type': 'low_dim'},
            'robot0_eye_in_hand_image': {'shape': [3, 140, 140], 'type': 'rgb'},
            'robot0_gripper_qpos':      {'shape': [2], 'type': 'low_dim'},
            'robot1_eef_pos':           {'shape': [3], 'type': 'low_dim'},
            'robot1_eef_quat':          {'shape': [4], 'type': 'low_dim'},
            'robot1_eye_in_hand_image': {'shape': [3, 140, 140], 'type': 'rgb'},
            'robot1_gripper_qpos':      {'shape': [2], 'type': 'low_dim'},
            'shouldercamera0_image':    {'shape': [3, 140, 140], 'type': 'rgb'},
            'shouldercamera1_image':    {'shape': [3, 140, 140], 'type': 'rgb'},
        },
        'action': {'shape': [20]},
    }

    rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

    try:
        replay_buffer = _convert_robomimic_to_replay(
            store=zarr.MemoryStore(),
            shape_meta=shape_meta,
            dataset_path=hdf5_path,
            abs_action=abs_action,
            rotation_transformer=rotation_transformer,
        )
        print("  Conversion succeeded ✓")
    except Exception as e:
        print(f"  Conversion FAILED ✗")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ---- check replay buffer keys ----
    print(f"\n  Replay buffer keys:")
    data_keys = list(replay_buffer.data.keys())
    for key in sorted(data_keys):
        arr = replay_buffer[key]
        print(f"    {key:40s} shape: {str(arr.shape):20s} dtype: {arr.dtype}")

    # ---- check critical keys for dynamics model ----
    critical_keys = ['action', 'abs_action']
    for k in critical_keys:
        if k not in replay_buffer.data:
            print(f"  ERROR: replay buffer missing '{k}'")
            return False
        else:
            print(f"  '{k}' present ✓  shape: {replay_buffer[k].shape}")

    # ---- check obs keys ----
    obs_keys_expected = [k for k, v in shape_meta['obs'].items()]
    for k in obs_keys_expected:
        if k not in replay_buffer.data:
            print(f"  ERROR: replay buffer missing obs '{k}'")
            return False

    # ---- check episode ends ----
    ep_ends = replay_buffer.episode_ends[:]
    print(f"\n  Episode ends: {ep_ends}")
    print(f"  Total steps: {ep_ends[-1] if len(ep_ends) > 0 else 0}")
    print(f"  N episodes:  {len(ep_ends)}")

    # ---- check abs_action conversion (14 -> 20) ----
    abs_actions = np.array(replay_buffer['abs_action'])
    print(f"\n  abs_action dim check: {abs_actions.shape[-1]} "
          f"({'✓ correct (20)' if abs_actions.shape[-1] == 20 else '✗ WRONG'})")

    # ---- check action (should be 14, unchanged) ----
    actions = np.array(replay_buffer['action'])
    print(f"  action dim check:     {actions.shape[-1]} "
          f"({'✓ correct (14)' if actions.shape[-1] == 14 else '✗ WRONG'})")

    print(f"\n  All checks passed ✓  — data is ready for dynamics model training")
    return True


def compare_with_expert(rollout_path, expert_path):
    """Compare rollout HDF5 structure with expert demo HDF5."""
    print(f"\n{'='*60}")
    print(f"  Structure comparison: rollout vs expert")
    print(f"{'='*60}")

    with h5py.File(rollout_path, 'r') as rf, h5py.File(expert_path, 'r') as ef:
        r_demo0 = rf['data'][sorted(rf['data'].keys(),
                            key=lambda x: int(x.split('_')[1]))[0]]
        e_demo0 = ef['data'][sorted(ef['data'].keys(),
                            key=lambda x: int(x.split('_')[1]))[0]]

        # Compare obs keys
        r_obs_keys = set(r_demo0['obs'].keys()) if 'obs' in r_demo0 else set()
        e_obs_keys = set(e_demo0['obs'].keys()) if 'obs' in e_demo0 else set()

        print(f"  Expert obs keys: {sorted(e_obs_keys)}")
        print(f"  Rollout obs keys: {sorted(r_obs_keys)}")

        missing = e_obs_keys - r_obs_keys
        extra = r_obs_keys - e_obs_keys
        if missing:
            print(f"  MISSING in rollout: {sorted(missing)}")
        if extra:
            print(f"  EXTRA in rollout: {sorted(extra)}")
        if not missing and not extra:
            print(f"  Obs keys match ✓")

        # Compare top-level keys
        r_keys = set(r_demo0.keys())
        e_keys = set(e_demo0.keys())
        missing_top = e_keys - r_keys
        if missing_top:
            print(f"  MISSING top-level keys: {sorted(missing_top)}")
        else:
            print(f"  Top-level keys match ✓")

        # Compare action shapes
        if 'actions' in r_demo0 and 'actions' in e_demo0:
            r_shape = r_demo0['actions'].shape
            e_shape = e_demo0['actions'].shape
            print(f"  Expert actions shape:  {e_shape}")
            print(f"  Rollout actions shape: {r_shape}")
            if r_shape[1:] == e_shape[1:]:
                print(f"  Action dims match ✓")
            else:
                print(f"  Action dims MISMATCH ✗")


if __name__ == '__main__':
    assert os.path.isfile(ROLLOUT_HDF5_PATH), f'Not found: {ROLLOUT_HDF5_PATH}'
    assert os.path.isfile(EXPERT_HDF5_PATH),  f'Not found: {EXPERT_HDF5_PATH}'

    # Step 1: structural check
    rollout_issues = check_structure(ROLLOUT_HDF5_PATH, "rollout data")
    expert_issues  = check_structure(EXPERT_HDF5_PATH,  "expert data")

    # Compare
    compare_with_expert(ROLLOUT_HDF5_PATH, EXPERT_HDF5_PATH)

    # Step 2: conversion test (the definitive test)
    ok = check_conversion(ROLLOUT_HDF5_PATH, abs_action=ABS_ACTION)

    # Summary
    print(f"\n{'='*60}")
    if ok and not rollout_issues:
        print("  RESULT: ✓ Data format is correct, ready for dynamics model training")
    elif ok:
        print("  RESULT: ⚠ Data converts successfully but has minor issues:")
        for issue in rollout_issues:
            print(f"    {issue}")
    else:
        print("  RESULT: ✗ Data format has problems, needs fixing")
        for issue in rollout_issues:
            print(f"    {issue}")
    print(f"{'='*60}")
