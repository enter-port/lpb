"""Count episodes and total steps in a robomimic HDF5 file.

Usage:
    python scripts/count_episodes.py /path/to/data.hdf5
"""
import sys
import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else 'data/transport/data/rollout/transport_rollout_and_demo.hdf5'

with h5py.File(path, 'r') as f:
    keys = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))
    n = len(keys)
    lens = [f[f'data/{k}/actions'].shape[0] for k in keys]

print(f"File:   {path}")
print(f"Episodes: {n}")
print(f"Steps:    {sum(lens)}")
if lens:
    print(f"Length:   min={min(lens)}  max={max(lens)}  mean={np.mean(lens):.1f}")
