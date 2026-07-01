"""Analyze rollout+demo combined dataset composition."""
import h5py
import numpy as np

# ============ Config ============
TRAIN_HDF5 = 'data/transport/data/rollout/transport_rollout_and_demo.hdf5'
VAL_HDF5 = 'data/transport/data/rollout/transport_val.hdf5'
# ================================


def analyze_hdf5(path, name):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"Path:   {path}")
    print(f"{'='*60}")

    with h5py.File(path, 'r') as f:
        # ---- Basic structure ----
        print("\nTop-level keys:", list(f.keys()))

        # Count episodes
        n_episodes = len(f['data'].keys())
        print(f"\nTotal episodes: {n_episodes}")

        # Episode lengths
        ep_lengths = []
        for ep_key in sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1])):
            actions = f[f'data/{ep_key}/actions']
            ep_lengths.append(actions.shape[0])
        ep_lengths = np.array(ep_lengths)
        total_steps = ep_lengths.sum()

        print(f"Total steps:    {total_steps}")
        print(f"Episode length: min={ep_lengths.min()}, max={ep_lengths.max()}, "
              f"mean={ep_lengths.mean():.1f}, median={np.median(ep_lengths):.1f}")

        # ---- Check mask (rollout vs demo split) ----
        print("\n--- Mask (rollout vs demo) ---")
        if 'mask' in f:
            mask_keys = list(f['mask'].keys())
            print(f"Mask keys: {mask_keys}")
            for mk in mask_keys:
                indices = f[f'mask/{mk}'][:]
                # indices might be array of episode indices
                if indices.dtype == bool or indices.dtype == np.bool_:
                    n = indices.sum()
                else:
                    n = len(indices)
                # compute steps for this subset
                if indices.dtype == bool or indices.dtype == np.bool_:
                    subset_steps = ep_lengths[indices].sum()
                    ep_names = [k for i, k in enumerate(sorted(f['data'].keys(),
                                 key=lambda x: int(x.split('_')[1]))) if indices[i]]
                else:
                    idx_list = [int(i) for i in indices]
                    subset_steps = ep_lengths[idx_list].sum()
                    ep_names = [f'demo_{i}' for i in idx_list]
                print(f"  {mk}: {n} episodes, {subset_steps} steps "
                      f"({100*subset_steps/total_steps:.1f}% of steps)")
        else:
            print("  No 'mask' group found. Cannot distinguish rollout vs demo.")

        # ---- Episode-level info ----
        print("\n--- Per-episode summary (first 20 + last 5) ---")
        sorted_keys = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))
        show_keys = sorted_keys[:20] + sorted_keys[-5:] if n_episodes > 25 else sorted_keys
        for ep_key in show_keys:
            T = f[f'data/{ep_key}/actions'].shape[0]
            # check if rewards/success available
            extra = ''
            if f'rewards' in f[f'data/{ep_key}']:
                rwd = f[f'data/{ep_key}/rewards'][:]
                extra += f', reward_sum={rwd.sum():.2f}'
            print(f"  {ep_key}: {T} steps{extra}")
        if n_episodes > 25:
            print(f"  ... ({n_episodes - 25} episodes omitted)")

        # ---- Sample one episode's obs keys ----
        sample_ep = sorted_keys[0]
        print(f"\n--- Sample episode '{sample_ep}' structure ---")
        sample_grp = f[f'data/{sample_ep}']
        for k in sample_grp.keys():
            item = sample_grp[k]
            if isinstance(item, h5py.Group):
                sub_keys = list(item.keys())
                print(f"  {k}/ (Group): {sub_keys}")
            else:
                print(f"  {k}: {item.shape} {item.dtype}")
        if 'obs' in sample_grp:
            print("  obs details:", list(sample_grp['obs'].keys()))

        # ---- Train/val ratio summary ----
        return n_episodes, total_steps, ep_lengths


def main():
    train_n, train_steps, train_lens = analyze_hdf5(TRAIN_HDF5, "TRAIN (rollout + demo)")
    val_n, val_steps, val_lens = analyze_hdf5(VAL_HDF5, "VAL")

    print(f"\n{'='*60}")
    print("TRAIN / VAL SPLIT SUMMARY")
    print(f"{'='*60}")
    total_eps = train_n + val_n
    total_steps_all = train_steps + val_steps
    print(f"Train: {train_n} eps ({100*train_n/total_eps:.1f}%), "
          f"{train_steps} steps ({100*train_steps/total_steps_all:.1f}%)")
    print(f"Val:   {val_n} eps ({100*val_n/total_eps:.1f}%), "
          f"{val_steps} steps ({100*val_steps/total_steps_all:.1f}%)")
    print(f"Total: {total_eps} eps, {total_steps_all} steps")


if __name__ == '__main__':
    main()
