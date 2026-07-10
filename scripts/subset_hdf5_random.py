"""
Subset a robomimic / diffusion_policy hdf5 by randomly selecting a fraction
of demos with a given random seed, and RENAME them to contiguous
demo_0 .. demo_{N-1}.

The contiguous renaming is REQUIRED: diffusion_policy's RobomimicReplayImageDataset
loads demos as demo_0, demo_1, ..., demo_{N-1} (see _convert_robomimic_to_replay).

Unlike a mask-based subset (which reads a pre-existing mask/ filter key), this
script generates the selection on the fly with np.random.RandomState(seed), so
you can create a reproducible 20% (or any fraction) subset from any HDF5
without needing a mask/ group.

Usage:
    python scripts/subset_hdf5_random.py \
        --input  square_ph_demo_v141.hdf5 \
        --output square_ph_demo_v141_20percent_seed42.hdf5 \
        --fraction 0.2 \
        --seed    42
"""
import argparse
import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="input hdf5 (full dataset)")
    ap.add_argument("--output", required=True, help="output hdf5 (random subset)")
    ap.add_argument("--fraction", type=float, default=0.2,
                    help="fraction of demos to keep, in (0, 1]")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for reproducible selection")
    args = ap.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        ap.error(f"--fraction must be in (0, 1], got {args.fraction}")

    f_in = h5py.File(args.input, "r")

    # ---- enumerate all demos in index order (demo_0, demo_1, ...) ----
    all_demos = sorted(f_in["data"].keys(), key=lambda x: int(x.split("_")[1]))
    n_total = len(all_demos)
    if n_total == 0:
        f_in.close()
        raise SystemExit("ERROR: input file has no demos under data/")

    # round to an integer count, always keep at least 1
    n_keep = max(1, int(round(n_total * args.fraction)))
    if n_keep > n_total:
        n_keep = n_total

    print(f"input:        {args.input}")
    print(f"total demos:  {n_total}")
    print(f"keeping:      {n_keep}  ({args.fraction*100:.1f}%)")
    print(f"seed:         {args.seed}")

    # ---- reproducible random selection ----
    rng = np.random.RandomState(args.seed)
    chosen_positions = sorted(rng.permutation(n_total)[:n_keep].tolist())
    sel_demos = [all_demos[i] for i in chosen_positions]

    preview = ", ".join(f"{p}->{d}" for p, d in zip(chosen_positions[:5], sel_demos[:5]))
    tail = " ..." if len(sel_demos) > 5 else ""
    print(f"selected:     [{preview}{tail}]")

    # ---- create output file ----
    f_out = h5py.File(args.output, "w")
    data_out = f_out.create_group("data")

    total = 0
    for new_idx, src_name in enumerate(sel_demos):
        new_name = f"demo_{new_idx}"
        # copy the whole demo group (obs/, actions, abs_actions, states, rewards,
        # dones, model_file attr, ...) preserving compression/dtypes
        f_in.copy(f"data/{src_name}", data_out, name=new_name)
        # num_samples is robomimic's per-demo timestep count; fall back to
        # actions.shape[0] if the attr is missing.
        demo_grp = data_out[new_name]
        if "num_samples" in demo_grp.attrs:
            total += int(demo_grp.attrs["num_samples"])
        elif "actions" in demo_grp:
            total += int(demo_grp["actions"].shape[0])
        if new_idx < 5 or new_idx == len(sel_demos) - 1:
            print(f"  {src_name} -> {new_name}")

    print(f"copied {len(sel_demos)} demos  (demo_0 .. demo_{len(sel_demos) - 1})")
    print(f"total samples: {total}")

    # ---- global attrs: keep env_args etc., recompute total ----
    for k, v in f_in["data"].attrs.items():
        if k == "total":
            continue
        data_out.attrs[k] = v
    data_out.attrs["total"] = total

    # NOTE: mask/ is deliberately NOT copied. The original mask arrays reference
    # demo names that are renamed here, and the whole point of this script is to
    # bypass the built-in mask with a fresh random selection.

    f_in.close()
    f_out.close()
    print(f"\nWROTE: {args.output}")
    print("Verify with:  python scripts/count_episodes.py", args.output)


if __name__ == "__main__":
    main()
