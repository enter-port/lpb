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
    # TODO: wire everything together in Task 9
    print("Config loaded:")
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    main()
