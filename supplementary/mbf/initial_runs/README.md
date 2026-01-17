# Initial Runs (run_0--run_8)

This folder contains lightweight artifacts from the initial ablation runs referenced in the paper.

## Contents

For each `run_k/`:
- `final_info.json`: seed-aggregated metrics (mean, stderr) for PTB and WikiText-2.
- `final_results_*.json`: per-seed final metrics.

We intentionally omit large per-batch logs and intermediate checkpoints to keep the repository lightweight.

## Run mapping (paper)

- `run_0`: Baseline
- `run_1--run_8`: MBF ablation variants (see manuscript Table 5 and text for exact configuration).

## Scripts

The corresponding prototype scripts are under `supplementary/mbf/scripts/initial_runs/`.
