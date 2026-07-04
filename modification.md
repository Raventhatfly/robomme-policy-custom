# RoboMME Policy Learning Local Modifications

This file records the local changes made on top of the original repository, plus the launch commands we use on the cluster.

## Scope

The baseline code path was not intentionally overwritten. The original `pi05_baseline` training script still exists at:

```bash
scripts/finetune_pi05_baseline.sh
```

The new SLURM training script described below is for `mme_vla_suite` memory variants. Use the baseline script or `scripts/train.py pi05_baseline` directly for baseline experiments.

## Environment And Paths

Policy/training environment:

```bash
conda activate robomme-openpi
```

RoboMME simulator/evaluation environment:

```bash
conda activate robomme
```

Dataset root:

```bash
/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme
```

Preprocessed dataset used by training:

```bash
/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme/robomme_preprocessed_data
```

Required dataset layout:

```bash
meta/stats.json
data/*.pkl
features/episode_*
```

Checkpoint storage is kept off the home filesystem. The repo path uses symlinks:

```bash
runs/ckpts/mme_vla_suite -> /n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/mme_vla_suite
runs/ckpts/pi05_baseline -> /n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/pi05_baseline
```

Evaluation outputs should also go to netscratch, for example:

```bash
/n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/evaluation
```

## Modified Original Files

### `scripts/train.py`

Changes:

- Removed the duplicate startup path that called `main(_config.cli(), tentative_run=True)`, slept, then called `main(_config.cli())` again. This duplicate call caused repeated checkpoint-directory creation and `FileExistsError`.
- Removed the hardcoded W&B entity `daiyp_umich`.
- W&B now reads the entity from `WANDB_ENTITY` if it is set; otherwise it lets W&B use the user's default account.

Optional W&B entity override:

```bash
export WANDB_ENTITY=your_entity
```

### `scripts/finetune_mme_vla_suite.sh`

Changes:

- Removed the invalid placeholder line:

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
```

- Added `--wandb-enabled` so this regular finetuning entrypoint uses W&B by default.

Use `wandb login` once, or set:

```bash
export WANDB_API_KEY=...
```

## Added Files

### `.vscode/launch.json`

VS Code helper launch entries for local smoke training, preprocessing, and rollout-style evaluation. These are convenience/debug entries, not the canonical long-run commands.

### `scripts/build_robomme_preprocessed.sh`

Wrapper for building the RoboMME preprocessed dataset from raw H5 data.

Default paths:

```bash
RAW_DATA_PATH=/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme/robomme_data_h5
PREPROCESSED_DATA_PATH=/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme/robomme_preprocessed_data
```

Usage:

```bash
POLICY_PYTHON=/n/holylabs/LABS/hankyang_lab/Lab/felix/.conda/envs/robomme-openpi/bin/python \
bash scripts/build_robomme_preprocessed.sh
```

It runs:

```bash
python scripts/build_dataset.py --dataset_type robomme_pkl ...
python scripts/compute_norm_stats.py --config-name mme_vla_suite ...
python scripts/compute_norm_stats.py --config-name pi05_baseline ...
```

### `scripts/train_single_gpu_smoke.sh`

Short single-GPU smoke training script. W&B is intentionally disabled by default here to avoid polluting W&B with tiny test runs.

Usage:

```bash
POLICY_PYTHON=/n/holylabs/LABS/hankyang_lab/Lab/felix/.conda/envs/robomme-openpi/bin/python \
CONFIG_NAME=pi05_baseline \
GPU_ID=0 \
NUM_TRAIN_STEPS=2 \
BATCH_SIZE=1 \
bash scripts/train_single_gpu_smoke.sh
```

For an MME-VLA smoke run, use:

```bash
POLICY_PYTHON=/n/holylabs/LABS/hankyang_lab/Lab/felix/.conda/envs/robomme-openpi/bin/python \
CONFIG_NAME=mme_vla_suite \
EXP_NAME=mme_vla_smoke \
GPU_ID=0 \
NUM_TRAIN_STEPS=2 \
BATCH_SIZE=1 \
bash scripts/train_single_gpu_smoke.sh
```

### `slurm_scripts/train_2gpu_mme_vla.sbatch`

Canonical 2-GPU SLURM training script for MME-VLA memory variants.

Features:

- Uses netscratch dataset by default.
- Uses timestamped experiment names by default.
- Uses W&B by default: `WANDB_ENABLED=true`.
- Supports `RESUME=true` and `OVERWRITE=true` for existing checkpoint directories.
- Writes checkpoints under `runs/ckpts/mme_vla_suite`, which is symlinked to netscratch.
- Uses the absolute `robomme-openpi` Python path if present.

Default memory type:

```bash
perceptual-framesamp-modul
```

Supported `MME_VLA_TYPE` examples:

```bash
perceptual-framesamp-modul
perceptual-framesamp-context
perceptual-framesamp-expert
perceptual-tokendrop-modul
recurrent-ttt-expert
recurrent-ttt-context
recurrent-rmt-expert
symbolic-grounded-subgoal
symbolic-simple-subgoal
```

Perceptual memory training:

```bash
MME_VLA_TYPE=perceptual-framesamp-modul \
BATCH_SIZE=32 \
NUM_TRAIN_STEPS=80000 \
SAVE_INTERVAL=10000 \
sbatch slurm_scripts/train_2gpu_mme_vla.sbatch
```

Recurrent memory training:

```bash
MME_VLA_TYPE=recurrent-ttt-expert \
BATCH_SIZE=16 \
NUM_TRAIN_STEPS=80000 \
SAVE_INTERVAL=10000 \
sbatch slurm_scripts/train_2gpu_mme_vla.sbatch
```

Recurrent smoke training:

```bash
MME_VLA_TYPE=recurrent-ttt-expert \
BATCH_SIZE=2 \
NUM_TRAIN_STEPS=2 \
SAVE_INTERVAL=1 \
WANDB_ENABLED=false \
sbatch slurm_scripts/train_2gpu_mme_vla.sbatch
```

Disable W&B for any run:

```bash
WANDB_ENABLED=false sbatch slurm_scripts/train_2gpu_mme_vla.sbatch
```

Resume an interrupted run:

```bash
MME_VLA_TYPE=recurrent-ttt-expert \
EXP_NAME=recurrent-ttt-expert_2gpu_YYYYMMDD_HHMMSS \
BATCH_SIZE=16 \
NUM_TRAIN_STEPS=80000 \
SAVE_INTERVAL=10000 \
RESUME=true \
sbatch slurm_scripts/train_2gpu_mme_vla.sbatch
```

### `scripts/eval_foreground.sh`

Foreground evaluation runner. This replaces the original tmux-based eval flow for debugging and smoke tests.

Features:

- Starts `scripts/serve_policy.py`.
- Runs `examples/robomme/eval.py`.
- Tails policy-server logs in the same terminal.
- Saves eval results outside the home filesystem by default.
- Supports `PORT=auto`.
- Supports symbolic aliases from the original script.

Common result layout:

```bash
${EVAL_SAVE_DIR}/${MODEL_TYPE}/ckpt${CKPT_ID}/seed${SEED}/log.json
${EVAL_SAVE_DIR}/${MODEL_TYPE}/ckpt${CKPT_ID}/seed${SEED}/progress.json
${EVAL_SAVE_DIR}/${MODEL_TYPE}/ckpt${CKPT_ID}/seed${SEED}/videos/
```

Perceptual smoke eval:

```bash
MODEL_TYPE=perceptual-framesamp-modul_2gpu_YYYYMMDD_HHMMSS \
CKPT_ID=20000 \
PORT=auto \
GPU_ID_SERVER=0 \
GPU_ID_CLIENT=1 \
EVAL_SAVE_DIR=/n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/evaluation_smoke \
EXTRA_ARGS='--args.only_tasks=BinFill --args.overwrite' \
bash scripts/eval_foreground.sh
```

Recurrent smoke eval:

```bash
MODEL_TYPE=recurrent-ttt-expert_2gpu_YYYYMMDD_HHMMSS \
CKPT_ID=20000 \
PORT=auto \
GPU_ID_SERVER=0 \
GPU_ID_CLIENT=1 \
EVAL_SAVE_DIR=/n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/evaluation_recurrent_smoke \
EXTRA_ARGS='--args.only_tasks=BinFill --args.overwrite' \
bash scripts/eval_foreground.sh
```

Symbolic oracle smoke eval:

```bash
MODEL_TYPE=symbolic_groundedSG_oracle \
CKPT_ID=20000 \
PORT=auto \
GPU_ID_SERVER=0 \
GPU_ID_CLIENT=1 \
EVAL_SAVE_DIR=/n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/evaluation_symbolic_smoke \
EXTRA_ARGS='--args.only_tasks=BinFill --args.overwrite' \
bash scripts/eval_foreground.sh
```

Note: `symbolic_groundedSG_oracle` maps to checkpoint directory `symbolic-grounded-subgoal`.

### `scripts/eval_memory_checkpoint.sh`

Convenience wrapper for evaluating an existing MME-VLA memory checkpoint in the foreground.

Defaults:

```bash
MODEL_TYPE=recurrent-ttt-expert_2gpu_20260703_143622
CKPT_ID=30000
MODE=full
ONLY_TASKS=BinFill
EVAL_SAVE_DIR=/n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/evaluation_memory
```

Full eval of the current recurrent memory checkpoint:

```bash
bash scripts/eval_memory_checkpoint.sh
```

Smoke eval on one task:

```bash
MODE=smoke ONLY_TASKS=PickXtimes bash scripts/eval_memory_checkpoint.sh
```

Full eval:

```bash
MODE=full bash scripts/eval_memory_checkpoint.sh
```

Evaluate another checkpoint:

```bash
MODEL_TYPE=recurrent-ttt-expert_2gpu_20260703_143622 \
CKPT_ID=20000 \
bash scripts/eval_memory_checkpoint.sh
```

### `slurm_scripts/eval_memory_checkpoint.sbatch`

SLURM version for evaluating an existing MME-VLA memory checkpoint. This is the preferred cluster entrypoint.

Defaults:

```bash
MODEL_TYPE=recurrent-ttt-expert_2gpu_20260703_143622
CKPT_ID=30000
MODE=full
ONLY_TASKS=BinFill
EVAL_SAVE_DIR=/n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/evaluation_memory
```

Full eval of the current recurrent memory checkpoint:

```bash
sbatch slurm_scripts/eval_memory_checkpoint.sbatch
```

Smoke eval on one task:

```bash
MODE=smoke ONLY_TASKS=PickXtimes sbatch slurm_scripts/eval_memory_checkpoint.sbatch
```

Explicit full eval:

```bash
MODE=full sbatch slurm_scripts/eval_memory_checkpoint.sbatch
```

Evaluate another checkpoint:

```bash
MODEL_TYPE=recurrent-ttt-expert_2gpu_20260703_143622 \
CKPT_ID=20000 \
sbatch slurm_scripts/eval_memory_checkpoint.sbatch
```

### `slurm_scripts/eval_mme_vla.sbatch`

Canonical 2-GPU SLURM evaluation script.

Features:

- GPU 0 runs the policy server.
- GPU 1 runs the RoboMME simulator/eval client.
- Uses absolute Python paths for both environments by default.
- Saves evaluation under netscratch by default.
- Supports the symbolic aliases used by `scripts/eval.sh` and `scripts/eval_foreground.sh`.

Formal perceptual eval:

```bash
MODEL_TYPE=perceptual-framesamp-modul_2gpu_YYYYMMDD_HHMMSS \
CKPT_ID=80000 \
EVAL_SAVE_DIR=/n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/evaluation \
EXTRA_EVAL_ARGS='--args.overwrite' \
sbatch slurm_scripts/eval_mme_vla.sbatch
```

Formal recurrent eval:

```bash
MODEL_TYPE=recurrent-ttt-expert_2gpu_YYYYMMDD_HHMMSS \
CKPT_ID=80000 \
EVAL_SAVE_DIR=/n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/evaluation_recurrent \
EXTRA_EVAL_ARGS='--args.overwrite' \
sbatch slurm_scripts/eval_mme_vla.sbatch
```

Formal symbolic GroundSG oracle eval:

```bash
MODEL_TYPE=symbolic_groundedSG_oracle \
CKPT_ID=80000 \
EVAL_SAVE_DIR=/n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/evaluation_symbolic \
EXTRA_EVAL_ARGS='--args.overwrite' \
sbatch slurm_scripts/eval_mme_vla.sbatch
```

If evaluating a timestamped symbolic checkpoint directly, use the real checkpoint directory name and pass oracle args:

```bash
MODEL_TYPE=symbolic-grounded-subgoal_2gpu_YYYYMMDD_HHMMSS \
CKPT_ID=80000 \
EVAL_SAVE_DIR=/n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/evaluation_symbolic \
EXTRA_EVAL_ARGS='--args.use-oracle --args.subgoal-type=grounded_subgoal --args.overwrite' \
sbatch slurm_scripts/eval_mme_vla.sbatch
```

## Runtime Directories

These directories may exist locally after running jobs:

```bash
logs/
wandb/
runs/
```

They are runtime outputs or symlink containers, not source-code changes. `logs/slurm` stores SLURM stdout/stderr and policy-server logs.

## Quick Result Lookup

Given:

```bash
MODEL_TYPE=perceptual-framesamp-modul_2gpu_YYYYMMDD_HHMMSS
CKPT_ID=20000
SEED=7
EVAL_SAVE_DIR=/n/netscratch/hankyang_lab/Lab/felix/ckpts/robomme_policy_ckpt/evaluation
```

Final summary:

```bash
${EVAL_SAVE_DIR}/${MODEL_TYPE}/ckpt${CKPT_ID}/seed${SEED}/log.json
```

Progress while running:

```bash
${EVAL_SAVE_DIR}/${MODEL_TYPE}/ckpt${CKPT_ID}/seed${SEED}/progress.json
```

Videos:

```bash
${EVAL_SAVE_DIR}/${MODEL_TYPE}/ckpt${CKPT_ID}/seed${SEED}/videos/
```

## Current Recommended Next Run

For a non-symbolic memory experiment, try recurrent TTT expert:

```bash
MME_VLA_TYPE=recurrent-ttt-expert \
BATCH_SIZE=16 \
NUM_TRAIN_STEPS=80000 \
SAVE_INTERVAL=10000 \
sbatch slurm_scripts/train_2gpu_mme_vla.sbatch
```
