# SLURM wrappers

What `.sbatch` files are: shell scripts whose first lines are `#SBATCH ...`
directives that the SLURM scheduler reads (job name, time limit, memory,
partition, etc.), followed by the actual commands to run. You submit a
`.sbatch` with `sbatch <file>`; the scheduler queues it, allocates a compute
node when resources are free, runs the script there, and writes the combined
stdout/stderr to `logs/<jobname>_<jobid>.out`.

## Partition choice

All density-pipeline wrappers and `run_mordor.sbatch` target
**`dcgp_usr_prod`** — Leonardo's production CPU partition. This is
**exclusive whole-node** allocation (one job owns ~112 cores and ~512 GB).

We do NOT use `lrd_all_serial` for batch work because its QOS allows only
1 submitted job per user at a time — array jobs and dependency chains
both blow that cap. `lrd_all_serial` is only for interactive Jupyter
sessions / quick one-offs.

## Wrapper shape

Each `.sbatch` allocates one whole dcgp node and runs all (model, snap)
combos in parallel inside that one job via bash `&` + `wait` throttled
to `N_PARALLEL` concurrent python processes. Per-combo stdout/stderr
goes to `logs/<jobname>_<jobid>_<MODEL>_<SNAP>.out`.

Edit `N_PARALLEL` at the top of any wrapper if you want more / fewer
concurrent processes per node.

## Logs

Logs go to **`$SCRATCH/master_thesis_project/logs/slurm/`** (kept out of
`$HOME` to avoid VS Code's filesystem watcher freezing on hundreds of
small files).

## Daily workflow

```bash
# 1. submit one stage
sbatch scripts/slurm/build_catalogs.sbatch

# 2. watch
squeue -u $USER
tail -f $SCRATCH/master_thesis_project/logs/slurm/build_cat_<jobid>.out
# per-combo log:
tail -f $SCRATCH/master_thesis_project/logs/slurm/build_cat_<jobid>_CDM_67.out

# 3. cancel
scancel <jobid>
```

## Files in this directory

| file                       | what it submits                                          |
|----------------------------|----------------------------------------------------------|
| `build_catalogs.sbatch`    | FP catalogs for all (model, snap) — one dcgp node, 2 h   |
| `build_catalogs_dmo.sbatch`| DMO catalogs matched to FP — needs FP catalogs first     |
| `compute_profiles.sbatch`  | Slice Despali profiles per (FP+DMO) catalog              |
| `compute_gamma.sbatch`     | gamma_DM per catalog (variants via `R_OUTER_KIND` var)   |
| `compute_rcore.sbatch`     | r_core cored-NFW fit per catalog                         |
| `run_mordor.sbatch`        | Full MORDOR pipeline for selected (model, snap)          |
| `pipeline_density.sh`      | Submit all 5 density stages in sequence (uses `--wait`)  |

## Pipeline launcher

`pipeline_density.sh` uses `sbatch --wait` so only **one** job is queued
at a time: it blocks on stage N before submitting stage N+1. Run inside
tmux so SSH drops don't kill the launcher:

```bash
tmux new -s pipeline
bash scripts/slurm/pipeline_density.sh
# Ctrl-b d to detach; `tmux attach -t pipeline` to reattach
```

## Editing for your runs

- Scope a run: edit the `MODELS` / `SNAPS` arrays at the top of the wrapper.
- Tune internal parallelism: edit `N_PARALLEL` (default 16).
- Change time / memory: edit the `#SBATCH -t` / `--mem` directives.
- Run a variant (e.g. `r_outer_kind=r200c`): edit the `R_OUTER_KIND`
  variable inside `compute_gamma.sbatch`.

The mass / particle cuts come from `config/scripts.yaml`'s `defaults:`
block. If you change them, also update `MSTAR_TAG` / `NDM_TAG` at the
top of each wrapper so the filename-matching for downstream stages
still resolves.

## Plot scripts

Plot scripts under `scripts/plots/` are fast and can be invoked directly
from a login node or any compute allocation — no SLURM wrapper needed:

```bash
python scripts/plots/density/plot_gamma_mosaic.py \
    --gamma-files $SCRATCH/master_thesis_project/data/processed/gamma_dm/*.hdf5
```

Output PDFs land in `figures/density/`, `figures/morphology/`, `figures/size_mass/`.
