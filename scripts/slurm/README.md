# SLURM wrappers

What `.sbatch` files are: shell scripts whose first lines are `#SBATCH ...`
directives that the SLURM scheduler reads (job name, time limit, memory,
partition, etc.), followed by the actual commands to run. You submit a
`.sbatch` with `sbatch <file>`; the scheduler queues it, allocates a compute
node when resources are free, runs the script there, and writes the combined
stdout/stderr to the `-o` log file.

## Habrok basics (what the headers assume)

- **No account** (`-A`) needed. Nodes: 128 cores, 512 GB (~4 GB/core).
- **`-p regular`** is a routing partition: the requested `-t` picks
  `regularshort` (≤ 8 h), `regularmedium` (≤ 3 d) or `regularlong` (≤ 10 d).
  Shorter requests start sooner, so every wrapper asks for a realistic time.
- **Always set `-t` and `--mem`**: defaults are 30 min and 2 GB per core.
- Ask for the cores a job actually uses (`-n 1 -c N`), not whole nodes.
- **`parallel`** (Omni-Path, ≤ 5 d) is for tightly coupled multi-node (MPI)
  jobs. Our work is embarrassingly parallel: bigger workloads use a **job
  array** on `regular` (see `scripts/mock/batch.sbatch`) — array tasks start
  independently in free gaps instead of waiting for several whole nodes.
- Array indices are capped at 1000 (`MaxArraySize=1001`); QOS allows 3000
  running / 6000 queued jobs per user.
- Nodes run a mix of AlmaLinux 8 and 9 (2026 migration); `~/software/habrok_env.sh`
  loads the `2023.01` stack so the environment works on both. Force one with
  `#SBATCH --constraint=alma8` (or `alma9`) if needed.

Every wrapper starts with `source "$HOME/software/habrok_env.sh"` (Python
module + venv, BBarolo on `PATH`, `AIDA_ROOT`).

## Wrapper shape

Each density/disc `.sbatch` runs all (model, snap) combos in parallel inside
one single-node job via bash `&` + `wait`, throttled to `N_PARALLEL` =
`$SLURM_CPUS_PER_TASK` concurrent single-threaded python processes (BLAS
pinned to 1 thread). `-c 18` = all FP combos at once. Per-combo stdout/stderr
goes to `<LOGS_DIR>/<jobname>_<jobid>_<MODEL>_<SNAP>.out`.

## Logs

Logs go to **`/scratch/s4636708/aida/derived/logs/slurm/`** (kept out of
`$HOME` to avoid VS Code's filesystem watcher freezing on hundreds of
small files). The directory must exist before submitting (SLURM does not
create the `-o` directory).

## Daily workflow

```bash
# 1. submit one stage
sbatch scripts/slurm/build_catalogs.sbatch

# 2. watch
squeue -u $USER
tail -f /scratch/s4636708/aida/derived/logs/slurm/build_cat_<jobid>.out
# per-combo log:
tail -f /scratch/s4636708/aida/derived/logs/slurm/build_cat_<jobid>_CDM_67.out

# 3. cancel
scancel <jobid>

# 4. after it ends: real CPU / memory efficiency -> right-size -c / --mem
jobinfo <jobid>
```

## Files in this directory

| file                              | what it submits                                             |
|-----------------------------------|-------------------------------------------------------------|
| `build_catalogs.sbatch`           | FP catalogs for all (model, snap) — 18 cores, 1 h           |
| `build_catalogs_dmo.sbatch`       | DMO catalogs matched to FP — needs FP catalogs + DMO data   |
| `compute_profiles.sbatch`         | Slice Despali profiles per (FP+DMO) catalog — 18 cores, 4 h |
| `compute_gamma.sbatch`            | gamma_DM per catalog (variants via `R_OUTER_KIND` var)      |
| `compute_rcore.sbatch`            | r_core cored-NFW fit per catalog                            |
| `build_mordor_samples.sbatch`     | MORDOR ASCII + catalog -> samples HDF5                      |
| `filter_disc_catalogs.sbatch`     | FP catalogs cut to MORDOR discs                             |
| `build_disc_catalogs_dmo.sbatch`  | DMO catalogs matched to the disc catalogs                   |
| `compute_{profiles,gamma,rcore}_disc.sbatch` | disc-catalog versions of the three stages        |
| `run_mordor.sbatch`               | Full MORDOR pipeline — one node, 64 cores / 320 GB, 12 h    |
| `recover_mordor.sbatch`           | Re-run stale MORDOR galaxies + rebuild samples              |
| `compute_vcirc.sbatch`            | v_circ in the disc plane for a galaxy list (`GALAXIES` var, needs `--export=ALL`) |
| `residual_split_all.sbatch`       | Residual figures (raw + floor-subtracted), all galaxies — 8 cores |
| `asymmetry_all.sbatch`            | 3D asymmetry table + figures, all galaxies — 8 cores        |
| `pipeline_density.sh`             | Submit the 5 density stages in sequence (uses `--wait`)     |
| `pipeline_disc.sh`                | Submit the 5 disc stages in sequence (uses `--wait`)        |

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
- Tune internal parallelism: change `#SBATCH -c` (and `--mem` with it);
  `N_PARALLEL` follows the allocation automatically.
- Change time / memory: edit the `#SBATCH -t` / `--mem` directives.
- Run a variant (e.g. `r_outer_kind=r200c`): edit the `R_OUTER_KIND`
  variable inside `compute_gamma.sbatch`.

The mass / particle cuts come from `config/scripts.yaml`'s `defaults:`
block. If you change them, also update `MSTAR_TAG` / `NDM_TAG` at the
top of each wrapper so the filename-matching for downstream stages
still resolves. Data paths in the wrappers (`/scratch/s4636708/aida/derived`)
mirror `paths:` in `config/scripts.yaml`, which the python scripts read.

## Plot scripts

Plot scripts under `scripts/plots/` are fast and can be invoked directly
from a login node or any compute allocation — no SLURM wrapper needed:

```bash
python scripts/plots/density/plot_gamma_mosaic.py \
    --gamma-files /scratch/s4636708/aida/derived/processed/gamma/*.hdf5
```

Output PDFs land in `figures/density/`, `figures/morphology/`, `figures/size_mass/`.
