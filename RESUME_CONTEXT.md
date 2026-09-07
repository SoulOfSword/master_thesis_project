# Resume context — mock-HI pipeline & disc classifier

Snapshot of where the MARTINI→BBarolo mock-HI work stands, so it can be picked
up on another machine. Thesis = galaxy morphology + angular momentum across DM
models (CDM / SIDM1 / vSIDM) in AIDA-TNG L35n1080, MORDOR morphology + mock HI
rotation curves → Tully-Fisher (TFR) and stellar-halo-mass (SHMR) relations.

## The active task (where we stopped): a disc-vs-perturbed gas classifier

**Why:** MORDOR labels a galaxy a "disc" from its *stars*; the HI cube images the
*gas*, which at high z can be perturbed even when the stars are a disc. Those give
untrustworthy BBarolo rotation curves and must be dropped from TFR/SHMR. We want a
number to flag them automatically (currently done by eye in
`config/problematic_discs.yaml`).

**Residual metrics** — `src/galaxy_sidm/mock/residuals.py`, three global data−model
statistics (whole cube, σ = std of the line-free end channels, no mask):
- `res1 = Σ(D−M)²/σ²` (χ²), `res2 = Σ|D−M|/σ` (χ), `res3 = Σ|D−M|/Σ|D|`.
- On 10 hand-labelled discs + 10 perturbed (CDM z2), **res3 separates perfectly**
  (threshold ≈ 0.98) while res1/res2 overlap. Figure: `scripts/plots/disc_quality/plot_residual_split.py`.

**But** the user wants **res1 (χ²)**, not res3, as the classifier — res3 lacks a
statistical meaning. res1/res2 overlap because all cubes share the same *noise*
(1e-5) → different *S/N*; a faint galaxy looks "bad" from low S/N, not perturbation.

**The fix = equalise S/N across galaxies** (so res1 differences are intrinsic):
1. Reference disc → build a **noiseless** cube; `signal_ref = spectral mean of the
   spatial mean over the line channels`; `ref_SNR = signal_ref / 1e-5`.
2. Each galaxy: same noiseless signal measurement → feed MARTINI
   `noise = signal_galaxy / ref_SNR`. All galaxies then sit at the same S/N.

**State:** `scripts/mock/snr_reference.py` does step 1 for one galaxy (hardcoded
`SUB_ID=29759`, CDM, snap 33; line channels `CHANNELS=(26,41)` picked by eye in
DS9 because MARTINI's Gaussian profile leaks flux into all 64 channels, so a `>0`
or fraction-of-peak auto-cut failed). **It has NOT been run in this final form** —
we don't have `ref_SNR` yet. (Its docstring header lines 9–12 are stale — still
describe the old `>0` logic.)

**Next steps (all need MARTINI on a compute node):**
1. Run `snr_reference.py` → get `ref_SNR`.
2. **Open design question:** the per-galaxy signal needs each galaxy's line
   channels, and line width varies ~8–25 channels — decide one automated rule vs
   20 hand-picked ranges.
3. Rebuild each of the 20 calibration cubes with `noise = signal_g / ref_SNR`.
4. Re-run the residual split; see if res1 now separates → set a threshold.

## Done & verified this session
- **TFR/SHMR:** `src/galaxy_sidm/mock/tables.py` (`assemble` → disc table; `v_flat`
  = mean of last 3 ring VROT), `scripts/plots/scaling/plot_tfr.py`,
  `plot_shmr.py`. Modified SHMR = `f_Mstar(M*,z)/f_Mstar(M*,z0) vs M*` with
  `f_Mstar=M*/M200c`, z0=0.5, one line per z (f_V ratio ≡ 1 by assumption — NO
  velocities). TFR slope shallow (~0.1 vs canonical 0.25) because v_flat is an
  inner value (rings reach only 1.8–4.7 kpc); possible switch to v_max.
- **BBarolo DISTANCE bug (found + fixed):** col-0 `RAD(Kpc)` was wrong — BBarolo
  guessed distance per-galaxy from Vsys (2.4–7.2 Mpc) because DISTANCE wasn't
  passed. Fixed: `write_par` now emits `DISTANCE` from `CubeParams().distance`
  (5 Mpc); `kinematics.py._rings` rebuilds kpc from col-1 arcsec × true distance
  (fixes existing on-disk fits with no refit). VROT/VDISP/VSYS and TFR/SHMR were
  never affected — only `info.json` V/σ + `kinematics.png` (refresh via
  `reprocess_kinematics.py`, no compute needed).
- **Sample facts:** MORDOR sample `R200c` is **comoving** kpc (÷(1+z) → physical);
  M200c/Mstar are plain Msun. v_flat ring grid (fixed 5 Mpc, 30″): 0.36, 1.09,
  1.82, 2.55, 3.27, 4.00, 4.73 kpc (z-independent).

## New/changed files (uncommitted — must be pushed before pulling elsewhere)
New: `src/galaxy_sidm/mock/{residuals.py,tables.py}`,
`scripts/mock/{snr_reference.py,collate_pv.py,disc_quality.py,run_bbarolo_plots.py}`,
`scripts/plots/{scaling/,disc_quality/}`, `config/{problematic_discs.yaml,residual_calibration.yaml}`.
Changed: `src/galaxy_sidm/mock/{barolo.py,cube.py,kinematics.py}`, `cosmology.py`
(+`delta_c_bryan_norman`), `scripts/mock/build_galaxy.py`, `scripts/mock/batch.sbatch`.

## Blockers / cluster notes
- **Leonardo compute budget `euhpc_r05_084` EXPIRED** → no BBarolo/MARTINI/srun.
  Everything past `snr_reference.py` (and any refit) is blocked until renewed.
- **Portability (see chat):** login-only downstream analysis (residuals, TFR/SHMR,
  plots) is portable if you copy the produced `martini/` output tree + MORDOR
  samples and repoint `config/scripts.yaml:paths`. The **cube-rebuild / SNR work is
  NOT easily portable** — it needs the raw AIDA-TNG snapshots (Leonardo `$WORK`),
  the `temet` loader (CINECA-local, not in pyproject), and a BBarolo rebuild.
