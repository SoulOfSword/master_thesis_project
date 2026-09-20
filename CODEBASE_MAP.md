# Codebase map — what every file does, grouped by thread

Purpose: turn "I have to understand everything" into a finite, prioritised list.
One-liners are each file's **own docstring** (not invented). Line counts show size.

**How to read the `role` column:**
- **core** — carries a result or is a step in a chain. *Understand these.*
- helper — support/glue/IO/style. Skim; know what it gives you, not its guts.
- diag/one-off — diagnostics, validations, data-maintenance, dead ends. *Ignore unless you need that specific check.*

**Priority rule:** understand the **core** files of the threads whose results are
in your thesis, first. Everything else can wait or be ignored. Threads are ordered
foundation → results.

---

## Infrastructure (used by everything — skim once, then trust)
| file | lines | role | what |
|---|---|---|---|
| `src/.../config.py`, `io/config.py` | 42, 25 | helper | load YAML config |
| `src/.../io/hdf5_store.py` | 143 | helper | read/write processed-data HDF5 |
| `src/.../cosmology.py` | 67 | helper | astropy cosmology, `critical_density`, `delta_c` |
| `src/.../viz/style.py` | 52 | helper | shared plot styling |
| `scripts/BayesLineFit_mod.py` | 432 | helper | Bayesian line fit (used to fit scaling relations) |
| `scripts/tng.py` | 19 | diag/dead | TNG web API — leftover from TNG50-4 familiarization |

## Thread 1 — Sim loading & catalogs (the foundation everything reads)
| file | lines | role | what |
|---|---|---|---|
| `src/.../data/aida_tng.py` | 211 | **core** | load AIDA-TNG catalogs/particles (uses `temet`) |
| `src/.../data/halos.py` | 84 | **core** | Halo/subhalo data structures |
| `scripts/data/build_catalog.py` | 124 | **core** | build the halo/galaxy catalog for a (model, snap) |
| `scripts/data/build_catalog_dmo.py` | 136 | helper | matched dark-matter-only catalog |
| `scripts/data/filter_disc_catalog.py` | 106 | helper | cut a catalog to the MORDOR-disc subset |
| `scripts/data/{migrate_galaxies_to_snap_dirs,mask_stale_mordor_rows,recover_stale_mordor_galaxies}.py` | 104,113,133 | diag/one-off | data-maintenance fixes — ignore |

## Thread 2 — Density profiles & inner DM slope γ_DM (Despali reproduction)
| file | lines | role | what |
|---|---|---|---|
| `src/.../observables/density.py` | 525 | **core** | measure density profiles from particles (also holds the cold-gas/temperature definitions — Thread 3) |
| `src/.../models/profiles.py` | 359 | **core** | NFW / cored-NFW / Einasto profile forms |
| `scripts/data/compute_gamma.py` | 169 | **core** | inner DM log-slope γ_DM per halo |
| `scripts/data/compute_profiles.py` | 135 | helper | slice a precomputed profile catalog |
| `scripts/data/compute_rcore.py` | 143 | helper | fit cored-NFW r_core per halo |
| `scripts/plots/density/plot_{density,gamma,rcore}_mosaic.py`, `plot_rcore_over_rhalf.py` | 544,182,157,211 | helper (figures) | the γ_DM / profile / r_core mosaics |

## Thread 3 — Cold gas / temperature
| file | lines | role | what |
|---|---|---|---|
| (inside `observables/density.py`) | — | **core** | cold-gas definitions / `temp_sfcold` live here, not a separate file |
| `scripts/plots/density/plot_sfms.py` | 144 | helper (figure) | star-forming main sequence |

## Thread 4 — Morphology / MORDOR (produces IsDisc + component masses)
| file | lines | role | what |
|---|---|---|---|
| `src/.../morphology/extract.py` | 123 | **core** | extract a subhalo's particles → Gadget-HDF5 (MORDOR input) |
| `src/.../morphology/runner.py` | 309 | **core** | invoke MORDOR on the per-galaxy files |
| `src/.../morphology/parse.py` | 78 | **core** | parse MORDOR's ASCII output |
| `src/.../morphology/classify.py` | 67 | **core** | disc-vs-spheroid → the `IsDisc` flag |
| `scripts/mordor/extract_galaxies.py` | 152 | **core** | batch the per-galaxy extraction |
| `scripts/mordor/run_mordor.py` / `run_mordor_all.py` | 432, 142 | **core** | run MORDOR (parallel / all combos) |
| `scripts/data/build_mordor_sample.py` | 175 | **core** | assemble the MORDOR sample HDF5 (what TFR/SHMR read) |
| `src/.../morphology/diagnostics.py` + `scripts/plots/morphology/*` (11 files) | 78 + ~1600 | diag/one-off | D/T mosaics, (η,E) diagnostics, energy-bump checks, interacting checks, `validate_te`, single-galaxy renders — mostly checks & one figure each |

## Thread 5 — Particle kinematics & angular momentum (your future AM work)
| file | lines | role | what |
|---|---|---|---|
| `src/.../observables/kinematics.py` | 380 | **core** | particle-level: dispersion, **angular momentum**, circularity, λ_R, disc fraction |
| `src/.../observables/scaling.py` | 131 | helper | scaling-relation fits (TFR, mass-size, Fall) |

## Thread 6 — Mock HI → TFR / SHMR (the recent work)
| file | lines | role | what |
|---|---|---|---|
| `src/.../mock/gas.py` | 146 | **core** | load one galaxy's gas + stars |
| `src/.../mock/cube.py` | 144 | **core** | MARTINI mock HI cube |
| `src/.../mock/barolo.py` | 252 | **core** | BBarolo tilted-ring fit |
| `src/.../mock/kinematics.py` | 308 | **core** | ring V/σ + the 4-panel kinematics figure |
| `src/.../mock/tables.py` | 122 | **core** | assemble the disc table → **feeds TFR/SHMR** |
| `src/.../mock/residuals.py` | 112 | **core** | data−model residuals, raw and noise-floor-subtracted (disc classifier) |
| `src/.../mock/asymmetry.py` | 119 | **core** | He+2026 3D asymmetry A (second disc classifier) |
| `src/.../mock/sphview.py` | 150 | helper | face-on/edge-on surface-density maps |
| `scripts/mock/build_galaxy.py` | 188 | **core** | the pipeline driver (gas→cube→barolo→kinematics) |
| `scripts/mock/make_manifest.py` | 52 | helper | build the batch work list |
| `scripts/mock/snr_reference.py` | 92 | **core (in progress)** | intrinsic-S/N reference — *the task you paused on* |
| `scripts/mock/{disc_quality,collate_pv,run_bbarolo_plots,reprocess_kinematics}.py` | 139,219,95,64 | diag/helper | residual ranking, PV review PDF, plot runners, V/σ refresh |
| `scripts/plots/disc_quality/plot_residual_split.py` / `plot_asymmetry.py` | 320, 308 | **core (figures)** | residuals and 3D asymmetry per calibration class and for all galaxies |
| `scripts/plots/scaling/plot_tfr.py` / `plot_shmr.py` | 140, 203 | **core (figures)** | **the TFR & SHMR thesis figures** |
| `scripts/plots/size_mass/plot_size_mass.py` | 235 | helper (figure) | size–mass relation |

## Thread 7 — SIDM physics models (reference / future)
| file | lines | role | what |
|---|---|---|---|
| `src/.../models/sidm.py` | 214 | helper | SIDM cross-section forms (physics reference) |
| `src/.../models/sam.py` | 317 | diag/future | semi-analytical model — **not current** (CLAUDE.md says far off) |
| `src/.../inference/mcmc.py` | 179 | helper | emcee wrapper for fits |

---

## What this means for you
- **~75 files, but only ~20 are "core"**, spread over 6 live threads. That's the real size of "understand it all."
- **Whole threads you can likely set aside:** the SAM (`models/sam.py`, future), `scripts/tng.py` (old), and most of `scripts/plots/morphology/*` + `scripts/data/*_stale_*` (diagnostics/maintenance).
- **Next step:** tell me which threads are actually **in your thesis** (my guess: Thread 2 γ_DM, Thread 4 morphology, Thread 6 TFR/SHMR — with Thread 5 AM coming). We mark those, and your "understand everything" collapses to the ~10–12 core files in the in-scope threads — a list you can finish.
