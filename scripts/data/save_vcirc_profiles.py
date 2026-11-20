import numpy as np
import pandas as pd
import json
import temet
from pathlib import Path
from galaxy_sidm.io import load_config, load_flat
from galaxy_sidm.observables import measure_subhalo_cold_gas_all
from galaxy_sidm.data.potential import COMPONENTS
from galaxy_sidm.mock.tables import gas_disc_sample

cfg = load_config(None)
h = float(cfg["cosmology"]["h"])
vdir = Path("/scratch/s4636708/aida/derived/processed/vcirc")
mdir = Path("/scratch/s4636708/aida/derived/mordor_galaxies/samples")
out = Path.home() / "tmp" / "vcirc_CDM_vSIDM_z1_z2_logMstar_gt10.6.csv"

gal = gas_disc_sample(cfg) 
sample = {(m, float(zz), int(sid)) for m, zz, sid in gal.loc[gal["in_sample"], ["model", "z", "subID"]].itertuples(index=False)}

rows = []
for model in ("CDM", "vSIDM"):
    run = "L35n1080_vSIDM_correa" if model == "vSIDM" else f"L35n1080_{model}"
    base = str(Path(cfg["paths"]["aida_root"]) / run / "output") + "/"
    for snap, z in ((50, 1), (33, 2)):
        sim = temet.sim(run="aida", variant=model, res=1080, snap=snap)
        a, _ = load_flat(vdir / f"vcirc_{model}_{snap:03d}.hdf5")
        s, _ = load_flat(mdir / f"mordor_sample_{model}_{snap:03d}.hdf5")
        logm = dict(zip(np.asarray(s["halo_ids"], int), np.log10(np.asarray(s["Mstar"], float))))
        for i, sub in enumerate(np.asarray(a["halo_ids"], int)):
            if logm[sub] <= 10.6 or (model, float(z), int(sub)) not in sample:
                continue
            ok = np.isfinite(a["vcirc_total"][i]) # nan where the total v^2 <= 0
            gas = measure_subhalo_cold_gas_all(base, snap, int(sub), h=h, sim=sim)
            info = json.loads((Path("/scratch/s4636708/aida/derived/martini") / f"z{z:g}" / model
                               / f"gal_{sub:06d}" / "info.json").read_text())
            # v_circ of each component, signed: negative where its net pull points outward
            v_comp = {c: np.sign(a[f"v2_{c}"][i]) * np.sqrt(np.abs(a[f"v2_{c}"][i])) for c in COMPONENTS}
            for j in np.flatnonzero(ok):
                R, v = a["R_direct"][i][j], a["vcirc_total"][i][j]
                rows.append(dict(model=model, z=z, subID=int(sub), log10_Mstar=round(logm[sub], 4),
                                 n_rings=info["nradii"],
                                 Mgas_Msun=gas["all"], M_SFgas_Msun=gas["SFR>0"],
                                 M_cold_SH03_Msun=gas["sh03"], M_neutral_hdec_Msun=info["M_neutral"],
                                 R_kpc=round(float(R), 4), vcirc_km_s=round(float(v), 3),
                                 **{f"v_{c}_km_s": round(float(v_comp[c][j]), 3) for c in COMPONENTS}))

df = pd.DataFrame(rows)
df.to_csv(out, index=False)
print(out, len(df), "rows")
print(df.groupby(["model", "z"])["subID"].nunique())