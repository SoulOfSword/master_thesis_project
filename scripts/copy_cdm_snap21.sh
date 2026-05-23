#!/bin/bash
# Build a shadow CDM tree in $SCRATCH that contains:
#   - real copies of snapdir_021/ and offsets_021.hdf5 (sourced from /leonardo_store)
#   - symlinks to everything else already in $WORK
# After this finishes, $DST/output/ is a complete illustris_python basePath.
#
# Run this on a LOGIN node inside tmux:
#   ssh leonardo
#   tmux new -s copy
#   bash ~/master_thesis_project/scripts/copy_cdm_snap21.sh
# Detach: Ctrl-b d   Reattach: tmux attach -t copy

set -euo pipefail

SRC=/leonardo_store/DRES_IF_Despa/AIDA/L35n1080/L35n1080_CDM
WRK=/leonardo_work/CNHPC_1478837/AIDA/L35n1080/L35n1080_CDM
DST=$SCRATCH/master_thesis_project/data/aida_tng/L35n1080_CDM

echo "SRC = $SRC"
echo "WRK = $WRK"
echo "DST = $DST"
echo

mkdir -p "$DST/output" "$DST/postprocessing/offsets"

echo "[1/4] Symlinking all $WRK/output/* into $DST/output/ ..."
for d in "$WRK"/output/*; do
    ln -sfn "$d" "$DST/output/$(basename "$d")"
done

echo "[2/4] Symlinking all existing offsets files ..."
for f in "$WRK"/postprocessing/offsets/*.hdf5; do
    ln -sfn "$f" "$DST/postprocessing/offsets/$(basename "$f")"
done

echo "[3/4] Symlinking other postprocessing entries (skip offsets dir) ..."
for x in "$WRK"/postprocessing/*; do
    [ "$(basename "$x")" = "offsets" ] && continue
    ln -sfn "$x" "$DST/postprocessing/$(basename "$x")"
done

echo "[4/4] Copying real data from $SRC ..."
echo "  snapdir_021  (~360 GB, this is the slow part)"
rsync -av --progress "$SRC"/output/snapdir_021/ "$DST"/output/snapdir_021/

echo "  offsets_021.hdf5"
cp "$SRC"/postprocessing/offsets/offsets_021.hdf5 "$DST"/postprocessing/offsets/

echo
echo "DONE. Quick sanity check:"
echo "  snapdir_021 should be a directory, not a symlink:"
ls -lad "$DST/output/snapdir_021"
echo
echo "  offsets_021.hdf5 should be a real file:"
ls -la "$DST/postprocessing/offsets/offsets_021.hdf5"
echo
echo "  All other snapdir_* and groups_* should be symlinks:"
ls -la "$DST/output/" | head
