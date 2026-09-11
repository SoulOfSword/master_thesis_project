#!/usr/bin/env bash
# Send the three DERIVED trees -- martini (incl. BBarolo), mordor_galaxies,
# processed -- from Leonardo scratch to Habrok in one pass, with visible progress.
#
# Safe to run ALONGSIDE the main snapshot transfer: it shares the same ssh master
# (the `habrok` alias) and works one folder at a time, so it adds only ONE extra
# channel -> won't blow past sshd's MaxSessions (~10).
#
#   mordor_galaxies, processed : rsync --info=progress2  (resumable, live % + rate)
#   martini (~207k small files): TAR-STREAMED -- rsync would crawl file-by-file
#                                over the long link. Progress = a record-count
#                                heartbeat (~every 50 MB) since `pv` isn't installed.
#                                NOT resumable: if it breaks, just rerun -- martini
#                                restarts, the other two resume from --partial.
#
# On a completed martini it touches the SAME $STATE/martini marker the main
# transfer uses, so the main script's tail SKIPS martini (no duplicate work).
#
# Run it in its own pane:  bash scripts/transfer/send_processed.sh
set -uo pipefail
HABROK="habrok"
DEST="/scratch/s4636708/aida"
SCR=/leonardo_scratch/large/userexternal/acosta01/master_thesis_project/data
STATE="$HOME/.habrok_xfer_state"; mkdir -p "$STATE"
SSHQ="ssh -o BatchMode=yes"

# one quick liveness check so we fail clearly instead of hanging on a dead master
if ! $SSHQ "$HABROK" true 2>/dev/null; then
  echo "master not usable -- bring it up in another pane (ssh -fN $HABROK), then rerun."
  exit 1
fi
$SSHQ "$HABROK" "mkdir -p $DEST/derived" 2>/dev/null

# 1) the two file-light trees: resumable rsync with a live progress bar
for d in mordor_galaxies processed; do
  if [ -d "$SCR/$d" ]; then
    echo "=== rsync $d  ($(date '+%T')) ==="
    rsync -a --partial --inplace --info=progress2 -e "$SSHQ" "$SCR/$d" "$HABROK:$DEST/derived/"
  else
    echo "=== $d: not present locally, skipping ==="
  fi
done

# 2) martini: tar-stream (fast for the huge small-file tree) with a heartbeat
if [ -f "$STATE/martini" ]; then
  echo "=== martini: already marked complete, skipping ==="
elif [ -d "$SCR/martini" ]; then
  echo "=== tar-streaming martini  ($(date '+%T')) -- heartbeat every ~50 MB ==="
  if tar cf - -C "$SCR" --checkpoint=5000 --checkpoint-action=echo --totals martini \
       | $SSHQ "$HABROK" "tar xf - -C $DEST/derived"; then
    touch "$STATE/martini"
    echo "=== martini done (marked; main transfer will now skip it) ==="
  else
    echo "=== martini tar FAILED -- rerun this script to retry just martini ==="
    exit 1
  fi
else
  echo "=== martini: not present locally ==="
fi

echo "=== $(date '+%F %T') derived transfer finished ==="
