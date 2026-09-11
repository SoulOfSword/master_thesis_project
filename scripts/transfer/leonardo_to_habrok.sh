#!/usr/bin/env bash
# Transfer the AIDA-TNG working set + derived outputs, Leonardo -> Habrok,
# with several snapshots copied IN PARALLEL. Resumable.
#
# RUN ON LEONARDO in tmux. Habrok needs password + Google Authenticator, so use
# ssh ControlMaster (authenticate once, reuse). In ~/.ssh/config on Leonardo:
#     Host habrok
#         HostName interactive1.hb.hpc.rug.nl
#         User s4636708
#         ControlMaster auto
#         ControlPath ~/.ssh/cm/%r@%h:%p
#         ControlPersist 24h
#         ServerAliveInterval 30
#         ServerAliveCountMax 6
#         TCPKeepAlive yes
# then:  mkdir -p ~/.ssh/cm && ssh -fN habrok   (enter the code once).
# All the parallel rsyncs then share that one authenticated connection.
set -uo pipefail
SELF="$(readlink -f "$0")"

# ==================== EDIT THESE ====================
HABROK="habrok"                        # the ssh-config alias above
DEST="/scratch/s4636708/aida"          # your 12 TB Habrok scratch dir
LOG="$HOME/habrok_transfer.log"
PAR="${PAR:-5}"                        # snapshots in flight at once. Keep <=8:
                                       # they share ONE ssh connection (server
                                       # MaxSessions is ~10). Lower if you see
                                       # "open failed: administratively prohibited".
# ===================================================

WORK=/leonardo_work/CNHPC_1478837/AIDA/L35n1080
STORE=/leonardo_store/DRES_IF_Despa/AIDA/L35n1080
SCR=/leonardo_scratch/large/userexternal/acosta01/master_thesis_project
# -a: archive; --partial+--inplace: resume large files in place. Complete chunks are skipped
# by the size+mtime quick-check with no re-read; partial ones resume via delta.
RS=(rsync -a --partial --inplace --timeout=120 -e "ssh -o BatchMode=yes")   # --timeout=120: abort a STALLED transfer after 120s with no data (this is the freeze -- rsync otherwise hangs forever on a half-open connection); BatchMode: fail fast, never hang on a prompt
STATE="$HOME/.habrok_xfer_state"; mkdir -p "$STATE"

# Is there a USABLE, code-free connection right now?  0 = yes (resume), 1 = a 2FA login is needed.
connection_usable(){
  ssh -O check "$HABROK" >/dev/null 2>&1 || return 1          # no master socket at all
  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$HABROK" true >/dev/null 2>&1 && return 0   # timeout 20: the probe itself can't hang on a stalled master
  ssh -O exit "$HABROK" >/dev/null 2>&1                       # stale corpse -> remove so -O check can't lie
  return 1
}

# Block until a usable connection exists. connection_usable resumes INSTANTLY with
# no code whenever the master is alive (survived a blip, or you re-logged elsewhere).
# We ALSO reattempt a code-free reconnect ourselves -- but GLOBALLY rate-limited by an
# atomic lock + timestamp so the PAR parallel workers can't burst failed logins and
# trip fail2ban. If a code is truly required the attempt just fails and we keep waiting.
RECON_EVERY=300      # min seconds between code-free reconnect attempts (all workers share this)
wait_for_master(){
  while ! connection_usable; do
    if mkdir "$STATE/.recon.lock" 2>/dev/null; then       # only one worker at a time past here
      now=$(date +%s); last=$(cat "$STATE/.recon.ts" 2>/dev/null || echo 0)
      if [ $((now - last)) -ge "$RECON_EVERY" ]; then
        echo "$now" > "$STATE/.recon.ts"
        if ssh -fN -o BatchMode=yes -o ConnectTimeout=15 "$HABROK" 2>/dev/null && connection_usable; then
          rmdir "$STATE/.recon.lock" 2>/dev/null; echo "[reconnected] code-free -- resuming"; return 0
        fi
        echo "[wait] master down; code-free reconnect not possible -- enter a code:  ssh -fN $HABROK"
      fi
      rmdir "$STATE/.recon.lock" 2>/dev/null
    fi
    sleep 30
  done
}

# Guarded remote ops for the serial tail (postprocessing/derived/software) below.
# Each re-checks the master FIRST, so if the link drops mid-tail we pause here --
# exactly like the worker loop does -- instead of firing ~two dozen fail-fast
# connection attempts back-to-back. That burst is what tripped fail2ban before.
g_ssh(){ wait_for_master; ssh -o BatchMode=yes "$HABROK" "$@" 2>>"$LOG"; }
g_rs(){  wait_for_master; "${RS[@]}" "$@" 2>>"$LOG"; }

# worker: transfer ONE snapshot (snapdir+groups), retry in place, mark done
if [ "${1:-}" = "--one" ]; then
  m="$2"; s="$3"; src="$4"; key="snap_${m}_${s}"
  [ -f "$STATE/$key" ] && { echo "[skip] $m $s (already complete)"; exit 0; }
  wait_for_master
  ssh -o BatchMode=yes "$HABROK" "mkdir -p $DEST/$m/output" 2>>"$LOG"
  for a in 1 2 3 4 5; do
    wait_for_master     # never open a connection while the link is down -> no storm
    "${RS[@]}" "$src/$m/output/snapdir_$s" "$HABROK:$DEST/$m/output/" 2>>"$LOG"
    "${RS[@]}" "$src/$m/output/groups_$s"  "$HABROK:$DEST/$m/output/" 2>>"$LOG"
    # VERIFY completeness: a dry-run must (a) connect AND (b) find NOTHING left to
    # send. This is what stops a killed / partial copy from ever being marked
    # "done" -- and if the check itself can't connect, we DON'T trust its "0".
    vout=$(rsync -an --out-format='%n' --timeout=120 -e "ssh -o BatchMode=yes" \
             "$src/$m/output/snapdir_$s" "$src/$m/output/groups_$s" \
             "$HABROK:$DEST/$m/output/" 2>/dev/null); vrc=$?
    left=$(printf '%s\n' "$vout" | grep -c '\.hdf5')
    if [ "$vrc" -eq 0 ] && [ "$left" -eq 0 ]; then
      touch "$STATE/$key"; echo "[done+verified] $m $s"; exit 0
    fi
    w=$((a * 30)); echo "[retry] $m $s ($a/5): $left chunk(s) left (rc=$vrc), waiting ${w}s"; sleep "$w"
  done
  echo "[FAIL] $m $s incomplete after 5 tries -- will retry on next run"; exit 1
fi

# main: build the snapshot job list, run PAR at a time
echo "=== $(date '+%F %T') start (parallel=$PAR) ===" | tee -a "$LOG"
rm -rf "$STATE/.recon.lock"   # clear any stale reconnect lock left by a killed run
JOBS="$(mktemp)"
{
  for s in 017 025 033 050 067; do echo "L35n1080_CDM $s $WORK"; done # CDM FP
  echo "L35n1080_CDM 021 $STORE" # z=4 from store
  for m in L35n1080_SIDM1 L35n1080_vSIDM_correa; do   # FP only -- DMO (-Dark) dropped, not needed
    for s in 017 021 025 033 050 067; do echo "$m $s $WORK"; done
  done
} > "$JOBS"
WANT=$(wc -l < "$JOBS"); echo "snapshots to transfer: $WANT" | tee -a "$LOG"
xargs -P "$PAR" -L1 bash "$SELF" --one < "$JOBS"
rm -f "$JOBS"

# offsets + FP<->DMO matching + profile catalogs. Every remote op below goes
# through g_ssh/g_rs, so a mid-tail drop pauses instead of storming Habrok.
echo "=== postprocessing ===" | tee -a "$LOG"
for m in L35n1080_CDM L35n1080_SIDM1 L35n1080_vSIDM_correa; do
  g_ssh "mkdir -p $DEST/$m/postprocessing"
  g_rs "$WORK/$m/postprocessing/offsets"               "$HABROK:$DEST/$m/postprocessing/"
  g_rs "$WORK/$m/postprocessing/SubhaloMatchingToDark" "$HABROK:$DEST/$m/postprocessing/"
  g_rs "$WORK/$m/postprocessing/"cat_halo_profiles_*   "$HABROK:$DEST/$m/postprocessing/"
done
g_rs "$STORE/L35n1080_CDM/postprocessing/offsets/" "$HABROK:$DEST/L35n1080_CDM/postprocessing/offsets/"
# (DMO offsets loop removed -- DMO dropped)

# derived outputs. martini is handled separately by send_processed.sh (it tar-streams
# the 207k-file tree and sets $STATE/martini), so it is NOT re-sent here. mordor_galaxies
# and processed are small; rsync re-checks them as a fast no-op if already present.
echo "=== derived outputs ===" | tee -a "$LOG"
g_ssh "mkdir -p $DEST/derived"
g_rs "$SCR/data/mordor_galaxies"             "$HABROK:$DEST/derived/"
g_rs "$SCR/data/processed"                   "$HABROK:$DEST/derived/"
g_rs "$SCR/data/coldgas_definitions_cdm.pkl" "$HABROK:$DEST/derived/"

# software
g_ssh "mkdir -p $DEST/software"
g_rs "$HOME/software/" "$HABROK:$DEST/software/"

# Exit non-zero while anything is still incomplete, so an outer 'until' loop
# keeps auto-resuming (it re-runs and picks up from the markers).
have=$(ls "$STATE"/snap_* 2>/dev/null | wc -l)
if [ "$have" -ge "${WANT:-0}" ]; then
  echo "=== $(date '+%F %T') ALL $WANT FP snapshots verified complete (martini handled by send_processed.sh) ===" | tee -a "$LOG"
  exit 0
fi
echo "=== $(date '+%F %T') incomplete ($have/${WANT:-?} snapshots) -- rerun to continue ===" | tee -a "$LOG"
exit 1
