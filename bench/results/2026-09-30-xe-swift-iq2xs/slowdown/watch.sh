#!/bin/bash
# watch.sh OUT: once a second, the time, vmstat's swap-in/out (pages/s), free RAM and page cache, and for a running
# engine its VmSwap, VmRSS and AnonHugePages (kB)
out=$1; echo "time si so free_kB cache_kB engine_pid VmSwap_kB VmRSS_kB AnonHuge_kB" > $out
vmstat -n 1 | while read -r r b sw fr bu ca si so rest; do
  [[ $r =~ ^[0-9]+$ ]] || continue
  p=$(pgrep -f "^\./build/xe/strata|^<repo>/build/xe/strata" | head -1)
  if [ -n "$p" ]; then
    vs=$(awk '/^VmSwap/{print $2}' /proc/$p/status 2>/dev/null); vr=$(awk '/^VmRSS/{print $2}' /proc/$p/status 2>/dev/null)
    ah=$(awk '/^AnonHugePages/{print $2}' /proc/$p/smaps_rollup 2>/dev/null)
  else vs=-; vr=-; ah=-; fi
  echo "$(date +%T) $si $so $fr $ca ${p:--} ${vs:--} ${vr:--} ${ah:--}" >> $out
done
