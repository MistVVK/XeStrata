#!/bin/bash
# m_ab.sh: E-9, the draft layer's prompt K/V in batches through the prompt path (bin/strata-k), against the drafter's
# own pass (STRATA_MTP_BATCH=0): the 26,292-token prompt with --mtp, then 128 tokens; prefill rate, the MTP prompt
# time, the drafts accepted, the decode rate and the generated tokens.  Two alternating rounds after a warm-up.
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin
export CACHE=10000
run() { local name=$1; shift
  env "$@" IDS=long_ids.txt NEW=128 PREFILL=auto NODUMP=1 EXE="$B/strata-k" "$H/run.sh" "$name" iq3_s --max-context 32768 --kv int8 --suffix-draft 0 --mtp "$H/../../mtp/rt" > /dev/null; }
NODUMP=1 IDS=port_ids.txt NEW=8 EXE="$B/strata-k" "$H/run.sh" m-jit iq3_s --suffix-draft 0 --mtp "$H/../../mtp/rt" > /dev/null
for r in 1 2; do for arm in batch own; do
  case $arm in batch) E=();; own) E=(STRATA_MTP_BATCH=0);; esac
  run "m-$arm-$r" "${E[@]}"
  f=$H/m-$arm-$r.txt
  echo "$r $arm prefill $(grep -E '^prefill  ' "$f" | grep -o '[0-9.]* tok/s') $(grep -o 'MTP prompt [0-9.]* ms' "$f") $(grep -o 'drafts accepted [0-9]* of [0-9]*' "$f") decode $(grep -E '^decode  ' "$f" | grep -o '[0-9.]* tok/s') out $(grep '^output  :' "$f" | md5sum | cut -c1-8)"
done; done
