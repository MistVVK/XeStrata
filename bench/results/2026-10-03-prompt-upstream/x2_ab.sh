#!/bin/bash
# x2_ab.sh: STRATA_PREFILL_BF16X2 (the BF16 projections' remainder products, upstream 61638c1 / 4e0592b).
# 1. the default against the binary before the change (bin/strata-pre-bf16x2): the 26,292-token prompt, IQ3_S,
#    int8 KV, 24 tokens, logits; two alternating rounds.
# 2. the first token's logits for modes 0, 2 and 1 (IQ2_XS: the 19-token chat and Japanese prompts, the 1,500-token
#    one; IQ3_S: the long prompt); KL against mode 1 with kl_first.py.
# 3. the default bitwise without the matrix engines (STRATA_NO_XMX=1) and on the small configuration.
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin
for r in 1 2; do for v in pre-bf16x2 bf16x2; do
  CACHE=10000 IDS=long_ids.txt NEW=24 PREFILL=auto EXE=$B/strata-$v "$H/run.sh" "x2-long-$v-$r" iq3_s --max-context 32768 --kv int8 --suffix-draft 0 > /dev/null
  echo "$v $r $(grep -E '^prefill  ' "$H/x2-long-$v-$r.txt" | grep -o '[0-9.]* tok/s') $(md5sum < "$H/x2-long-$v-$r.logits" | cut -c1-8)"
done; done
for ids in chat short ja; do for md in 0 2 1; do
  STRATA_PREFILL_BF16X2=$md NEW=1 IDS=${ids}_ids.txt EXE=$B/strata-bf16x2 "$H/run.sh" "x2-$ids-m$md" iq2_xs --suffix-draft 0 > /dev/null
done; python3 "$H/kl_first.py" "$H/x2-$ids-m1.logits" "$H/x2-$ids-m0.logits" "$H/x2-$ids-m2.logits"; done
for md in 0 2 1; do
  CACHE=10000 STRATA_PREFILL_BF16X2=$md IDS=long_ids.txt NEW=1 PREFILL=auto EXE=$B/strata-bf16x2 "$H/run.sh" "x2-long-m$md" iq3_s --max-context 32768 --kv int8 --suffix-draft 0 > /dev/null
  echo "long m$md $(grep -E '^prefill  ' "$H/x2-long-m$md.txt" | grep -o '[0-9.]* tok/s')"
done
python3 "$H/kl_first.py" "$H/x2-long-m1.logits" "$H/x2-long-m0.logits" "$H/x2-long-m2.logits"
for v in pre-bf16x2 bf16x2; do
  STRATA_NO_XMX=1 NEW=1 IDS=short_ids.txt EXE=$B/strata-$v "$H/run.sh" "x2-nx-$v" iq2_xs --suffix-draft 0 > /dev/null
done
cmp "$H/x2-nx-pre-bf16x2.logits" "$H/x2-nx-bf16x2.logits" && echo "no XMX: default bitwise the same"
for md in 0 2; do
  STRATA_VRAM_LIMIT_MIB=8192 STRATA_MAX_ALLOC_MIB=4096 STRATA_NO_XMX=1 STRATA_PREFILL_BF16X2=$md CACHE=auto NEW=8 IDS=short_ids.txt EXE=$B/strata-bf16x2 "$H/run.sh" "x2-small-m$md" iq2_xs --suffix-draft 0 > /dev/null
  echo "small m$md $(grep -E '^prefill  ' "$H/x2-small-m$md.txt" | grep -o '[0-9.]* tok/s')"
done
