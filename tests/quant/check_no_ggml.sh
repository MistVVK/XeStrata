#!/usr/bin/env bash
# tests/quant/check_no_ggml.sh - with STRATA_OWN_CPU_QUANT the CPU expert pool, the --mtp-q4 requantizer and the DeepSeek CPU lane must call no ggml function.
#
#   tests/quant/check_no_ggml.sh <build dir> [extra object-name patterns...]
#
# Looks at the undefined symbols of the pool's objects (strata_kernels_cpu, strata_quant, mtp.cpp, ds4_gg_cpu.cpp, and any extra pattern) and fails if one
# starts with ggml_ or is one of ggml-cpu's quantizer / dot entry points.  The tests that use ggml-cpu as the oracle are not looked at.
set -u
B=${1:?usage: check_no_ggml.sh <build dir> [pattern...]}
shift
PATS=("strata_kernels_cpu.dir/.*\.o$" "strata_quant.dir/.*\.o$" "/mtp\.cpp\.o$" "/ds4_gg_cpu\.cpp\.o$" "$@")
bad=0
n=0
while IFS= read -r o; do
  ok=0
  for p in "${PATS[@]}"; do [[ $o =~ $p ]] && ok=1; done
  [ $ok = 1 ] || continue
  n=$((n + 1))
  syms=$(nm -u "$o" 2>/dev/null | awk '{print $2}' | grep -E '^(ggml_|quantize_row_|dequantize_row_|ggml_vec_dot)' || true)
  if [ -n "$syms" ]; then
    echo "GGML symbols in $o:"; echo "$syms" | sed 's/^/   /'
    bad=1
  fi
done < <(find "$B" -name '*.o' -o -name '*.obj')
echo "check_no_ggml: $n objects looked at, $([ $bad = 0 ] && echo clean || echo 'ggml symbols found')"
[ $n -gt 0 ] || { echo "no objects matched"; exit 2; }
exit $bad
