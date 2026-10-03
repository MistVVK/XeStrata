#!/bin/bash
# enc_ab.sh: the image encoders alone (cpu: 8 threads; sycl: ggml-sycl in oneAPI's environment; vulkan: ggml-vulkan,
# the B70 by PCI address), the images of bench/results/2026-09-30-xe-vision-*, caps 300 and 1024, two alternating
# rounds; each image twice a process after the warm-up.  ENC times and the embeddings (<variant>-<cap>-<image>-<i>.sve).
set -u
H=$(cd "$(dirname "$0")" && pwd); R="<repo>"; D="<data>"   # the repository and the data folder
MM=$D/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf; MD=$D/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
IMG=$R/bench/results/2026-09-30-xe-vision-cpu; BIG=$R/bench/results/2026-09-30-xe-vision-sycl/landscape-large.png
enc() {   # variant cap round
  local v=$1 cap=$2 r=$3 cmd
  # sycl: the inner shell expands $0 and $@ (the encoder and its arguments) after loading oneAPI's environment
  # shellcheck disable=SC2016
  case $v in
    cpu) cmd=("$R/engine/strata-vision-cpu" --threads 8) ;;
    sycl) cmd=(bash -c 'set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; GGML_SYCL_ENABLE_DNN=0 exec "$0" "$@"' "$R/engine/strata-vision-sycl" --gpu) ;;
    vulkan) cmd=("$H/../bin/strata-vision-vulkan" --gpu --gpu-pci 0000:03:00.0) ;;
  esac
  if [ "$cap" = 300 ]; then
    reqs="ENC $IMG/landscape.png $H/$v-300-landscape-1.sve
ENC $IMG/landscape.png $H/$v-300-landscape-2.sve
ENC $IMG/portrait.png $H/$v-300-portrait-3.sve
ENC $IMG/portrait.png $H/$v-300-portrait-4.sve"
  else
    reqs="ENC $BIG $H/$v-1024-landscape-large-1.sve
ENC $BIG $H/$v-1024-landscape-large-2.sve"
  fi
  out=$(printf '%s\nQUIT\n' "$reqs" | "${cmd[@]}" --mmproj "$MM" --model "$MD" --max-tokens "$cap" 2>"$H/$v-$cap-$r.err" | tr '\n' ' ')
  echo "round $r $v cap $cap: $out"
}
for r in 1 2; do for v in cpu sycl vulkan; do for cap in 300 1024; do enc $v $cap $r; done; done; done
