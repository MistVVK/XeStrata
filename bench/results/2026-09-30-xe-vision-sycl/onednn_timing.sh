#!/bin/bash
# timing.sh ROUNDS: encoder variants alternating; per process: READY, then images, "ENC" ms per image
S=<scratch>; V=$S/vision; D=$S/dnn
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
MM=<data>/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf
MO=<data>/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
E=<repo>/engine
run() {  # tag cap envs... -- exe args...
  local tag=$1 cap=$2; shift 2; local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local imgs; if [ $cap = 300 ]; then imgs="landscape landscape portrait portrait"; else imgs="landscape-large landscape-large"; fi
  local cmds="" i=0; for im in $imgs; do i=$((i+1)); cmds+="ENC $V/$im.png $D/out/$tag-$cap-$im-$i.sve"$'\n'; done; cmds+="QUIT"$'\n'
  local t0=$(date +%s.%N)
  local out=$(printf '%s' "$cmds" | env "${envs[@]}" "$@" --mmproj $MM --model $MO --max-tokens $cap 2>>$D/out/$tag-$cap.err | tr '\n' ' ')
  local t1=$(date +%s.%N)
  printf 'round %s %-8s cap %4s: %s (process %.1f s)\n' "$R" $tag $cap "$out" $(awk "BEGIN{print $t1 - $t0}")
}
mkdir -p $D/out
for R in $(seq 1 ${1:-2}); do
  for cap in 300 1024; do
    run cpu $cap X=1 -- $E/strata-vision-cpu --threads 8
    run nodnn $cap X=1 -- $D/build-nodnn/bin/strata-vision --gpu
    run dnn $cap X=1 -- $E/strata-vision-sycl --gpu
    run dnnoff $cap GGML_SYCL_ENABLE_DNN=0 GGML_SYCL_FA_ONEDNN=0 -- $E/strata-vision-sycl --gpu
  done
done
