# packaging/engine_targets.sh - the engine targets a configured build tree has (sourced, POSIX sh).
#
#   targets=$(engine_targets <cmake> <build dir>)
#   "$cmake" --build "$build" --target $targets ...
#
# Prints "strata strata_kq" when the tree has the strata_kq target (libstrata_kq.so, the K-quant prompt kernels of a build with
# llama.cpp's MMQ, -DSTRATA_MMQ=ON) and "strata" when it has not (STRATA_MMQ=OFF, the default). The build tree is asked
# (a dry run of strata_kq), not the configure line: whether strata_kq exists also depends on STRATA_MMQ_KQUANTS and on defaults
# the line need not name. Any other failure of the dry run stops the build instead of leaving the library out.
engine_targets() {
  if _et_probe=$("$1" --build "$2" --target strata_kq -- -n 2>&1); then
    echo "strata strata_kq"
    return 0
  fi
  case "$_et_probe" in
    *"unknown target 'strata_kq'"*|*"No rule to make target"*strata_kq*) echo "strata" ;;
    *) printf '%s\n' "$_et_probe" >&2
       echo "engine_targets.sh: cannot tell whether the build tree $2 has the strata_kq target" >&2
       return 1 ;;
  esac
}
