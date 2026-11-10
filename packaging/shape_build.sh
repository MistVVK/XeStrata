# packaging/shape_build.sh - SHAPE=<geometry> for the engine build scripts (sourced, POSIX sh).
#
#   SHAPE=q4_lite packaging/build_cuda_linux.sh       # the engine of a model shape: its own component strata-q4-lite (strata_control/shape_engine.py)
#
# With SHAPE set, a build script configures the engine for that geometry (-DSTRATA_GEOMETRY=<SHAPE>, unless STRATA_CMAKE_EXTRA already names
# one, e.g. -DSTRATA_GEOMETRY_FILE=<the header made from the model's file>), checks that the built program's `strata --geometry` names SHAPE,
# and packages it with package_engine.py --shape (no DeepSeek engine in it: a shape package is the Strata engine program built for that shape).
# Without SHAPE nothing changes: both variables are empty and the check returns at once.
#
#   shape_setup <repo>      sets shape_cmake (one configure argument or empty) and shape_pkg (package_engine.py arguments or empty)
#   shape_check <program> [args...]   with SHAPE set: `<program> [args...] --geometry` must print SHAPE, else the build stops
shape_setup() {
  shape_cmake=""
  shape_pkg=""
  [ -n "${SHAPE:-}" ] || return 0
  case "$SHAPE" in
    [a-z]*) ;;
    *) echo "shape_build.sh: SHAPE=$SHAPE is not a geometry name (lower case, digits and _, like q4_lite)" >&2; exit 2 ;;
  esac
  case "$SHAPE" in
    *[!a-z0-9_]*) echo "shape_build.sh: SHAPE=$SHAPE is not a geometry name (lower case, digits and _, like q4_lite)" >&2; exit 2 ;;
  esac
  case " ${STRATA_CMAKE_EXTRA:-} " in
    *-DSTRATA_GEOMETRY*) ;;           # the header is named there (a file made from the model's GGUF): its kGeometryName must be SHAPE (shape_check)
    *) [ -f "$1/include/strata/core/geometries/$SHAPE.hpp" ] || {
         echo "shape_build.sh: no header include/strata/core/geometries/$SHAPE.hpp; name the model's header with STRATA_CMAKE_EXTRA=-DSTRATA_GEOMETRY_FILE=<abs path>" >&2; exit 2; }
       shape_cmake="-DSTRATA_GEOMETRY=$SHAPE" ;;
  esac
  shape_pkg="--shape $SHAPE"
}

shape_check() {
  [ -n "${SHAPE:-}" ] || return 0
  named=$("$@" --geometry 2>/dev/null | head -n 1)
  [ "$named" = "$SHAPE" ] || {
    echo "shape_build.sh: the engine built here says --geometry '${named}', not '$SHAPE' (the header's kGeometryName must be the shape's name)" >&2; exit 1; }
  echo "shape_build.sh: the engine is built for $SHAPE (strata --geometry)"
}
