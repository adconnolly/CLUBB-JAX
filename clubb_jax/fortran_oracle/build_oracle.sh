#!/usr/bin/env bash
# Build the CLUBB Fortran oracle Python extension via F2PY.
#
# Usage:  bash build_oracle.sh
# Output: clubb_oracle.so (importable as `import clubb_oracle`)
#
# Requires: gfortran (conda-forge), numpy (with f2py)
set -euo pipefail

GFORTRAN=/c/Users/adac/anaconda3/Library/bin/gfortran.exe
PYTHON=/c/Users/adac/anaconda3/python.exe
F2PY="$PYTHON -m numpy.f2py"

SRC=/c/Users/adac/OneDrive/Documents/Claude/CLUBB-JAX/clubb_release/src/CLUBB_core
HERE=$(dirname "$(realpath "$0")")

# Preprocessor flag: core_rknd = double precision
DP_FLAG="-DCLUBB_REAL_TYPE=selected_real_kind(p=12)"

cd "$HERE"
mkdir -p build && cd build

echo "=== Compiling CLUBB dependency chain ==="

FFLAGS="-fPIC -O2 $DP_FLAG -cpp"

for SRC_FILE in \
    clubb_precision.F90 \
    error_code.F90 \
    model_flags.F90 \
    constants_clubb.F90 \
    err_info_type_module.F90 \
    file_functions.F90 \
    interpolation.F90 \
    grid_class.F90 \
    diffusion.F90
do
    echo "  Compiling $SRC_FILE ..."
    $GFORTRAN $FFLAGS -c "$SRC/$SRC_FILE"
done

echo "=== Compiling oracle wrapper ==="
$GFORTRAN $FFLAGS -c "$HERE/oracle_wrapper.F90"

echo "=== Building F2PY extension ==="
OBJS="clubb_precision.o error_code.o model_flags.o constants_clubb.o \
      err_info_type_module.o file_functions.o interpolation.o \
      grid_class.o diffusion.o oracle_wrapper.o"

# Python 3.12+ F2PY uses meson backend; set FC and add meson/ninja to PATH
export FC=$GFORTRAN
export PATH="/c/Users/adac/anaconda3/Scripts:$PATH"

$F2PY -m clubb_oracle -c \
    --f90flags="$DP_FLAG -fPIC -O2 -cpp" \
    $OBJS \
    "$HERE/oracle_wrapper.F90" \
    2>&1

echo "=== Done ==="
ls -lh clubb_oracle*.so clubb_oracle*.pyd 2>/dev/null || echo "WARNING: no .so/.pyd found"
