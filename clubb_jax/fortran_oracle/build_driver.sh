#!/usr/bin/env bash
# Build standalone Fortran oracle executable (no F2PY, no meson needed).
set -euo pipefail

GFORTRAN=/c/Users/adac/anaconda3/Library/bin/gfortran.exe
SRC=/c/Users/adac/OneDrive/Documents/Claude/CLUBB-JAX/clubb_release/src/CLUBB_core
HERE=$(dirname "$(realpath "$0")")

DP_FLAG="-DCLUBB_REAL_TYPE=selected_real_kind(p=12)"
FFLAGS="-O2 $DP_FLAG -cpp"

cd "$HERE"
mkdir -p build_exe && cd build_exe

echo "=== Compiling CLUBB dependency chain ==="
for F in \
    clubb_precision.F90 \
    error_code.F90 \
    model_flags.F90 \
    constants_clubb.F90 \
    err_info_type_module.F90 \
    file_functions.F90 \
    interpolation.F90 \
    grid_class.F90 \
    diffusion.F90 \
    tridiag_lu_solver.F90
do
    echo "  $F"
    $GFORTRAN $FFLAGS -c "$SRC/$F"
done

echo "=== Compiling oracle driver ==="
$GFORTRAN $FFLAGS -c "$HERE/oracle_driver.F90"

echo "=== Linking oracle executable ==="
$GFORTRAN $FFLAGS \
    clubb_precision.o error_code.o model_flags.o constants_clubb.o \
    err_info_type_module.o file_functions.o interpolation.o \
    grid_class.o diffusion.o tridiag_lu_solver.o oracle_driver.o \
    -o oracle_driver.exe

echo "=== Done: build_exe/oracle_driver.exe ==="
ls -lh oracle_driver.exe
