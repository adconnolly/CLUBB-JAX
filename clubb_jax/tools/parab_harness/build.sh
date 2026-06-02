#!/usr/bin/env bash
# Build a standalone harness for the Fortran oracle's parabolic-cylinder function Dv_fnc
# (KK_utilities.F90), linking ONLY the ACM-850 `parab` machinery — no full CLUBB rebuild.
#
# Purpose: generate ground-truth Dv values at BOTH tolerances the oracle supports:
#   epss = 1.0e-4   (the SCM-run default — l_high_accuracy_parab_cyl_fnc=.false., Parabolic.f90:20)
#   epss = 1.0e-15  (the unit-test / high-accuracy path)
# Use it to (a) prove the do/ds non-bit-faithfulness is the epss=1e-4 truncation, and
# (b) validate any future JAX port of `parab` at epss=1e-4. See DESIGN.md §do/ds.
#
# Usage:
#   bash build.sh                       # builds ./dvtest  (needs intel ifx + the build modules)
#   echo "ORDER ARGUMENT" | ./dvtest    # prints: order argument Dv(1e-4) Dv(1e-15)
#   ./dvtest < pairs.txt                # one "order argument" pair per line
# (order/argument are Dv_fnc's args, i.e. D_v(z) with order=v, argument=z.)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
KK="$HERE/../../../clubb_release/src/Microphys/KK_microphys"
cd "$HERE"
ifx -O2 -fp-model=precise -c \
    clubb_precision.F90 \
    "$KK/Parabolic_constants.f90" \
    "$KK/AiryFunction.f90" \
    "$KK/Parabolic.f90"
ifx -O2 -fp-model=precise dvtest.f90 \
    clubb_precision.o Parabolic_constants.o AiryFunction.o Parabolic.o -o dvtest
echo "built $HERE/dvtest"
