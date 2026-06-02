# parab_harness — oracle parabolic-cylinder ground truth

A standalone Fortran harness around the CLUBB oracle's `Dv_fnc` (parabolic cylinder
function `D_v`, `KK_utilities.F90`), linking only the ACM-850 `parab` machinery
(`Parabolic.f90` + `AiryFunction.f90` + `Parabolic_constants.f90`) — no full CLUBB build.

## Why

The KK upscaled-covariance integrals call `Dv_fnc`. In an SCM run the oracle computes
`D_v` with `epss = 1.0e-4` (the module default `l_high_accuracy_parab_cyl_fnc=.false.`,
`Parabolic.f90:20`); only the G-unit test sets `epss = 1.0e-15`. The JAX
`dv_parabolic_cylinder` uses a *different* algorithm (DLMF 12.4/12.9 series) that is
bit-faithful to the **true** `D_v` (≈ the `epss=1e-15` value, to ~1e-14). So for the
`dycoms2_rf02_do/ds` cases the JAX is **more accurate** than the run, and the
~1e-6 `D_v` gap is amplified ~16× by the covariance near-cancellation → the 1e-5–1e-4
covar-source failures. This harness produces the `epss=1e-4` ground truth that proves
that and that any future faithful `parab` port must reproduce.

## Build / run

    bash build.sh                    # needs intel ifx + the casper build modules
    printf "%s\n" "-3.470 35.435" | ./dvtest
    ./dvtest < pairs.txt             # one "order argument" pair per line

`dvtest` prints, per line: `order  argument  Dv(epss=1e-4)  Dv(epss=1e-15)`
(order=v, argument=z of `D_v(z)`, matching `Dv_fnc(order, argument)`).

The `do`-run hits order ∈ [−4.47, −2.0], argument ∈ [−37, 37]; the gap appears only for
argument > 0 (the chi<0 half), max rel ~1e-6.
