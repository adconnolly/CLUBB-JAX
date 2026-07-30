#!/usr/bin/env python3
# Doesn't break when running run_scm.py using old version of python
from __future__ import annotations # This means that | for strings doesn't break in old versions
import argparse
import os
import re
import subprocess
import sys

# This script lives in clubb_jax/run_scripts/.
# clubb_jax/ and clubb_release/ are siblings under the same parent (CLUBB-JAX/).
RUN_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
JAX_ROOT    = os.path.normpath(os.path.join(RUN_SCRIPTS, "../.."))   # CLUBB-JAX/
CLUBB_ROOT  = os.path.normpath(os.path.join(JAX_ROOT, "clubb_release"))

# Output convention: JAX-produced stats live under clubb_jax/output/, the Fortran
# oracle under clubb_release/output/. The JAX driver is the DEFAULT (no executable
# flag); it defaults to its OWN directory so a bare JAX run can never clobber the
# stored Fortran oracle (clubb_release/output/<case>_stats.nc) — see DESIGN.md
# "Oracle-protection convention". Run the oracle with -fortran / -legacy / -exe.
DEFAULT_OUTPUT_DIR     = os.path.join(CLUBB_ROOT, "output")          # Fortran / legacy / exe
DEFAULT_JAX_OUTPUT_DIR = os.path.join(JAX_ROOT, "clubb_jax", "output")  # JAX driver

# create_multi_col_params.py lives in clubb_release/run_scripts (not yet translated)
multi_col_params_script = os.path.join(CLUBB_ROOT, "run_scripts", "create_multi_col_params.py")

HR_SPEC_RE = re.compile(
    r"^[A-Za-z_]\w*/[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?:[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?/\d+"
    r"(?:,[A-Za-z_]\w*/[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?:[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?/\d+)*$"
)


def strip_comments_and_remove_keys(content: str, keys_to_remove=None) -> str:
    """Remove Fortran namelist comments (!) and specified keys."""
    if keys_to_remove is None:
        keys_to_remove = []
    lines = []
    for line in content.splitlines():
        line = re.sub(r"!.*", "", line)  # strip comments
        if any(key in line for key in keys_to_remove):
            continue
        lines.append(line)
    return "\n".join(lines)

def run_case(run_cmd, run_cwd, case_name, namelist_file, output_dir, run_env=None):

    if not run_cmd:
        print("No run command was provided.")
        return 1

    # clubb requires the output directory to exist prior to running
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, f"{case_name}_log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            run_cmd + [namelist_file],
            cwd=run_cwd,
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )

        # Stream model output to terminal and file at the same time.
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
            process.stdout.close()

        process.wait()

    if process.returncode not in (0, 6):
        return 1

    stats_file = os.path.join(output_dir, f"{case_name}_stats.nc")
    if not os.path.isfile(stats_file):
        print(f"WARNING: stats output not found: {stats_file}")

    return 0


def prepare_fortran_run_cwd(output_dir: str) -> str:
    """Create a local Fortran cwd whose ../input resolves to clubb_release/input."""
    work_root = os.path.join(output_dir, "_clubb_run")
    run_scripts_dir = os.path.join(work_root, "run_scripts")
    os.makedirs(run_scripts_dir, exist_ok=True)

    input_link = os.path.join(work_root, "input")
    target = os.path.join(CLUBB_ROOT, "input")
    if os.path.lexists(input_link):
        if os.path.islink(input_link) and os.readlink(input_link) == target:
            return run_scripts_dir
        if os.path.isdir(input_link) and not os.path.islink(input_link):
            sys.exit(f"Cannot prepare Fortran run cwd; path exists and is not a symlink: {input_link}")
        os.unlink(input_link)
    os.symlink(target, input_link)
    return run_scripts_dir


def read_model_times(model_file):
    """Read time_initial, time_final, dt_main from a model file if present."""
    values = {}
    line_re = re.compile(r'^\s*([a-zA-Z_]\w*)\s*=\s*([-+0-9.eEdD]+)')
    with open(model_file) as f:
        for line in f:
            m = line_re.match(line)
            if m:
                key, val = m.groups()
                val = val.replace("D", "E").replace("d", "e")
                try:
                    values[key.lower()] = float(val)
                except ValueError:
                    pass
    return values

def parse_multicol_arg(value: str):
    """Parse -multicol as either an integer ngrdcol or an hr-style spec string."""
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("must provide a non-empty -multicol value")

    try:
        ngrdcol = int(stripped)
    except ValueError:
        if HR_SPEC_RE.fullmatch(stripped):
            return {"ngrdcol": None, "hr_spec": stripped}
        raise argparse.ArgumentTypeError(
            "must be either an integer column count or an hr spec like C8/0.2:0.8/4"
        )

    if ngrdcol < 1:
        raise argparse.ArgumentTypeError("integer -multicol value must be >= 1")

    return {"ngrdcol": ngrdcol, "hr_spec": None}


def convert_to_multi_col(
    params_file: str,
    case_name: str,
    output_dir: str,
    ngrdcol: int | None,
    hr_spec: str | None = None,
) -> str:
    """
    Create a temporary multi-column params file and return its path.

    If ``hr_spec`` is provided, it is forwarded directly to
    ``create_multi_col_params.py -hr``. Otherwise ``ngrdcol`` uses the legacy
    dup_tweak path.
    """

    if not os.path.isfile(multi_col_params_script):
        sys.exit(f"Missing helper script: {multi_col_params_script}")

    out_file = os.path.join(output_dir, f"{case_name}_multicol_params.in")

    cmd = [sys.executable, multi_col_params_script,
             "-param_file", params_file,
             "-out_file", out_file]
    if hr_spec:
        cmd.extend(["-hr", hr_spec])
    else:
        cmd.extend(["-mode", "dup_tweak", "-n", str(ngrdcol)])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"create_multi_col_params failed (exit {e.returncode}). "
                    f"Command was:\n  {' '.join(cmd)}")

    if not os.path.isfile(out_file):
        sys.exit(f"Expected output file not created: {out_file}")

    return out_file


def override_value(override_string, clubb_in_text):
    """
    Apply overrides from -override KEY1=val1,KEY2=val2,... to the aggregate text.
    """
    for pair in override_string.split(","):
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        key, val = key.strip(), val.strip()
        replacement = f"{key} = {val}"

        assignment_re = re.compile(
            rf"(?im)^(\s*){re.escape(key)}\s*=.*$"
        )

        clubb_in_text, replacements = assignment_re.subn(
            lambda match: f"{match.group(1)}{key} = {val}",
            clubb_in_text,
        )

        if replacements == 0:
            clubb_in_text += f"\n{replacement}\n"
    return clubb_in_text


def setup_files_and_aggregate(args, output_dir):
    """Resolve file paths, validate, and create the aggregate namelist."""

    # Model file
    model_file = os.path.join(CLUBB_ROOT, f"input/case_setups/{args.case_name}_model.in")
    if not os.path.isfile(model_file):
        sys.exit(f"{model_file} does not exist")

    # Config dir, default is input/tunable_parameters
    config_dir = (os.path.abspath(args.config) if args.config
                    else os.path.join(CLUBB_ROOT, "input/tunable_parameters"))

    if not os.path.isdir(config_dir):
        sys.exit(f"--config directory does not exist: {config_dir}")

    # A case may select a non-default tunable-parameters file via `parameter_file`
    # in its &model_setting namelist (e.g. ekman/coriolis_test/atex_long). That key
    # is a run-script directive, not a Fortran namelist variable, so it must be used
    # to pick the params file AND stripped from the aggregate (else the Fortran
    # namelist read errors). Path is relative to clubb_release/run_scripts/.
    case_param_file = None
    with open(model_file) as _mf:
        m = re.search(r'^\s*parameter_file\s*=\s*["\']([^"\']+)["\']',
                      _mf.read(), re.MULTILINE)
    if m:
        case_param_file = os.path.normpath(
            os.path.join(CLUBB_ROOT, "run_scripts", m.group(1)))
        # Some upstream model.in files carry a stale parameter_file directory
        # (e.g. coriolis_test points at input/tunable_parameters/ but the file
        # actually lives in input/tunable_parameters_coriolis_cases/). If the
        # literal path is missing, fall back to locating the basename anywhere
        # under input/ so the case still runs.
        if not os.path.isfile(case_param_file):
            _base = os.path.basename(case_param_file)
            for _root, _dirs, _files in os.walk(os.path.join(CLUBB_ROOT, "input")):
                if _base in _files:
                    case_param_file = os.path.join(_root, _base)
                    break

    # Files (respect overrides)
    params_file       = args.params       or case_param_file \
                                          or os.path.join(config_dir, "tunable_parameters.in")
    flags_file        = args.flags        or os.path.join(config_dir, "configurable_model_flags.in")
    silhs_params_file = args.silhs_params or os.path.join(config_dir, "silhs_parameters.in")
    stats_arg = (args.stats or "").strip()
    disable_stats = stats_arg.lower() == "none"
    stats_file = (None if disable_stats
                    else (args.stats or os.path.join(CLUBB_ROOT, "input/stats/standard_stats.in")))

    # The Fortran binary resolves "../input/..." relative to its CWD and writes
    # scratch files such as fort.10 there. Use a local generated cwd so a
    # symlinked clubb_release checkout remains read-only.
    run_cwd = prepare_fortran_run_cwd(output_dir)
    run_env = os.environ.copy()
    run_env["GIT_DIR"] = os.path.join(CLUBB_ROOT, ".git")
    run_env["GIT_WORK_TREE"] = CLUBB_ROOT
    run_cmd = None

    if args.exe:
        # Use the users input from -exe to determine which executable to use
        if args.legacy:
            print(f"-legacy overriden by -exe entry: {args.exe}")
        executable  = os.path.abspath(args.exe)
        if not os.path.isfile(executable):
            sys.exit(f"{executable} not found (did you re-compile?)")
        run_cmd = [executable]
    elif args.legacy:
        # The legacy install location is /bin/clubb_standalone
        executable  = os.path.join(CLUBB_ROOT, f"bin/clubb_standalone")
        if not os.path.isfile(executable):
            sys.exit(f"{executable} not found (did you re-compile?)")
        run_cmd = [executable]
    elif args.python:
        python_driver = os.path.join(CLUBB_ROOT, "clubb_python_driver", "clubb_standalone.py")
        if not os.path.isfile(python_driver):
            sys.exit(f"Python standalone driver not found: {python_driver}")
        clubb_python_api_dir = os.path.join(CLUBB_ROOT, "clubb_python_api")
        if not os.path.isdir(clubb_python_api_dir):
            sys.exit(f"Python API directory not found: {clubb_python_api_dir}")
        run_cwd = RUN_SCRIPTS
        executable = f"{sys.executable} -m clubb_python_driver.clubb_standalone"
        run_cmd = [sys.executable, "-m", "clubb_python_driver.clubb_standalone"]
        run_env = os.environ.copy()
        existing_pythonpath = run_env.get("PYTHONPATH", "")
        pythonpath_entries = [CLUBB_ROOT, clubb_python_api_dir]
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        run_env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    elif args.fortran or args.driver_test:
        # The compiled Fortran oracle from install/latest (the last compiled binary).
        # This was the old no-flag default; the no-flag default is now the JAX driver,
        # so the oracle is selected explicitly with -fortran (or -driver_test).
        if args.driver_test:
            executable  = os.path.join(CLUBB_ROOT, f"install/latest/clubb_driver_test")
        else:
            executable  = os.path.join(CLUBB_ROOT, f"install/latest/clubb_standalone")
        if not os.path.isfile(executable):
            sys.exit(f"{executable} not found (did you re-compile?)")
        run_cmd = [executable]
    else:
        # DEFAULT: the JAX standalone driver (the model this repo is). No flag needed.
        jax_driver = os.path.join(JAX_ROOT, "clubb_jax", "src", "clubb_standalone.py")
        if not os.path.isfile(jax_driver):
            sys.exit(f"JAX standalone driver not found: {jax_driver}")
        clubb_python_api_dir = os.path.join(CLUBB_ROOT, "clubb_python_api")
        if not os.path.isdir(clubb_python_api_dir):
            sys.exit(f"Python API directory not found: {clubb_python_api_dir}")
        # The JAX driver still calls Fortran-backed prescribe_forcings, which
        # resolves ../input relative to CWD.
        run_cwd = prepare_fortran_run_cwd(output_dir)
        executable = f"{sys.executable} -m clubb_jax.src.clubb_standalone"
        run_cmd = [sys.executable, "-m", "clubb_jax.src.clubb_standalone"]
        run_env = os.environ.copy()
        existing_pythonpath = run_env.get("PYTHONPATH", "")
        pythonpath_entries = [JAX_ROOT, CLUBB_ROOT, clubb_python_api_dir]
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        run_env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    print(f" - using executable: {executable}")

    # Expand multi-column params if requested
    if args.multicol is not None:
        params_file = convert_to_multi_col(
            params_file,
            args.case_name,
            output_dir,
            args.multicol["ngrdcol"],
            args.multicol["hr_spec"],
        )

    # Validate files
    files_to_validate = [
        ("-params", params_file),
        ("-flags", flags_file),
        ("-silhs_params", silhs_params_file),
    ]
    if not (args.python or args.is_jax):
        files_to_validate.append(("-exe", executable))
    if not disable_stats:
        files_to_validate.append(("-stats", stats_file))

    for opt, f in files_to_validate:
        if not os.path.isfile(f):
            sys.exit(f"Required file for {opt} not found: {f}")

    # Aggregate into <output_dir>/CASE.in
    os.makedirs(output_dir, exist_ok=True)
    clubb_input_namelist = os.path.join(output_dir, f"{args.case_name}.in")
    with open(clubb_input_namelist, "w") as out:
        files_to_aggregate = [params_file, silhs_params_file, flags_file, model_file]
        if not disable_stats:
            files_to_aggregate.append(stats_file)
        for f in files_to_aggregate:
            with open(f) as src:
                # `parameter_file` is a run-script directive, not a Fortran namelist
                # variable — strip it so the Fortran namelist read does not error.
                out.write(strip_comments_and_remove_keys(
                    src.read(), keys_to_remove=["parameter_file"]))
                out.write("\n")

    return clubb_input_namelist, model_file, run_cmd, run_cwd, run_env


def _set_stats_output_dir(clubb_in: str, output_dir: str) -> str:
    """Ensure &stats_setting contains output_dir."""
    out_norm = output_dir.replace("\\", "/")
    stats_match = re.search(r"(?is)(&\s*stats_setting\b)(.*?)(/)", clubb_in)
    if not stats_match:
        clubb_in += (
            "\n&stats_setting\n"
            f"output_dir = '{out_norm}',\n"
            "/\n"
        )
        return clubb_in

    header = stats_match.group(1)
    body = stats_match.group(2)
    end = stats_match.group(3)
    if re.search(r"(?im)^\s*output_dir\s*=", body):
        body = re.sub(
            r"(?im)^\s*output_dir\s*=.*$",
            f"output_dir = '{out_norm}',",
            body,
        )
    else:
        body = body.rstrip() + f"\noutput_dir = '{out_norm}',\n"

    return clubb_in[:stats_match.start()] + header + body + end + clubb_in[stats_match.end():]


def edit_namelist(args, clubb_input_namelist, model_file, output_dir):
    """Apply modifications to the aggregate namelist."""

    with open(clubb_input_namelist) as f:
        clubb_in = f.read()

    # Timestep modifications
    if args.dt_main is not None:
        clubb_in = re.sub(r"dt_main\s*=.*", f"dt_main = {args.dt_main}.0", clubb_in)

    if args.dt_rad is not None:
        clubb_in = re.sub(r"dt_rad\s*=.*", f"dt_rad = {args.dt_rad}.0", clubb_in)

    # Explicit grids
    if args.zt_grid is not None:
        clubb_in += f"\nnzmax = {args.nzmax}\nzt_grid_fname = '{args.zt_grid}'\ngrid_type = 2\n"

    if args.zm_grid is not None:
        clubb_in += f"\nnzmax = {args.nzmax}\nzm_grid_fname = '{args.zm_grid}'\ngrid_type = 3\n"

    # Stats output control, args.tout defines output frequency, and setting to 0 disables output.
    disable_stats = ((args.stats or "").strip().lower() == "none")
    if args.tout is not None:
        if args.tout == 0:
            disable_stats = True
        else:
            clubb_in = re.sub(r"stats_tout\s*=.*", f"stats_tout = {args.tout}.0", clubb_in)

    if disable_stats:
        if re.search(r"l_stats\s*=.*", clubb_in):
            clubb_in = re.sub(r"l_stats\s*=.*", "l_stats = .false.", clubb_in)
        else:
            clubb_in += "\nl_stats = .false.\n"

    # Debug level
    if args.debug is not None:
        clubb_in = re.sub(r"debug_level\s*=.*", f"debug_level = {args.debug}", clubb_in)

    # Iteration control
    if args.max_iters is not None:
        vals = read_model_times(model_file)
        time_initial = vals["time_initial"]
        dt_main_val = args.dt_main if args.dt_main else vals.get("dt_main")
        new_time_final = time_initial + dt_main_val * args.max_iters

        # only update time_final if it's less than the current one\
        if ( vals["time_final"] >= new_time_final ):
            clubb_in = re.sub(r"time_final\s*=.*", f"time_final = {new_time_final}", clubb_in)

    # Overrides
    if args.override:
        clubb_in = override_value(args.override, clubb_in)

    # Route all CLUBB output files into the selected directory.
    clubb_in = _set_stats_output_dir(clubb_in, output_dir)

    # Save back
    with open(clubb_input_namelist, "w") as f:
        f.write(clubb_in)


def apply_jax_device_flags(args, run_env):
    """Translate -cpu/-gpu into JAX backend env vars + a CPU-core affinity cap.

    Why this is needed: the JAX driver runs through XLA, whose CPU backend executes
    every array op on an Eigen thread pool sized to the machine's logical core count.
    So even a single-column run spreads each elementwise / reduction / tridiagonal-solve
    kernel across ALL cores with no code-level threading — that is the "implicit
    parallelism" a bare run exhibits. In jaxlib 0.10.2 the XLA_FLAGS / OMP_NUM_THREADS
    knobs are ignored by the CPU runtime, so the only reliable cap is OS CPU affinity
    (os.sched_setaffinity), applied here on the parent and inherited by the child JAX
    process. The GPU backend runs on a single device today, so -gpu N only restricts
    visibility (CUDA_VISIBLE_DEVICES); it does not yet shard columns across GPUs.
    See DESIGN.md "Backend & device control" for the trade-offs.
    """
    if args.cpu is not None:
        run_env["JAX_PLATFORMS"] = "cpu"
        if args.cpu and args.cpu > 0 and hasattr(os, "sched_getaffinity"):
            avail = sorted(os.sched_getaffinity(0))
            n = min(args.cpu, len(avail))
            if args.cpu > len(avail):
                print(f"WARNING: requested -cpu {args.cpu} but only {len(avail)} core(s) "
                      f"available; using {n}.")
            cores = set(avail[:n])
            os.sched_setaffinity(0, cores)   # inherited by the child JAX process
            print(f" - JAX backend: cpu, pinned to {n} core(s): {sorted(cores)}")
        else:
            navail = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "all"
            print(f" - JAX backend: cpu, using all {navail} available core(s)")
    elif args.gpu is not None:
        run_env.pop("JAX_PLATFORMS", None)   # force GPU (override any env cpu setting)
        if args.gpu and args.gpu > 0:
            visible = ",".join(str(i) for i in range(args.gpu))
            run_env["CUDA_VISIBLE_DEVICES"] = visible
            print(f" - JAX backend: gpu, CUDA_VISIBLE_DEVICES={visible} "
                  f"(driver uses a single device; >1 GPU is not yet sharded)")
        else:
            print(" - JAX backend: gpu, using all visible GPU(s)")


def main():

    parser = argparse.ArgumentParser( description="Run the standalone CLUBB model")

    # The "config" folder is assumed to contain the tunable parameters,
    # silhs parameters, and config flags. This is meant to be a quick way to input all these
    # files without having to specify them individually.
    # Individually specificed files will overwrite the ones found from here
    parser.add_argument("-config", metavar="[DIR]",
        help=("Directory containing all three tunable files:\n"
            "  tunable_parameters.in\n"
            "  configurable_model_flags.in\n"
            "  silhs_parameters.in\n"
            "Defaults to input/tunable_parameters if not given."))

    # per-file overrides (params/flags already existed; add silhs_params)
    parser.add_argument("-params", metavar="[FILE]",
        help=("Define the tunable parameters.\n"
            "Used to override params file defined by --config"))
    parser.add_argument("-flags", metavar="[FILE]",
        help=("Model flags file.\n"
            "Used to override flags file defined by --config"))
    parser.add_argument("-silhs_params", metavar="[FILE]",
        help=("SILHS parameters file.\n"
            "Used to override silhs_params file defined by --config"))

    # Options to input grid files, and nzmax defines the maximum number of vertical levels
    # when inputting grid files like this
    parser.add_argument("-zt_grid", metavar="[FILE]",
        help="Specify a zt grid file from input/grid.\nDefault: unused")
    parser.add_argument("-zm_grid", metavar="[FILE]",
        help="Specify a zm grid file from input/grid.\nDefault: unused")
    parser.add_argument("-nzmax", metavar="[NUM]", type=int,
        help="Max number of levels (required if specifying a zt/zm grid)")

    # Stats fiules define which fields to output, can be overriden here
    parser.add_argument("-stats", metavar="[FILE]",
        help=("Stats file defining fields to output.\n"
              "Default: input/stats/standard_stats.in.\n"
              "Use 'none' to disable stats output."))

    # This script will try to figure out the right executable to use based on the
    # compiler in the environment but inputting a specific executable will
    # override that with the specified one
    parser.add_argument("-exe", metavar="[EXECUTABLE]",
        help="CLUBB executable to use.\nDefault: clubb_release/install/latest/clubb_standalone")

    parser.add_argument("-driver_test", action="store_true",
        help="Runs the clubb_driver_test executable instead of clubb_standalone"
    )

    parser.add_argument("-python", action="store_true",
        help="Run the Python standalone driver (python -m clubb_python_driver.clubb_standalone)")

    # The JAX standalone driver is the DEFAULT (run with no executable-selector flag);
    # the old explicit `-jax` flag is retired. To run the compiled Fortran oracle
    # instead, use -fortran (install/latest), -legacy (bin/), or -exe PATH.
    parser.add_argument("-fortran", action="store_true",
        help="Run the compiled Fortran oracle (install/latest/clubb_standalone).\n"
             "Was the old no-flag default; the no-flag default is now the JAX driver.")

    # The old method of compile clubb resulted in the executable "clubb/bin/clubb_standalone"
    # this option causes that to be the prefered executable, unless -exe is specified
    parser.add_argument(
        "-legacy",
        action="store_true",
        help="Runs the legacy compiled version of clubb_standalone (with compile.bash)"
    )

    # JAX compute-backend selection + device-count cap (JAX runs only; mutually
    # exclusive). The optional integer caps how many devices are used:
    #   -cpu N  → JAX_PLATFORMS=cpu and pin the process to N cores (sched_setaffinity)
    #   -gpu N  → GPU backend with CUDA_VISIBLE_DEVICES limited to N GPUs
    # Bare -cpu / -gpu selects the backend but uses all available devices. With
    # neither flag the backend follows the environment (e.g. jaxenv.sh).
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument("-cpu", nargs="?", type=int, const=0, default=None,
        metavar="N",
        help="Run JAX on CPU. Optional N caps the run to N CPU cores (default: all).\n"
             "See DESIGN.md 'Backend & device control' for why JAX uses every core "
             "by default and the trade-offs of capping.")
    backend_group.add_argument("-gpu", nargs="?", type=int, const=0, default=None,
        metavar="N",
        help="Run JAX on GPU. Optional N limits to N visible GPUs (default: all).\n"
             "Note: the driver currently runs on a single device; >1 GPU does not "
             "yet shard columns.")

    # Allow a custom output directory to be used for all generated files.
    parser.add_argument("-out_dir", metavar="[DIR]",
        help="Output directory for results.\n"
             "Default: clubb_jax/output/ for the JAX driver, clubb_release/output/ "
             "for the Fortran oracle.")

    # Runtime options
    parser.add_argument("-debug", metavar="[NUM]",
        help="Debug level (0–3) that controls CLUBB's runtime checks (0 is no checks)."
                "\nDefault specified in model file.")
    parser.add_argument("-max_iters", metavar="[NUM]", type=int,
        help="Maximum number of iterations")
    parser.add_argument("-dt_main", metavar="[SECONDS]", type=int,
        help="Main timestep (s).\nDefault from model file.")
    parser.add_argument("-dt_rad", metavar="[SECONDS]", type=int,
        help="Radiation timestep (s).\nDefault from model file.")
    parser.add_argument("-tout", metavar="[SECONDS]", type=int,
        help="Stats output interval (s). Use 0 to disable.\nDefault from model file.")

    # Setting -multicol will call create_multi_col_params.py to
        # generate a multi-column parameter file.
    # Integer input uses the legacy dup_tweak path. A string matching the hr syntax is forwarded to
    # create_multi_col_params.py -hr for hypergrid generation.
    parser.add_argument("-multicol", metavar="[NUM|SPEC]", type=parse_multicol_arg,
        help=("Generate a multi-column parameter file. "
              "Use an integer for dup_tweak mode, e.g. -multicol 4, or an hr spec like "
              "-multicol C8/0.2:0.8/4"))

    # This can be used to override pretty much any settings in the aggregate namelist
    parser.add_argument(
        "-override",
        help="Comma-separated key=value pairs, e.g. -override FLAG1=true,C2=2.0,...",
    )

    parser.add_argument("case_name", help="Name of the case to run")
    args = parser.parse_args()

    # Error check
    ndefined = (sum(bool(x) for x in
                        [args.exe, args.legacy, args.driver_test, args.python, args.fortran]))
    if ndefined > 1:
        parser.error("Only one of -exe, -legacy, -driver_test, -python, or -fortran may be specified.")

    # JAX is the default driver: no executable-selector flag means run JAX.
    args.is_jax = (ndefined == 0)

    # -cpu/-gpu configure the JAX compute backend; they are meaningless for a
    # Fortran/Python run.
    if (args.cpu is not None or args.gpu is not None) and not args.is_jax:
        parser.error("-cpu/-gpu select the JAX compute backend and cannot be combined "
                     "with -exe/-legacy/-driver_test/-python/-fortran.")
    # Validate grid options
    if args.zt_grid and args.zm_grid:
        sys.exit(f"\n\033[91mERROR: Cannot specify both a ZT grid and a ZM grid\033[0m")
    if args.nzmax and not (args.zt_grid or args.zm_grid):
        print("\n\033[93mWARNING: Specifying --nzmax will have no effect without "
                "specifying a --zm_grid or --zt_grid\033[0m")

    # The default JAX driver writes to clubb_jax/output/ (its own dir); the Fortran
    # oracle (-fortran/-legacy/-exe) writes to clubb_release/output/ (the oracle home).
    # -out_dir overrides either.
    if args.out_dir:
        output_dir = os.path.abspath(args.out_dir)
    elif args.is_jax:
        output_dir = os.path.abspath(DEFAULT_JAX_OUTPUT_DIR)
    else:
        output_dir = os.path.abspath(DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: setup and aggregate namelist files into <output_dir>/CASE.in
    (clubb_input_namelist, model_file,
            run_cmd, run_cwd, run_env) = setup_files_and_aggregate(args, output_dir)

    # Step 2: edit clubb_input_namelist based on input specifications
    edit_namelist(args, clubb_input_namelist, model_file, output_dir)

    # Step 2b: apply -cpu/-gpu backend + device-count selection (JAX runs only).
    if args.is_jax:
        apply_jax_device_flags(args, run_env)

    # Step 3: run model
    print(f"=================== Running {args.case_name} ===================")
    result = run_case(run_cmd, run_cwd, args.case_name, clubb_input_namelist, output_dir, run_env)

    sys.exit(result)


if __name__ == "__main__":
    main()
