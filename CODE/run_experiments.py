#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

CODE_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from common import paths

EXP_DIR = CODE_DIR / "EXPERIMENTS"

@dataclass
class Experiment:
    key: str
    title: str
    runner: str
    script: pathlib.Path
    forward: tuple = field(default_factory=tuple)

    @property
    def available(self) -> bool:
        return self.script.exists()

EXPERIMENTS = {
    "e1": Experiment(
        "e1", "DAG sensitivity of the analysis (re-decompose existing SA results)",
        "R", EXP_DIR / "E1_dag_sensitivity.R", forward=("attr", "generator")),
    "e3": Experiment(
        "e3", "Template-generator ablation (deterministic TEMPLATE generator)",
        "python", EXP_DIR / "E3_template_generator.py", forward=("attr",)),
    "gc": Experiment(
        "gc", "Generator comparison (CHATGPT4/GEMINIPRO/TEMPLATE, aggregate + per-scenario)",
        "R", EXP_DIR / "generator_comparison.R", forward=("attr",)),
}

ALL_KEYS = ("e1", "gc")

def _find_rscript() -> str | None:
    env = os.environ.get("RSCRIPT")
    if env and pathlib.Path(env).exists():
        return env
    on_path = shutil.which("Rscript")
    if on_path:
        return on_path
    patterns = [
        r"C:\Program Files\R\R-*\bin\x64\Rscript.exe",
        r"C:\Program Files\R\R-*\bin\Rscript.exe",
        r"C:\Program Files\R\R-*\bin\Rscript",
        "/usr/local/bin/Rscript",
        "/usr/bin/Rscript",
        "/opt/R/*/bin/Rscript",
    ]
    found = []
    for pat in patterns:
        found.extend(glob.glob(pat))
    found = [f for f in found if pathlib.Path(f).exists()]
    found.sort(reverse=True)
    return found[0] if found else None

def _build_command(exp: Experiment, attr, generator, passthrough, plots_only=False):
    if exp.runner == "python":
        cmd = [sys.executable, str(exp.script)]
    elif exp.runner == "R":
        rscript = _find_rscript()
        if not rscript:
            raise SystemExit(
                "Rscript not found. Install R (>= 4.4) and either add it to PATH, "
                "or set the RSCRIPT env var to the Rscript executable, e.g.\n"
                '  PowerShell:  $env:RSCRIPT = "C:\\Program Files\\R\\R-4.6.1\\bin\\Rscript.exe"\n'
                '  bash:        export RSCRIPT=/usr/bin/Rscript')
        cmd = [rscript, str(exp.script)]
    else:
        raise SystemExit(f"Unknown runner '{exp.runner}' for {exp.key}")

    if "attr" in exp.forward:
        cmd += ["--attr", attr]
    if "generator" in exp.forward:
        cmd += ["--generator", generator]
    if plots_only and exp.runner == "R":
        cmd += ["--plots-only"]
    cmd += list(passthrough)
    return cmd

def _print_list():
    print("Major-revision experiments (python CODE/run_experiments.py <key> ...):\n")
    for key, exp in EXPERIMENTS.items():
        tag = "py " if exp.runner == "python" else "R  "
        status = "" if exp.available else "   [not yet implemented]"
        fwd = ", ".join(exp.forward) if exp.forward else "none"
        print(f"  {key}  [{tag}]  {exp.title}{status}")
        print(f"        forwards: {fwd}   script: {exp.script.relative_to(CODE_DIR.parent)}")
    print(f"\n  all  ->  {', '.join(ALL_KEYS)} (data-reusing / analytical; E3 must be run explicitly)")
    print("\n--generator accepts CHATGPT4 | GEMINIPRO | TEMPLATE | all "
          "(all = run once per generator).")
    print("--attr / --generator also read from ATTR / GENERATOR env vars.")
    print("Anything after `--` is forwarded to the experiment script.")

def _resolve_generators(spec):
    raw = spec if spec is not None else os.environ.get("GENERATOR")
    if raw is not None and str(raw).strip().lower() == "all":
        return list(paths.GENERATORS)
    return [paths.resolve_generator(spec, default="CHATGPT4")]

def _select_keys(spec):
    s = (spec or "").strip().lower()
    if s == "all":
        return list(ALL_KEYS)
    keys = [k.strip() for k in s.split(",") if k.strip()]
    bad = [k for k in keys if k not in EXPERIMENTS]
    if bad:
        raise SystemExit(f"Unknown experiment(s): {', '.join(bad)}. "
                         f"Valid: {', '.join(EXPERIMENTS)}, all")
    return keys

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="run_experiments", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?", default=None,
                    help="e1 | e3 | gc | comma list | all")
    ap.add_argument("--attr", "-a", default=None, help="GENDER | RACE (env: ATTR)")
    ap.add_argument("--generator", "-g", default=None,
                    help="CHATGPT4 | GEMINIPRO | TEMPLATE | all (env: GENERATOR; default CHATGPT4)")
    ap.add_argument("--list", action="store_true", help="list experiments and exit")
    ap.add_argument("--plots-only", action="store_true",
                    help="R experiments (e1, gc): redraw figures from the saved CSVs "
                         "without recomputing any decomposition (fast)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved command(s) without running them")
    args, passthrough = ap.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if args.list or not args.experiment:
        _print_list()
        return 0

    keys = _select_keys(args.experiment)

    need_attr = any("attr" in EXPERIMENTS[k].forward for k in keys)
    need_gen = any("generator" in EXPERIMENTS[k].forward for k in keys)
    attr = paths.resolve_attr(args.attr) if need_attr else None
    generators = _resolve_generators(args.generator) if need_gen else [None]

    rc_total = 0
    for key in keys:
        exp = EXPERIMENTS[key]
        if not exp.available:
            print(f"[skip] {key}: {exp.script} does not exist yet.")
            rc_total = rc_total or 1
            continue
        gens = generators if "generator" in exp.forward else [None]
        for generator in gens:
            cmd = _build_command(exp, attr, generator, passthrough, plots_only=args.plots_only)
            scope = []
            if "attr" in exp.forward:
                scope.append(attr)
            if "generator" in exp.forward:
                scope.append(generator)
            print("=" * 70)
            print(f"{key.upper()} | {exp.title}" + (f"  [{' / '.join(scope)}]" if scope else ""))
            print("  " + " ".join(cmd))
            print("=" * 70)
            if args.dry_run:
                continue
            rc = subprocess.run(cmd, cwd=str(paths.repo_root())).returncode
            if rc != 0:
                print(f"[error] {key}"
                      + (f" / {generator}" if generator else "")
                      + f" exited with code {rc}")
                rc_total = rc_total or rc
    return rc_total

if __name__ == "__main__":
    raise SystemExit(main())
