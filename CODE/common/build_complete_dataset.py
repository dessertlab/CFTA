from __future__ import annotations

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import paths

import pandas as pd

def build(attr: str, generator: str) -> pathlib.Path:
    frames = []
    for s in paths.scenarios(attr):
        f = paths.generated_csv(attr, generator, s)
        if not f.exists():
            raise SystemExit(f"Missing scenario file: {f}")
        df = pd.read_csv(f)
        df["Scenario"] = s
        if len(df) != 800:
            print(f"  [warn] {attr}/{generator}/{s}: {len(df)} rows (expected 800)")
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    out = paths.complete_csv(attr, generator)
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out, index=False)
    print(f"  [ok] {attr}/{generator}: {len(full)} rows, {len(frames)} scenarios -> {out}")
    return out

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attr", default="all", help="GENDER | RACE | all  (env: ATTR)")
    ap.add_argument("--generator", default=None, help="CHATGPT4 | GEMINIPRO | TEMPLATE  (env: GENERATOR)")
    args = ap.parse_args(argv)

    generator = paths.resolve_generator(args.generator)
    attrs = list(paths.ATTRS) if str(args.attr).lower() == "all" else [paths.resolve_attr(args.attr)]

    for attr in attrs:
        build(attr, generator)

if __name__ == "__main__":
    main()
