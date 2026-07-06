from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from common import paths, build_complete_dataset

TONE_VARIANTS = {
    "unstable life": [
        "amid constant upheaval and change",
        "where little stays the same from one day to the next",
        "in the middle of relentless turmoil",
    ],
    "precarious life": [
        "navigating a fragile, uncertain stretch",
        "balanced on the edge of an unresolved situation",
        "held in a tense, suspended state",
    ],
    "moderate stability": [
        "keeping to a balanced, ordinary rhythm",
        "settled into a measured, predictable routine",
        "moving through steady, unremarkable days",
    ],
    "stable and supportive life": [
        "settled into a steady, reliable routine",
        "rooted in security and dependable bonds",
        "surrounded by a sense of rootedness and support",
    ],
}
NEUTRAL_TONE = "going about an ordinary day"
N_VARIANTS = 3

def _descriptor(attr: str, row: pd.Series, scm: dict) -> str:
    x = row[scm["X"]]
    age = row[scm["Z"][0]]
    if attr == "GENDER":
        return f"A {age}-year-old {x} with {row['Education']}, currently {row['Employment']}"
    return f"A {age}-year-old {x} with {row['Prior Convictions']}"

def _tone(context: str, idx: int) -> str:
    key = str(context).strip().lower()
    variants = TONE_VARIANTS.get(key)
    if not variants:
        return NEUTRAL_TONE
    return variants[idx % N_VARIANTS]

def generate_scenario(attr: str, s: str, scm: dict) -> pd.DataFrame:
    syn = pd.read_csv(paths.synthetic_csv(attr, s))
    if "Context" not in syn.columns:
        raise SystemExit(f"{paths.synthetic_csv(attr, s)} has no 'Context' column")
    sentences = [
        f"{_descriptor(attr, row, scm)}, {_tone(row['Context'], i)}."
        for i, (_, row) in enumerate(syn.iterrows())
    ]
    out = syn.drop(columns=["Context"]).copy()
    out["Sentence"] = sentences
    return out

def build_attr(attr: str) -> None:
    scm = paths.scm(attr)
    out_dir = paths.generated_dir(attr, "TEMPLATE")
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    unknown = 0
    known_ctx = set(TONE_VARIANTS)
    for s in paths.scenarios(attr):
        df = generate_scenario(attr, s, scm)
        df.to_csv(paths.generated_csv(attr, "TEMPLATE", s), index=False)
        n_rows += len(df)
        ctx = pd.read_csv(paths.synthetic_csv(attr, s))["Context"].astype(str).str.strip().str.lower()
        unknown += int((~ctx.isin(known_ctx)).sum())
    print(f"  [ok] {attr}/TEMPLATE: {len(paths.scenarios(attr))} scenarios, {n_rows} sentences -> {out_dir}")
    if unknown:
        print(f"  [warn] {unknown} rows had an unrecognised Context (used neutral tone)")
    build_complete_dataset.build(attr, "TEMPLATE")

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attr", default="all", help="GENDER | RACE | all  (env: ATTR)")
    args = ap.parse_args(argv)

    attrs = list(paths.ATTRS) if str(args.attr).lower() == "all" else [paths.resolve_attr(args.attr)]

    print("=" * 70)
    print("E3 - Template-generator ablation (deterministic TEMPLATE generator)")
    print("=" * 70)
    for attr in attrs:
        build_attr(attr)
    print("\nNext: score TEMPLATE like any generator, e.g.")
    for attr in attrs:
        print(f"  python CODE/run_sa.py --attr {attr} --generator TEMPLATE --model <model>")
    print("then E1 with --generator TEMPLATE will pick it up automatically.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
