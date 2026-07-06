from __future__ import annotations

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import paths, sa_models, sa_core

def parse_scenarios(spec, attr):
    valid = paths.scenarios(attr)
    if spec is None or str(spec).strip().lower() in ("all", ""):
        return valid
    valid_set = set(valid)
    out = []

    def add(tok):
        s = tok if tok.startswith("s") else f"s{tok}"
        if s not in valid_set:
            raise SystemExit(f"Scenario '{tok}' out of range for {attr} (valid: s1..{valid[-1]})")
        if s not in out:
            out.append(s)

    for tok in str(spec).split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            a = int(a[1:] if a.startswith("s") else a)
            b = int(b[1:] if b.startswith("s") else b)
            for n in range(a, b + 1):
                add(str(n))
        else:
            add(tok)
    return out

def _print_list():
    print("Available sentiment/toxicity models (--model <key|alias>):\n")
    print("  HuggingFace (local, needs torch/transformers):")
    for k in sa_models.HF_KEYS:
        print(f"    - {k}")
    print("\n  API (hosted chat models, needs OPENAI_API_KEY / GROQ_API_KEY):")
    for k in sa_models.API_KEYS:
        print(f"    - {k}")
    print("\n  Groups: all | hf | api")
    print("  Aliases:")
    inv = {}
    for a, k in sa_models.ALIASES.items():
        inv.setdefault(k, []).append(a)
    for k, al in inv.items():
        print(f"    {k}: {', '.join(al)}")

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="run_sa", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attr", "-a", default=None, help="GENDER | RACE (env: ATTR)")
    ap.add_argument("--generator", "-g", default=None,
                    help="CHATGPT4 | GEMINIPRO | TEMPLATE (env: GENERATOR)")
    ap.add_argument("--model", "-m", default=None,
                    help="model key/alias, comma list, or all|hf|api")
    ap.add_argument("--scenarios", "-s", default="all",
                    help="all | s1,s3 | 1-5 | s1-s5 (default: all)")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="reuse checkpoints and only score missing rows (default: on)")
    ap.add_argument("--force", action="store_true", help="recompute from scratch, overwrite")
    ap.add_argument("--limit", type=int, default=None, help="max rows per scenario (quick test)")
    ap.add_argument("--flush-rows", type=int, default=25, help="checkpoint cadence in rows")
    ap.add_argument("--list", action="store_true", help="list models and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        _print_list()
        return 0

    attr = paths.resolve_attr(args.attr)
    generator = paths.resolve_generator(args.generator)
    model_keys = sa_models.resolve_models(args.model)
    scen = parse_scenarios(args.scenarios, attr)
    verbose = not args.quiet

    print("=" * 70)
    print(f"Sentiment analysis  |  {attr} / {generator}")
    print(f"Models   : {', '.join(model_keys)}")
    print(f"Scenarios: {', '.join(scen)}")
    print(f"Resume   : {args.resume}   Force: {args.force}"
          + (f"   Limit: {args.limit}" if args.limit else ""))
    print("=" * 70)

    grand = {"ok": 0, "complete": 0, "scored": 0, "nan": 0, "init_failed": 0}
    for mk in model_keys:
        res = sa_core.run_model(mk, attr, generator, scenarios=scen,
                                resume=args.resume, force=args.force,
                                limit=args.limit, flush_rows=args.flush_rows,
                                verbose=verbose)
        for r in res:
            if r["status"] == "init_failed":
                grand["init_failed"] += 1
                continue
            grand["scored"] += r["scored"]
            grand["nan"] += r["nan"]
            if r["status"] == "ok":
                grand["ok"] += 1
            elif r["status"] == "complete":
                grand["complete"] += 1

    print("-" * 70)
    print(f"Done. scenario-runs ok={grand['ok']} already-complete={grand['complete']} "
          f"| rows scored={grand['scored']} | rows still NaN={grand['nan']}")
    if grand["init_failed"]:
        print(f"{grand['init_failed']} scenario-run(s) skipped: model init failed "
              "(missing API key, or torch/transformers not installed). See errors above.")
    if grand["ok"] and grand["nan"]:
        print("Some scored rows are still NaN (API failures/rate limits). Re-run the "
              "same command with --resume to fill them in.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
