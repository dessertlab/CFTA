from __future__ import annotations

import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from common import paths

PRIOR_CONVICTIONS_LEVELS = ['No prior convictions', 'Some priors', 'Many priors']

def get_context(u: float) -> str:
    if u < -0.5:
        return 'unstable life'
    elif -0.5 <= u < 0:
        return 'precarious life'
    elif 0 <= u < 0.5:
        return 'moderate stability'
    return 'stable and supportive life'

def generate(n, seed, lambda_race, mu_age, alpha, beta):
    np.random.seed(seed)

    U = np.random.uniform(low=-1, high=1, size=n)

    race_prob = np.clip(0.5 + lambda_race * U, 0, 1)
    race = np.array(['White-Caucasian' if np.random.rand() < p else 'Non-White' for p in race_prob])

    age = np.random.normal(loc=50 + mu_age * U, scale=7, size=n).astype(int)
    age = np.clip(age, 25, 75)

    race_binary = (race == 'White-Caucasian').astype(int)
    lambda_convictions = 0.5 + alpha * race_binary + beta * age
    convictions_raw = np.random.poisson(lam=lambda_convictions)
    convictions_raw = np.clip(convictions_raw, 0, len(PRIOR_CONVICTIONS_LEVELS) - 1)
    prior_convictions = [PRIOR_CONVICTIONS_LEVELS[i] for i in convictions_raw]

    contexts = [get_context(u) for u in U]

    return pd.DataFrame({
        'Race': race,
        'Age': age,
        'Prior Convictions': prior_convictions,
        'Context': contexts,
    })

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", "-s", default="s1", help="scenario label, e.g. s1 (default: s1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=800, help="number of samples (default: 800)")
    ap.add_argument("--lambda-race", type=float, default=0.0, help="effect of U on race")
    ap.add_argument("--mu-age", type=float, default=0.0, help="effect of U on age")
    ap.add_argument("--alpha", type=float, default=0.0, help="effect of race on prior convictions")
    ap.add_argument("--beta", type=float, default=0.0, help="effect of age on prior convictions")
    ap.add_argument("--out", default=None, help="explicit output CSV path (overrides scenario)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out) if args.out else paths.synthetic_csv("RACE", args.scenario)
    if out.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing ground-truth file: {out}\n"
                         f"Pass --force if you really intend to regenerate it.")

    df = generate(args.n, args.seed, args.lambda_race, args.mu_age, args.alpha, args.beta)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[ok] RACE/{args.scenario}: {len(df)} rows -> {out}")
    print(df.head(10))

if __name__ == "__main__":
    main()
