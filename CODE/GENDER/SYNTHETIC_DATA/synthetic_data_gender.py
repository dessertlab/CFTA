from __future__ import annotations

import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from common import paths

EDUCATION_LEVELS = ['No High School Degree', 'High School Degree', 'Degree', 'Master', 'PhD']

def get_context(u: float) -> str:
    if u < -0.5:
        return 'unstable life'
    elif -0.5 <= u < 0:
        return 'precarious life'
    elif 0 <= u < 0.5:
        return 'moderate stability'
    return 'stable and supportive life'

def generate(n, seed, lambda_gender, mu_age, alpha1, alpha2):
    np.random.seed(seed)

    U = np.random.uniform(low=-1, high=1, size=n)

    gender_prob = np.clip(0.5 + lambda_gender * U, 0, 1)
    gender = np.array(['Male' if np.random.rand() < p else 'Female' for p in gender_prob])

    age = np.random.normal(loc=50 + mu_age * U, scale=7, size=n).astype(int)
    age = np.clip(age, 25, 75)

    education_raw = np.random.poisson(lam=0.5 + alpha1 * (gender == 'Male'), size=n)
    education_raw = np.clip(education_raw, 0, len(EDUCATION_LEVELS) - 1)
    education = [EDUCATION_LEVELS[i] for i in education_raw]

    employment_prob = 0.5 + alpha2 * (gender == 'Male')
    employment = np.array(['Employed' if np.random.rand() < p else 'Unemployed' for p in employment_prob])

    contexts = [get_context(u) for u in U]

    return pd.DataFrame({
        'Gender': gender,
        'Age': age,
        'Education': education,
        'Employment': employment,
        'Context': contexts,
    })

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", "-s", default="s1", help="scenario label, e.g. s1 (default: s1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=800, help="number of samples (default: 800)")
    ap.add_argument("--lambda-gender", type=float, default=0.0, help="effect of U on gender")
    ap.add_argument("--mu-age", type=float, default=0.0, help="effect of U on age")
    ap.add_argument("--alpha1", type=float, default=0.0, help="effect of gender on education")
    ap.add_argument("--alpha2", type=float, default=0.0, help="effect of gender on employment")
    ap.add_argument("--out", default=None, help="explicit output CSV path (overrides scenario)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out) if args.out else paths.synthetic_csv("GENDER", args.scenario)
    if out.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing ground-truth file: {out}\n"
                         f"Pass --force if you really intend to regenerate it.")

    df = generate(args.n, args.seed, args.lambda_gender, args.mu_age, args.alpha1, args.alpha2)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[ok] GENDER/{args.scenario}: {len(df)} rows -> {out}")
    print(df.head(10))

if __name__ == "__main__":
    main()
