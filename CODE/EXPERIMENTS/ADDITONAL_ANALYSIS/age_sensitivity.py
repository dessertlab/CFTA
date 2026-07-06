#!/usr/bin/env python3
"""
Age-sensitivity analysis: why is the spurious component never
significant?

For a spurious effect (x-SE) to reach significance, two factors must
combine: (1) the injected X-Z correlation in the test data, and (2) the
sensitivity of the scorer's output Y to Z (age). This script measures
both and reports the implied spurious-effect magnitude:

  1. OLS of the sentiment score on Age, controlling for the remaining
     attributes and scenario fixed effects. The MAIN estimate is
     computed on the "clean" scenarios, where the spurious-group
     coefficients are inactive (c_{U,X} = c_{U,Z} = 0), so that Age is
     independent of the latent context by construction and the estimate
     is not confounded by the context tone. Reported: raw coefficient
     per year with 95% CI and p-value, standardized effect (SD of Y per
     SD of Age), and the ratio to the protected-attribute effect.
  2. Partial Spearman correlation (both Y and Age residualized on the
     categorical covariates and scenario), as a non-parametric check.
  3. Non-linearity check: F-test for age-bin dummies (<35, 35-55, >55).
  4. Implied |SE| = |b(Age)| x max |E[Age|X=x1] - E[Age|X=x0]| across
     the spurious scenarios (the injected age separation).

Usage:
  python age_sensitivity.py gender scores_scorerA.csv [scores_scorerB.csv ...]
  python age_sensitivity.py race   scores_scorerA.csv [...]

Expected columns:
  gender: Scenario, Gender, Age, Education, Employment, sentiment_score
  race:   Scenario, Race, Age, Prior Convictions, sentiment_score
(Spaces in column names are handled; extra columns are ignored.)

Requirements: pandas, numpy, scipy, statsmodels.
"""

import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.stats.anova import anova_lm

CONFIG = {
    "gender": {
        "protected": "Gender",
        "covariates": "C(Gender) + C(Education) + C(Employment)",
        # Scenarios with the spurious group fully inactive
        # (c_{U,X} = c_{U,Z} = 0): baseline + indirect-only group.
        "clean_scenarios": ["S1", "S8", "S9", "S10", "S11", "S12", "S13"],
        # Scenarios of the spurious group (used for the injected age gap).
        "spurious_scenarios": ["S2", "S3", "S4", "S5", "S6", "S7"],
    },
    "race": {
        "protected": "Race",
        "covariates": "C(Race) + C(Prior_Convictions)",
        "clean_scenarios": ["S1", "S8", "S9", "S10", "S11"],
        "spurious_scenarios": ["S2", "S3", "S4", "S5", "S6", "S7"],
    },
}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["sentiment_score"])
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def age_effect(sub: pd.DataFrame, cfg: dict) -> dict:
    """OLS + partial Spearman + bin F-test on a scenario subset."""
    covs = cfg["covariates"]
    m = smf.ols(f"sentiment_score ~ Age + {covs} + C(Scenario)",
                data=sub).fit()
    b = m.params["Age"]
    lo, hi = m.conf_int().loc["Age"]
    y_sd = sub["sentiment_score"].std()
    beta_std = b * sub["Age"].std() / y_sd
    prot_term = next(k for k in m.params.index
                     if k.startswith(f"C({cfg['protected']})"))
    prot_std = abs(m.params[prot_term]) / y_sd

    r_y = smf.ols(f"sentiment_score ~ {covs} + C(Scenario)",
                  data=sub).fit().resid
    r_a = smf.ols(f"Age ~ {covs} + C(Scenario)", data=sub).fit().resid
    rho, p_rho = spearmanr(r_a, r_y)

    binned = sub.copy()
    binned["age_bin"] = pd.cut(binned["Age"], [0, 35, 55, 200],
                               labels=["young", "mid", "old"])
    m0 = smf.ols(f"sentiment_score ~ {covs} + C(Scenario)",
                 data=binned).fit()
    m1 = smf.ols(f"sentiment_score ~ C(age_bin) + {covs} + C(Scenario)",
                 data=binned).fit()
    p_bins = anova_lm(m0, m1)["Pr(>F)"][1]

    return dict(n=len(sub), b=b, lo=lo, hi=hi, p=m.pvalues["Age"],
                beta_std=beta_std, prot_std=prot_std,
                rho=rho, p_rho=p_rho, p_bins=p_bins)


def injected_age_gap(df: pd.DataFrame, cfg: dict) -> float:
    """Max |E[Age|x1] - E[Age|x0]| across the spurious scenarios."""
    gaps = []
    for s in cfg["spurious_scenarios"]:
        g = df[df["Scenario"] == s].groupby(cfg["protected"])["Age"].mean()
        if len(g) == 2:
            gaps.append(abs(g.iloc[0] - g.iloc[1]))
    return max(gaps) if gaps else float("nan")


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in CONFIG:
        sys.exit(__doc__)
    cfg = CONFIG[sys.argv[1]]

    header = (f"{'file':44s} {'set':6s} {'n':>6s} {'b(Age)/yr':>10s} "
              f"{'95% CI':>21s} {'p':>7s} {'beta_std':>9s} "
              f"{'prot_std':>9s} {'ratio':>6s} {'p(rho)':>7s} "
              f"{'p(bins)':>8s} {'implied|SE|':>11s}")
    print(header)
    for path in sys.argv[2:]:
        df = load(path)
        gap = injected_age_gap(df, cfg)
        name = path.split("/")[-1][:43]
        for label, sub in (("all", df),
                           ("clean",
                            df[df["Scenario"].isin(cfg["clean_scenarios"])])):
            r = age_effect(sub, cfg)
            ratio = r["prot_std"] / max(abs(r["beta_std"]), 1e-9)
            implied = abs(r["b"]) * gap if label == "clean" else float("nan")
            print(f"{name:44s} {label:6s} {r['n']:6d} {r['b']:10.5f} "
                  f"[{r['lo']:8.5f},{r['hi']:8.5f}] {r['p']:7.3f} "
                  f"{r['beta_std']:9.4f} {r['prot_std']:9.4f} "
                  f"{ratio:6.1f} {r['p_rho']:7.3f} {r['p_bins']:8.3f} "
                  f"{implied:11.4f}")
        print(f"  -> injected age gap (max over spurious scenarios): "
              f"{gap:.2f} years")


if __name__ == "__main__":
    main()
