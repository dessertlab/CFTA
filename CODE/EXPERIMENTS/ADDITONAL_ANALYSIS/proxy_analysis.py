#!/usr/bin/env python3
"""
Proxy analysis: is the protected attribute recoverable from the
non-attribute content of the generated sentences?

For each corpus (Gender / Race), all attribute-bearing terms are removed
from the sentences, and a classifier (TF-IDF unigrams+bigrams, logistic
regression, 5-fold stratified CV) is trained to predict the protected
attribute from the residual content. Results are reported separately for:
  - scenarios where X is independent of the latent context U by
    construction (c_{U,X} = 0): accuracy at chance level is expected if
    the generator introduces no attribute proxy;
  - scenarios where an X-U correlation is injected by design
    (c_{U,X} != 0): above-chance accuracy is expected, driven by the
    context-tone vocabulary (the designed correlation, not a confound).

Usage:
  python proxy_analysis.py gender path/to/generated_sentences_gender_complete.csv
  python proxy_analysis.py race   path/to/generated_sentences_race_complete.csv

Requirements: pandas, scikit-learn, numpy.
"""

import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------
# Configuration per protected attribute
# ----------------------------------------------------------------------
AGE_TERMS = r"\b\d+(-year-old)?\b|\baged?\b"

CONFIG = {
    "gender": {
        "label_col": "Gender",
        "positive_class": "Male",
        # Scenarios with c_{U,X} = 0 (X independent of U by construction)
        "indep_scenarios": ["s1", "s5", "s7", "s8", "s9", "s10", "s11", "s12", "s13"],
        # Scenarios with c_{U,X} != 0 (X-U correlation injected by design)
        "dep_scenarios": ["s2", "s3", "s4", "s6", "s14"],
        "strip_patterns": [
            # education (strip BEFORE gender: multi-word phrases first)
            r"\b(no\s+high\s+school\s+degree|high\s+school\s+degree|"
            r"bachelor'?s?\s+degree|master'?s?\s+degree|phd|no\s+degree|"
            r"bachelor'?s?|master'?s?|degree|diploma)\b",
            # gender markers
            r"\b(male|female|man|woman|men|women|he|she|his|her|hers|him|"
            r"himself|herself|mr|mrs|ms|gentleman|lady|boy|girl)\b",
            # employment
            r"\b(employed|unemployed|employment|job|jobless)\b",
            AGE_TERMS,
        ],
    },
    "race": {
        "label_col": "Race",
        "positive_class": "White-Caucasian",
        "indep_scenarios": ["s1", "s5", "s7", "s8", "s9", "s10", "s11"],
        "dep_scenarios": ["s2", "s3", "s4", "s6", "s12"],
        "strip_patterns": [
            # prior convictions (multi-word phrases first)
            r"\b(no\s+prior\s+convictions?|some\s+priors?|many\s+priors?|"
            r"priors?|convictions?|convicted|criminal|record)\b",
            # race markers
            r"\b(non[\s-]?white|white[\s-]?caucasian|caucasian|white|black|"
            r"race|racial|ethnic|ethnicity)\b",
            AGE_TERMS,
        ],
    },
}


def strip_attributes(sentence: str, patterns) -> str:
    """Remove all attribute-bearing terms and non-alphabetic characters."""
    s = sentence.lower()
    for pat in patterns:
        s = re.sub(pat, " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                      sublinear_tf=True)),
            ("lr", LogisticRegression(max_iter=2000)),
        ]
    )


def evaluate(sub: pd.DataFrame, y: np.ndarray, label: str) -> None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(make_pipeline(), sub["clean"], y, cv=cv,
                          scoring="accuracy")
    majority = max(y.mean(), 1 - y.mean())
    print(f"{label:32s} n={len(sub):5d}  majority={majority:.3f}  "
          f"acc={acc.mean():.3f} +/- {acc.std():.3f}")


def top_features(sub: pd.DataFrame, y: np.ndarray, k: int = 12) -> None:
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    X = tf.fit_transform(sub["clean"])
    lr = LogisticRegression(max_iter=2000).fit(X, y)
    feat = np.array(tf.get_feature_names_out())
    coef = lr.coef_[0]
    print(f"\nTop {k} features predicting the positive class:",
          feat[np.argsort(coef)[-k:]][::-1].tolist())
    print(f"Top {k} features predicting the negative class:",
          feat[np.argsort(coef)[:k]].tolist())


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in CONFIG:
        sys.exit(__doc__)
    cfg = CONFIG[sys.argv[1]]
    df = pd.read_csv(sys.argv[2])

    df["clean"] = df["Sentence"].astype(str).map(
        lambda s: strip_attributes(s, cfg["strip_patterns"])
    )
    y_all = (df[cfg["label_col"]] == cfg["positive_class"]).astype(int)

    # Sanity check: no residual leakage of the protected-attribute terms
    # (patterns[1] holds the attribute markers in both configs).
    leakage = df["clean"].str.contains(cfg["strip_patterns"][1],
                                       regex=True).sum()
    print(f"Residual attribute-term leakage: {leakage} (must be 0)")
    print(f"Class balance: {df[cfg['label_col']].value_counts().to_dict()}\n")

    indep = df[df["Scenario"].isin(cfg["indep_scenarios"])]
    dep = df[df["Scenario"].isin(cfg["dep_scenarios"])]

    evaluate(indep, y_all.loc[indep.index].values,
             "X independent of U (pooled)")
    evaluate(dep, y_all.loc[dep.index].values,
             "X correlated with U (pooled)")
    print()
    for s in cfg["indep_scenarios"]:
        sub = df[df["Scenario"] == s]
        evaluate(sub, y_all.loc[sub.index].values, f"  scenario {s}")

    top_features(dep, y_all.loc[dep.index].values)


if __name__ == "__main__":
    main()
