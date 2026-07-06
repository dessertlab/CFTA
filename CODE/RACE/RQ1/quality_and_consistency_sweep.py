import os
import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from tqdm import tqdm
from readability import Readability
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

from common import paths

DEFAULT_ATTR = "RACE"

RAREFACTION_STEP = 200
RAREFACTION_MAX = 5000
RAREFACTION_REPEATS = 5

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_float(x, ndigits=2, default=0.0):
    try:
        return round(float(x), ndigits)
    except Exception:
        return default

def word_length_per_sentence(df: pd.DataFrame, column: str = "Sentence") -> pd.Series:
    return df[column].astype(str).apply(lambda s: len(s.split()))

def readability_scores_on_text(text: str) -> dict:
    r = Readability(text)
    out = {}
    try:
        fk = r.flesch_kincaid()
        out["flesch_kincaid_score"] = fk.score
        out["flesch_kincaid_grade"] = fk.grade_level
    except Exception:
        out["flesch_kincaid_score"] = None
        out["flesch_kincaid_grade"] = None
    try:
        ari = r.ari()
        out["ari_score"] = ari.score
    except Exception:
        out["ari_score"] = None
    try:
        gf = r.gunning_fog()
        out["gunning_fog_score"] = gf.score
        out["gunning_fog_grade"] = gf.grade_level
    except Exception:
        out["gunning_fog_score"] = None
        out["gunning_fog_grade"] = None
    return out

def readability_in_samples(df: pd.DataFrame, sample_size: int = 200, repeats: int = 5, column: str = "Sentence") -> dict:
    texts = df[column].astype(str)
    n = len(texts)
    sample_size = min(sample_size, max(1, n))
    agg = {}
    for _ in range(repeats):
        sample = texts.sample(sample_size, replace=True).tolist()
        scores = readability_scores_on_text(" ".join(sample))
        for k, v in scores.items():
            if v is None:
                continue
            agg.setdefault(k, []).append(safe_float(v, ndigits=2, default=np.nan))
    out = {}
    for k, vals in agg.items():
        arr = np.array([x for x in vals if x is not None and not pd.isna(x)], dtype=float)
        out[k] = {"mean": None, "std": None} if arr.size == 0 else {
            "mean": float(np.round(arr.mean(), 2)),
            "std": float(np.round(arr.std(ddof=0), 2))
        }
    return out

def unique_token_counts_in_samples(df: pd.DataFrame, sample_size: int = 200, repeats: int = 5, column: str = "Sentence") -> tuple[float, float]:
    texts = df[column].astype(str)
    n = len(texts)
    sample_size = min(sample_size, max(1, n))
    vec = CountVectorizer()
    counts = []
    for _ in range(repeats):
        sample = texts.sample(sample_size, replace=True).tolist()
        X = vec.fit_transform(sample)
        vocab = vec.get_feature_names_out()
        counts.append(len(vocab))
    return float(np.round(np.mean(counts), 2)), float(np.round(np.std(counts, ddof=0), 2))

def compute_quality_metrics(df: pd.DataFrame, label: str) -> pd.DataFrame:
    wl = word_length_per_sentence(df)
    wl_mean = float(np.round(wl.mean(), 2))
    wl_std  = float(np.round(wl.std(ddof=0), 2))
    ut_mean, ut_std = unique_token_counts_in_samples(df)
    read_scores = readability_in_samples(df)
    row = {
        "Label": label,
        "WordLength_Mean": wl_mean,
        "WordLength_SD": wl_std,
        "UniqueTokens_Mean": ut_mean,
        "UniqueTokens_SD": ut_std,
    }
    for k, stats in read_scores.items():
        row[f"{k}_Mean"] = stats["mean"]
        row[f"{k}_SD"] = stats["std"]
    return pd.DataFrame([row])

def check_consistency_race(original_df: pd.DataFrame, generated_df: pd.DataFrame) -> pd.DataFrame:
    errors = []
    for idx, (orig, gen) in enumerate(zip(original_df.itertuples(index=False), generated_df.itertuples(index=False))):
        sentence = str(getattr(gen, "Sentence", ""))
        missing = []
        if str(getattr(orig, "Race", "")).strip() not in sentence:
            missing.append("Race")
        if str(getattr(orig, "Age", "")).strip() not in sentence:
            missing.append("Age")
        if str(getattr(orig, "Prior Convictions", "")).strip() not in sentence:
            missing.append("Prior Convictions")

        ctx = str(getattr(orig, "Context", "")).strip()
        context_mentioned = ctx.lower() in sentence.lower() if ctx else False

        if missing or context_mentioned:
            errors.append({
                "RowIndex": idx,
                "MissingFields": ", ".join(missing),
                "ContextMentioned": bool(context_mentioned),
                "Sentence": sentence,
            })
    return pd.DataFrame(errors)

def rarefaction_curve_data(df: pd.DataFrame, column: str = "Sentence",
                           step: int = RAREFACTION_STEP, max_samples: int = RAREFACTION_MAX,
                           repeats: int = RAREFACTION_REPEATS) -> pd.DataFrame:
    texts = df[column].astype(str).tolist()
    n_total = len(texts)
    if n_total == 0:
        return pd.DataFrame(columns=["NumSentences", "UniqueTokens_Mean", "UniqueTokens_SD"])
    sizes = list(range(step, min(max_samples, n_total) + 1, step))
    vec = CountVectorizer()
    rows = []
    for size in sizes:
        counts = []
        for _ in range(repeats):
            sample = np.random.choice(texts, size=size, replace=False)
            X = vec.fit_transform(sample)
            vocab = vec.get_feature_names_out()
            counts.append(len(vocab))
        rows.append({
            "NumSentences": size,
            "UniqueTokens_Mean": float(np.round(np.mean(counts), 2)),
            "UniqueTokens_SD": float(np.round(np.std(counts, ddof=0), 2)),
        })
    return pd.DataFrame(rows)

def plot_rarefaction_comparison(df_raw: pd.DataFrame, df_dedup: pd.DataFrame, out_png: str, title: str):
    plt.figure(figsize=(9, 5))
    if not df_raw.empty:
        x = df_raw["NumSentences"].values
        y = df_raw["UniqueTokens_Mean"].values
        sd = df_raw["UniqueTokens_SD"].values
        plt.plot(x, y, label="Raw corpus (mean)")
        plt.fill_between(x, y - sd, y + sd, alpha=0.15)
    if not df_dedup.empty:
        x2 = df_dedup["NumSentences"].values
        y2 = df_dedup["UniqueTokens_Mean"].values
        sd2 = df_dedup["UniqueTokens_SD"].values
        plt.plot(x2, y2, label="Deduplicated corpus (mean)")
        plt.fill_between(x2, y2 - sd2, y2 + sd2, alpha=0.15)
    plt.xlabel("Number of sampled sentences")
    plt.ylabel("Unique tokens")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_metrics_summary_two(quality_raw: pd.DataFrame, quality_dedup: pd.DataFrame, out_png: str):
    keys = ["WordLength_Mean", "UniqueTokens_Mean",
            "flesch_kincaid_score_Mean", "ari_score_Mean", "gunning_fog_score_Mean"]
    labels = ["Word length", "Unique tokens", "Flesch-Kincaid", "ARI", "Gunning-Fog"]

    vals_raw  = [quality_raw.loc[0].get(k, np.nan) for k in keys]
    vals_ded  = [quality_dedup.loc[0].get(k, np.nan) for k in keys]

    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, vals_raw, width, label="Raw")
    plt.bar(x + width/2, vals_ded, width, label="Deduplicated")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Value")
    plt.title("Aggregate quality metrics (raw vs. deduplicated)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def main(argv=None):
    ap = argparse.ArgumentParser(description="RQ1 quality & consistency sweep")
    ap.add_argument("--attr", "-a", default=None, help="GENDER | RACE (env: ATTR)")
    ap.add_argument("--generator", "-g", default=None,
                    help="CHATGPT4 | GEMINIPRO | TEMPLATE (env: GENERATOR)")
    args = ap.parse_args(argv)

    attr = paths.resolve_attr(args.attr) if args.attr or os.environ.get("ATTR") else DEFAULT_ATTR
    generator = paths.resolve_generator(args.generator, default="CHATGPT4")

    SYNTHETIC_DIR = str(paths.synthetic_dir(attr))
    GENERATED_DIR = str(paths.generated_dir(attr, generator))
    RESULT_BASE = str(paths.result_dir(attr, generator, "RQ1"))
    AGG_DIR = os.path.join(RESULT_BASE, "aggregate")

    print(f"RQ1 quality/consistency | {attr} / {generator}")
    print(f"  generated: {GENERATED_DIR}")
    print(f"  synthetic: {SYNTHETIC_DIR}")
    print(f"  output   : {RESULT_BASE}")

    for suffix in tqdm(paths.scenarios(attr), desc="Scenarios"):
        gen_path   = os.path.join(GENERATED_DIR, f"generated_sentences_{suffix}.csv")
        synth_path = os.path.join(SYNTHETIC_DIR,  f"synthetic_data_{suffix}.csv")
        out_dir = os.path.join(RESULT_BASE, f"scenario_{suffix}")
        ensure_dir(out_dir)

        if not os.path.isfile(gen_path) or not os.path.isfile(synth_path):
            pd.DataFrame([{
                "Label": suffix,
                "generated_exists": os.path.isfile(gen_path),
                "synthetic_exists": os.path.isfile(synth_path)
            }]).to_csv(os.path.join(out_dir, f"missing_inputs_{suffix}.csv"), index=False)
            continue

        try:
            generated_df = pd.read_csv(gen_path)
            original_df  = pd.read_csv(synth_path)
        except Exception as e:
            pd.DataFrame([{"Label": suffix, "load_error": str(e)}]).to_csv(
                os.path.join(out_dir, f"load_error_{suffix}.csv"), index=False
            )
            continue

        try:
            quality_df = compute_quality_metrics(generated_df, label=suffix)
            quality_df.to_csv(os.path.join(out_dir, f"quality_metrics_{suffix}.csv"), index=False)
        except Exception as e:
            pd.DataFrame([{"Label": suffix, "quality_error": str(e)}]).to_csv(
                os.path.join(out_dir, f"quality_error_{suffix}.csv"), index=False
            )

        try:
            errors_df = check_consistency_race(original_df, generated_df)
            if errors_df.empty:
                pd.DataFrame([{"Label": suffix, "Consistency": "No errors detected"}]).to_csv(
                    os.path.join(out_dir, f"consistency_report_{suffix}.csv"), index=False
                )
            else:
                errors_df.to_csv(os.path.join(out_dir, f"consistency_errors_{suffix}.csv"), index=False)
        except Exception as e:
            pd.DataFrame([{"Label": suffix, "consistency_error": str(e)}]).to_csv(
                os.path.join(out_dir, f"consistency_error_{suffix}.csv"), index=False
            )

    ensure_dir(AGG_DIR)
    gen_files = [os.path.join(GENERATED_DIR, f) for f in os.listdir(GENERATED_DIR)
                 if f.startswith("generated_sentences_s") and f.endswith(".csv")]
    gen_files.sort()

    if not gen_files:
        pd.DataFrame([{"note": "No generated files found for aggregate step."}]).to_csv(
            os.path.join(AGG_DIR, "aggregate_missing_inputs.csv"), index=False
        )
        return

    df_list = []
    for fpath in gen_files:
        try:
            df_list.append(pd.read_csv(fpath))
        except Exception:
            pass
    if not df_list:
        pd.DataFrame([{"note": "All generated files failed to load."}]).to_csv(
            os.path.join(AGG_DIR, "aggregate_load_error.csv"), index=False
        )
        return

    corpus_raw = pd.concat(df_list, ignore_index=True)
    corpus_raw["Sentence"] = corpus_raw["Sentence"].astype(str)
    corpus_dedup = corpus_raw.drop_duplicates(subset="Sentence").reset_index(drop=True)

    corpus_raw.to_csv(os.path.join(AGG_DIR, "corpus_raw.csv"), index=False)
    corpus_dedup.to_csv(os.path.join(AGG_DIR, "corpus_deduplicated.csv"), index=False)

    n_total = len(corpus_raw)
    n_unique = len(corpus_dedup)
    dup_pct = round(100 * (1 - n_unique / max(1, n_total)), 2)
    with open(os.path.join(AGG_DIR, "duplicate_rate.txt"), "w", encoding="utf-8") as f:
        f.write(f"Total sentences: {n_total}\n")
        f.write(f"Unique sentences: {n_unique}\n")
        f.write(f"Duplicate rate: {dup_pct}%\n")

    q_raw   = compute_quality_metrics(corpus_raw,   label="corpus_raw")
    q_dedup = compute_quality_metrics(corpus_dedup, label="corpus_deduplicated")
    q_raw.to_csv(os.path.join(AGG_DIR, "metrics_corpus_raw.csv"), index=False)
    q_dedup.to_csv(os.path.join(AGG_DIR, "metrics_corpus_deduplicated.csv"), index=False)

    plot_metrics_summary_two(q_raw, q_dedup, os.path.join(AGG_DIR, "metrics_summary.png"))

    curve_raw   = rarefaction_curve_data(corpus_raw)
    curve_dedup = rarefaction_curve_data(corpus_dedup)

    merged = pd.DataFrame({
        "NumSentences": curve_raw["NumSentences"],
        "UniqueTokens_Raw_Mean": curve_raw["UniqueTokens_Mean"],
        "UniqueTokens_Raw_SD": curve_raw["UniqueTokens_SD"],
        "UniqueTokens_Deduplicated_Mean": curve_dedup["UniqueTokens_Mean"] if not curve_dedup.empty else np.nan,
        "UniqueTokens_Deduplicated_SD": curve_dedup["UniqueTokens_SD"] if not curve_dedup.empty else np.nan,
    })
    merged.to_csv(os.path.join(AGG_DIR, "rarefaction_data.csv"), index=False)

    plot_rarefaction_comparison(curve_raw, curve_dedup,
                                os.path.join(AGG_DIR, "rarefaction_curve.png"),
                                title="Rarefaction curve (raw vs. deduplicated)")

if __name__ == "__main__":
    main()
