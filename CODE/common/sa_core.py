from __future__ import annotations

import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import paths, sa_models

import numpy as np
import pandas as pd

def _atomic_write(df, out_path):
    tmp = str(out_path) + ".tmp"
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, out_path)
    except Exception:
        df.to_csv(out_path, index=False)
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

def _load_working(base, out_path, out_columns, resume, force):
    if out_path.exists() and resume and not force:
        try:
            work = pd.read_csv(out_path, low_memory=False)
        except Exception:
            work = None
        if work is not None and len(work) == len(base):
            for c in out_columns:
                if c not in work.columns:
                    work[c] = np.nan
            return work, True
    work = base.copy()
    for c in out_columns:
        work[c] = np.nan
    return work, False

def run_one(model_key, attr, generator, scenario, resume=True, force=False,
            limit=None, flush_rows=25, verbose=True):
    scorer = sa_models.get_scorer(model_key)
    if getattr(scorer, "_init_failed", False):
        return {"scenario": scenario, "status": "init_failed", "scored": 0, "todo": 0, "nan": 0}
    out_dir = paths.sa_result_dir(attr, generator, model_key)
    out_path = out_dir / scorer.filename(scenario)
    in_path = paths.generated_csv(attr, generator, scenario)

    if not in_path.exists():
        if verbose:
            print(f"    [skip] {scenario}: no input {in_path}")
        return {"scenario": scenario, "status": "missing_input", "scored": 0, "todo": 0, "nan": 0}

    base = pd.read_csv(in_path, low_memory=False)
    if "Sentence" not in base.columns or base.empty:
        if verbose:
            print(f"    [skip] {scenario}: no 'Sentence' column / empty")
        return {"scenario": scenario, "status": "no_sentence", "scored": 0, "todo": 0, "nan": 0}

    work, did_resume = _load_working(base, out_path, scorer.out_columns, resume, force)

    na_mask = work[scorer.out_columns].isna().any(axis=1)
    todo = [i for i in work.index[na_mask] if (limit is None or i < limit)]

    if not todo:
        if verbose:
            print(f"    [done] {scenario}: already complete ({len(work)} rows){' (resumed)' if did_resume else ''}")
        return {"scenario": scenario, "status": "complete", "scored": 0,
                "todo": 0, "nan": int(na_mask.sum())}

    if not scorer._ready:
        try:
            scorer.prepare()
        except Exception as e:
            scorer._init_failed = True
            print(f"    [error] cannot initialize '{model_key}': {e}")
            return {"scenario": scenario, "status": "init_failed", "scored": 0,
                    "todo": len(todo), "nan": int(na_mask.sum())}

    if verbose:
        tag = f" (resume: {len(todo)} of {len(work)} remaining)" if did_resume else ""
        print(f"    [run ] {scenario}: scoring {len(todo)} rows{tag}")

    out_dir.mkdir(parents=True, exist_ok=True)
    texts = work["Sentence"].astype(str)
    scored = 0
    since_flush = 0
    bs = max(1, scorer.batch_size)
    for bstart in range(0, len(todo), bs):
        idxs = todo[bstart:bstart + bs]
        batch_texts = [texts.loc[i] for i in idxs]
        try:
            results = scorer.score(batch_texts)
        except Exception as e:
            if verbose:
                print(f"        [warn] batch failed ({e}); left as NaN")
            results = [None] * len(idxs)
        for k, i in enumerate(idxs):
            r = results[k] if k < len(results) else None
            if r is not None:
                for col, val in zip(scorer.out_columns, r):
                    work.at[i, col] = val
                scored += 1
        since_flush += len(idxs)
        if since_flush >= flush_rows:
            _atomic_write(work, out_path)
            since_flush = 0
        if scorer.delay:
            import time
            time.sleep(scorer.delay)

    _atomic_write(work, out_path)
    nan_left = int(work[scorer.out_columns].isna().any(axis=1).sum())
    if verbose:
        print(f"    [ok  ] {scenario}: {len(work) - nan_left}/{len(work)} scored"
              + (f", {nan_left} still NaN (rerun --resume)" if nan_left else "")
              + f"  -> {out_path.name}")
    return {"scenario": scenario, "status": "ok", "scored": scored,
            "todo": len(todo), "nan": nan_left}

def run_model(model_key, attr, generator, scenarios=None, resume=True, force=False,
              limit=None, flush_rows=25, verbose=True):
    scen = scenarios or paths.scenarios(attr)
    if verbose:
        print(f"  Model: {model_key}  |  {attr}/{generator}  |  {len(scen)} scenario(s)")
    results = []
    for s in scen:
        results.append(run_one(model_key, attr, generator, s, resume=resume,
                                force=force, limit=limit, flush_rows=flush_rows, verbose=verbose))
    return results
