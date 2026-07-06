from __future__ import annotations

import os
from pathlib import Path

ATTRS = ("GENDER", "RACE")
GENERATORS = ("CHATGPT4", "GEMINIPRO", "TEMPLATE")
N_SCENARIOS = {"GENDER": 14, "RACE": 12}

SCM = {
    "GENDER": {"X": "Gender", "Z": ["Age"], "W": ["Employment", "Education"],
               "x0": "Female", "x1": "Male"},
    "RACE": {"X": "Race", "Z": ["Age"], "W": ["Prior Convictions"],
             "x0": "White-Caucasian", "x1": "Non-White"},
}

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def dataset_root() -> Path:
    return repo_root() / "DATASET"

def result_root() -> Path:
    return repo_root() / "RESULT"

def scenarios(attr: str) -> list[str]:
    return [f"s{i}" for i in range(1, N_SCENARIOS[attr.upper()] + 1)]

def synthetic_dir(attr: str) -> Path:
    return dataset_root() / attr.upper() / "SYNTHETIC_DATA"

def synthetic_csv(attr: str, s: str) -> Path:
    return synthetic_dir(attr) / f"synthetic_data_{s}.csv"

def generated_dir(attr: str, generator: str) -> Path:
    return dataset_root() / attr.upper() / generator.upper() / "GENERATED_SENTENCES"

def generated_csv(attr: str, generator: str, s: str) -> Path:
    return generated_dir(attr, generator) / f"generated_sentences_{s}.csv"

def complete_dir(attr: str, generator: str) -> Path:
    return dataset_root() / attr.upper() / generator.upper() / "COMPLETE_DATASET"

def complete_csv(attr: str, generator: str) -> Path:
    return complete_dir(attr, generator) / f"generated_sentences_{attr.lower()}_complete.csv"

def result_dir(attr: str, generator: str, *parts: str) -> Path:
    base = result_root() / attr.upper() / generator.upper()
    return base.joinpath(*parts) if parts else base

def sa_result_dir(attr: str, generator: str, model: str) -> Path:
    return result_dir(attr, generator, "RQ2", "SA", model)

def resolve_choice(value, env_var: str, choices, default=None, name: str = "value") -> str:
    raw = value if value is not None else os.environ.get(env_var)
    if raw is None:
        raw = default
    if raw is None:
        raise SystemExit(f"Missing --{name.lower()} (or ${env_var}). Choose one of: {', '.join(choices)}")
    norm = str(raw).upper()
    if norm not in choices:
        raise SystemExit(f"Invalid {name} '{raw}'. Choose one of: {', '.join(choices)}")
    return norm

def resolve_attr(value=None) -> str:
    return resolve_choice(value, "ATTR", ATTRS, name="attr")

def resolve_generator(value=None, default=None) -> str:
    return resolve_choice(value, "GENERATOR", GENERATORS, default=default, name="generator")

def scm(attr: str) -> dict:
    return SCM[resolve_attr(attr)]
