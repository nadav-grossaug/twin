"""Adult Census Income (UCI, 1994 census extract; 32,561 rows, 14 features + income).

Preprocessing decisions:

- Column names: dots replaced by underscores (``education.num`` -> ``education_num``),
  so names are valid identifiers everywhere downstream.
- ``"?"`` (missing ``workclass``/``occupation``/``native_country``, ~1.8k/1.8k/0.6k rows) is
  kept as its own category. It is not missing at random: every ``workclass == "?"`` row
  also has ``occupation == "?"``, these rows skew young/old and have a >50K rate of 10%
  vs 25% for the rest, so the marker carries signal. Synthcity generators also do not
  handle NaN. A generator should reproduce it like any other category.
- ``education_num`` is kept although it is a 1:1 recoding of ``education``: it is part
  of the canonical dataset, and whether a generator preserves the deterministic
  mapping is an interpretable fidelity check (the ``education_num_mapping`` constraint).
- Target ``income`` is encoded as int 0/1 (1 = ``>50K``). Synthcity label-encodes
  object columns anyway, but an explicit 0/1 target keeps the meaning of "positive
  class" unambiguous for the AUROC-based utility metrics and the conditional samplers.
- Sensitive features: ``sex`` and ``race``, the protected attributes of the fairness
  literature on Adult. Synthcity's k-anonymity/l-diversity treat every other column as
  a quasi-identifier and ask how many distinct sensitive values each cluster holds.
  ``income`` is not used: it is the utility target and binary, so its l-diversity is
  capped at 2 and would not discriminate between generators.
- Exact duplicate rows (24) are kept: they are part of the canonical data.
"""

import pandas as pd

from synth_bench.config import dataset_dir
from synth_bench.datasets.base import DatasetSpec

NAME = "adult-census"
TARGET = "income"
CATEGORICAL = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
NUMERICAL = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


def load_adult() -> pd.DataFrame:
    """Read ``data/adult.csv`` and return the cleaned, typed DataFrame."""
    df = pd.read_csv(dataset_dir(NAME) / "data" / "adult.csv", skipinitialspace=True)
    df.columns = [c.replace(".", "_") for c in df.columns]
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip()
    df[NUMERICAL] = df[NUMERICAL].astype(int)
    df[TARGET] = (df[TARGET].str.strip().str.rstrip(".") == ">50K").astype(int)
    return df


# Domain rules (True = violation). Real-data violation rate is ~0: 3 of 32,561 rows
# break the Husband/Wife-vs-sex rule, none break the others.
EDUCATION_NUM = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
    "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
    "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16,
}  # fmt: skip
MARRIED = ["Married-civ-spouse", "Married-AF-spouse"]
SPOUSE = ["Husband", "Wife"]

CONSTRAINTS = {
    "education_num_mapping": lambda df: df["education"].map(EDUCATION_NUM) != df["education_num"],
    "spouse_sex": lambda df: ((df["relationship"] == "Husband") & (df["sex"] != "Male"))
    | ((df["relationship"] == "Wife") & (df["sex"] != "Female")),
    "spouse_married": lambda df: df["relationship"].isin(SPOUSE) & ~df["marital_status"].isin(MARRIED),
    "unmarried_not_married": lambda df: (df["relationship"] == "Unmarried")
    & df["marital_status"].isin(MARRIED),
    "workclass_occupation_missing": lambda df: (df["workclass"] == "?") & (df["occupation"] != "?"),
    "gain_and_loss": lambda df: (df["capital_gain"] > 0) & (df["capital_loss"] > 0),
    "max_gain_high_income": lambda df: (df["capital_gain"] == 99999) & (df[TARGET] != 1),
}

ADULT_CENSUS = DatasetSpec(
    name=NAME,
    dir=dataset_dir(NAME),
    target=TARGET,
    task_type="classification",
    categorical=CATEGORICAL,
    numerical=NUMERICAL,
    sensitive_features=["sex", "race"],
    loader=load_adult,
    constraints=CONSTRAINTS,
)
