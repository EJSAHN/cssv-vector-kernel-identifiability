from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_LONG_COLUMNS = {
    "treatment",
    "year",
    "zone",
    "analysis_ring_range",
    "baseline_state",
    "severe_1A_infected_count",
    "not_severe_1A_count",
    "total_observed_count",
}

REQUIRED_GEOMETRY_COLUMNS = {
    "treatment",
    "replicate",
    "row",
    "col",
    "x_m",
    "y_m",
    "original_perimeter",
    "analysis_ring",
    "baseline_state",
}


class DataValidationError(ValueError):
    """Raised when input data fail hard consistency checks."""


def parse_ring_range(value: object) -> list[int]:
    """Parse strings like '1-5' or '8' into integer ring lists."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:
            raise DataValidationError(f"Invalid ring range {value!r}: lower bound exceeds upper bound")
        return list(range(lo, hi + 1))
    if re.fullmatch(r"\d+", text):
        return [int(text)]
    raise DataValidationError(f"Cannot parse analysis ring range: {value!r}")


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def load_zone_timeseries(root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "crig_cssv_zone_timeseries_long.csv"
    df = pd.read_csv(path)
    missing = REQUIRED_LONG_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required long-data columns in {path}: {sorted(missing)}")

    int_cols = ["year", "severe_1A_infected_count", "not_severe_1A_count", "total_observed_count"]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    df["treatment"] = df["treatment"].astype(str)
    df["zone"] = df["zone"].astype(str)
    df["baseline_state"] = df["baseline_state"].astype(str)
    df["treatment_zone"] = df["treatment"] + ":" + df["zone"]
    df["is_N1_protected"] = df["baseline_state"].eq("N1_mild_protected").astype(int)
    df["analysis_rings"] = df["analysis_ring_range"].map(parse_ring_range)
    return df.sort_values(["treatment", "zone", "year"]).reset_index(drop=True)


def load_geometry(root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "plot_geometry_by_treatment_3_reps.csv"
    geom = pd.read_csv(path)
    missing = REQUIRED_GEOMETRY_COLUMNS - set(geom.columns)
    if missing:
        raise DataValidationError(f"Missing required geometry columns in {path}: {sorted(missing)}")
    for col in ["replicate", "row", "col", "original_perimeter"]:
        geom[col] = pd.to_numeric(geom[col], errors="raise").astype(int)
    for col in ["x_m", "y_m"]:
        geom[col] = pd.to_numeric(geom[col], errors="raise").astype(float)
    geom["analysis_ring_num"] = pd.to_numeric(geom["analysis_ring"], errors="coerce")
    if "is_monitored_inner_tree" in geom.columns:
        geom["is_monitored_inner_tree"] = _coerce_bool_series(geom["is_monitored_inner_tree"])
    else:
        geom["is_monitored_inner_tree"] = geom["original_perimeter"].ge(3)
    return geom


def load_expected_counts(root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "treatment_design_expected_counts.csv"
    exp = pd.read_csv(path)
    exp["expected_count_3_reps"] = pd.to_numeric(exp["expected_count_3_reps"], errors="raise").astype(int)
    return exp


def load_all(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_zone_timeseries(root), load_geometry(root), load_expected_counts(root)


def validate_zone_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Return an audit table and raise only for hard analytical impossibilities.

    We intentionally do not silently repair printed totals. The CRIG/PLOS tables have
    small denominator irregularities; these are scientific audit items rather than code bugs.
    """
    rows: list[dict[str, object]] = []
    hard_errors: list[str] = []
    for key, g in df.groupby("treatment_zone", sort=True):
        g = g.sort_values("year")
        prev_y = 0
        prev_year = 0
        for _, r in g.iterrows():
            y = int(r["severe_1A_infected_count"])
            n_not = int(r["not_severe_1A_count"])
            n_total = int(r["total_observed_count"])
            recomputed_total = y + n_not
            delta = y - prev_y
            hard = []
            warn = []
            if int(r["year"]) <= prev_year:
                hard.append("non_increasing_year")
            if y < 0 or n_not < 0 or n_total <= 0:
                hard.append("negative_or_zero_count")
            if recomputed_total != n_total:
                hard.append("printed_total_mismatch")
            if delta < 0:
                hard.append("cumulative_count_decreases")
            if y > n_total:
                hard.append("infected_exceeds_total")
            if n_total != g["total_observed_count"].iloc[0]:
                warn.append("zone_total_varies_over_years")
            rows.append(
                {
                    "treatment_zone": key,
                    "treatment": r["treatment"],
                    "zone": r["zone"],
                    "year": int(r["year"]),
                    "infected": y,
                    "not_severe": n_not,
                    "total": n_total,
                    "new_infections_since_previous_year": delta,
                    "hard_flags": ";".join(hard),
                    "warning_flags": ";".join(warn),
                }
            )
            for h in hard:
                hard_errors.append(f"{key}, year {int(r['year'])}: {h}")
            prev_y = y
            prev_year = int(r["year"])
    audit = pd.DataFrame(rows)
    fatal = [x for x in hard_errors if "printed_total_mismatch" not in x]
    if fatal:
        raise DataValidationError("Hard validation errors:\n" + "\n".join(fatal))
    return audit


def attach_expected_counts(long_df: pd.DataFrame, expected_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["treatment", "zone", "expected_count_3_reps"]
    out = long_df.merge(expected_df[cols], on=["treatment", "zone"], how="left", validate="many_to_one")
    if out["expected_count_3_reps"].isna().any():
        missing = out.loc[out["expected_count_3_reps"].isna(), ["treatment", "zone"]].drop_duplicates()
        raise DataValidationError(f"Missing expected design counts for:\n{missing}")
    out["expected_count_3_reps"] = out["expected_count_3_reps"].astype(int)
    out["observed_minus_expected_total"] = out["total_observed_count"] - out["expected_count_3_reps"]
    return out


def normalize_denominator_mode(denominator_mode: str) -> str:
    """Return the public denominator-mode label used by the workflow."""
    aliases = {
        "reported": "reported",
        "table": "reported",
        "reconstructed": "reconstructed",
        "expected_geometry": "reconstructed",
    }
    try:
        return aliases[str(denominator_mode)]
    except KeyError as exc:
        raise ValueError("denominator_mode must be 'reported' or 'reconstructed'") from exc


def make_interval_data(
    long_df: pd.DataFrame,
    expected_df: pd.DataFrame,
    denominator_mode: str = "reported",
) -> pd.DataFrame:
    """Convert cumulative counts into interval-level binomial survival data.

    For each zone and year t, the likelihood contribution is:
        new_infections_t ~ Binomial(at_risk_t, q_t)

    This is preferable to treating cumulative yearly counts as independent binomials.
    """
    denominator_mode = normalize_denominator_mode(denominator_mode)

    df = attach_expected_counts(long_df.copy(), expected_df)
    rows: list[dict[str, object]] = []
    for key, g in df.groupby("treatment_zone", sort=True):
        g = g.sort_values("year")
        prev_y = 0
        for _, r in g.iterrows():
            y = int(r["severe_1A_infected_count"])
            if denominator_mode == "reported":
                denominator = int(r["total_observed_count"])
            else:
                denominator = int(r["expected_count_3_reps"])
            new_y = y - prev_y
            at_risk = denominator - prev_y
            if new_y < 0:
                raise DataValidationError(f"{key} year {int(r['year'])}: cumulative infected count decreased")
            if y > denominator:
                raise DataValidationError(
                    f"{key} year {int(r['year'])}: infected count {y} exceeds denominator {denominator}"
                )
            if new_y > at_risk:
                raise DataValidationError(
                    f"{key} year {int(r['year'])}: new infections {new_y} exceed at-risk count {at_risk}"
                )
            row = r.to_dict()
            row.update(
                {
                    "denominator_mode": denominator_mode,
                    "denominator_used": denominator,
                    "cumulative_infected_previous_year": prev_y,
                    "new_infections": int(new_y),
                    "at_risk": int(at_risk),
                    "interval_hazard_observed": (float(new_y) / float(at_risk)) if at_risk > 0 else np.nan,
                }
            )
            rows.append(row)
            prev_y = y
    out = pd.DataFrame(rows)
    out["year"] = out["year"].astype(int)
    out["is_N1_protected"] = out["baseline_state"].eq("N1_mild_protected").astype(int)
    return out.sort_values(["treatment", "zone", "year"]).reset_index(drop=True)


def audit_expected_denominators(long_df: pd.DataFrame, expected_df: pd.DataFrame) -> pd.DataFrame:
    df = attach_expected_counts(long_df, expected_df)
    df = df.copy()
    df["matches_expected_count"] = df["observed_minus_expected_total"].eq(0)
    return df[
        [
            "treatment",
            "zone",
            "year",
            "total_observed_count",
            "expected_count_3_reps",
            "observed_minus_expected_total",
            "matches_expected_count",
        ]
    ].sort_values(["treatment", "zone", "year"])


def zone_final_summary(long_df: pd.DataFrame, expected_df: pd.DataFrame) -> pd.DataFrame:
    df = attach_expected_counts(long_df, expected_df)
    final = df.sort_values("year").groupby("treatment_zone", as_index=False).tail(1)
    final = final.copy()
    final["final_fraction_table_denominator"] = final["severe_1A_infected_count"] / final["total_observed_count"]
    final["final_fraction_expected_denominator"] = final["severe_1A_infected_count"] / final["expected_count_3_reps"]
    return final[
        [
            "treatment",
            "zone",
            "baseline_state",
            "analysis_ring_range",
            "severe_1A_infected_count",
            "total_observed_count",
            "expected_count_3_reps",
            "final_fraction_table_denominator",
            "final_fraction_expected_denominator",
        ]
    ].sort_values(["treatment", "zone"])
