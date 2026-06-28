from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .data import parse_ring_range, load_geometry


def kernel_values(
    distances_m: np.ndarray,
    local_scale_m: float,
    jump_weight: float = 0.0,
    jump_scale_m: float = 50.0,
) -> np.ndarray:
    """Mixture exponential kernel used as an effective source-pressure score.

    This is not claimed to be a directly observed mealybug dispersal distribution. It is a
    compact scalar representation of distance-mediated exposure from the outer severe-1A
    source perimeters to a monitored target zone.
    """
    if local_scale_m <= 0:
        raise ValueError("local_scale_m must be positive")
    if jump_scale_m <= 0:
        raise ValueError("jump_scale_m must be positive")
    if not 0 <= jump_weight < 1:
        raise ValueError("jump_weight must be in [0, 1)")
    local = np.exp(-distances_m / local_scale_m)
    if jump_weight == 0:
        return local
    jump = np.exp(-distances_m / jump_scale_m)
    return (1.0 - jump_weight) * local + jump_weight * jump


def _pairwise_distances(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    dx = a_xy[:, None, 0] - b_xy[None, :, 0]
    dy = a_xy[:, None, 1] - b_xy[None, :, 1]
    return np.sqrt(dx * dx + dy * dy)


def zone_source_exposure(
    geometry: pd.DataFrame,
    treatment: str,
    analysis_rings: list[int],
    local_scale_m: float,
    jump_weight: float = 0.0,
    jump_scale_m: float = 50.0,
) -> dict[str, float]:
    """Compute mean source exposure from original perimeters 1-2 to a target ring set."""
    if not analysis_rings:
        raise ValueError("analysis_rings must be non-empty for monitored zones")

    g = geometry.loc[geometry["treatment"].eq(treatment)].copy()
    if g.empty:
        raise ValueError(f"No geometry found for treatment {treatment!r}")

    rep_rows: list[dict[str, float]] = []
    for rep, gr in g.groupby("replicate", sort=True):
        src = gr.loc[gr["original_perimeter"].le(2), ["x_m", "y_m"]].to_numpy(float)
        tgt = gr.loc[gr["analysis_ring_num"].isin(analysis_rings), ["x_m", "y_m"]].to_numpy(float)
        if src.size == 0 or tgt.size == 0:
            raise ValueError(
                f"Missing source or target trees for treatment={treatment}, rep={rep}, rings={analysis_rings}"
            )
        d = _pairwise_distances(tgt, src)
        k = kernel_values(d, local_scale_m=local_scale_m, jump_weight=jump_weight, jump_scale_m=jump_scale_m)
        exposure_by_target = k.sum(axis=1)
        mean_min_distance = d.min(axis=1).mean()
        mean_distance = d.mean()
        rep_rows.append(
            {
                "replicate": float(rep),
                "target_tree_count": float(tgt.shape[0]),
                "source_tree_count": float(src.shape[0]),
                "mean_source_pressure": float(exposure_by_target.mean()),
                "median_source_pressure": float(np.median(exposure_by_target)),
                "mean_nearest_source_distance_m": float(mean_min_distance),
                "mean_pairwise_source_distance_m": float(mean_distance),
            }
        )

    rep_df = pd.DataFrame(rep_rows)
    return {
        "target_tree_count_per_rep_mean": float(rep_df["target_tree_count"].mean()),
        "source_tree_count_per_rep_mean": float(rep_df["source_tree_count"].mean()),
        "kernel_source_pressure": float(rep_df["mean_source_pressure"].mean()),
        "kernel_source_pressure_median": float(rep_df["median_source_pressure"].mean()),
        "mean_nearest_source_distance_m": float(rep_df["mean_nearest_source_distance_m"].mean()),
        "mean_pairwise_source_distance_m": float(rep_df["mean_pairwise_source_distance_m"].mean()),
    }


def exposure_table_for_zones(
    interval_df: pd.DataFrame,
    geometry: pd.DataFrame,
    local_scale_m: float,
    jump_weight: float = 0.0,
    jump_scale_m: float = 50.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    unique = interval_df[["treatment", "zone", "analysis_ring_range"]].drop_duplicates()
    for _, r in unique.iterrows():
        rings = parse_ring_range(r["analysis_ring_range"])
        ex = zone_source_exposure(
            geometry=geometry,
            treatment=str(r["treatment"]),
            analysis_rings=rings,
            local_scale_m=local_scale_m,
            jump_weight=jump_weight,
            jump_scale_m=jump_scale_m,
        )
        rows.append(
            {
                "treatment": r["treatment"],
                "zone": r["zone"],
                "analysis_ring_range": r["analysis_ring_range"],
                "local_scale_m": float(local_scale_m),
                "jump_weight": float(jump_weight),
                "jump_scale_m": float(jump_scale_m),
                **ex,
            }
        )
    out = pd.DataFrame(rows)
    out["log_kernel_source_pressure"] = np.log(out["kernel_source_pressure"].clip(lower=1e-300))
    return out.sort_values(["treatment", "zone"]).reset_index(drop=True)


def attach_exposure(
    interval_df: pd.DataFrame,
    geometry: pd.DataFrame,
    local_scale_m: float,
    jump_weight: float = 0.0,
    jump_scale_m: float = 50.0,
) -> pd.DataFrame:
    exp = exposure_table_for_zones(
        interval_df=interval_df,
        geometry=geometry,
        local_scale_m=local_scale_m,
        jump_weight=jump_weight,
        jump_scale_m=jump_scale_m,
    )
    return interval_df.merge(
        exp,
        on=["treatment", "zone", "analysis_ring_range"],
        how="left",
        validate="many_to_one",
    )
