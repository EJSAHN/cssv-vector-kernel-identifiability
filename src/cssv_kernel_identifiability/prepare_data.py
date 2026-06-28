
from __future__ import annotations

from pathlib import Path
import pandas as pd


def _add_row(rows, treatment, year, zone, analysis_ring_range, original_perimeter_range,
             baseline_state, infected, not_severe, source_table):
    infected = int(infected)
    not_severe = int(not_severe)
    total = infected + not_severe
    rows.append({
        "treatment": treatment,
        "year": int(year),
        "zone": zone,
        "analysis_ring_range": analysis_ring_range,
        "original_perimeter_range": original_perimeter_range,
        "baseline_state": baseline_state,
        "severe_1A_infected_count": infected,
        "not_severe_1A_count": not_severe,
        "total_observed_count": total,
        "severe_1A_fraction": infected / total if total else None,
        "source_table": source_table,
    })


def build_processed_data(project_root: Path) -> None:
    """Create analysis-ready CSV files from transcribed source tables.

    The code keeps the published table counts unchanged. Denominator differences
    between printed tables and the reconstructed geometry are handled later as an
    explicit sensitivity analysis.
    """
    project_root = Path(project_root)
    source_dir = project_root / "data" / "source_tables"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t1 = pd.read_csv(source_dir / "treatment_T1_table1_transcribed.csv")
    for _, r in t1.iterrows():
        _add_row(rows, "T1", r.year, "n1_rows1_5", "1-5", "3-7", "N1_mild_protected",
                 r.n1_rows1_5_infected_1A, r.n1_rows1_5_not_severe_1A, "10.1371/journal.pone.0294579.t001")
        _add_row(rows, "T1", r.year, "susceptible_rows6_8", "6-8", "8-10", "susceptible_no_N1",
                 r.susceptible_rows6_8_infected_1A, r.susceptible_rows6_8_not_severe_1A, "10.1371/journal.pone.0294579.t001")

    t2 = pd.read_csv(source_dir / "treatment_T2_table2_transcribed.csv")
    for _, r in t2.iterrows():
        _add_row(rows, "T2", r.year, "n1_rows1_3", "1-3", "3-5", "N1_mild_protected",
                 r.n1_rows1_3_infected_1A, r.n1_rows1_3_not_severe_1A, "10.1371/journal.pone.0294579.t002")
        _add_row(rows, "T2", r.year, "susceptible_rows4_8", "4-8", "6-10", "susceptible_no_N1",
                 r.susceptible_rows4_8_infected_1A, r.susceptible_rows4_8_not_severe_1A, "10.1371/journal.pone.0294579.t002")

    t3 = pd.read_csv(source_dir / "treatment_T3_table3_transcribed.csv")
    for _, r in t3.iterrows():
        _add_row(rows, "T3", r.year, "susceptible_rows1_8", "1-8", "3-10", "susceptible_no_N1",
                 r.susceptible_rows1_8_infected_1A, r.susceptible_rows1_8_not_severe_1A, "10.1371/journal.pone.0294579.t003")

    long = pd.DataFrame(rows).sort_values(["treatment", "year", "zone"]).reset_index(drop=True)
    long.to_csv(processed_dir / "crig_cssv_zone_timeseries_long.csv", index=False)

    geom_rows = []
    for treatment in ["T1", "T2", "T3", "T4"]:
        for replicate in [1, 2, 3]:
            for row in range(1, 20):
                for col in range(1, 20):
                    original_perimeter = min(row, 20 - row, col, 20 - col)
                    if original_perimeter <= 2:
                        analysis_ring = ""
                        baseline_state = "severe_1A_source_outer_perimeter"
                    else:
                        analysis_ring_num = original_perimeter - 2
                        analysis_ring = analysis_ring_num
                        if treatment == "T1" and 1 <= analysis_ring_num <= 5:
                            baseline_state = "N1_mild_protected"
                        elif treatment == "T2" and 1 <= analysis_ring_num <= 3:
                            baseline_state = "N1_mild_protected"
                        elif treatment == "T4" and 1 <= analysis_ring_num <= 8:
                            baseline_state = "N1_mild_protected"
                        else:
                            baseline_state = "susceptible_no_N1"
                    geom_rows.append({
                        "treatment": treatment,
                        "replicate": replicate,
                        "tree_id": f"{treatment}_R{replicate}_r{row:02d}_c{col:02d}",
                        "row": row,
                        "col": col,
                        "x_m": round((col - 1) * 2.4, 6),
                        "y_m": round((row - 1) * 2.4, 6),
                        "original_perimeter": original_perimeter,
                        "analysis_ring": analysis_ring,
                        "is_monitored_inner_tree": original_perimeter >= 3,
                        "baseline_state": baseline_state,
                    })
    geom = pd.DataFrame(geom_rows)
    geom.to_csv(processed_dir / "plot_geometry_by_treatment_3_reps.csv", index=False)
    pd.read_csv(source_dir / "treatment_design_expected_counts.csv").to_csv(processed_dir / "treatment_design_expected_counts.csv", index=False)

    ring_rows = []
    for original_perimeter in range(1, 11):
        side = 21 - 2 * original_perimeter
        tree_count_one_rep = 1 if side == 1 else 4 * side - 4
        ring_rows.append({
            "original_perimeter": original_perimeter,
            "analysis_ring": "" if original_perimeter <= 2 else original_perimeter - 2,
            "side_length": side,
            "tree_count_one_rep": tree_count_one_rep,
            "tree_count_three_reps": tree_count_one_rep * 3,
            "role": "severe_1A_source_outer_perimeter" if original_perimeter <= 2 else "monitored_inner_tree",
        })
    pd.DataFrame(ring_rows).to_csv(processed_dir / "ring_counts_19x19_one_rep.csv", index=False)
