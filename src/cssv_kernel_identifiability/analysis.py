
from __future__ import annotations

from pathlib import Path
import math
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .prepare_data import build_processed_data
from .data import load_all, load_zone_timeseries, load_geometry, load_expected_counts, make_interval_data, validate_zone_timeseries, zone_final_summary, parse_ring_range, normalize_denominator_mode
from .kernel import attach_exposure, kernel_values
from .models import fit_hazard_model, fit_and_score, build_design, cloglog_inv, binomial_logpmf, beta_binomial_logpmf, prediction_frame
from .optimizer_check import check_zone_profile, check_tree_profile, summarize_optimizer_check, check_empirical_fits

DEFAULT_SCALES = [2.4, 3.6, 4.8, 7.2, 9.6, 14.4, 19.2, 28.8]
EPS = 1e-10


def parse_scales(text: str | None) -> list[float]:
    if text is None or str(text).strip() == "":
        return list(DEFAULT_SCALES)
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("Kernel scales must be positive numbers")
    return vals


def aicc(aic: float, n_params: int, n_obs: int) -> float:
    if n_obs <= n_params + 1:
        return float("nan")
    return float(aic + (2.0 * n_params * (n_params + 1.0)) / (n_obs - n_params - 1.0))


def exp_clip(x: float) -> float:
    return float(np.exp(np.clip(float(x), -50.0, 50.0)))


def fit_profile(interval: pd.DataFrame, geometry: pd.DataFrame, scales: list[float], family: str):
    fits = []
    coefs = []
    rows = []
    for scale in scales:
        df = attach_exposure(interval, geometry, local_scale_m=scale)
        name = "M2_year_N1_kernel" if family == "binomial" else "M3_beta_binomial_year_N1_kernel"
        fit, X, info = fit_hazard_model(
            df, model_name=name, include_protection=True, include_kernel=True,
            family=family, local_scale_m=scale,
        )
        fits.append((fit, df, X, info))
        coefs.append(fit.coefficient_frame())
        rows.append(fit.summary_row())
    return fits, coefs, rows


def fit_models_for_denominator(project_root: Path, denominator_mode: str, scales: list[float]) -> dict[str, pd.DataFrame]:
    long_df, geometry, expected = load_all(project_root)
    interval = make_interval_data(long_df, expected, denominator_mode=denominator_mode)
    fits = []
    coef_frames = []

    fit0, X0, info0 = fit_hazard_model(interval, "M0_year_only", False, False, family="binomial")
    fits.append((fit0, interval, X0, info0)); coef_frames.append(fit0.coefficient_frame())
    fit1, X1, info1 = fit_hazard_model(interval, "M1_year_N1", True, False, family="binomial")
    fits.append((fit1, interval, X1, info1)); coef_frames.append(fit1.coefficient_frame())

    m2_fits, m2_coefs, m2_rows = fit_profile(interval, geometry, scales, family="binomial")
    m3_fits, m3_coefs, m3_rows = fit_profile(interval, geometry, scales, family="beta_binomial")
    fits.extend(m2_fits); fits.extend(m3_fits)
    coef_frames.extend(m2_coefs); coef_frames.extend(m3_coefs)

    comparison = pd.DataFrame([x[0].summary_row() for x in fits]).sort_values(["aic", "bic"]).reset_index(drop=True)
    comparison["aicc"] = [aicc(r.aic, int(r.n_params), int(r.n_obs)) for r in comparison.itertuples()]
    comparison["delta_aic"] = comparison["aic"] - comparison["aic"].min()
    comparison["delta_aicc"] = comparison["aicc"] - comparison["aicc"].min()

    coefficient_frames = []
    for frame in coef_frames:
        if not frame.empty:
            coefficient_frames.append(frame.dropna(axis=1, how="all"))
    coefficients = pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame()
    kernel_profile = pd.DataFrame(m2_rows + m3_rows).sort_values(["family", "local_scale_m"]).reset_index(drop=True)
    kernel_profile["aicc"] = [aicc(r.aic, int(r.n_params), int(r.n_obs)) for r in kernel_profile.itertuples()]
    kernel_profile["delta_aic_within_family"] = kernel_profile.groupby("family")["aic"].transform(lambda x: x - x.min())
    kernel_profile["delta_aicc_within_family"] = kernel_profile.groupby("family")["aicc"].transform(lambda x: x - x.min())

    successful = [x for x in fits if x[0].success]
    best_fit, best_df, best_X, best_info = min(successful, key=lambda x: x[0].aic)
    predictions = prediction_frame(best_df, best_fit, best_X)

    return {
        "interval_data": interval,
        "model_comparison": comparison,
        "model_coefficients": coefficients,
        "kernel_profile": kernel_profile,
        "predictions_best_aic": predictions,
    }


def coefficient_digest(coefficients: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    successful = comparison.loc[comparison["success"].astype(bool)].copy()
    if successful.empty:
        return pd.DataFrame()
    best = successful.sort_values(["aicc", "aic"]).iloc[0]
    co = coefficients.loc[
        coefficients["model_name"].eq(best["model_name"]) & coefficients["family"].eq(best["family"])
    ].copy()
    if pd.notna(best.get("local_scale_m")):
        co = co.loc[np.isclose(pd.to_numeric(co["local_scale_m"], errors="coerce"), float(best["local_scale_m"]), equal_nan=False)]
    def term_value(term):
        s = co.loc[co["term"].eq(term), "estimate"]
        return float(s.iloc[0]) if len(s) else np.nan
    n1 = term_value("N1_protected")
    pressure = term_value("z_log_source_pressure")
    icc = term_value("intracluster_correlation_derived")
    return pd.DataFrame([{
        "aicc_leader": best["model_name"],
        "aicc_leader_family": best["family"],
        "aicc_leader_scale_m": best.get("local_scale_m", np.nan),
        "beta_binomial_icc": icc,
        "N1_coef": n1,
        "N1_cumulative_hazard_ratio": exp_clip(n1) if pd.notna(n1) else np.nan,
        "source_pressure_coef": pressure,
        "source_pressure_log_hazard_ratio": pressure,
        "source_pressure_cumulative_hazard_ratio": exp_clip(pressure) if pd.notna(pressure) else np.nan,
    }])


def kernel_identifiability(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prof = comparison.loc[comparison["model_name"].str.contains("kernel", na=False)].copy()
    for (model_name, family), g in prof.groupby(["model_name", "family"], dropna=False):
        g = g.dropna(subset=["local_scale_m", "aic", "aicc"]).sort_values(["aicc", "aic"])
        if g.empty:
            continue
        min_aic = float(g["aic"].min())
        min_aicc = float(g["aicc"].min())
        rows.append({
            "model_name": model_name,
            "family": family,
            "best_scale_by_aicc_m": float(g.iloc[0]["local_scale_m"]),
            "best_scale_by_aic_m": float(g.sort_values("aic").iloc[0]["local_scale_m"]),
            "n_scales_with_delta_aic_le_2": int((g["aic"] <= min_aic + 2).sum()),
            "n_scales_with_delta_aicc_le_2": int((g["aicc"] <= min_aicc + 2).sum()),
            "scales_with_delta_aicc_le_2": ";".join(str(float(x)) for x in g.loc[g["aicc"] <= min_aicc + 2, "local_scale_m"]),
            "interpretation": "weak scale identification" if int((g["aicc"] <= min_aicc + 2).sum()) > 1 else "sharp scale support on this grid",
        })
    return pd.DataFrame(rows)


def loo_cv_all_families(interval: pd.DataFrame, geometry: pd.DataFrame, scales: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    zones = sorted(interval["treatment_zone"].unique())
    base_specs = [
        ("M0_year_only", False, False, "binomial", None),
        ("M1_year_N1", True, False, "binomial", None),
        ("M0_year_only_beta_binomial", False, False, "beta_binomial", None),
        ("M1_year_N1_beta_binomial", True, False, "beta_binomial", None),
    ]
    for heldout in zones:
        train = interval.loc[interval["treatment_zone"].ne(heldout)].copy()
        test = interval.loc[interval["treatment_zone"].eq(heldout)].copy()
        for model_name, inc_prot, inc_kernel, family, scale in base_specs:
            try:
                score = fit_and_score(train, test, model_name, inc_prot, inc_kernel, family=family, local_scale_m=scale)
                score["heldout_zone"] = heldout
                score["scale_selection"] = "not_applicable"
                rows.append(score)
            except Exception as exc:
                rows.append({"heldout_zone": heldout, "model_name": model_name, "family": family, "error": str(exc)})
        for family, model_name in [("binomial", "M2_year_N1_kernel"), ("beta_binomial", "M3_beta_binomial_year_N1_kernel")]:
            candidates = []
            for scale in scales:
                try:
                    train_s = attach_exposure(train, geometry, local_scale_m=scale)
                    test_s = attach_exposure(test, geometry, local_scale_m=scale)
                    score = fit_and_score(train_s, test_s, model_name, True, True, family=family, local_scale_m=scale)
                    candidates.append(score)
                except Exception as exc:
                    rows.append({"heldout_zone": heldout, "model_name": model_name, "family": family, "local_scale_m": scale, "error": str(exc)})
            if candidates:
                best = max(candidates, key=lambda x: x["train_log_likelihood"])
                best["heldout_zone"] = heldout
                best["scale_selection"] = "best_training_likelihood_within_fold"
                rows.append(best)
    cv = pd.DataFrame(rows)
    if "heldout_log_likelihood" in cv.columns:
        summary = (cv.dropna(subset=["heldout_log_likelihood"])
                   .groupby(["model_name", "family"], dropna=False)["heldout_log_likelihood"]
                   .sum().reset_index().sort_values("heldout_log_likelihood", ascending=False))
    else:
        summary = pd.DataFrame()
    return cv, summary


def nll_constrained(theta, X, y, n, family):
    p = cloglog_inv(X @ theta[:X.shape[1]])
    if family == "binomial":
        return -float(np.sum(binomial_logpmf(y, n, p)))
    return -float(np.sum(beta_binomial_logpmf(y, n, p, np.exp(theta[-1]))))


def fit_positive_pressure(df: pd.DataFrame, family: str, scale: float) -> dict[str, object]:
    fit, _, info_un = fit_hazard_model(df, "positive_pressure_diagnostic", True, True, family=family, local_scale_m=scale)
    X, info = build_design(df, True, True, design_info=info_un)
    y = pd.to_numeric(df["new_infections"], errors="raise").to_numpy(float)
    n = pd.to_numeric(df["at_risk"], errors="raise").to_numpy(float)
    names = list(info.feature_names)
    pidx = names.index("z_log_source_pressure")
    theta0 = np.asarray(fit.params, dtype=float).copy()
    theta0[pidx] = max(theta0[pidx], 1e-8)
    bounds = [(None, None)] * len(theta0)
    bounds[pidx] = (0.0, None)
    if family == "beta_binomial":
        bounds[-1] = (-4.0, 12.0)
    res = minimize(lambda th: nll_constrained(th, X, y, n, family), theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000, "ftol": 1e-9})
    theta = np.asarray(res.x, dtype=float)
    ll = -float(nll_constrained(theta, X, y, n, family))
    k = len(theta)
    aic = 2 * k - 2 * ll
    pressure = float(theta[pidx])
    n1 = float(theta[names.index("N1_protected")])
    row = {
        "family": family,
        "local_scale_m": scale,
        "success": bool(res.success),
        "log_likelihood": ll,
        "n_params": k,
        "aic": aic,
        "aicc": aicc(aic, k, len(df)),
        "pressure_coef_constrained_nonnegative": pressure,
        "pressure_at_zero_boundary": abs(pressure) < 1e-6,
        "N1_coef": n1,
        "N1_cumulative_hazard_ratio": exp_clip(n1),
        "unconstrained_pressure_coef": float(fit.params[pidx]),
    }
    if family == "beta_binomial":
        kappa = exp_clip(float(theta[-1]))
        row["intracluster_correlation"] = 1.0 / (kappa + 1.0)
    return row


def constrained_positive_pressure_profile(interval: pd.DataFrame, geometry: pd.DataFrame, scales: list[float]) -> pd.DataFrame:
    rows = []
    for scale in scales:
        df = attach_exposure(interval, geometry, local_scale_m=scale)
        for family in ["binomial", "beta_binomial"]:
            try:
                rows.append(fit_positive_pressure(df, family, scale))
            except Exception as exc:
                rows.append({"family": family, "local_scale_m": scale, "success": False, "error": str(exc)})
    return pd.DataFrame(rows).sort_values(["family", "aicc"], na_position="last")


def run_empirical_analysis(project_root: Path, scales: list[float], denominator_mode: str = "reported",
                           check_optimizer: bool = False) -> dict[str, pd.DataFrame]:
    project_root = Path(project_root)
    denominator_mode = normalize_denominator_mode(denominator_mode)
    build_processed_data(project_root)
    long_df, geometry, expected = load_all(project_root)
    audit = validate_zone_timeseries(long_df)
    final_summary = zone_final_summary(long_df, expected)
    interval_table = make_interval_data(long_df, expected, denominator_mode="reported")
    denominator_audit = interval_table[["treatment", "zone", "year", "total_observed_count", "expected_count_3_reps", "observed_minus_expected_total"]].copy()
    denominator_audit = denominator_audit.drop_duplicates().sort_values(["treatment", "zone", "year"])

    mode = denominator_mode
    print(f"Running empirical denominator mode: {mode}", flush=True)
    mode_res = fit_models_for_denominator(project_root, mode, scales)
    results = {
        "data_audit_zone_year": audit,
        "denominator_audit_observed_vs_expected": denominator_audit,
        "zone_final_summary": final_summary,
        f"interval_data_{mode}": mode_res["interval_data"],
        f"model_comparison_{mode}": mode_res["model_comparison"],
        f"model_coefficients_{mode}": mode_res["model_coefficients"],
        f"kernel_profile_{mode}": mode_res["kernel_profile"],
        f"predictions_best_aic_{mode}": mode_res["predictions_best_aic"],
        f"coefficient_digest_{mode}": coefficient_digest(mode_res["model_coefficients"], mode_res["model_comparison"]),
        f"kernel_identifiability_{mode}": kernel_identifiability(mode_res["model_comparison"]),
    }
    cv, cv_summary = loo_cv_all_families(mode_res["interval_data"], geometry, scales)
    results[f"leave_one_zone_cv_{mode}"] = cv
    results[f"leave_one_zone_cv_summary_{mode}"] = cv_summary
    results[f"constrained_positive_pressure_{mode}"] = constrained_positive_pressure_profile(mode_res["interval_data"], geometry, scales)
    if check_optimizer:
        results[f"optimizer_check_empirical_{mode}"] = check_empirical_fits(mode_res["interval_data"], geometry, scales)
    return results


# Simulation functions.
def pairwise_distances(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    dx = a_xy[:, None, 0] - b_xy[None, :, 0]
    dy = a_xy[:, None, 1] - b_xy[None, :, 1]
    return np.sqrt(dx * dx + dy * dy)


def tree_source_pressure_for_treatment(geometry_t: pd.DataFrame, target_rows: pd.DataFrame, scale_m: float) -> pd.Series:
    pieces = []
    for rep, target_rep in target_rows.groupby("replicate", sort=True):
        geom_rep = geometry_t.loc[geometry_t["replicate"].eq(rep)].copy()
        source_xy = geom_rep.loc[geom_rep["original_perimeter"].le(2), ["x_m", "y_m"]].to_numpy(float)
        target_xy = target_rep[["x_m", "y_m"]].to_numpy(float)
        d = pairwise_distances(target_xy, source_xy)
        pressure = kernel_values(d, local_scale_m=scale_m).sum(axis=1)
        pieces.append(pd.Series(pressure, index=target_rep.index))
    return pd.concat(pieces).sort_index()


def build_target_tree_panel(long_df: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    zone_defs = long_df[["treatment", "zone", "analysis_ring_range", "baseline_state"]].drop_duplicates()
    geom = geometry.copy()
    if "analysis_ring_num" not in geom.columns:
        geom["analysis_ring_num"] = pd.to_numeric(geom["analysis_ring"], errors="coerce")
    parts = []
    for _, z in zone_defs.iterrows():
        rings = parse_ring_range(z["analysis_ring_range"])
        g = geom.loc[geom["treatment"].eq(z["treatment"]) & geom["analysis_ring_num"].isin(rings)].copy()
        g["zone"] = str(z["zone"])
        g["analysis_ring_range"] = str(z["analysis_ring_range"])
        g["baseline_state_zone"] = str(z["baseline_state"])
        g["is_N1_protected"] = int(str(z["baseline_state"]) == "N1_mild_protected")
        g["treatment_zone"] = g["treatment"].astype(str) + ":" + g["zone"].astype(str)
        parts.append(g)
    panel = pd.concat(parts, ignore_index=True).sort_values(["treatment", "replicate", "zone", "row", "col"]).reset_index(drop=True)
    panel["tree_uid"] = np.arange(len(panel), dtype=int)
    panel["replicate_zone"] = panel["treatment_zone"].astype(str) + ":rep" + panel["replicate"].astype(str)
    return panel


def add_pressure_columns(panel: pd.DataFrame, geometry: pd.DataFrame, scales: list[float]) -> pd.DataFrame:
    out = panel.copy()
    for scale in scales:
        vals = []
        for treatment, target_t in out.groupby("treatment", sort=False):
            geom_t = geometry.loc[geometry["treatment"].eq(treatment)].copy()
            vals.append(tree_source_pressure_for_treatment(geom_t, target_t, scale))
        p = pd.concat(vals).sort_index()
        out[f"log_pressure_scale_{scale:g}"] = np.log(np.clip(p.to_numpy(float), 1e-300, np.inf))
    return out


def fit_calibration_year_effects(interval: pd.DataFrame) -> dict[int, float]:
    fit, _, _ = fit_hazard_model(interval, "calibration_year_only_beta_binomial", False, False, family="beta_binomial")
    coefs = dict(zip(fit.feature_names, fit.params[:len(fit.feature_names)]))
    out = {1: float(coefs["intercept"])}
    for year in range(2, 8):
        out[year] = float(coefs["intercept"] + coefs.get(f"year_{year}", 0.0))
    return out


def simulate_tree_epidemic(panel: pd.DataFrame, true_scale: float, year_eta: dict[int, float], n1_log_hr: float,
                           pressure_coef: float, frailty_sd: float, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    col = f"log_pressure_scale_{true_scale:g}"
    z = pd.to_numeric(panel[col], errors="raise").to_numpy(float)
    z = (z - z.mean()) / (z.std() if z.std() > 0 else 1.0)
    infected = np.zeros(len(panel), dtype=bool)
    records = []
    for year in range(1, 8):
        at_idx = np.where(~infected)[0]
        if len(at_idx) == 0:
            break
        at = panel.iloc[at_idx].copy()
        eta = np.full(len(at), float(year_eta[year]), dtype=float)
        eta += float(n1_log_hr) * at["is_N1_protected"].to_numpy(float)
        eta += float(pressure_coef) * z[at_idx]
        if frailty_sd > 0:
            keys = at["replicate_zone"].astype(str).to_numpy()
            frail = {k: rng.normal(0.0, frailty_sd) for k in sorted(set(keys))}
            eta += np.array([frail[k] for k in keys], dtype=float)
        p = cloglog_inv(eta)
        new = rng.binomial(1, p).astype(int)
        infected[at_idx] = infected[at_idx] | new.astype(bool)
        rec = at[["tree_uid", "treatment", "replicate", "zone", "treatment_zone", "analysis_ring_range", "baseline_state_zone", "is_N1_protected", "row", "col"]].copy()
        rec["year"] = year
        rec["at_risk"] = 1
        rec["new_infections"] = new
        records.append(rec)
    tree_year = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    denom = panel.groupby(["treatment", "zone", "treatment_zone", "analysis_ring_range", "baseline_state_zone"], as_index=False).size().rename(columns={"size": "denominator_used"})
    inc = tree_year.groupby(["treatment", "zone", "treatment_zone", "year"], as_index=False)["new_infections"].sum()
    rows = []
    for _, d in denom.iterrows():
        for year in range(1, 8):
            rows.append({
                "treatment": d.treatment,
                "zone": d.zone,
                "treatment_zone": d.treatment_zone,
                "analysis_ring_range": d.analysis_ring_range,
                "baseline_state": d.baseline_state_zone,
                "is_N1_protected": int(str(d.baseline_state_zone) == "N1_mild_protected"),
                "denominator_used": int(d.denominator_used),
                "year": year,
            })
    zdf = pd.DataFrame(rows).merge(inc, on=["treatment", "zone", "treatment_zone", "year"], how="left")
    zdf["new_infections"] = zdf["new_infections"].fillna(0).astype(int)
    zdf = zdf.sort_values(["treatment", "zone", "year"]).reset_index(drop=True)
    prev, cum = [], []
    for _, g in zdf.groupby("treatment_zone", sort=False):
        c = 0
        for new in g["new_infections"].to_numpy(int):
            prev.append(c)
            c += int(new)
            cum.append(c)
    zdf["cumulative_infected_previous_year"] = prev
    zdf["severe_1A_infected_count"] = cum
    zdf["at_risk"] = zdf["denominator_used"] - zdf["cumulative_infected_previous_year"]
    zdf["total_observed_count"] = zdf["denominator_used"]
    zdf["not_severe_1A_count"] = zdf["denominator_used"] - zdf["severe_1A_infected_count"]
    zdf["denominator_mode"] = "simulated_reconstructed_geometry"
    return tree_year, zdf


def fit_zone_scale_profile(sim_zone: pd.DataFrame, geometry: pd.DataFrame, candidate_scales: list[float]) -> pd.DataFrame:
    rows = []
    for scale in candidate_scales:
        df = attach_exposure(sim_zone, geometry, local_scale_m=scale)
        try:
            fit, _, _ = fit_hazard_model(df, "sim_zone_beta_binomial_N1_kernel", True, True, family="beta_binomial", local_scale_m=scale)
            pidx = fit.feature_names.index("z_log_source_pressure")
            n1idx = fit.feature_names.index("N1_protected")
            rows.append({"analysis_unit": "zone_level_aggregate", "candidate_scale_m": scale, "success": fit.success,
                         "log_likelihood": fit.log_likelihood, "n_obs": fit.n_obs, "n_params": fit.n_params,
                         "aic": fit.aic, "aicc": aicc(fit.aic, fit.n_params, fit.n_obs),
                         "pressure_coef": float(fit.params[pidx]), "N1_coef": float(fit.params[n1idx]),
                         "icc": fit.intra_cluster_correlation})
        except Exception as exc:
            rows.append({"analysis_unit": "zone_level_aggregate", "candidate_scale_m": scale, "success": False, "error": str(exc)})
    out = pd.DataFrame(rows)
    if "aicc" in out.columns and out["aicc"].notna().any():
        out["delta_aicc"] = out["aicc"] - out["aicc"].min(skipna=True)
    return out


def tree_grouped_design_for_scale(tree_year: pd.DataFrame, panel: pd.DataFrame, scale: float) -> pd.DataFrame:
    col = f"log_pressure_scale_{scale:g}"
    df = tree_year.merge(panel[["tree_uid", col]], on="tree_uid", how="left", validate="many_to_one")
    df["log_kernel_source_pressure"] = pd.to_numeric(df[col], errors="raise").round(10)
    grouped = (df.groupby(["year", "is_N1_protected", "log_kernel_source_pressure"], as_index=False)
               .agg(new_infections=("new_infections", "sum"), at_risk=("at_risk", "sum")))
    grouped["treatment"] = "tree_grouped"
    grouped["zone"] = "tree_grouped"
    grouped["treatment_zone"] = "tree_grouped"
    grouped["analysis_ring_range"] = "all"
    grouped["baseline_state"] = np.where(grouped["is_N1_protected"].eq(1), "N1_mild_protected", "susceptible_no_N1")
    grouped["denominator_used"] = grouped["at_risk"]
    return grouped


def fit_tree_grouped_scale_profile(tree_year: pd.DataFrame, panel: pd.DataFrame, candidate_scales: list[float]) -> pd.DataFrame:
    rows = []
    for scale in candidate_scales:
        df = tree_grouped_design_for_scale(tree_year, panel, scale)
        try:
            fit, _, _ = fit_hazard_model(df, "sim_tree_grouped_binomial_N1_kernel", True, True, family="binomial", local_scale_m=scale)
            pidx = fit.feature_names.index("z_log_source_pressure")
            n1idx = fit.feature_names.index("N1_protected")
            rows.append({"analysis_unit": "tree_level_grouped", "candidate_scale_m": scale, "success": fit.success,
                         "log_likelihood": fit.log_likelihood, "n_obs": fit.n_obs, "n_params": fit.n_params,
                         "aic": fit.aic, "aicc": aicc(fit.aic, fit.n_params, fit.n_obs),
                         "pressure_coef": float(fit.params[pidx]), "N1_coef": float(fit.params[n1idx]), "icc": np.nan})
        except Exception as exc:
            rows.append({"analysis_unit": "tree_level_grouped", "candidate_scale_m": scale, "success": False, "error": str(exc)})
    out = pd.DataFrame(rows)
    if "aicc" in out.columns and out["aicc"].notna().any():
        out["delta_aicc"] = out["aicc"] - out["aicc"].min(skipna=True)
    return out


def summarize_scale_profile(profile: pd.DataFrame, true_scale: float, sim_id: int) -> dict[str, object]:
    valid = pd.Series(True, index=profile.index)
    for column in ("aicc", "aic", "candidate_scale_m", "pressure_coef", "N1_coef"):
        if column not in profile.columns:
            valid[:] = False
            break
        valid &= np.isfinite(pd.to_numeric(profile[column], errors="coerce"))
    ok = profile.loc[valid].copy()
    unit = str(profile["analysis_unit"].dropna().iloc[0]) if "analysis_unit" in profile and profile["analysis_unit"].notna().any() else "unknown"
    n_fitted = int(len(ok))
    if n_fitted == 0 or n_fitted < len(profile):
        return {"sim_id": sim_id, "true_scale_m": true_scale, "analysis_unit": unit, "profile_completed": False,
                "n_candidate_scales_fitted": n_fitted}
    ok = ok.sort_values(["aicc", "aic", "candidate_scale_m"])
    best = ok.iloc[0]
    min_aicc = float(ok["aicc"].min())
    support = ok.loc[ok["aicc"] <= min_aicc + 2.0]
    return {
        "sim_id": sim_id,
        "true_scale_m": float(true_scale),
        "analysis_unit": unit,
        "profile_completed": True,
        "n_candidate_scales_fitted": n_fitted,
        "selected_optimizer_success": bool(best["success"]),
        "selected_scale_m": float(best["candidate_scale_m"]),
        "selected_pressure_coef": float(best["pressure_coef"]),
        "selected_N1_coef": float(best["N1_coef"]),
        "selected_pressure_positive": bool(float(best["pressure_coef"]) > 0.0),
        "exact_scale_recovered": bool(abs(float(best["candidate_scale_m"]) - true_scale) < 1e-9),
        "true_scale_in_delta_aicc_2_set": bool(np.any(np.isclose(support["candidate_scale_m"].to_numpy(float), true_scale))),
        "delta_aicc_support_size": int(len(support)),
        "abs_log_scale_error": float(abs(math.log(float(best["candidate_scale_m"]) / float(true_scale)))),
        "min_aicc": min_aicc,
    }


def aggregate_recovery(rep_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (true_scale, unit), g in rep_results.groupby(["true_scale_m", "analysis_unit"], sort=True):
        ok = g.loc[g["profile_completed"].astype(bool)].copy()
        rows.append({
            "true_scale_m": true_scale,
            "analysis_unit": unit,
            "n_sims_requested": int(len(g)),
            "n_replicates_analysed": int(len(ok)),
            "n_selected_fits_with_optimizer_success": int(ok["selected_optimizer_success"].sum()) if len(ok) else 0,
            "exact_scale_recovery_rate": float(ok["exact_scale_recovered"].mean()) if len(ok) else np.nan,
            "true_scale_in_delta_aicc_2_rate": float(ok["true_scale_in_delta_aicc_2_set"].mean()) if len(ok) else np.nan,
            "positive_pressure_rate": float(ok["selected_pressure_positive"].mean()) if len(ok) else np.nan,
            "median_delta_aicc_support_size": float(ok["delta_aicc_support_size"].median()) if len(ok) else np.nan,
            "median_abs_log_scale_error": float(ok["abs_log_scale_error"].median()) if len(ok) else np.nan,
            "median_selected_pressure_coef": float(ok["selected_pressure_coef"].median()) if len(ok) else np.nan,
            "median_selected_N1_coef": float(ok["selected_N1_coef"].median()) if len(ok) else np.nan,
        })
    return pd.DataFrame(rows)


def scale_selection_counts(rep_results: pd.DataFrame) -> pd.DataFrame:
    ok = rep_results.loc[rep_results["profile_completed"].astype(bool)].copy()
    if ok.empty:
        return pd.DataFrame(columns=["true_scale_m", "analysis_unit", "selected_scale_m", "n_selected",
                                     "n_replicates_analysed", "selection_fraction"])
    counts = ok.groupby(["true_scale_m", "analysis_unit", "selected_scale_m"], as_index=False).size().rename(columns={"size": "n_selected"})
    denom = ok.groupby(["true_scale_m", "analysis_unit"], as_index=False).size().rename(columns={"size": "n_replicates_analysed"})
    out = counts.merge(denom, on=["true_scale_m", "analysis_unit"], how="left")
    out["selection_fraction"] = out["n_selected"] / out["n_replicates_analysed"]
    return out.sort_values(["true_scale_m", "analysis_unit", "selected_scale_m"])


def run_simulation_study(project_root: Path, scales: list[float], true_scales: list[float], n_sim: int, seed: int,
                         n1_log_hr: float = -0.40, pressure_coef: float = 0.30, frailty_sd: float = 0.30,
                         check_optimizer: bool = False, year_effects_file: Path | None = None) -> dict[str, pd.DataFrame]:
    project_root = Path(project_root)
    build_processed_data(project_root)
    long_df = load_zone_timeseries(project_root)
    geometry = load_geometry(project_root)
    expected = load_expected_counts(project_root)
    interval = make_interval_data(long_df, expected, denominator_mode="expected_geometry")
    if year_effects_file is not None:
        ye = pd.read_csv(year_effects_file)
        year_eta = {int(y): float(e) for y, e in zip(ye["year"], ye["eta_cloglog"])}
    else:
        year_eta = fit_calibration_year_effects(interval)
    panel = build_target_tree_panel(long_df, geometry)
    panel = add_pressure_columns(panel, geometry, sorted(set(scales + true_scales)))
    rng = np.random.default_rng(seed)
    rep_rows = []
    event_rows = []
    check_rows = []
    for true_scale in true_scales:
        for sim_id in range(1, n_sim + 1):
            if sim_id == 1 or sim_id % max(1, n_sim // 5) == 0:
                print(f"Simulation true_scale={true_scale:g} m, replicate {sim_id}/{n_sim}", flush=True)
            tree_year, sim_zone = simulate_tree_epidemic(panel, true_scale, year_eta, n1_log_hr, pressure_coef, frailty_sd, rng)
            final_counts = sim_zone.sort_values("year").groupby("treatment_zone").tail(1)["severe_1A_infected_count"].sum()
            event_rows.append({"sim_id": sim_id, "true_scale_m": true_scale, "tree_year_rows": len(tree_year), "zone_rows": len(sim_zone),
                               "total_simulated_infections": int(final_counts), "final_infection_fraction": float(final_counts / panel.shape[0])})
            zp = fit_zone_scale_profile(sim_zone, geometry, scales)
            tp = fit_tree_grouped_scale_profile(tree_year, panel, scales)
            rep_rows.append(summarize_scale_profile(zp, true_scale, sim_id))
            rep_rows.append(summarize_scale_profile(tp, true_scale, sim_id))
            if check_optimizer:
                for row in (check_zone_profile(sim_zone, geometry, scales, true_scale),
                            check_tree_profile(tree_year, panel, scales, true_scale, tree_grouped_design_for_scale)):
                    row["sim_id"] = sim_id
                    check_rows.append(row)
    rep = pd.DataFrame(rep_rows)
    summary = aggregate_recovery(rep)
    counts = scale_selection_counts(rep)
    settings = pd.DataFrame([{
        "n_sim_per_true_scale": n_sim,
        "seed": seed,
        "candidate_scales_m": ",".join(str(x) for x in scales),
        "true_scales_m": ",".join(str(x) for x in true_scales),
        "n1_log_hazard_ratio_true": n1_log_hr,
        "n1_hazard_ratio_true": exp_clip(n1_log_hr),
        "standardized_pressure_coef_true": pressure_coef,
        "pressure_hazard_ratio_per_1sd_true": exp_clip(pressure_coef),
        "frailty_sd_cloglog_scale": frailty_sd,
        "n_target_trees": int(panel.shape[0]),
        "year_effects_source": Path(year_effects_file).name if year_effects_file is not None else "calibrated in this run",
        "year_eta_calibration": json.dumps({int(k): float(v) for k, v in year_eta.items()}),
    }])
    out = {
        "simulation_replicate_results": rep,
        "design_power_recovery_summary": summary,
        "design_power_scale_selection_counts": counts,
        "design_power_simulated_epidemic_sizes": pd.DataFrame(event_rows),
        "design_power_settings": settings,
    }
    if check_optimizer:
        checks = pd.DataFrame(check_rows)
        out["optimizer_check_replicates"] = checks
        out["optimizer_check_summary"] = summarize_optimizer_check(checks)
    return out
