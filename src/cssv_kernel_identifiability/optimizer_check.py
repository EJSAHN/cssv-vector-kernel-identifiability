"""Refit models with tighter optimisation and compare the results with the original fits.

Binomial (complementary log-log) models are refitted by Fisher scoring with step
halving, targeting a maximum absolute score below 1e-9 and reporting whether
that tolerance was reached. Beta-binomial models
are refitted from three starting values with L-BFGS-B followed by a bounded
trust-region search, using analytic gradients; the best solution is kept and its
optimiser message and maximum absolute gradient are recorded.

The comparison is always between the original fitting routine, run in the
current session on the same data, and the refit. Comparison with previously
reported values requires merging these outputs with the reported tables.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, betaln, digamma

from .kernel import attach_exposure
from .models import fit_hazard_model, build_design

SCORE_TOLERANCE = 1e-9
BETA_BINOMIAL_LOG_KAPPA_BOUNDS = (-4.0, 12.0)
EPS_PROB = 1e-10


def _aicc(aic: float, k: int, n: int) -> float:
    return aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)


def _binomial_loglik(beta: np.ndarray, X: np.ndarray, y: np.ndarray, n: np.ndarray) -> float:
    eta = np.clip(X @ beta, -30.0, 5.0)
    p = np.clip(-np.expm1(-np.exp(eta)), 1e-10, 1 - 1e-10)
    return float(np.sum(gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1) + y * np.log(p) + (n - y) * np.log1p(-p)))


def _binomial_score(beta: np.ndarray, X: np.ndarray, y: np.ndarray, n: np.ndarray) -> np.ndarray:
    mu = np.exp(X @ beta)
    q = np.exp(-mu)
    one_minus_q = np.clip(-np.expm1(-mu), 1e-300, None)
    return X.T @ (mu * (y * q / one_minus_q - (n - y)))


def fisher_scoring_binomial(X: np.ndarray, y: np.ndarray, n: np.ndarray, beta0: np.ndarray,
                            tol: float = SCORE_TOLERANCE, max_iter: int = 200) -> dict[str, object]:
    """Complementary log-log binomial fit by Fisher scoring with step halving."""
    beta = np.asarray(beta0, dtype=float).copy()
    iterations = 0
    for iterations in range(1, max_iter + 1):
        score = _binomial_score(beta, X, y, n)
        if np.max(np.abs(score)) < tol:
            break
        mu = np.exp(X @ beta)
        q = np.exp(-mu)
        one_minus_q = np.clip(-np.expm1(-mu), 1e-300, None)
        info = X.T @ (X * (n * q * mu ** 2 / one_minus_q)[:, None])
        step = np.linalg.solve(info, score)
        ll0 = _binomial_loglik(beta, X, y, n)
        t = 1.0
        while t > 1e-8 and _binomial_loglik(beta + t * step, X, y, n) < ll0 - 1e-12:
            t /= 2.0
        beta = beta + t * step
        if np.max(np.abs(t * step)) < 1e-12:
            break
    max_score = float(np.max(np.abs(_binomial_score(beta, X, y, n))))
    return {"params": beta, "log_likelihood": _binomial_loglik(beta, X, y, n), "max_abs_score": max_score,
            "converged": bool(max_score < tol), "iterations": iterations,
            "message": "score below tolerance" if max_score < tol else "score tolerance not reached"}


def _beta_binomial_nll_and_grad(theta: np.ndarray, X: np.ndarray, y: np.ndarray, n: np.ndarray) -> tuple[float, np.ndarray]:
    """Negative log-likelihood of the beta-binomial complementary log-log model and its analytic gradient.

    theta holds the regression coefficients followed by log(kappa); the mean and linear
    predictor are clipped exactly as in the fitting routine of the main analysis.
    """
    beta, log_kappa = theta[:-1], theta[-1]
    kappa_raw = np.exp(log_kappa)
    kappa = max(kappa_raw, 1e-6)
    eta = X @ beta
    eta_c = np.clip(eta, -30.0, 5.0)
    e = np.exp(eta_c)
    mu_raw = -np.expm1(-e)
    mu = np.clip(mu_raw, EPS_PROB, 1.0 - EPS_PROB)
    a_raw = mu * kappa
    b_raw = (1.0 - mu) * kappa
    a = np.clip(a_raw, 1e-8, np.inf)
    b = np.clip(b_raw, 1e-8, np.inf)
    loglik = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1) + betaln(y + a, n - y + b) - betaln(a, b)
    d_a = (digamma(y + a) - digamma(n + a + b) - digamma(a) + digamma(a + b)) * (a_raw > 1e-8)
    d_b = (digamma(n - y + b) - digamma(n + a + b) - digamma(b) + digamma(a + b)) * (b_raw > 1e-8)
    active = (eta > -30.0) & (eta < 5.0) & (mu_raw > EPS_PROB) & (mu_raw < 1.0 - EPS_PROB)
    dmu_deta = np.where(active, e * np.exp(-e), 0.0)
    grad_beta = X.T @ (kappa * (d_a - d_b) * dmu_deta)
    grad_log_kappa = np.sum(kappa * (mu * d_a + (1.0 - mu) * d_b)) * (kappa_raw > 1e-6)
    return -float(np.sum(loglik)), -np.r_[grad_beta, grad_log_kappa]


def _newton_decrement(theta: np.ndarray, X: np.ndarray, y: np.ndarray, n: np.ndarray, free: np.ndarray,
                      h: float = 1e-5) -> float:
    """Curvature-scaled gradient diagnostic over the selected free parameters.

    The Hessian is obtained by central differences of the analytic gradient.
    The value is half the sum of squared gradient projections divided by the
    absolute Hessian eigenvalues; directions with absolute eigenvalue at most
    1e-10 are omitted. It is not a convergence certificate or a bound on the
    remaining likelihood improvement.
    """
    k = len(theta)
    _, g = _beta_binomial_nll_and_grad(theta, X, y, n)
    H = np.zeros((k, k))
    for j in range(k):
        e = np.zeros(k)
        e[j] = h
        H[:, j] = (_beta_binomial_nll_and_grad(theta + e, X, y, n)[1] - _beta_binomial_nll_and_grad(theta - e, X, y, n)[1]) / (2 * h)
    H = 0.5 * (H + H.T)[np.ix_(free, free)]
    lam, V = np.linalg.eigh(H)
    proj = V.T @ g[free]
    keep = np.abs(lam) > 1e-10
    return float(0.5 * np.sum(proj[keep] ** 2 / np.abs(lam[keep])))


def multistart_beta_binomial(X: np.ndarray, y: np.ndarray, n: np.ndarray, theta0: np.ndarray) -> dict[str, object]:
    """Beta-binomial refit with analytic gradients from three starting values of log(kappa).

    Each start is optimised with L-BFGS-B and then with a bounded trust-region method; the
    best solution is kept. The reported gradient and Newton decrement exclude log(kappa) when it
    lies on a bound.
    """
    bounds = [(None, None)] * X.shape[1] + [BETA_BINOMIAL_LOG_KAPPA_BOUNDS]
    best = None
    for start in (theta0, np.r_[theta0[:-1], 3.0], np.r_[theta0[:-1], 9.0]):
        r1 = minimize(_beta_binomial_nll_and_grad, start, args=(X, y, n), jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 50000, "ftol": 1e-15, "gtol": 1e-9, "maxcor": 30})
        with warnings.catch_warnings():
            # quasi-Newton update skipped when consecutive gradients coincide at the optimum
            warnings.filterwarnings("ignore", message="delta_grad == 0.0", category=UserWarning)
            r2 = minimize(_beta_binomial_nll_and_grad, r1.x, args=(X, y, n), jac=True, method="trust-constr", bounds=bounds,
                          options={"maxiter": 5000, "gtol": 1e-10, "xtol": 1e-15})
        for r in (r1, r2):
            if best is None or r.fun < best.fun:
                best = r
    f, g = _beta_binomial_nll_and_grad(best.x, X, y, n)
    at_bound = bool(np.isclose(best.x[-1], BETA_BINOMIAL_LOG_KAPPA_BOUNDS[0]) or np.isclose(best.x[-1], BETA_BINOMIAL_LOG_KAPPA_BOUNDS[1]))
    free = np.arange(len(g) - 1) if at_bound else np.arange(len(g))
    return {"params": best.x, "log_likelihood": -f, "optimizer_success": bool(best.success), "message": str(best.message),
            "max_abs_gradient": float(np.max(np.abs(g[free]))), "newton_decrement": _newton_decrement(best.x, X, y, n, free),
            "log_kappa_at_bound": at_bound}


def _refit(df: pd.DataFrame, family: str, include_protection: bool, include_kernel: bool,
           scale: float | None, model_name: str) -> dict[str, object]:
    try:
        return _refit_unchecked(df, family, include_protection, include_kernel, scale, model_name)
    except Exception as exc:  # recorded so that one failed candidate does not stop the check
        return {"error": str(exc), "original_success": False, "loglik_gain": np.nan,
                "aicc_original": np.nan, "aicc_refit": np.nan, "coef_original": np.nan, "coef_refit": np.nan}


def _refit_unchecked(df: pd.DataFrame, family: str, include_protection: bool, include_kernel: bool,
                     scale: float | None, model_name: str) -> dict[str, object]:
    fit, _, _ = fit_hazard_model(df, model_name, include_protection, include_kernel, family=family, local_scale_m=scale)
    X, info = build_design(df, include_protection=include_protection, include_kernel=include_kernel)
    y = df["new_infections"].to_numpy(float)
    n = df["at_risk"].to_numpy(float)
    if family == "binomial":
        res = fisher_scoring_binomial(X, y, n, fit.params)
        diag = {"refit_converged": res["converged"], "refit_max_abs_score": res["max_abs_score"],
                "refit_message": res["message"]}
    else:
        res = multistart_beta_binomial(X, y, n, fit.params)
        diag = {"refit_optimizer_success": res["optimizer_success"], "refit_max_abs_gradient": res["max_abs_gradient"],
                "refit_newton_decrement": res["newton_decrement"], "refit_log_kappa_at_bound": res["log_kappa_at_bound"], "refit_message": res["message"]}
    k = len(res["params"])
    out = {
        "original_success": bool(fit.success),
        "original_message": fit.message,
        "loglik_original": fit.log_likelihood,
        "loglik_refit": res["log_likelihood"],
        "loglik_gain": res["log_likelihood"] - fit.log_likelihood,
        "max_abs_coef_change": float(np.max(np.abs(np.asarray(res["params"])[:len(fit.params)] - fit.params))),
        "aicc_original": _aicc(fit.aic, fit.n_params, fit.n_obs),
        "aicc_refit": _aicc(2 * k - 2 * res["log_likelihood"], k, len(y)),
    }
    if include_kernel:
        pidx = info.feature_names.index("z_log_source_pressure")
        out["coef_original"] = float(fit.params[pidx])
        out["coef_refit"] = float(res["params"][pidx])
    out.update(diag)
    return out


def check_empirical_fits(interval: pd.DataFrame, geometry: pd.DataFrame, scales: list[float]) -> pd.DataFrame:
    """Refit the empirical model ladder (both families, with and without the kernel term)."""
    rows = []
    for family in ("binomial", "beta_binomial"):
        for name, prot in (("Year only", False), ("Year + N1", True)):
            r = _refit(interval, family, prot, False, None, name)
            rows.append({"family": family, "model": name, "candidate_scale_m": np.nan, **r})
        for scale in scales:
            df = attach_exposure(interval, geometry, local_scale_m=scale)
            r = _refit(df, family, True, True, scale, "Year + N1 + kernel")
            rows.append({"family": family, "model": "Year + N1 + kernel", "candidate_scale_m": scale, **r})
    return pd.DataFrame(rows)


def _finite_rows(profile: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    valid = pd.Series(True, index=profile.index)
    for column in columns:
        if column not in profile.columns:
            return pd.Series(False, index=profile.index)
        valid &= np.isfinite(pd.to_numeric(profile[column], errors="coerce"))
    return valid


def _select(profile: pd.DataFrame, aicc_col: str, coef_col: str, true_scale: float) -> dict[str, object]:
    valid = _finite_rows(profile, (aicc_col, coef_col, "candidate_scale_m"))
    ordered = profile.loc[valid].sort_values([aicc_col, "candidate_scale_m"])
    if ordered.empty:
        return {"selected_scale_m": np.nan, "min_aicc": np.nan, "exact_scale_recovered": False,
                "true_scale_in_delta_aicc_2_set": False, "delta_aicc_support_size": 0, "selected_pressure_positive": False}
    best = ordered.iloc[0]
    support = ordered.loc[ordered[aicc_col] <= ordered[aicc_col].min() + 2.0]
    return {
        "selected_scale_m": float(best["candidate_scale_m"]),
        "min_aicc": float(best[aicc_col]),
        "exact_scale_recovered": bool(np.isclose(best["candidate_scale_m"], true_scale)),
        "true_scale_in_delta_aicc_2_set": bool(np.any(np.isclose(support["candidate_scale_m"], true_scale))),
        "delta_aicc_support_size": int(len(support)),
        "selected_pressure_positive": bool(best[coef_col] > 0.0),
    }


def _compare(profile: pd.DataFrame, true_scale: float, unit: str) -> dict[str, object]:
    orig = _select(profile, "aicc_original", "coef_original", true_scale)
    new = _select(profile, "aicc_refit", "coef_refit", true_scale)
    valid = _finite_rows(profile, ("aicc_original", "aicc_refit", "coef_original", "coef_refit", "candidate_scale_m"))
    if "error" in profile:
        valid &= profile["error"].isna()
    out = {"analysis_unit": unit, "true_scale_m": float(true_scale),
           "n_candidates_failed": int((~valid).sum()),
           "n_candidates_original_flag_false": int((~profile["original_success"].astype(bool)).sum()),
           "max_loglik_gain": float(profile["loglik_gain"].max())}
    if "refit_converged" in profile:
        out["n_candidates_refit_not_converged"] = int((profile["refit_converged"] == False).sum())
        out["max_abs_score_refit"] = float(profile["refit_max_abs_score"].max())
    if "refit_max_abs_gradient" in profile:
        out["max_abs_gradient_refit"] = float(profile["refit_max_abs_gradient"].max())
        out["max_newton_decrement_refit"] = float(profile["refit_newton_decrement"].max())
        out["n_candidates_log_kappa_at_bound"] = int((profile["refit_log_kappa_at_bound"] == True).sum())
        out["n_candidates_refit_flag_false"] = int((profile["refit_optimizer_success"] == False).sum())
    for key in orig:
        out[f"{key}_original"] = orig[key]
        out[f"{key}_refit"] = new[key]
    # a replicate is compared only when every candidate scale was refitted
    out["comparable"] = bool(out["n_candidates_failed"] == 0 and np.isfinite(orig["selected_scale_m"])
                             and np.isfinite(new["selected_scale_m"]))
    out["selection_changed"] = bool(out["comparable"] and orig["selected_scale_m"] != new["selected_scale_m"])
    return out


def check_zone_profile(sim_zone: pd.DataFrame, geometry: pd.DataFrame, scales: list[float], true_scale: float) -> dict[str, object]:
    rows = []
    for scale in scales:
        df = attach_exposure(sim_zone, geometry, local_scale_m=scale)
        rows.append({"candidate_scale_m": scale,
                     **_refit(df, "beta_binomial", True, True, scale, "sim_zone_beta_binomial_N1_kernel")})
    return _compare(pd.DataFrame(rows), true_scale, "zone_level_aggregate")


def check_tree_profile(tree_year: pd.DataFrame, panel: pd.DataFrame, scales: list[float], true_scale: float,
                       grouped_design) -> dict[str, object]:
    rows = []
    for scale in scales:
        df = grouped_design(tree_year, panel, scale)
        rows.append({"candidate_scale_m": scale,
                     **_refit(df, "binomial", True, True, scale, "sim_tree_grouped_binomial_N1_kernel")})
    return _compare(pd.DataFrame(rows), true_scale, "tree_level_grouped")


def summarize_optimizer_check(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (true_scale, unit), g_all in rows.groupby(["true_scale_m", "analysis_unit"], sort=True):
        g = g_all.loc[g_all["comparable"].astype(bool)]
        rec = {"true_scale_m": true_scale, "analysis_unit": unit, "n_replicates": int(len(g_all)),
               "n_replicates_compared": int(len(g)),
               "n_selection_changed": int(g["selection_changed"].sum()),
               "max_loglik_gain": float(g["max_loglik_gain"].max())}
        for col in ("n_candidates_refit_not_converged", "n_candidates_refit_flag_false", "n_candidates_log_kappa_at_bound"):
            if col in g and g[col].notna().any():
                rec[col] = int(g[col].sum())
        for col in ("max_abs_score_refit", "max_abs_gradient_refit", "max_newton_decrement_refit"):
            if col in g and g[col].notna().any():
                rec[col] = float(g[col].max())
        for metric in ["exact_scale_recovered", "true_scale_in_delta_aicc_2_set", "selected_pressure_positive"]:
            rec[f"{metric}_rate_original"] = float(g[f"{metric}_original"].mean())
            rec[f"{metric}_rate_refit"] = float(g[f"{metric}_refit"].mean())
        rec["median_support_size_original"] = float(g["delta_aicc_support_size_original"].median())
        rec["median_support_size_refit"] = float(g["delta_aicc_support_size_refit"].median())
        out.append(rec)
    return pd.DataFrame(out)
