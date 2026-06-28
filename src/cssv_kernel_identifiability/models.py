from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, betaln


EPS = 1e-10


def cloglog_inv(eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    eta_clip = np.clip(eta, -30.0, 5.0)
    p = 1.0 - np.exp(-np.exp(eta_clip))
    return np.clip(p, EPS, 1.0 - EPS)


def binomial_logpmf(y: np.ndarray, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return gammaln(n + 1.0) - gammaln(y + 1.0) - gammaln(n - y + 1.0) + y * np.log(p) + (n - y) * np.log1p(-p)


def beta_binomial_logpmf(y: np.ndarray, n: np.ndarray, mu: np.ndarray, kappa: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), EPS, 1.0 - EPS)
    kappa = max(float(kappa), 1e-6)
    alpha = np.clip(mu * kappa, 1e-8, np.inf)
    beta = np.clip((1.0 - mu) * kappa, 1e-8, np.inf)
    return (
        gammaln(n + 1.0)
        - gammaln(y + 1.0)
        - gammaln(n - y + 1.0)
        + betaln(y + alpha, n - y + beta)
        - betaln(alpha, beta)
    )


@dataclass
class DesignInfo:
    include_protection: bool
    include_kernel: bool
    kernel_mean: float | None
    kernel_sd: float | None
    feature_names: list[str]


@dataclass
class FitResult:
    model_name: str
    family: str
    feature_names: list[str]
    params: np.ndarray
    success: bool
    message: str
    log_likelihood: float
    n_obs: int
    n_trials: int
    n_params: int
    aic: float
    bic: float
    design_info: DesignInfo
    local_scale_m: float | None = None
    jump_weight: float = 0.0
    jump_scale_m: float = 50.0

    @property
    def kappa(self) -> float | None:
        if self.family != "beta_binomial":
            return None
        return float(np.exp(self.params[-1]))

    @property
    def intra_cluster_correlation(self) -> float | None:
        if self.kappa is None:
            return None
        return float(1.0 / (self.kappa + 1.0))

    def coefficient_frame(self) -> pd.DataFrame:
        names = list(self.feature_names)
        params = list(self.params[: len(names)])
        rows = [
            {
                "model_name": self.model_name,
                "family": self.family,
                "term": name,
                "estimate": float(value),
                "local_scale_m": self.local_scale_m,
                "jump_weight": self.jump_weight,
                "jump_scale_m": self.jump_scale_m,
            }
            for name, value in zip(names, params)
        ]
        if self.family == "beta_binomial":
            rows.append(
                {
                    "model_name": self.model_name,
                    "family": self.family,
                    "term": "log_kappa_overdispersion",
                    "estimate": float(self.params[-1]),
                    "local_scale_m": self.local_scale_m,
                    "jump_weight": self.jump_weight,
                    "jump_scale_m": self.jump_scale_m,
                }
            )
            rows.append(
                {
                    "model_name": self.model_name,
                    "family": self.family,
                    "term": "intracluster_correlation_derived",
                    "estimate": self.intra_cluster_correlation,
                    "local_scale_m": self.local_scale_m,
                    "jump_weight": self.jump_weight,
                    "jump_scale_m": self.jump_scale_m,
                }
            )
        return pd.DataFrame(rows)

    def summary_row(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "family": self.family,
            "success": self.success,
            "message": self.message,
            "log_likelihood": self.log_likelihood,
            "n_obs": self.n_obs,
            "n_trials": self.n_trials,
            "n_params": self.n_params,
            "aic": self.aic,
            "bic": self.bic,
            "local_scale_m": self.local_scale_m,
            "jump_weight": self.jump_weight,
            "jump_scale_m": self.jump_scale_m,
            "kappa": self.kappa,
            "intracluster_correlation": self.intra_cluster_correlation,
        }


def build_design(
    df: pd.DataFrame,
    include_protection: bool,
    include_kernel: bool,
    design_info: DesignInfo | None = None,
) -> tuple[np.ndarray, DesignInfo]:
    X = pd.DataFrame(index=df.index)
    X["intercept"] = 1.0
    years = sorted(pd.to_numeric(df["year"], errors="raise").astype(int).unique())
    # Fixed 1-7 coding keeps train/test designs compatible in cross-validation.
    for year in range(2, 8):
        X[f"year_{year}"] = (pd.to_numeric(df["year"]).astype(int) == year).astype(float)

    kernel_mean = None
    kernel_sd = None
    if include_protection:
        X["N1_protected"] = df["is_N1_protected"].astype(float)

    if include_kernel:
        if "log_kernel_source_pressure" not in df.columns:
            raise ValueError("include_kernel=True requires a log_kernel_source_pressure column")
        raw = pd.to_numeric(df["log_kernel_source_pressure"], errors="raise").astype(float)
        if design_info is None or design_info.kernel_mean is None or design_info.kernel_sd is None:
            kernel_mean = float(raw.mean())
            kernel_sd = float(raw.std(ddof=0))
            if kernel_sd <= 0 or not np.isfinite(kernel_sd):
                kernel_sd = 1.0
        else:
            kernel_mean = float(design_info.kernel_mean)
            kernel_sd = float(design_info.kernel_sd)
        X["z_log_source_pressure"] = (raw - kernel_mean) / kernel_sd

    feature_names = list(X.columns)
    if design_info is not None:
        # Keep exact training order. Missing columns are zeros; extra columns are discarded.
        for c in design_info.feature_names:
            if c not in X.columns:
                X[c] = 0.0
        X = X[design_info.feature_names]
        feature_names = design_info.feature_names
        kernel_mean = design_info.kernel_mean
        kernel_sd = design_info.kernel_sd

    info = DesignInfo(
        include_protection=include_protection,
        include_kernel=include_kernel,
        kernel_mean=kernel_mean,
        kernel_sd=kernel_sd,
        feature_names=feature_names,
    )
    return X.to_numpy(float), info


def _nll_binomial(beta: np.ndarray, X: np.ndarray, y: np.ndarray, n: np.ndarray) -> float:
    p = cloglog_inv(X @ beta)
    return -float(np.sum(binomial_logpmf(y, n, p)))


def _nll_beta_binomial(theta: np.ndarray, X: np.ndarray, y: np.ndarray, n: np.ndarray) -> float:
    beta = theta[:-1]
    log_kappa = float(theta[-1])
    p = cloglog_inv(X @ beta)
    return -float(np.sum(beta_binomial_logpmf(y, n, p, np.exp(log_kappa))))


def fit_hazard_model(
    df: pd.DataFrame,
    model_name: str,
    include_protection: bool,
    include_kernel: bool,
    family: str = "binomial",
    design_info: DesignInfo | None = None,
    local_scale_m: float | None = None,
    jump_weight: float = 0.0,
    jump_scale_m: float = 50.0,
) -> tuple[FitResult, np.ndarray, DesignInfo]:
    if family not in {"binomial", "beta_binomial"}:
        raise ValueError("family must be 'binomial' or 'beta_binomial'")
    X, info = build_design(df, include_protection=include_protection, include_kernel=include_kernel, design_info=design_info)
    y = pd.to_numeric(df["new_infections"], errors="raise").to_numpy(float)
    n = pd.to_numeric(df["at_risk"], errors="raise").to_numpy(float)
    if np.any(y < 0) or np.any(n < 0) or np.any(y > n):
        raise ValueError("Invalid interval counts: require 0 <= new_infections <= at_risk")

    # Robust starting values from the marginal interval hazards.
    p0 = np.clip((y.sum() + 0.5) / (n.sum() + 1.0), 1e-5, 1 - 1e-5)
    intercept0 = np.log(-np.log1p(-p0))
    beta0 = np.zeros(X.shape[1], dtype=float)
    beta0[0] = intercept0

    if family == "binomial":
        obj = lambda b: _nll_binomial(b, X, y, n)
        res = minimize(obj, beta0, method="BFGS", options={"maxiter": 2000, "gtol": 1e-6})
        params = np.asarray(res.x, dtype=float)
        log_lik = -float(obj(params))
    else:
        theta0 = np.r_[beta0, 6.0]  # kappa ~= 403; close to binomial but estimable.
        obj = lambda th: _nll_beta_binomial(th, X, y, n)
        bounds = [(None, None)] * X.shape[1] + [(-4.0, 12.0)]
        res = minimize(obj, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000, "ftol": 1e-9})
        params = np.asarray(res.x, dtype=float)
        log_lik = -float(obj(params))

    n_params = len(params)
    n_obs = int(len(y))
    n_trials = int(np.sum(n))
    aic = 2 * n_params - 2 * log_lik
    bic = np.log(max(n_obs, 1)) * n_params - 2 * log_lik
    fit = FitResult(
        model_name=model_name,
        family=family,
        feature_names=info.feature_names,
        params=params,
        success=bool(res.success),
        message=str(res.message),
        log_likelihood=log_lik,
        n_obs=n_obs,
        n_trials=n_trials,
        n_params=n_params,
        aic=float(aic),
        bic=float(bic),
        design_info=info,
        local_scale_m=local_scale_m,
        jump_weight=jump_weight,
        jump_scale_m=jump_scale_m,
    )
    return fit, X, info


def predict_prob(fit: FitResult, X: np.ndarray) -> np.ndarray:
    beta = fit.params[: len(fit.feature_names)]
    return cloglog_inv(X @ beta)


def prediction_frame(df: pd.DataFrame, fit: FitResult, X: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    p = predict_prob(fit, X)
    out["model_name"] = fit.model_name
    out["family"] = fit.family
    out["predicted_interval_hazard"] = p
    out["expected_new_infections_given_observed_history"] = out["at_risk"].astype(float) * p

    seq_rows: list[pd.DataFrame] = []
    for key, g in out.groupby("treatment_zone", sort=True):
        g = g.sort_values("year").copy()
        prev = 0.0
        pred_cum = []
        pred_new = []
        for _, r in g.iterrows():
            denom = float(r["denominator_used"])
            at_risk_pred = max(denom - prev, 0.0)
            new = at_risk_pred * float(r["predicted_interval_hazard"])
            prev = min(prev + new, denom)
            pred_new.append(new)
            pred_cum.append(prev)
        g["dynamic_predicted_new_infections"] = pred_new
        g["dynamic_predicted_cumulative_infections"] = pred_cum
        g["dynamic_predicted_cumulative_fraction"] = g["dynamic_predicted_cumulative_infections"] / g["denominator_used"].astype(float)
        seq_rows.append(g)
    return pd.concat(seq_rows, ignore_index=True)


def heldout_log_likelihood(fit: FitResult, X: np.ndarray, df: pd.DataFrame) -> float:
    y = pd.to_numeric(df["new_infections"], errors="raise").to_numpy(float)
    n = pd.to_numeric(df["at_risk"], errors="raise").to_numpy(float)
    p = predict_prob(fit, X)
    if fit.family == "binomial":
        return float(np.sum(binomial_logpmf(y, n, p)))
    assert fit.family == "beta_binomial"
    return float(np.sum(beta_binomial_logpmf(y, n, p, fit.kappa if fit.kappa is not None else 1e6)))


def fit_and_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    include_protection: bool,
    include_kernel: bool,
    family: str,
    local_scale_m: float | None = None,
) -> dict[str, object]:
    fit, _, info = fit_hazard_model(
        train_df,
        model_name=model_name,
        include_protection=include_protection,
        include_kernel=include_kernel,
        family=family,
        local_scale_m=local_scale_m,
    )
    X_test, _ = build_design(test_df, include_protection=include_protection, include_kernel=include_kernel, design_info=info)
    return {
        "model_name": model_name,
        "family": family,
        "heldout_log_likelihood": heldout_log_likelihood(fit, X_test, test_df),
        "train_log_likelihood": fit.log_likelihood,
        "n_test_rows": len(test_df),
        "local_scale_m": local_scale_m,
        "success": fit.success,
    }
