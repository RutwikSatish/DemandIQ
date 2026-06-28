"""
DemandIQ — Demand Sensing & Forecast Accuracy Engine
Computation utilities: all metric formulas, model fitting, and derived calculations.

Sources:
  FPP3   — Hyndman & Athanasopoulos, Forecasting: Principles and Practice, 3rd ed. (OTexts, 2021)
  DFBP   — Vandeput, Demand Forecasting Best Practices (Manning, 2023)
  FDPF   — Jain, Fundamentals of Demand Planning & Forecasting (Graceway, 2020)
  DMBP   — Crum & Palmatier, Demand Management Best Practices (J. Ross Publishing)
"""

import numpy as np
import pandas as pd
from itertools import product
import streamlit as st

# ─── SAMPLE DATA ────────────────────────────────────────────────────────────

SAMPLE_DATA = pd.DataFrame({
    "Period": list(range(1, 25)),
    "SKU": ["Component-A"] * 24,
    "Demand": [
        380, 420, 390, 510, 480, 440, 395, 620, 580, 430, 410, 490,
        400, 445, 415, 535, 505, 460, 415, 645, 600, 450, 430, 515
    ]
})

# ─── DEMAND PROFILE ─────────────────────────────────────────────────────────

def compute_profile(demand: np.ndarray) -> dict:
    """Summary statistics for demand array. Source: DFBP Ch. 13."""
    n = len(demand)
    total = float(np.sum(demand))
    mean = total / n
    std = float(np.std(demand, ddof=1)) if n > 1 else 0.0
    cv = std / mean if mean > 0 else 0.0
    return {"n": n, "total": total, "mean": mean, "std": std, "cv": cv}


def classify_cv(cv: float) -> dict:
    """
    ABC-XYZ demand pattern classification by CV.
    Source: Vandeput, DFBP Ch. 13.
    """
    if cv < 0.2:
        return {
            "label": "X — Stable",
            "description": "Stable demand — statistical forecasting is highly reliable",
            "color": "#16a34a",
            "category": "X"
        }
    elif cv <= 0.5:
        return {
            "label": "Y — Variable",
            "description": "Variable demand — use adaptive methods with caution",
            "color": "#d97706",
            "category": "Y"
        }
    else:
        return {
            "label": "Z — Lumpy",
            "description": "Highly variable / lumpy — consider Croston method or manual override",
            "color": "#dc2626",
            "category": "Z"
        }


def compute_trend_slope(demand: np.ndarray, periods: int = 3) -> dict:
    """Compute slope over last N periods for S&OP trend direction."""
    if len(demand) < periods:
        return {"slope": 0.0, "pct": 0.0, "direction": "Flat demand"}
    recent = demand[-periods:]
    x = np.arange(periods, dtype=float)
    slope = float(np.polyfit(x, recent, 1)[0])
    base = float(recent[0]) if recent[0] != 0 else 1.0
    pct = (slope * periods / base) * 100
    if slope > 0.5:
        direction = f"Upward trend (+{pct:.1f}% over last {periods} periods)"
    elif slope < -0.5:
        direction = f"Downward trend ({pct:.1f}% over last {periods} periods)"
    else:
        direction = "Flat demand"
    return {"slope": slope, "pct": pct, "direction": direction}


# ─── SEASONALITY DETECTION ───────────────────────────────────────────────────

@st.cache_data
def run_seasonal_decompose(demand: tuple, period: int = 12):
    """
    Additive seasonal decomposition.
    Source: FPP3 Ch. 3.
    Returns statsmodels DecomposeResult or None on failure.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    arr = np.array(demand, dtype=float)
    if len(arr) < 2 * period:
        return None
    try:
        result = seasonal_decompose(arr, model="additive", period=period, extrapolate_trend="freq")
        return result
    except Exception:
        return None


def detect_seasonality(decomp_result) -> dict:
    """
    Flag seasonal pattern if seasonal std > 10% of trend std.
    Source: FPP3 Ch. 3.
    """
    if decomp_result is None:
        return {"detected": False, "message": "Insufficient data for decomposition (requires 24+ periods)."}
    seasonal_std = float(np.nanstd(decomp_result.seasonal))
    trend_std = float(np.nanstd(decomp_result.trend))
    if trend_std == 0:
        return {"detected": False, "message": "No significant seasonality detected. Non-seasonal models are appropriate."}
    if seasonal_std > 0.10 * trend_std:
        return {
            "detected": True,
            "message": "Seasonal pattern detected (estimated period: 12). Use Holt-Winters or seasonal ARIMA."
        }
    return {"detected": False, "message": "No significant seasonality detected. Non-seasonal models are appropriate."}


# ─── FORECASTING MODELS ──────────────────────────────────────────────────────

@st.cache_data
def fit_naive(demand: tuple, n_train: int) -> dict:
    """
    Naive forecast: F_{t+1} = D_t.
    Source: FPP3 Ch. 5.2.
    """
    arr = np.array(demand, dtype=float)
    n = len(arr)
    fitted = np.full(n, np.nan)
    for t in range(1, n):
        fitted[t] = arr[t - 1]
    holdout_actual = arr[n_train:]
    holdout_forecast = fitted[n_train:]
    return {
        "method": "Naive",
        "params": "F_{t+1} = D_{t}",
        "fitted": fitted,
        "holdout_actual": holdout_actual,
        "holdout_forecast": holdout_forecast,
        "model_obj": None
    }


@st.cache_data
def fit_sma(demand: tuple, n_train: int, k: int = 3) -> dict:
    """
    Simple Moving Average: F_{t+1} = mean(D_{t-k+1}..D_t).
    Source: Jain, FDPF — Statistical Methods chapter.
    """
    arr = np.array(demand, dtype=float)
    n = len(arr)
    fitted = np.full(n, np.nan)
    for t in range(k, n):
        fitted[t] = np.mean(arr[t - k:t])
    holdout_actual = arr[n_train:]
    holdout_forecast = fitted[n_train:]
    return {
        "method": f"SMA(k={k})",
        "params": f"k={k}",
        "fitted": fitted,
        "holdout_actual": holdout_actual,
        "holdout_forecast": holdout_forecast,
        "model_obj": None,
        "k": k
    }


@st.cache_data
def fit_ses(demand: tuple, n_train: int) -> dict:
    """
    Simple Exponential Smoothing with MLE-optimized alpha.
    Source: FPP3 Ch. 8.1.
    """
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    arr = np.array(demand, dtype=float)
    train = arr[:n_train]
    try:
        model = SimpleExpSmoothing(train).fit(optimized=True)
        alpha = float(model.params["smoothing_level"])
        fitted_train = np.array(model.fittedvalues)
        # Generate holdout one-step-ahead forecasts by recursive application
        n_holdout = len(arr) - n_train
        holdout_fc = []
        last_level = float(model.level[-1])
        for i in range(n_holdout):
            fc = last_level
            holdout_fc.append(fc)
            actual = arr[n_train + i]
            last_level = alpha * actual + (1 - alpha) * last_level
        holdout_forecast = np.array(holdout_fc)
        holdout_actual = arr[n_train:]
        fitted = np.concatenate([fitted_train, holdout_forecast])
        return {
            "method": "SES",
            "params": f"alpha={alpha:.3f}",
            "alpha": alpha,
            "fitted": fitted,
            "holdout_actual": holdout_actual,
            "holdout_forecast": holdout_forecast,
            "model_obj": model
        }
    except Exception as e:
        return _failed_result("SES", str(e))


@st.cache_data
def fit_holt(demand: tuple, n_train: int) -> dict:
    """
    Holt's Linear Exponential Smoothing (trend, no seasonality).
    Source: FPP3 Ch. 8.2.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    arr = np.array(demand, dtype=float)
    train = arr[:n_train]
    try:
        model = ExponentialSmoothing(
            train, trend="add", seasonal=None,
            initialization_method="heuristic"
        ).fit(optimized=True)
        alpha = float(model.params["smoothing_level"])
        beta = float(model.params["smoothing_trend"])
        fitted_train = np.array(model.fittedvalues)
        n_holdout = len(arr) - n_train
        fc_steps = model.forecast(n_holdout)
        holdout_forecast = np.array(fc_steps)
        holdout_actual = arr[n_train:]
        fitted = np.concatenate([fitted_train, holdout_forecast])
        return {
            "method": "Holt Linear",
            "params": f"alpha={alpha:.3f}, beta={beta:.3f}",
            "alpha": alpha,
            "beta": beta,
            "fitted": fitted,
            "holdout_actual": holdout_actual,
            "holdout_forecast": holdout_forecast,
            "model_obj": model
        }
    except Exception as e:
        return _failed_result("Holt Linear", str(e))


@st.cache_data
def fit_holtwinters(demand: tuple, n_train: int, seasonal_periods: int = 12) -> dict:
    """
    Holt-Winters Triple Exponential Smoothing (additive trend + additive seasonality).
    Only valid when len(demand) >= 2 * seasonal_periods.
    Source: FPP3 Ch. 8.5.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    arr = np.array(demand, dtype=float)
    n = len(arr)
    if n < 2 * seasonal_periods:
        return _failed_result(
            "Holt-Winters",
            f"Insufficient data for seasonal model (requires {2 * seasonal_periods}+ periods)."
        )
    train = arr[:n_train]
    if len(train) < 2 * seasonal_periods:
        return _failed_result(
            "Holt-Winters",
            f"Training set too small for seasonal model (requires {2 * seasonal_periods}+ periods)."
        )
    try:
        model = ExponentialSmoothing(
            train, trend="add", seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="heuristic"
        ).fit(optimized=True)
        alpha = float(model.params["smoothing_level"])
        beta = float(model.params["smoothing_trend"])
        gamma = float(model.params["smoothing_seasonal"])
        fitted_train = np.array(model.fittedvalues)
        n_holdout = len(arr) - n_train
        holdout_forecast = np.array(model.forecast(n_holdout))
        holdout_actual = arr[n_train:]
        fitted = np.concatenate([fitted_train, holdout_forecast])
        return {
            "method": "Holt-Winters",
            "params": f"alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}",
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "fitted": fitted,
            "holdout_actual": holdout_actual,
            "holdout_forecast": holdout_forecast,
            "model_obj": model
        }
    except Exception as e:
        return _failed_result("Holt-Winters", str(e))


@st.cache_data
def fit_arima(demand: tuple, n_train: int) -> dict:
    """
    ARIMA with AICc-based order selection over p in [0,1,2], d in [0,1], q in [0,1,2].
    AICc = AIC + (2k^2 + 2k) / (n - k - 1).
    Source: FPP3 Ch. 9.
    """
    from statsmodels.tsa.arima.model import ARIMA
    arr = np.array(demand, dtype=float)
    train = arr[:n_train]
    n = len(train)
    best_aicc = np.inf
    best_order = (1, 1, 1)
    best_model = None
    for p, d, q in product(range(3), range(2), range(3)):
        try:
            res = ARIMA(train, order=(p, d, q)).fit()
            aic = res.aic
            k = p + q + d + 1  # approx number of free params
            if n - k - 1 > 0:
                aicc = aic + (2 * k ** 2 + 2 * k) / (n - k - 1)
            else:
                aicc = np.inf
            if aicc < best_aicc:
                best_aicc = aicc
                best_order = (p, d, q)
                best_model = res
        except Exception:
            continue
    if best_model is None:
        return _failed_result("ARIMA", "No ARIMA order converged.")
    try:
        n_holdout = len(arr) - n_train
        fc = best_model.forecast(steps=n_holdout)
        holdout_forecast = np.array(fc)
        holdout_actual = arr[n_train:]
        fitted_train = np.array(best_model.fittedvalues)
        fitted = np.concatenate([fitted_train, holdout_forecast])
        return {
            "method": f"ARIMA{best_order}",
            "params": f"order={best_order}, AICc={best_aicc:.1f}",
            "order": best_order,
            "aicc": best_aicc,
            "fitted": fitted,
            "holdout_actual": holdout_actual,
            "holdout_forecast": holdout_forecast,
            "model_obj": best_model
        }
    except Exception as e:
        return _failed_result("ARIMA", str(e))


def _failed_result(method: str, reason: str) -> dict:
    return {
        "method": method,
        "params": "N/A",
        "fitted": None,
        "holdout_actual": None,
        "holdout_forecast": None,
        "model_obj": None,
        "error": reason
    }


# ─── ACCURACY METRICS ────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    """
    Compute MAE, Bias, Vandeput Score, WMAPE, RMSE, FA%, MAPE.

    Sources:
      MAE, Bias, Vandeput Score, WMAPE — Vandeput, DFBP Ch. 8–9
      RMSE — FPP3 Ch. 5.8
      FA% — Jain, FDPF
      MAPE — displayed for reference only (DFBP Ch. 8 caution)
    """
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)
    # Drop NaN pairs
    mask = ~(np.isnan(actual) | np.isnan(forecast))
    actual = actual[mask]
    forecast = forecast[mask]
    if len(actual) == 0:
        return {}

    errors = actual - forecast
    abs_errors = np.abs(errors)
    n = len(errors)

    mae = float(np.mean(abs_errors))
    bias = float(np.mean(errors))
    vandeput_score = mae + abs(bias)
    sum_actual = float(np.sum(actual))
    wmape = float(np.sum(abs_errors) / sum_actual) if sum_actual > 0 else np.nan
    fa_pct = (1 - wmape) * 100 if not np.isnan(wmape) else np.nan
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE — only where actual > 0
    nonzero_mask = actual > 0
    excluded_zero = int(np.sum(~nonzero_mask))
    if np.any(nonzero_mask):
        mape = float(np.mean(abs_errors[nonzero_mask] / actual[nonzero_mask]) * 100)
    else:
        mape = np.nan

    bias_direction = (
        f"Under-forecasting by {abs(bias):.1f} units on average"
        if bias > 0
        else f"Over-forecasting by {abs(bias):.1f} units on average"
    )

    return {
        "n": n,
        "mae": mae,
        "bias": bias,
        "bias_direction": bias_direction,
        "vandeput_score": vandeput_score,
        "wmape": wmape,
        "fa_pct": fa_pct,
        "rmse": rmse,
        "mape": mape,
        "mape_excluded_zero": excluded_zero,
    }


def detect_bias_trend(errors: np.ndarray, window: int = 3) -> dict:
    """
    Flag if last N errors all have the same sign.
    Source: Jain, FDPF — Monitoring and Revising Forecasts.
    """
    if len(errors) < window:
        return {"detected": False}
    recent = errors[-window:]
    if np.all(recent > 0):
        return {
            "detected": True,
            "direction": "under-forecasting",
            "n": window,
            "message": (
                f"Systematic Bias Detected — your forecast has been under-forecasting for "
                f"{window} consecutive periods. Review your demand assumptions and consider "
                "adjusting your baseline before the next S&OP cycle."
            )
        }
    elif np.all(recent < 0):
        return {
            "detected": True,
            "direction": "over-forecasting",
            "n": window,
            "message": (
                f"Systematic Bias Detected — your forecast has been over-forecasting for "
                f"{window} consecutive periods. Review your demand assumptions and consider "
                "adjusting your baseline before the next S&OP cycle."
            )
        }
    return {"detected": False}


# ─── FORWARD FORECAST ────────────────────────────────────────────────────────

@st.cache_data
def generate_forward_forecast(
    demand: tuple,
    method_key: str,
    horizon: int,
    k: int = 3,
    seasonal_periods: int = 12
) -> dict:
    """
    Refit the selected model on the full dataset and produce H-period forward forecast
    with 80% and 95% prediction intervals.
    Source: FPP3 Ch. 5.5 — sigma_h = RMSE * sqrt(h).
    """
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA

    arr = np.array(demand, dtype=float)
    n = len(arr)

    point_fc = np.full(horizon, np.nan)
    rmse_proxy = None

    try:
        if method_key == "Naive":
            last_val = arr[-1]
            point_fc = np.full(horizon, last_val)
            residuals = np.diff(arr)
            rmse_proxy = float(np.sqrt(np.mean(residuals ** 2)))

        elif method_key.startswith("SMA"):
            window = np.mean(arr[-k:])
            point_fc = np.full(horizon, window)
            # use rolling MAE as proxy for sigma
            fitted = np.array([np.mean(arr[max(0, i - k):i]) for i in range(1, n + 1)])
            errors = arr[k:] - fitted[k - 1:n - 1]
            rmse_proxy = float(np.sqrt(np.mean(errors ** 2))) if len(errors) > 0 else float(np.std(arr))

        elif method_key == "SES":
            model = SimpleExpSmoothing(arr).fit(optimized=True)
            point_fc = np.array(model.forecast(horizon))
            rmse_proxy = float(np.sqrt(np.mean(np.array(model.resid) ** 2)))

        elif method_key == "Holt Linear":
            model = ExponentialSmoothing(
                arr, trend="add", seasonal=None,
                initialization_method="heuristic"
            ).fit(optimized=True)
            point_fc = np.array(model.forecast(horizon))
            rmse_proxy = float(np.sqrt(np.mean(np.array(model.resid) ** 2)))

        elif method_key == "Holt-Winters":
            if n < 2 * seasonal_periods:
                return {"error": f"Insufficient data for Holt-Winters (requires {2 * seasonal_periods}+ periods)."}
            model = ExponentialSmoothing(
                arr, trend="add", seasonal="add",
                seasonal_periods=seasonal_periods,
                initialization_method="heuristic"
            ).fit(optimized=True)
            point_fc = np.array(model.forecast(horizon))
            rmse_proxy = float(np.sqrt(np.mean(np.array(model.resid) ** 2)))

        elif method_key.startswith("ARIMA"):
            # Re-run AICc selection on full data
            best_aicc = np.inf
            best_order = (1, 1, 1)
            best_res = None
            for p, d, q in product(range(3), range(2), range(3)):
                try:
                    res = ARIMA(arr, order=(p, d, q)).fit()
                    k_p = p + q + d + 1
                    aic = res.aic
                    if n - k_p - 1 > 0:
                        aicc = aic + (2 * k_p ** 2 + 2 * k_p) / (n - k_p - 1)
                    else:
                        aicc = np.inf
                    if aicc < best_aicc:
                        best_aicc = aicc
                        best_order = (p, d, q)
                        best_res = res
                except Exception:
                    continue
            if best_res is None:
                return {"error": "ARIMA fitting failed on full dataset."}
            point_fc = np.array(best_res.forecast(steps=horizon))
            rmse_proxy = float(np.sqrt(np.mean(np.array(best_res.resid) ** 2)))

    except Exception as e:
        return {"error": str(e)}

    if rmse_proxy is None or rmse_proxy == 0:
        rmse_proxy = float(np.std(arr)) if np.std(arr) > 0 else 1.0

    h_arr = np.arange(1, horizon + 1, dtype=float)
    sigma_h = rmse_proxy * np.sqrt(h_arr)

    return {
        "point": point_fc,
        "lower_80": point_fc - 1.282 * sigma_h,
        "upper_80": point_fc + 1.282 * sigma_h,
        "lower_95": point_fc - 1.960 * sigma_h,
        "upper_95": point_fc + 1.960 * sigma_h,
        "rmse_proxy": rmse_proxy
    }


# ─── SAFETY STOCK ────────────────────────────────────────────────────────────

def compute_safety_stock(
    sigma_d: float,
    lead_time: int,
    service_level_pct: float,
    mean_demand: float
) -> dict:
    """
    Safety stock for normally distributed demand.
    SS = Z * sigma_D * sqrt(L)
    ROP = D_bar * L + SS
    Source: Jain, FDPF.
    """
    z_map = {90: 1.282, 95: 1.645, 99: 2.326}
    z = z_map.get(int(service_level_pct), 1.645)
    sigma_lt = sigma_d * np.sqrt(lead_time)
    ss = z * sigma_lt
    rop = mean_demand * lead_time + ss
    return {
        "z": z,
        "sigma_d": sigma_d,
        "sigma_lt": sigma_lt,
        "ss": ss,
        "rop": rop,
        "lead_time": lead_time,
        "service_level": service_level_pct
    }


# ─── RESULTS TABLE BUILDER ───────────────────────────────────────────────────

def build_results_table(results_list: list) -> pd.DataFrame:
    """Assemble ranked comparison table sorted by Vandeput Score (ascending)."""
    rows = []
    for r in results_list:
        if r.get("error") or r.get("holdout_forecast") is None:
            continue
        actual = r["holdout_actual"]
        forecast = r["holdout_forecast"]
        if actual is None or forecast is None:
            continue
        valid = ~(np.isnan(actual) | np.isnan(forecast))
        if not np.any(valid):
            continue
        m = compute_metrics(actual[valid], forecast[valid])
        if not m:
            continue
        mape_display = f"{m['mape']:.1f}%" if not np.isnan(m.get("mape", np.nan)) else "N/A"
        rows.append({
            "Method": r["method"],
            "Parameters": r.get("params", ""),
            "Vandeput Score": round(m["vandeput_score"], 2),
            "MAE": round(m["mae"], 2),
            "Bias": round(m["bias"], 2),
            "WMAPE": f"{m['wmape'] * 100:.1f}%" if not np.isnan(m.get("wmape", np.nan)) else "N/A",
            "FA%": round(m["fa_pct"], 1) if not np.isnan(m.get("fa_pct", np.nan)) else np.nan,
            "RMSE": round(m["rmse"], 2),
            "MAPE": mape_display,
            "_metrics": m
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("Vandeput Score").reset_index(drop=True)
    return df


# ─── S&OP SUMMARY TEXT ───────────────────────────────────────────────────────

def generate_sopp_summary(
    profile: dict,
    cv_class: dict,
    seasonality: dict,
    trend: dict,
    best_method_name: str,
    best_metrics: dict,
    forecast_table: pd.DataFrame,
    ss_result: dict | None,
    alert_counts: dict | None
) -> str:
    """
    Generate plain-text S&OP executive summary.
    Source: Crum & Palmatier, DMBP; Jain, FDPF.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("DemandIQ — S&OP EXECUTIVE SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    # Section 1
    lines.append("SECTION 1: DEMAND SIGNAL")
    lines.append(f"  Demand Pattern: {cv_class['label']}  |  CV: {profile['cv'] * 100:.1f}%")
    lines.append(f"  Trend Direction: {trend['direction']}")
    lines.append(f"  Seasonality: {'Detected' if seasonality['detected'] else 'Not detected'}")
    lines.append("")

    # Section 2
    lines.append("SECTION 2: RECOMMENDED FORECAST METHOD")
    lines.append(f"  Method: {best_method_name}")
    lines.append(f"  Vandeput Score: {best_metrics.get('vandeput_score', 'N/A'):.2f} units (MAE + |Bias|)")
    fa = best_metrics.get("fa_pct", np.nan)
    lines.append(f"  Forecast Accuracy: {fa:.1f}%" if not np.isnan(fa) else "  Forecast Accuracy: N/A")
    lines.append(f"  Bias Direction: {best_metrics.get('bias_direction', 'N/A')}")
    # Rationale
    cat = cv_class["category"]
    if cat == "X":
        rationale = "Stable demand (CV < 0.2) supports statistical methods; SES or ARIMA are efficient choices (FPP3 Ch. 8.1)."
    elif cat == "Y":
        rationale = "Variable demand (0.2 <= CV <= 0.5) warrants adaptive smoothing — Holt or SES with careful monitoring (DFBP Ch. 13)."
    else:
        rationale = "Highly variable demand (CV > 0.5) challenges statistical methods; safety stock buffers are critical (DFBP Ch. 13)."
    lines.append(f"  Rationale: {rationale}")
    lines.append("")

    # Section 3
    lines.append("SECTION 3: FORWARD OUTLOOK (next 4 periods)")
    if not forecast_table.empty:
        preview = forecast_table.head(4)
        lines.append(f"  {'Period':<10} {'Point Fc':>10} {'Lower 80%':>12} {'Upper 80%':>12}")
        lines.append(f"  {'-'*46}")
        for _, row in preview.iterrows():
            lines.append(
                f"  {str(row.get('Period', '')):<10} {row.get('Point Forecast', 0):>10.0f} "
                f"{row.get('Lower 80%', 0):>12.0f} {row.get('Upper 80%', 0):>12.0f}"
            )
    else:
        lines.append("  Run Tab 3 (Forward Forecast) to populate this section.")
    lines.append("")

    # Section 4
    lines.append("SECTION 4: RISKS AND ALERTS")
    if alert_counts:
        r = alert_counts.get("red", 0)
        a = alert_counts.get("amber", 0)
        g = alert_counts.get("green", 0)
        if r + a == 0:
            lines.append("  No demand alerts in the forecast horizon.")
        else:
            lines.append(f"  Stockout Risk (below min threshold): {r} period(s)")
            lines.append(f"  Overstock Risk (above max threshold): {a} period(s)")
            lines.append(f"  Within Plan: {g} period(s)")
    else:
        lines.append("  Set alert thresholds in Tab 3 to populate this section.")
    lines.append("")

    # Section 5
    lines.append("SECTION 5: RECOMMENDED ACTIONS")
    actions = []
    cv = profile["cv"]
    bias = best_metrics.get("bias", 0)
    mean_d = profile["mean"]

    if cv > 0.5:
        ss_str = f"{ss_result['ss']:.0f}" if ss_result else "N/A"
        actions.append(
            f"Demand variability (CV={cv * 100:.1f}%) is high. Recommend maintaining safety stock "
            f"of {ss_str} units and reviewing forecast weekly rather than monthly. "
            "(Source: Vandeput, DFBP Ch. 13)"
        )
    if mean_d > 0 and abs(bias) > 0.1 * mean_d:
        actions.append(
            f"Systematic bias of {bias:.1f} units detected. Recommend reviewing baseline demand "
            "assumptions with the commercial team before the next S&OP cycle. (Source: Jain, FDPF)"
        )
    if not np.isnan(fa) and fa < 70:
        actions.append(
            f"Forecast accuracy of {fa:.1f}% is below the manufacturing world-class threshold of 80%. "
            f"Recommend switching to {best_method_name} and running a forecast value-add review. "
            "(Source: Vandeput, DFBP Ch. 12)"
        )
    if not actions:
        actions.append("Forecast performance is within acceptable bounds. Maintain current method and cadence.")
    for i, a in enumerate(actions, 1):
        lines.append(f"  {i}. {a}")
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
