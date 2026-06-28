"""
DemandIQ — Demand Sensing & Forecast Accuracy Engine
Computation utilities.

Sources:
  FPP3  — Hyndman & Athanasopoulos, Forecasting: Principles and Practice, 3rd ed. (OTexts, 2021)
  DFBP  — Vandeput, Demand Forecasting Best Practices (Manning, 2023)
  FDPF  — Jain, Fundamentals of Demand Planning & Forecasting (Graceway, 2020)
  DMBP  — Crum & Palmatier, Demand Management Best Practices (J. Ross Publishing)
"""

import numpy as np
import pandas as pd
from itertools import product
import streamlit as st

# ─── SAMPLE DATA ─────────────────────────────────────────────────────────────

SAMPLE_DATA = pd.DataFrame({
    "Period": list(range(1, 25)),
    "SKU": ["Component-A"] * 24,
    "Demand": [
        380, 420, 390, 510, 480, 440, 395, 620, 580, 430, 410, 490,
        400, 445, 415, 535, 505, 460, 415, 645, 600, 450, 430, 515
    ]
})

# ─── DEMAND PROFILE ──────────────────────────────────────────────────────────

def compute_profile(demand: np.ndarray) -> dict:
    n = len(demand)
    total = float(np.sum(demand))
    mean = total / n
    std = float(np.std(demand, ddof=1)) if n > 1 else 0.0
    cv = std / mean if mean > 0 else 0.0
    return {"n": n, "total": total, "mean": mean, "std": std, "cv": cv}


def classify_cv(cv: float) -> dict:
    if cv < 0.2:
        return {"label": "X — Stable", "description": "Statistical forecasting is highly reliable",
                "color": "#16a34a", "category": "X"}
    elif cv <= 0.5:
        return {"label": "Y — Variable", "description": "Use adaptive methods with caution",
                "color": "#d97706", "category": "Y"}
    else:
        return {"label": "Z — Lumpy", "description": "Consider Croston method or manual override",
                "color": "#dc2626", "category": "Z"}


def compute_trend_slope(demand: np.ndarray, periods: int = 3) -> dict:
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

# ─── SEASONALITY ─────────────────────────────────────────────────────────────

@st.cache_data
def run_seasonal_decompose(demand: tuple, period: int = 12):
    from statsmodels.tsa.seasonal import seasonal_decompose
    arr = np.array(demand, dtype=float)
    if len(arr) < 2 * period:
        return None
    try:
        return seasonal_decompose(arr, model="additive", period=period, extrapolate_trend="freq")
    except Exception:
        return None


def detect_seasonality(decomp_result) -> dict:
    if decomp_result is None:
        return {"detected": False, "message": "Insufficient data for decomposition (requires 24+ periods)."}
    seasonal_std = float(np.nanstd(decomp_result.seasonal))
    trend_std = float(np.nanstd(decomp_result.trend))
    if trend_std == 0:
        return {"detected": False, "message": "No significant seasonality detected."}
    if seasonal_std > 0.10 * trend_std:
        return {"detected": True, "message": "Seasonal pattern detected. Use Holt-Winters or seasonal ARIMA."}
    return {"detected": False, "message": "No significant seasonality detected. Non-seasonal models are appropriate."}

# ─── MODELS ──────────────────────────────────────────────────────────────────

def _failed_result(method: str, reason: str) -> dict:
    return {"method": method, "params": "N/A", "fitted": None,
            "holdout_actual": None, "holdout_forecast": None,
            "model_obj": None, "error": reason}


@st.cache_data
def fit_naive(demand: tuple, n_train: int) -> dict:
    arr = np.array(demand, dtype=float)
    n = len(arr)
    fitted = np.full(n, np.nan)
    for t in range(1, n):
        fitted[t] = arr[t - 1]
    return {"method": "Naive", "params": "F(t+1) = D(t)",
            "fitted": fitted, "holdout_actual": arr[n_train:],
            "holdout_forecast": fitted[n_train:], "model_obj": None}


@st.cache_data
def fit_sma(demand: tuple, n_train: int, k: int = 3) -> dict:
    arr = np.array(demand, dtype=float)
    n = len(arr)
    fitted = np.full(n, np.nan)
    for t in range(k, n):
        fitted[t] = np.mean(arr[t - k:t])
    return {"method": f"SMA(k={k})", "params": f"k={k}",
            "fitted": fitted, "holdout_actual": arr[n_train:],
            "holdout_forecast": fitted[n_train:], "model_obj": None, "k": k}


@st.cache_data
def fit_ses(demand: tuple, n_train: int) -> dict:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    arr = np.array(demand, dtype=float)
    train = arr[:n_train]
    try:
        model = SimpleExpSmoothing(train).fit(optimized=True)
        alpha = float(model.params["smoothing_level"])
        fitted_train = np.array(model.fittedvalues)
        n_holdout = len(arr) - n_train
        holdout_fc = []
        last_level = float(model.level[-1])
        for i in range(n_holdout):
            holdout_fc.append(last_level)
            actual = arr[n_train + i]
            last_level = alpha * actual + (1 - alpha) * last_level
        holdout_forecast = np.array(holdout_fc)
        fitted = np.concatenate([fitted_train, holdout_forecast])
        return {"method": "SES", "params": f"alpha={alpha:.3f}",
                "alpha": alpha, "fitted": fitted,
                "holdout_actual": arr[n_train:], "holdout_forecast": holdout_forecast,
                "model_obj": model}
    except Exception as e:
        return _failed_result("SES", str(e))


@st.cache_data
def fit_holt(demand: tuple, n_train: int) -> dict:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    arr = np.array(demand, dtype=float)
    train = arr[:n_train]
    try:
        model = ExponentialSmoothing(train, trend="add", seasonal=None,
                                     initialization_method="heuristic").fit(optimized=True)
        alpha = float(model.params["smoothing_level"])
        beta = float(model.params["smoothing_trend"])
        fitted_train = np.array(model.fittedvalues)
        n_holdout = len(arr) - n_train
        holdout_forecast = np.array(model.forecast(n_holdout))
        fitted = np.concatenate([fitted_train, holdout_forecast])
        return {"method": "Holt Linear", "params": f"alpha={alpha:.3f}, beta={beta:.3f}",
                "alpha": alpha, "beta": beta, "fitted": fitted,
                "holdout_actual": arr[n_train:], "holdout_forecast": holdout_forecast,
                "model_obj": model}
    except Exception as e:
        return _failed_result("Holt Linear", str(e))


@st.cache_data
def fit_holtwinters(demand: tuple, n_train: int, seasonal_periods: int = 12) -> dict:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    arr = np.array(demand, dtype=float)
    n = len(arr)
    if n < 2 * seasonal_periods:
        return _failed_result("Holt-Winters",
                              f"Insufficient data (requires {2*seasonal_periods}+ periods).")
    train = arr[:n_train]
    if len(train) < 2 * seasonal_periods:
        return _failed_result("Holt-Winters",
                              f"Training set too small (requires {2*seasonal_periods}+ periods).")
    try:
        model = ExponentialSmoothing(train, trend="add", seasonal="add",
                                     seasonal_periods=seasonal_periods,
                                     initialization_method="heuristic").fit(optimized=True)
        alpha = float(model.params["smoothing_level"])
        beta = float(model.params["smoothing_trend"])
        gamma = float(model.params["smoothing_seasonal"])
        fitted_train = np.array(model.fittedvalues)
        n_holdout = len(arr) - n_train
        holdout_forecast = np.array(model.forecast(n_holdout))
        fitted = np.concatenate([fitted_train, holdout_forecast])
        return {"method": "Holt-Winters",
                "params": f"alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}",
                "alpha": alpha, "beta": beta, "gamma": gamma, "fitted": fitted,
                "holdout_actual": arr[n_train:], "holdout_forecast": holdout_forecast,
                "model_obj": model}
    except Exception as e:
        return _failed_result("Holt-Winters", str(e))


@st.cache_data
def fit_arima(demand: tuple, n_train: int) -> dict:
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
            k = p + q + d + 1
            aicc = res.aic + (2*k**2 + 2*k) / (n - k - 1) if n - k - 1 > 0 else np.inf
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
        holdout_forecast = np.array(best_model.forecast(steps=n_holdout))
        fitted_train = np.array(best_model.fittedvalues)
        fitted = np.concatenate([fitted_train, holdout_forecast])
        return {"method": f"ARIMA{best_order}",
                "params": f"order={best_order}, AICc={best_aicc:.1f}",
                "order": best_order, "aicc": best_aicc, "fitted": fitted,
                "holdout_actual": arr[n_train:], "holdout_forecast": holdout_forecast,
                "model_obj": best_model}
    except Exception as e:
        return _failed_result("ARIMA", str(e))

# ─── METRICS ─────────────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(forecast))
    actual = actual[mask]
    forecast = forecast[mask]
    if len(actual) == 0:
        return {}
    errors = actual - forecast
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    bias = float(np.mean(errors))
    vandeput_score = mae + abs(bias)
    sum_actual = float(np.sum(actual))
    wmape = float(np.sum(abs_errors) / sum_actual) if sum_actual > 0 else np.nan
    fa_pct = (1 - wmape) * 100 if not np.isnan(wmape) else np.nan
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    nonzero_mask = actual > 0
    mape = float(np.mean(abs_errors[nonzero_mask] / actual[nonzero_mask]) * 100) if np.any(nonzero_mask) else np.nan
    bias_direction = (f"Under-forecasting by {abs(bias):.1f} units" if bias > 0
                      else f"Over-forecasting by {abs(bias):.1f} units")
    return {"n": len(errors), "mae": mae, "bias": bias, "bias_direction": bias_direction,
            "vandeput_score": vandeput_score, "wmape": wmape, "fa_pct": fa_pct,
            "rmse": rmse, "mape": mape}


def detect_bias_trend(errors: np.ndarray, window: int = 3) -> dict:
    if len(errors) < window:
        return {"detected": False}
    recent = errors[-window:]
    if np.all(recent > 0):
        return {"detected": True, "direction": "under-forecasting", "n": window,
                "message": f"Systematic Bias Detected — under-forecasting for {window} consecutive periods. Review demand assumptions before the next S&OP cycle."}
    elif np.all(recent < 0):
        return {"detected": True, "direction": "over-forecasting", "n": window,
                "message": f"Systematic Bias Detected — over-forecasting for {window} consecutive periods. Review demand assumptions before the next S&OP cycle."}
    return {"detected": False}

# ─── FORWARD FORECAST ────────────────────────────────────────────────────────

@st.cache_data
def generate_forward_forecast(demand: tuple, method_key: str, horizon: int,
                               k: int = 3, seasonal_periods: int = 12) -> dict:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    arr = np.array(demand, dtype=float)
    n = len(arr)
    point_fc = np.full(horizon, np.nan)
    rmse_proxy = None
    try:
        if method_key == "Naive":
            point_fc = np.full(horizon, arr[-1])
            rmse_proxy = float(np.sqrt(np.mean(np.diff(arr) ** 2)))
        elif method_key.startswith("SMA"):
            point_fc = np.full(horizon, np.mean(arr[-k:]))
            fitted = np.array([np.mean(arr[max(0, i-k):i]) for i in range(1, n+1)])
            errors = arr[k:] - fitted[k-1:n-1]
            rmse_proxy = float(np.sqrt(np.mean(errors**2))) if len(errors) > 0 else float(np.std(arr))
        elif method_key == "SES":
            model = SimpleExpSmoothing(arr).fit(optimized=True)
            point_fc = np.array(model.forecast(horizon))
            rmse_proxy = float(np.sqrt(np.mean(np.array(model.resid)**2)))
        elif method_key == "Holt Linear":
            model = ExponentialSmoothing(arr, trend="add", seasonal=None,
                                         initialization_method="heuristic").fit(optimized=True)
            point_fc = np.array(model.forecast(horizon))
            rmse_proxy = float(np.sqrt(np.mean(np.array(model.resid)**2)))
        elif method_key == "Holt-Winters":
            if n < 2 * seasonal_periods:
                return {"error": f"Insufficient data for Holt-Winters (requires {2*seasonal_periods}+ periods)."}
            model = ExponentialSmoothing(arr, trend="add", seasonal="add",
                                         seasonal_periods=seasonal_periods,
                                         initialization_method="heuristic").fit(optimized=True)
            point_fc = np.array(model.forecast(horizon))
            rmse_proxy = float(np.sqrt(np.mean(np.array(model.resid)**2)))
        elif method_key.startswith("ARIMA"):
            best_aicc, best_model = np.inf, None
            for p, d, q in product(range(3), range(2), range(3)):
                try:
                    res = ARIMA(arr, order=(p, d, q)).fit()
                    kk = p + q + d + 1
                    aicc = res.aic + (2*kk**2+2*kk)/(n-kk-1) if n-kk-1 > 0 else np.inf
                    if aicc < best_aicc:
                        best_aicc, best_model = aicc, res
                except Exception:
                    continue
            if best_model is None:
                return {"error": "ARIMA fitting failed."}
            point_fc = np.array(best_model.forecast(steps=horizon))
            rmse_proxy = float(np.sqrt(np.mean(np.array(best_model.resid)**2)))
    except Exception as e:
        return {"error": str(e)}
    if not rmse_proxy or rmse_proxy == 0:
        rmse_proxy = float(np.std(arr)) or 1.0
    h_arr = np.arange(1, horizon+1, dtype=float)
    sigma_h = rmse_proxy * np.sqrt(h_arr)
    return {"point": point_fc, "lower_80": point_fc - 1.282*sigma_h,
            "upper_80": point_fc + 1.282*sigma_h, "lower_95": point_fc - 1.960*sigma_h,
            "upper_95": point_fc + 1.960*sigma_h, "rmse_proxy": rmse_proxy}

# ─── SAFETY STOCK ────────────────────────────────────────────────────────────

def compute_safety_stock(sigma_d: float, lead_time: int,
                          service_level_pct: float, mean_demand: float) -> dict:
    z_map = {90: 1.282, 95: 1.645, 99: 2.326}
    z = z_map.get(int(service_level_pct), 1.645)
    sigma_lt = sigma_d * np.sqrt(lead_time)
    ss = z * sigma_lt
    rop = mean_demand * lead_time + ss
    return {"z": z, "sigma_d": sigma_d, "sigma_lt": sigma_lt,
            "ss": ss, "rop": rop, "lead_time": lead_time,
            "service_level": service_level_pct}

# ─── RESULTS TABLE ───────────────────────────────────────────────────────────

def build_results_table(results_list: list) -> pd.DataFrame:
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
            "Method": r["method"], "Parameters": r.get("params", ""),
            "Vandeput Score": round(m["vandeput_score"], 2),
            "MAE": round(m["mae"], 2), "Bias": round(m["bias"], 2),
            "WMAPE": f"{m['wmape']*100:.1f}%" if not np.isnan(m.get("wmape", np.nan)) else "N/A",
            "FA%": round(m["fa_pct"], 1) if not np.isnan(m.get("fa_pct", np.nan)) else np.nan,
            "RMSE": round(m["rmse"], 2), "MAPE*": mape_display,
            "_metrics": m
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Vandeput Score").reset_index(drop=True)

# ─── S&OP SUMMARY ────────────────────────────────────────────────────────────

def generate_sopp_summary(profile, cv_class, seasonality, trend,
                           best_method_name, best_metrics, forecast_table,
                           ss_result, alert_counts) -> str:
    lines = ["="*60, "DemandIQ — S&OP EXECUTIVE SUMMARY", "="*60, ""]
    lines += ["SECTION 1: DEMAND SIGNAL",
              f"  Pattern: {cv_class['label']}  |  CV: {profile['cv']*100:.1f}%",
              f"  Trend: {trend['direction']}",
              f"  Seasonality: {'Detected' if seasonality['detected'] else 'Not detected'}", ""]
    fa = best_metrics.get("fa_pct", float("nan"))
    lines += ["SECTION 2: RECOMMENDED METHOD",
              f"  Method: {best_method_name}",
              f"  Vandeput Score: {best_metrics.get('vandeput_score', 0):.2f} units",
              f"  Forecast Accuracy: {fa:.1f}%" if not np.isnan(fa) else "  Forecast Accuracy: N/A",
              f"  Bias: {best_metrics.get('bias_direction', 'N/A')}", ""]
    lines.append("SECTION 3: FORWARD OUTLOOK (next 4 periods)")
    if forecast_table is not None and not forecast_table.empty:
        for _, row in forecast_table.head(4).iterrows():
            lines.append(f"  {str(row.get('Period','')):<8} Fc={row.get('Point Forecast',0):.0f}  80%PI=[{row.get('Lower 80%',0):.0f}, {row.get('Upper 80%',0):.0f}]")
    else:
        lines.append("  Run Tab 3 to populate.")
    lines.append("")
    lines.append("SECTION 4: RISKS AND ALERTS")
    if alert_counts:
        r, a, g = alert_counts.get("red",0), alert_counts.get("amber",0), alert_counts.get("green",0)
        lines += [f"  Stockout Risk: {r} period(s)", f"  Overstock Risk: {a} period(s)", f"  Within Plan: {g} period(s)"]
    else:
        lines.append("  Set thresholds in Tab 3.")
    lines += ["", "SECTION 5: RECOMMENDED ACTIONS"]
    actions = []
    if profile["cv"] > 0.5:
        ss_str = f"{ss_result['ss']:.0f}" if ss_result else "N/A"
        actions.append(f"High variability (CV={profile['cv']*100:.1f}%). Maintain safety stock of {ss_str} units. Review weekly. (DFBP Ch. 13)")
    if profile["mean"] > 0 and abs(best_metrics.get("bias",0)) > 0.1*profile["mean"]:
        actions.append(f"Systematic bias of {best_metrics.get('bias',0):+.1f} units. Review baseline assumptions with commercial team. (FDPF)")
    if not np.isnan(fa) and fa < 70:
        actions.append(f"FA% of {fa:.1f}% below 80% world-class. Run forecast value-add review. (DFBP Ch. 12)")
    if not actions:
        actions.append("Performance within acceptable bounds. Maintain current method and cadence.")
    for i, a in enumerate(actions, 1):
        lines.append(f"  {i}. {a}")
    lines += ["", "="*60]
    return "\n".join(lines)
