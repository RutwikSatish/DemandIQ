# requirements:
#   streamlit>=1.32
#   pandas>=2.0
#   numpy>=1.26
#   plotly>=5.20
#   statsmodels>=0.14

"""
DemandIQ — Demand Sensing & Forecast Accuracy Engine
Enterprise demand planning portfolio application.

Sources:
  FPP3  — Hyndman & Athanasopoulos, Forecasting: Principles and Practice, 3rd ed. (OTexts, 2021)
  DFBP  — Vandeput, Demand Forecasting Best Practices (Manning, 2023)
  FDPF  — Jain, Fundamentals of Demand Planning & Forecasting (Graceway, 2020)
  DMBP  — Crum & Palmatier, Demand Management Best Practices (J. Ross Publishing)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from io import StringIO, BytesIO

import utils

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DemandIQ — Demand Sensing & Forecast Accuracy Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  /* Base */
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #0f1923 !important;
  }
  [data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
  }
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stSelectbox label {
    color: #94a3b8 !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  [data-testid="stSidebarContent"] hr {
    border-color: #1e3a5f !important;
  }

  /* Main panel */
  .main .block-container {
    background: #f8f9fb;
    padding-top: 1.5rem;
  }

  /* Cards */
  .card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
  }
  .card-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 0.25rem;
  }
  .card-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #0f172a;
    line-height: 1.2;
  }
  .card-sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.2rem;
  }

  /* Metric grid */
  .metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
  .metric-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
  }

  /* Badge */
  .badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #fff;
  }

  /* Mono values */
  .mono {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
  }

  /* Interpretation text */
  .interpretation {
    background: #f0f4ff;
    border-left: 3px solid #1B6EF3;
    padding: 0.5rem 0.85rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.82rem;
    color: #1e3a5f;
    margin-top: 0.5rem;
  }

  /* Alert banners */
  .alert-danger {
    background: #fef2f2;
    border: 1px solid #fca5a5;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #991b1b;
    font-weight: 500;
  }
  .alert-warning {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #92400e;
    font-weight: 500;
  }
  .alert-success {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #166534;
    font-weight: 500;
  }
  .alert-info {
    background: #eff6ff;
    border: 1px solid #93c5fd;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #1e3a8a;
    font-weight: 500;
  }

  /* Table styling */
  .stDataFrame { border-radius: 8px; overflow: hidden; }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    border-bottom: 2px solid #e2e8f0;
  }
  .stTabs [data-baseweb="tab"] {
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    color: #64748b;
  }
  .stTabs [aria-selected="true"] {
    color: #1B6EF3 !important;
    border-bottom: 2px solid #1B6EF3;
  }

  /* Section headings */
  .section-heading {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #e2e8f0;
  }

  /* FA% color coding */
  .fa-green { color: #16a34a; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
  .fa-amber { color: #d97706; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
  .fa-red   { color: #dc2626; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }

  /* Source cite */
  .cite {
    font-size: 0.72rem;
    color: #94a3b8;
    font-style: italic;
    margin-top: 0.35rem;
  }

  /* Summary box */
  .summary-box {
    background: #0f1923;
    color: #f1f5f9;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.6;
    padding: 1.25rem;
    border-radius: 10px;
    white-space: pre-wrap;
  }
</style>
""", unsafe_allow_html=True)

# ─── CHART THEME ─────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter", size=12, color="#334155"),
    margin=dict(l=40, r=20, t=40, b=40),
    hovermode="x unified",
    xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
    yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#e2e8f0", borderwidth=1)
)

ACCENT_BLUE = "#1B6EF3"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
ORANGE = "#ea580c"

# ─── SESSION STATE ────────────────────────────────────────────────────────────

if "df" not in st.session_state:
    st.session_state["df"] = None
if "forward_fc" not in st.session_state:
    st.session_state["forward_fc"] = None
if "forward_method" not in st.session_state:
    st.session_state["forward_method"] = None
if "best_method_key" not in st.session_state:
    st.session_state["best_method_key"] = None
if "best_metrics" not in st.session_state:
    st.session_state["best_metrics"] = {}
if "ss_result" not in st.session_state:
    st.session_state["ss_result"] = None
if "alert_counts" not in st.session_state:
    st.session_state["alert_counts"] = None

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## DemandIQ")
    st.markdown('<div style="font-size:0.75rem;color:#64748b;margin-top:-0.5rem;margin-bottom:1rem;">Demand Sensing & Forecast Accuracy Engine</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;margin-bottom:0.5rem;">Upload Demand Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            cols_lower = {c: c.strip().lower() for c in raw.columns}
            raw = raw.rename(columns={c: cols_lower[c] for c in raw.columns})
            if "demand" not in raw.columns:
                st.error("CSV must contain a 'Demand' column.")
            else:
                raw["demand"] = pd.to_numeric(raw["demand"], errors="coerce")
                neg_mask = raw["demand"] < 0
                if neg_mask.any():
                    st.warning(f"{neg_mask.sum()} negative demand value(s) dropped.")
                    raw = raw[~neg_mask]
                raw = raw.dropna(subset=["demand"])
                if "period" not in raw.columns:
                    raw["period"] = range(1, len(raw) + 1)
                if "sku" not in raw.columns:
                    raw["sku"] = "SKU-1"
                st.session_state["df"] = raw.rename(
                    columns={"period": "Period", "demand": "Demand", "sku": "SKU"}
                )[["Period", "SKU", "Demand"]].reset_index(drop=True)
                st.success(f"Loaded {len(raw)} periods.")
        except Exception as ex:
            st.error(f"Could not read file: {ex}")

    if st.button("Use Sample Data (Component-A)", use_container_width=True):
        st.session_state["df"] = utils.SAMPLE_DATA.copy()
        st.success("Sample data loaded.")

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;margin-bottom:0.5rem;">Forecast Settings</div>', unsafe_allow_html=True)
    sma_k = st.slider("SMA window (k)", min_value=2, max_value=8, value=3)
    seasonal_periods = st.slider("Seasonal period", min_value=4, max_value=12, value=12)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.65rem;color:#475569;line-height:1.6;">'
        'Built by <b>Rutwik Satish</b><br>'
        'MS Engineering Management<br>'
        'Graduate Certificate in Supply Chain<br>'
        'Northeastern University'
        '</div>',
        unsafe_allow_html=True
    )

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

df = st.session_state.get("df")
if df is None:
    st.markdown("""
    <div style="text-align:center;margin-top:4rem;">
      <div style="font-size:2rem;font-weight:700;color:#0f172a;">DemandIQ</div>
      <div style="font-size:1rem;color:#64748b;margin-top:0.5rem;">Demand Sensing & Forecast Accuracy Engine</div>
      <div style="font-size:0.85rem;color:#94a3b8;margin-top:1.5rem;">
        Upload a CSV or click <b>Use Sample Data</b> in the sidebar to begin.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

demand_arr = df["Demand"].values.astype(float)
periods = df["Period"].values
n = len(demand_arr)
n_train = max(int(n * 0.8), n - 4)
n_holdout = n - n_train

# Pre-compute profile for use across tabs
profile = utils.compute_profile(demand_arr)
cv_class = utils.classify_cv(profile["cv"])

# ─── TABS ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Profile & Segmentation",
    "Forecast Engine",
    "Forward Forecast & Safety Stock",
    "Forecast Accuracy Tracker",
    "S&OP Executive Summary"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA PROFILE & DEMAND SEGMENTATION
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-heading">Demand Data</div>', unsafe_allow_html=True)
    sku_label = df["SKU"].iloc[0] if "SKU" in df.columns else "—"
    st.markdown(f'<div style="font-size:0.8rem;color:#64748b;margin-bottom:0.5rem;">SKU: <b>{sku_label}</b> &nbsp;|&nbsp; {n} periods loaded</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=200)

    # ── Demand Profile Summary ──────────────────────────────────────────────
    st.markdown('<div class="section-heading">Demand Profile Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    metric_defs = [
        (c1, "Total Periods", f"{profile['n']}", "n", ""),
        (c2, "Total Demand", f"{profile['total']:,.0f}", "units", "sum(D_t)"),
        (c3, "Mean Demand", f"{profile['mean']:,.1f}", "units/period", "D-bar"),
        (c4, "Std Deviation", f"{profile['std']:,.1f}", "units", "s (sample)"),
    ]
    for col, title, val, sub, formula in metric_defs:
        with col:
            st.markdown(f"""
            <div class="card">
              <div class="card-title">{title}</div>
              <div class="card-value mono">{val}</div>
              <div class="card-sub">{sub}{(' — ' + formula) if formula else ''}</div>
            </div>""", unsafe_allow_html=True)

    # ── Demand Pattern Classification ──────────────────────────────────────
    st.markdown('<div class="section-heading">Demand Pattern Classification</div>', unsafe_allow_html=True)
    col_badge, col_info = st.columns([1, 2])
    with col_badge:
        st.markdown(f"""
        <div class="card" style="border-left: 4px solid {cv_class['color']};">
          <div class="card-title">CV Classification</div>
          <span class="badge" style="background:{cv_class['color']};">{cv_class['label']}</span>
          <div style="margin-top:0.75rem;font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:600;color:#0f172a;">
            CV = {profile['cv']:.3f} <span style="font-size:0.9rem;color:#64748b;">({profile['cv']*100:.1f}%)</span>
          </div>
          <div style="font-size:0.82rem;color:#334155;margin-top:0.5rem;">{cv_class['description']}</div>
          <div class="cite">Source: Vandeput, DFBP Ch. 13 — ABC-XYZ segmentation framework</div>
        </div>""", unsafe_allow_html=True)
    with col_info:
        st.markdown("""
        <div class="card">
          <div class="card-title">Classification Guide</div>
          <table style="width:100%;font-size:0.8rem;border-collapse:collapse;margin-top:0.5rem;">
            <tr style="border-bottom:1px solid #e2e8f0;">
              <th style="text-align:left;padding:0.4rem 0.5rem;color:#64748b;">Band</th>
              <th style="text-align:left;padding:0.4rem 0.5rem;color:#64748b;">CV Range</th>
              <th style="text-align:left;padding:0.4rem 0.5rem;color:#64748b;">Interpretation</th>
            </tr>
            <tr style="border-bottom:1px solid #f1f5f9;">
              <td style="padding:0.4rem 0.5rem;"><span style="color:#16a34a;font-weight:700;">X</span></td>
              <td style="padding:0.4rem 0.5rem;font-family:'IBM Plex Mono',monospace;">&lt; 0.20</td>
              <td style="padding:0.4rem 0.5rem;">Stable — statistical methods highly reliable</td>
            </tr>
            <tr style="border-bottom:1px solid #f1f5f9;">
              <td style="padding:0.4rem 0.5rem;"><span style="color:#d97706;font-weight:700;">Y</span></td>
              <td style="padding:0.4rem 0.5rem;font-family:'IBM Plex Mono',monospace;">0.20 – 0.50</td>
              <td style="padding:0.4rem 0.5rem;">Variable — use adaptive methods with caution</td>
            </tr>
            <tr>
              <td style="padding:0.4rem 0.5rem;"><span style="color:#dc2626;font-weight:700;">Z</span></td>
              <td style="padding:0.4rem 0.5rem;font-family:'IBM Plex Mono',monospace;">&gt; 0.50</td>
              <td style="padding:0.4rem 0.5rem;">Lumpy — Croston method or manual override</td>
            </tr>
          </table>
        </div>""", unsafe_allow_html=True)

    # ── Seasonality Detection ───────────────────────────────────────────────
    st.markdown('<div class="section-heading">Seasonality Detection</div>', unsafe_allow_html=True)
    decomp = utils.run_seasonal_decompose(tuple(demand_arr), period=seasonal_periods)
    season_info = utils.detect_seasonality(decomp)

    if season_info["detected"]:
        st.markdown(f'<div class="alert-warning">Seasonal pattern detected (estimated period: {seasonal_periods}). Use Holt-Winters or seasonal ARIMA.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-success">{season_info["message"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="cite" style="margin-top:0.35rem;">Source: Hyndman & Athanasopoulos, FPP3 Ch. 3 — Seasonal decomposition of time series</div>', unsafe_allow_html=True)

    if decomp is not None:
        fig_decomp = go.Figure()
        x_vals = list(range(n))
        panels = [
            ("Observed", demand_arr, ACCENT_BLUE),
            ("Trend", decomp.trend, "#7c3aed"),
            ("Seasonal", decomp.seasonal, "#0891b2"),
            ("Residual", decomp.resid, "#94a3b8"),
        ]
        from plotly.subplots import make_subplots
        fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                    subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
                                    vertical_spacing=0.06)
        for i, (name, vals, color) in enumerate(panels, 1):
            fig_decomp.add_trace(go.Scatter(
                x=x_vals, y=vals, name=name,
                line=dict(color=color, width=1.5),
                showlegend=False
            ), row=i, col=1)
        fig_decomp.update_layout(
            height=520,
            **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
            title=None,
        )
        for i in range(1, 5):
            fig_decomp.update_xaxes(gridcolor="#f1f5f9", linecolor="#e2e8f0", row=i, col=1)
            fig_decomp.update_yaxes(gridcolor="#f1f5f9", linecolor="#e2e8f0", row=i, col=1)
        st.plotly_chart(fig_decomp, use_container_width=True)
    else:
        st.info("Decomposition requires at least 24 periods. Upload more data to enable.")

    # ── Demand History Chart ────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Demand History</div>', unsafe_allow_html=True)
    sma6 = pd.Series(demand_arr).rolling(6).mean().values

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=list(range(n)), y=demand_arr,
        name="Demand", mode="lines+markers",
        line=dict(color=ACCENT_BLUE, width=2),
        marker=dict(size=5)
    ))
    fig_hist.add_trace(go.Scatter(
        x=list(range(n)), y=sma6,
        name="6-Period SMA", mode="lines",
        line=dict(color=AMBER, width=2, dash="dash")
    ))
    fig_hist.update_layout(
        height=320,
        xaxis_title="Period", yaxis_title="Demand (units)",
        **CHART_LAYOUT
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown('<div class="interpretation">The 6-period moving average removes noise to reveal the underlying demand trend — look for sustained directional moves in the dashed line rather than reacting to individual spikes.</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORECAST ENGINE
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    # ── Cross-Validation Setup ──────────────────────────────────────────────
    st.markdown('<div class="section-heading">Cross-Validation Setup</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
      <div style="font-size:0.85rem;color:#334155;">
        <b>Training:</b> periods 1–{n_train} &nbsp;|&nbsp; <b>Holdout:</b> periods {n_train+1}–{n}
        &nbsp;|&nbsp; Holdout size: {n_holdout} period(s)
      </div>
      <div class="cite" style="margin-top:0.5rem;">
        Source: Hyndman & Athanasopoulos, FPP3 Ch. 5 — Time series cross-validation (rolling origin evaluation).
        Any method that cannot outperform the Naive benchmark is not considered useful.
      </div>
    </div>
    """, unsafe_allow_html=True)

    demand_tuple = tuple(demand_arr)

    # ── Fit all models ──────────────────────────────────────────────────────
    with st.spinner("Fitting models and computing holdout metrics..."):
        r_naive = utils.fit_naive(demand_tuple, n_train)
        r_sma   = utils.fit_sma(demand_tuple, n_train, k=sma_k)
        r_ses   = utils.fit_ses(demand_tuple, n_train)
        r_holt  = utils.fit_holt(demand_tuple, n_train)
        r_hw    = utils.fit_holtwinters(demand_tuple, n_train, seasonal_periods=seasonal_periods)
        r_arima = utils.fit_arima(demand_tuple, n_train)

    all_results = [r_naive, r_sma, r_ses, r_holt, r_hw, r_arima]

    # Display model parameter cards
    st.markdown('<div class="section-heading">Fitted Model Parameters</div>', unsafe_allow_html=True)
    param_cols = st.columns(3)
    for i, r in enumerate([r_ses, r_holt, r_hw]):
        with param_cols[i]:
            if r.get("error"):
                st.markdown(f"""
                <div class="card">
                  <div class="card-title">{r['method']}</div>
                  <div style="color:#dc2626;font-size:0.8rem;">{r['error']}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="card">
                  <div class="card-title">{r['method']}</div>
                  <div class="mono" style="font-size:0.85rem;color:#0f172a;">{r.get('params','')}</div>
                </div>""", unsafe_allow_html=True)

    arima_col = st.columns(1)[0]
    with arima_col:
        if r_arima.get("error"):
            st.markdown(f'<div class="alert-warning">ARIMA: {r_arima["error"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card">
              <div class="card-title">ARIMA (AICc-selected)</div>
              <div class="mono" style="font-size:0.85rem;">{r_arima.get('params','')}</div>
              <div class="cite">Source: FPP3 Ch. 9 — AICc = AIC + (2k² + 2k) / (n − k − 1)</div>
            </div>""", unsafe_allow_html=True)

    # ── Forecast vs Actuals chart ───────────────────────────────────────────
    st.markdown('<div class="section-heading">Holdout Period — Forecast vs Actuals</div>', unsafe_allow_html=True)
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=list(range(n)), y=demand_arr,
        name="Actual", mode="lines+markers",
        line=dict(color="#334155", width=2),
        marker=dict(size=5)
    ))
    colors_fc = [ACCENT_BLUE, AMBER, GREEN, "#7c3aed", "#0891b2", "#dc2626"]
    dashes_fc = ["solid", "dash", "dot", "dashdot", "longdash", "dash"]
    for i, r in enumerate(all_results):
        if r.get("error") or r.get("fitted") is None:
            continue
        fitted = r["fitted"]
        # only show holdout region
        holdout_fitted = np.full(n, np.nan)
        holdout_fitted[n_train:] = fitted[n_train:] if len(fitted) > n_train else fitted[-n_holdout:]
        fig_fc.add_trace(go.Scatter(
            x=list(range(n)), y=holdout_fitted,
            name=r["method"], mode="lines",
            line=dict(color=colors_fc[i % len(colors_fc)], width=1.5,
                      dash=dashes_fc[i % len(dashes_fc)])
        ))
    # shading for holdout
    fig_fc.add_vrect(
        x0=n_train - 0.5, x1=n - 0.5,
        fillcolor="rgba(27,110,243,0.05)", line_width=0,
        annotation_text="Holdout", annotation_position="top left"
    )
    fig_fc.update_layout(height=360, xaxis_title="Period", yaxis_title="Demand (units)", **CHART_LAYOUT)
    st.plotly_chart(fig_fc, use_container_width=True)
    st.markdown('<div class="interpretation">Shaded region shows the holdout evaluation window — compare forecast lines to the black actual line here to visually judge which method tracks demand most closely.</div>', unsafe_allow_html=True)

    # ── Results Table ───────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Holdout Accuracy — Ranked by Vandeput Score</div>', unsafe_allow_html=True)

    results_df = utils.build_results_table(all_results)

    if results_df.empty:
        st.warning("No models produced valid holdout forecasts. Try uploading more data.")
    else:
        # Build display df
        display_df = results_df.drop(columns=["_metrics"], errors="ignore").copy()

        # FA% color annotation
        def fa_color(val):
            if pd.isna(val):
                return ""
            if val >= 80:
                return "color: #16a34a; font-weight: 700;"
            elif val >= 65:
                return "color: #d97706; font-weight: 700;"
            else:
                return "color: #dc2626; font-weight: 700;"

        styled = display_df.style.apply(
            lambda row: ["background: #f0fdf4;" if row.name == 0 else "" for _ in row],
            axis=1
        ).applymap(
            fa_color, subset=["FA%"]
        ).format({
            "Vandeput Score": "{:.2f}",
            "MAE": "{:.2f}",
            "Bias": "{:.2f}",
            "RMSE": "{:.2f}",
        })
        st.dataframe(styled, use_container_width=True)

        st.markdown("""
        <div class="cite">
          Vandeput Score = MAE + |Bias| — primary ranking metric (Vandeput, DFBP Ch. 8–9) |
          WMAPE = weighted MAE / total demand — handles near-zero demand (DFBP Ch. 9) |
          MAPE displayed for reference only — do not use for ranking (DFBP Ch. 8) |
          RMSE used for safety stock calculations only (FPP3 Ch. 5.8)
        </div>""", unsafe_allow_html=True)

        # Best method
        best_row = results_df.iloc[0]
        best_method_name = best_row["Method"]
        best_metrics = best_row["_metrics"]
        st.session_state["best_method_key"] = best_method_name
        st.session_state["best_metrics"] = best_metrics

        # ── Benchmark Panel ─────────────────────────────────────────────────
        st.markdown('<div class="section-heading">Industry Benchmark Comparison</div>', unsafe_allow_html=True)
        user_fa = best_metrics.get("fa_pct", np.nan)

        benchmarks = [
            ("Manufacturing", 80, "CV < 0.3 — component / discrete manufacturing"),
            ("FMCG / Consumer Goods", 85, "CV 0.2–0.5 — high-volume consumer products"),
            ("High-tech / Electronics", 70, "CV > 0.4 — short lifecycles, demand shocks"),
        ]
        b_cols = st.columns(3)
        for i, (industry, wc, desc) in enumerate(benchmarks):
            with b_cols[i]:
                gap = user_fa - wc if not np.isnan(user_fa) else None
                if gap is None:
                    color = "#94a3b8"; status = "N/A"
                elif gap >= 0:
                    color = GREEN; status = f"+{gap:.1f}pp above world-class"
                elif gap >= -10:
                    color = AMBER; status = f"{gap:.1f}pp below world-class"
                else:
                    color = RED; status = f"{gap:.1f}pp below world-class"
                fa_str = f"{user_fa:.1f}%" if not np.isnan(user_fa) else "N/A"
                st.markdown(f"""
                <div class="card">
                  <div class="card-title">{industry}</div>
                  <div style="font-size:0.78rem;color:#64748b;margin-bottom:0.5rem;">{desc}</div>
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                      <div style="font-size:0.7rem;color:#94a3b8;">World-class</div>
                      <div class="mono" style="font-size:1.2rem;color:#334155;">{wc}%</div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-size:0.7rem;color:#94a3b8;">Your FA%</div>
                      <div class="mono" style="font-size:1.2rem;color:{color};">{fa_str}</div>
                    </div>
                  </div>
                  <div style="font-size:0.75rem;color:{color};margin-top:0.4rem;font-weight:600;">{status}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown('<div class="cite">Source: Vandeput, DFBP Ch. 10 — Forecasting benchmarks by industry segment</div>', unsafe_allow_html=True)

        # ── Auto-Recommendation ─────────────────────────────────────────────
        st.markdown('<div class="section-heading">Automated Method Recommendation</div>', unsafe_allow_html=True)
        cat = cv_class["category"]
        pattern_name = cv_class["label"]
        bias_dir = best_metrics.get("bias_direction", "N/A")
        fa_str2 = f"{user_fa:.0f}%" if not np.isnan(user_fa) else "N/A"

        if cat == "X":
            rationale = "Stable demand patterns are well-suited to statistical smoothing; exponential smoothing methods efficiently track the level without overfitting noise (FPP3 Ch. 8.1)."
        elif cat == "Y":
            rationale = "Variable demand benefits from adaptive methods that update level and trend estimates continuously; monitor bias closely and retrain monthly (Vandeput, DFBP Ch. 13)."
        else:
            rationale = "Highly variable demand may exceed the assumptions of statistical methods; supplement with safety stock buffers and manual review (Vandeput, DFBP Ch. 13)."

        vs = best_metrics.get("vandeput_score", 0)
        rec_html = f"""
        <div class="card" style="border-left: 4px solid {ACCENT_BLUE};">
          <div style="font-size:0.82rem;color:#334155;line-height:1.7;">
            Based on your demand pattern (<b>{pattern_name}</b>, CV={profile['cv']*100:.1f}%) and holdout
            evaluation across <b>{n_holdout} periods</b>, <b>{best_method_name}</b> performs best with a
            Vandeput Score of <span class="mono">{vs:.2f} units</span>,
            Forecast Accuracy of <b>{fa_str2}</b>, and a Bias of
            <span class="mono">{best_metrics.get('bias', 0):+.1f} units</span>
            ({bias_dir.split(' by ')[0].lower()} tendency).
            {rationale}
          </div>
        </div>
        """
        st.markdown(rec_html, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — FORWARD FORECAST & SAFETY STOCK
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-heading">Forward Forecast Configuration</div>', unsafe_allow_html=True)

    method_options_map = {
        "Naive": "Naive",
        f"SMA (k={sma_k})": "SMA",
        "SES": "SES",
        "Holt Linear": "Holt Linear",
        "Holt-Winters": "Holt-Winters",
        "ARIMA (AICc-selected)": "ARIMA"
    }
    # Determine default index
    default_label = "SES"
    bk = st.session_state.get("best_method_key", "")
    for label, key in method_options_map.items():
        if bk and bk.startswith(key):
            default_label = label
            break
    method_labels = list(method_options_map.keys())
    def_idx = method_labels.index(default_label) if default_label in method_labels else 0

    col_method, col_horizon = st.columns(2)
    with col_method:
        method_sel = st.selectbox("Forecast Method", method_labels, index=def_idx)
    with col_horizon:
        horizon = st.slider("Forecast Horizon (periods)", min_value=1, max_value=24, value=6)

    method_key = method_options_map[method_sel]
    fc_result = utils.generate_forward_forecast(
        tuple(demand_arr), method_key, horizon, k=sma_k,
        seasonal_periods=seasonal_periods
    )

    if fc_result.get("error"):
        st.markdown(f'<div class="alert-danger">{fc_result["error"]}</div>', unsafe_allow_html=True)
    else:
        point = fc_result["point"]
        lo80 = fc_result["lower_80"]
        hi80 = fc_result["upper_80"]
        lo95 = fc_result["lower_95"]
        hi95 = fc_result["upper_95"]
        st.session_state["forward_fc"] = fc_result
        st.session_state["forward_method"] = method_sel

        # ── Alert Zones ──────────────────────────────────────────────────────
        st.markdown('<div class="section-heading">Alert Zone Configuration</div>', unsafe_allow_html=True)
        col_min, col_max = st.columns(2)
        with col_min:
            min_thresh = st.number_input(
                "Minimum Demand Threshold (stockout risk below this)",
                min_value=0, value=int(profile["mean"] * 0.7), step=10
            )
        with col_max:
            max_thresh = st.number_input(
                "Maximum Demand Threshold (overstock risk above this)",
                min_value=0, value=int(profile["mean"] * 1.3), step=10
            )

        zone_colors = []
        for p in point:
            if p < min_thresh:
                zone_colors.append("red")
            elif p > max_thresh:
                zone_colors.append("amber")
            else:
                zone_colors.append("green")

        alert_counts = {
            "red": zone_colors.count("red"),
            "amber": zone_colors.count("amber"),
            "green": zone_colors.count("green")
        }
        st.session_state["alert_counts"] = alert_counts

        # ── Forecast Chart ────────────────────────────────────────────────
        future_x = list(range(n, n + horizon))
        hist_x = list(range(n))

        fig_fwd = go.Figure()
        # Historical
        fig_fwd.add_trace(go.Scatter(
            x=hist_x, y=demand_arr, name="Historical Demand",
            mode="lines+markers", line=dict(color="#334155", width=2),
            marker=dict(size=4)
        ))
        # 95% PI shading
        fig_fwd.add_trace(go.Scatter(
            x=future_x + future_x[::-1],
            y=list(hi95) + list(lo95[::-1]),
            fill="toself", fillcolor="rgba(27,110,243,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% PI", showlegend=True, hoverinfo="skip"
        ))
        # 80% PI shading
        fig_fwd.add_trace(go.Scatter(
            x=future_x + future_x[::-1],
            y=list(hi80) + list(lo80[::-1]),
            fill="toself", fillcolor="rgba(27,110,243,0.14)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% PI", showlegend=True, hoverinfo="skip"
        ))
        # Alert zone vertical coloring
        for i, (fx, p) in enumerate(zip(future_x, point)):
            color_fill = {"red": "rgba(220,38,38,0.12)", "amber": "rgba(217,119,6,0.12)", "green": "rgba(22,163,74,0.08)"}[zone_colors[i]]
            fig_fwd.add_vrect(x0=fx - 0.5, x1=fx + 0.5, fillcolor=color_fill, line_width=0)
        # Point forecast
        fig_fwd.add_trace(go.Scatter(
            x=future_x, y=point, name="Point Forecast",
            mode="lines+markers", line=dict(color=ACCENT_BLUE, width=2.5),
            marker=dict(size=7, color=[
                {"red": RED, "amber": AMBER, "green": GREEN}[z] for z in zone_colors
            ])
        ))
        # Threshold lines
        fig_fwd.add_hline(y=min_thresh, line_dash="dot", line_color=RED, annotation_text="Min Threshold", annotation_position="bottom right")
        fig_fwd.add_hline(y=max_thresh, line_dash="dot", line_color=ORANGE, annotation_text="Max Threshold", annotation_position="top right")

        fig_fwd.update_layout(height=400, xaxis_title="Period", yaxis_title="Demand (units)", **CHART_LAYOUT)
        st.plotly_chart(fig_fwd, use_container_width=True)
        st.markdown(
            '<div class="interpretation">Shaded PI bands reflect forecast uncertainty that grows with horizon (sigma_h = RMSE × sqrt(h)); '
            'red markers indicate stockout risk, amber markers indicate overstock risk — prioritize supply actions for red periods first.</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="cite">Prediction intervals assume normally distributed forecast errors (FPP3 Ch. 5.5). '
            'For non-normal or intermittent demand, treat intervals as approximate.</div>',
            unsafe_allow_html=True
        )

        # Zone summary
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            st.markdown(f'<div class="card" style="border-left:4px solid {RED};"><div class="card-title">Stockout Risk Periods</div><div class="card-value mono">{alert_counts["red"]}</div><div class="card-sub">Below min threshold</div></div>', unsafe_allow_html=True)
        with ac2:
            st.markdown(f'<div class="card" style="border-left:4px solid {AMBER};"><div class="card-title">Overstock Risk Periods</div><div class="card-value mono">{alert_counts["amber"]}</div><div class="card-sub">Above max threshold</div></div>', unsafe_allow_html=True)
        with ac3:
            st.markdown(f'<div class="card" style="border-left:4px solid {GREEN};"><div class="card-title">Within Plan Periods</div><div class="card-value mono">{alert_counts["green"]}</div><div class="card-sub">Between thresholds</div></div>', unsafe_allow_html=True)

        # ── Forecast Table ────────────────────────────────────────────────
        st.markdown('<div class="section-heading">Forward Forecast Table</div>', unsafe_allow_html=True)
        fc_table = pd.DataFrame({
            "Period": [f"t+{i+1}" for i in range(horizon)],
            "Point Forecast": np.round(point, 0).astype(int),
            "Lower 80%": np.round(lo80, 0).astype(int),
            "Upper 80%": np.round(hi80, 0).astype(int),
            "Lower 95%": np.round(lo95, 0).astype(int),
            "Upper 95%": np.round(hi95, 0).astype(int),
            "Zone": [{"red": "Stockout Risk", "amber": "Overstock Risk", "green": "Within Plan"}[z] for z in zone_colors]
        })
        # Store for S&OP
        st.session_state["fc_table"] = fc_table
        st.dataframe(fc_table, use_container_width=True, hide_index=True)

    # ── Safety Stock Calculator ───────────────────────────────────────────────
    st.markdown('<div class="section-heading">Safety Stock Calculator</div>', unsafe_allow_html=True)

    ss_col1, ss_col2 = st.columns(2)
    with ss_col1:
        service_level = st.select_slider(
            "Target Service Level",
            options=[90, 95, 99], value=95,
            format_func=lambda x: f"{x}%"
        )
    with ss_col2:
        lead_time = st.slider("Lead Time (periods)", min_value=1, max_value=12, value=2)

    # sigma_D from holdout forecast errors
    bm = st.session_state.get("best_metrics", {})
    rmse_val = bm.get("rmse", profile["std"])
    if rmse_val == 0 or np.isnan(rmse_val):
        rmse_val = profile["std"]
    sigma_d = rmse_val  # one-period forecast error std as proxy for demand uncertainty

    ss_res = utils.compute_safety_stock(sigma_d, lead_time, service_level, profile["mean"])
    st.session_state["ss_result"] = ss_res

    z_labels = {90: "1.282", 95: "1.645", 99: "2.326"}
    z_str = z_labels.get(service_level, "1.645")

    col_ss1, col_ss2 = st.columns(2)
    with col_ss1:
        st.markdown(f"""
        <div class="card" style="border-left: 4px solid {ACCENT_BLUE};">
          <div class="card-title">Safety Stock</div>
          <div class="card-value mono">{ss_res['ss']:,.0f} units</div>
          <div class="card-sub">{service_level}% service level | {lead_time}-period lead time</div>
          <div style="background:#f8fafc;border-radius:6px;padding:0.6rem;margin-top:0.75rem;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#334155;line-height:1.8;">
            SS = Z × σ_D × √L<br>
            SS = {z_str} × {sigma_d:.1f} × √{lead_time}<br>
            SS = {z_str} × {sigma_d:.1f} × {np.sqrt(lead_time):.3f}<br>
            SS = <b>{ss_res['ss']:,.1f} units</b>
          </div>
          <div class="cite">Source: Jain, FDPF — Safety stock for normally distributed demand. This formula assumes fixed lead time.</div>
        </div>""", unsafe_allow_html=True)
    with col_ss2:
        st.markdown(f"""
        <div class="card" style="border-left: 4px solid {GREEN};">
          <div class="card-title">Reorder Point</div>
          <div class="card-value mono">{ss_res['rop']:,.0f} units</div>
          <div class="card-sub">Trigger replenishment when inventory falls to this level</div>
          <div style="background:#f8fafc;border-radius:6px;padding:0.6rem;margin-top:0.75rem;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#334155;line-height:1.8;">
            ROP = D-bar × L + SS<br>
            ROP = {profile['mean']:.1f} × {lead_time} + {ss_res['ss']:.1f}<br>
            ROP = {profile['mean'] * lead_time:.1f} + {ss_res['ss']:.1f}<br>
            ROP = <b>{ss_res['rop']:,.1f} units</b>
          </div>
          <div class="cite">Note: For variable lead times, consult your supply planner (Jain, FDPF).</div>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — FORECAST ACCURACY TRACKER
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-heading">Forecast vs Actual Input</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input method",
        ["Manual Entry", "Upload CSV"],
        horizontal=True
    )

    tracker_df = None

    if input_mode == "Upload CSV":
        tracker_file = st.file_uploader(
            "Upload CSV with columns: Period, Forecast, Actual",
            type=["csv"], key="tracker_upload"
        )
        if tracker_file:
            try:
                raw_t = pd.read_csv(tracker_file)
                raw_t.columns = [c.strip().lower() for c in raw_t.columns]
                raw_t = raw_t.rename(columns={"period": "Period", "forecast": "Forecast", "actual": "Actual"})
                if not all(c in raw_t.columns for c in ["Forecast", "Actual"]):
                    st.error("CSV must contain Forecast and Actual columns.")
                else:
                    tracker_df = raw_t[["Period", "Forecast", "Actual"]].dropna()
                    tracker_df["Forecast"] = pd.to_numeric(tracker_df["Forecast"], errors="coerce")
                    tracker_df["Actual"] = pd.to_numeric(tracker_df["Actual"], errors="coerce")
                    tracker_df = tracker_df.dropna()
            except Exception as ex:
                st.error(f"Could not read file: {ex}")
    else:
        default_entries = pd.DataFrame({
            "Period": list(range(1, 7)),
            "Forecast": [450, 470, 430, 490, 510, 480],
            "Actual": [420, 490, 410, 520, 495, 500]
        })
        tracker_df_raw = st.data_editor(
            default_entries,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Period": st.column_config.TextColumn("Period"),
                "Forecast": st.column_config.NumberColumn("Forecast", min_value=0),
                "Actual": st.column_config.NumberColumn("Actual", min_value=0),
            }
        )
        tracker_df = tracker_df_raw.dropna()

    if tracker_df is not None and len(tracker_df) > 0:
        actual_t = tracker_df["Actual"].values.astype(float)
        forecast_t = tracker_df["Forecast"].values.astype(float)
        errors_t = actual_t - forecast_t
        metrics_t = utils.compute_metrics(actual_t, forecast_t)

        # ── Summary metrics ─────────────────────────────────────────────────
        st.markdown('<div class="section-heading">Accuracy Metrics</div>', unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        def _fa_class(v):
            if np.isnan(v): return "#94a3b8"
            if v >= 80: return GREEN
            if v >= 65: return AMBER
            return RED

        vs_val = metrics_t.get("vandeput_score", np.nan)
        mae_val = metrics_t.get("mae", np.nan)
        bias_val = metrics_t.get("bias", np.nan)
        fa_val = metrics_t.get("fa_pct", np.nan)
        wmape_val = metrics_t.get("wmape", np.nan)
        rmse_val_t = metrics_t.get("rmse", np.nan)

        with mc1:
            st.markdown(f'<div class="card"><div class="card-title">Vandeput Score</div><div class="card-value mono">{vs_val:.2f}</div><div class="card-sub">MAE + |Bias| — primary metric</div></div>', unsafe_allow_html=True)
        with mc2:
            st.markdown(f'<div class="card"><div class="card-title">Forecast Accuracy</div><div class="card-value mono" style="color:{_fa_class(fa_val)};">{fa_val:.1f}%</div><div class="card-sub">(1 − WMAPE) × 100</div></div>', unsafe_allow_html=True)
        with mc3:
            st.markdown(f'<div class="card"><div class="card-title">Bias</div><div class="card-value mono">{bias_val:+.2f}</div><div class="card-sub">{metrics_t.get("bias_direction","")}</div></div>', unsafe_allow_html=True)
        with mc4:
            wmape_disp = f"{wmape_val*100:.1f}%" if not np.isnan(wmape_val) else "N/A"
            st.markdown(f'<div class="card"><div class="card-title">WMAPE</div><div class="card-value mono">{wmape_disp}</div><div class="card-sub">Weighted — handles near-zero demand</div></div>', unsafe_allow_html=True)

        # ── Bias Trend Detection ─────────────────────────────────────────────
        bias_trend = utils.detect_bias_trend(errors_t)
        if bias_trend["detected"]:
            st.markdown(f'<div class="alert-danger" style="margin:0.75rem 0;">{bias_trend["message"]}</div>', unsafe_allow_html=True)

        # ── Error Waterfall ──────────────────────────────────────────────────
        st.markdown('<div class="section-heading">Forecast Error Waterfall</div>', unsafe_allow_html=True)
        bar_colors = [ACCENT_BLUE if e >= 0 else ORANGE for e in errors_t]
        fig_wf = go.Figure()
        fig_wf.add_trace(go.Bar(
            x=list(tracker_df["Period"].astype(str)),
            y=errors_t,
            marker_color=bar_colors,
            name="Error (A − F)",
            text=[f"{e:+.0f}" for e in errors_t],
            textposition="outside"
        ))
        fig_wf.add_hline(y=0, line_width=1.5, line_color="#334155")
        fig_wf.update_layout(height=320, xaxis_title="Period", yaxis_title="Error (Actual − Forecast)", **CHART_LAYOUT)
        st.plotly_chart(fig_wf, use_container_width=True)
        st.markdown(
            '<div class="interpretation">Bars above zero mean you under-forecast (stockout risk); '
            'bars below zero mean you over-forecast (overstock / working capital risk). '
            'Persistent bars on one side signal systematic bias requiring process correction.</div>',
            unsafe_allow_html=True
        )

        # ── Cumulative Accuracy Chart ────────────────────────────────────────
        st.markdown('<div class="section-heading">Rolling Forecast Accuracy</div>', unsafe_allow_html=True)
        cum_fa = []
        for i in range(1, len(actual_t) + 1):
            m_i = utils.compute_metrics(actual_t[:i], forecast_t[:i])
            cum_fa.append(m_i.get("fa_pct", np.nan))

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=list(tracker_df["Period"].astype(str)),
            y=cum_fa, mode="lines+markers",
            line=dict(color=ACCENT_BLUE, width=2.5),
            marker=dict(size=7),
            name="Cumulative FA%"
        ))
        fig_cum.add_hline(y=80, line_dash="dot", line_color=GREEN, annotation_text="Manufacturing WC 80%")
        fig_cum.add_hline(y=70, line_dash="dot", line_color=AMBER, annotation_text="High-tech WC 70%")
        fig_cum.update_layout(height=300, xaxis_title="Period", yaxis_title="Cumulative FA%", **CHART_LAYOUT)
        st.plotly_chart(fig_cum, use_container_width=True)
        st.markdown('<div class="interpretation">A declining trajectory signals that recent periods are being forecast less accurately — investigate whether demand patterns have shifted before the next planning cycle.</div>', unsafe_allow_html=True)

        # ── Industry Benchmark Table ─────────────────────────────────────────
        st.markdown('<div class="section-heading">Industry Benchmark Table</div>', unsafe_allow_html=True)
        bmark_data = [
            ("Manufacturing", 80),
            ("FMCG / Consumer Goods", 85),
            ("High-tech / Electronics", 70),
        ]
        bmark_rows = []
        for industry, wc in bmark_data:
            if not np.isnan(fa_val):
                gap = fa_val - wc
                if gap >= 0:
                    status = "Above world-class"
                    color_css = "color:#16a34a;font-weight:700;"
                elif gap >= -10:
                    status = "Within 10pp of world-class"
                    color_css = "color:#d97706;font-weight:700;"
                else:
                    status = "Below world-class"
                    color_css = "color:#dc2626;font-weight:700;"
                fa_html = f'<span style="{color_css}">{fa_val:.1f}%</span>'
            else:
                status = "N/A"
                fa_html = "N/A"
            bmark_rows.append({"Industry": industry, "World-Class FA%": f"{wc}%", "Your FA%": f"{fa_val:.1f}%" if not np.isnan(fa_val) else "N/A", "Status": status})
        st.dataframe(pd.DataFrame(bmark_rows), use_container_width=True, hide_index=True)
        st.markdown('<div class="cite">Source: Vandeput, DFBP Ch. 10 — Forecasting benchmarks by industry segment</div>', unsafe_allow_html=True)

    else:
        st.info("Enter forecast vs. actual data above to compute accuracy metrics.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — S&OP EXECUTIVE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<div class="section-heading">S&OP Executive Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.8rem;color:#64748b;margin-bottom:1rem;">'
        'Source: Crum & Palmatier, DMBP — consensus demand planning and S&OP review structure | '
        'Jain, FDPF — How to report, present and sell forecasts to management'
        '</div>',
        unsafe_allow_html=True
    )

    bm_key = st.session_state.get("best_method_key", "—")
    bm_metrics = st.session_state.get("best_metrics", {})
    fwd_fc = st.session_state.get("forward_fc")
    ss_r = st.session_state.get("ss_result")
    ac = st.session_state.get("alert_counts")
    fc_tbl = st.session_state.get("fc_table", pd.DataFrame())
    trend_info = utils.compute_trend_slope(demand_arr, periods=3)

    decomp_s = utils.run_seasonal_decompose(tuple(demand_arr), period=seasonal_periods)
    season_info_s = utils.detect_seasonality(decomp_s)

    # Section 1
    st.markdown("""<div style="font-size:0.85rem;font-weight:700;color:#0f172a;margin:1.25rem 0 0.5rem 0;text-transform:uppercase;letter-spacing:0.06em;">Section 1 — Demand Signal</div>""", unsafe_allow_html=True)
    s1c1, s1c2, s1c3 = st.columns(3)
    with s1c1:
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Demand Pattern</div>
          <span class="badge" style="background:{cv_class['color']};">{cv_class['label']}</span>
          <div class="mono" style="margin-top:0.5rem;font-size:0.9rem;">CV = {profile['cv']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with s1c2:
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Trend Direction</div>
          <div style="font-size:0.85rem;font-weight:600;color:#0f172a;margin-top:0.25rem;">{trend_info['direction']}</div>
        </div>""", unsafe_allow_html=True)
    with s1c3:
        scolor = GREEN if not season_info_s["detected"] else AMBER
        slabel = "Detected" if season_info_s["detected"] else "Not Detected"
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Seasonality</div>
          <div style="font-size:0.85rem;font-weight:600;color:{scolor};margin-top:0.25rem;">{slabel}</div>
        </div>""", unsafe_allow_html=True)

    # Section 2
    st.markdown("""<div style="font-size:0.85rem;font-weight:700;color:#0f172a;margin:1.25rem 0 0.5rem 0;text-transform:uppercase;letter-spacing:0.06em;">Section 2 — Recommended Forecast Method</div>""", unsafe_allow_html=True)
    if bm_key != "—" and bm_metrics:
        fa_bm = bm_metrics.get("fa_pct", np.nan)
        vs_bm = bm_metrics.get("vandeput_score", np.nan)
        bias_bm = bm_metrics.get("bias", np.nan)
        bias_dir_bm = bm_metrics.get("bias_direction", "N/A")
        fa_color_bm = GREEN if (not np.isnan(fa_bm) and fa_bm >= 80) else (AMBER if (not np.isnan(fa_bm) and fa_bm >= 65) else RED)

        s2c1, s2c2, s2c3, s2c4 = st.columns(4)
        with s2c1:
            st.markdown(f'<div class="card"><div class="card-title">Method</div><div style="font-size:1rem;font-weight:700;color:#0f172a;margin-top:0.25rem;">{bm_key}</div></div>', unsafe_allow_html=True)
        with s2c2:
            st.markdown(f'<div class="card"><div class="card-title">Vandeput Score</div><div class="card-value mono">{vs_bm:.2f}</div><div class="card-sub">units (MAE + |Bias|)</div></div>', unsafe_allow_html=True)
        with s2c3:
            st.markdown(f'<div class="card"><div class="card-title">Forecast Accuracy</div><div class="card-value mono" style="color:{fa_color_bm};">{fa_bm:.1f}%</div></div>', unsafe_allow_html=True)
        with s2c4:
            st.markdown(f'<div class="card"><div class="card-title">Bias</div><div class="card-value mono">{bias_bm:+.1f}</div><div class="card-sub">{bias_dir_bm}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-info">Run Tab 2 (Forecast Engine) to populate this section.</div>', unsafe_allow_html=True)

    # Section 3
    st.markdown("""<div style="font-size:0.85rem;font-weight:700;color:#0f172a;margin:1.25rem 0 0.5rem 0;text-transform:uppercase;letter-spacing:0.06em;">Section 3 — Forward Outlook (Next 4 Periods)</div>""", unsafe_allow_html=True)
    if not fc_tbl.empty:
        st.dataframe(fc_tbl.head(4)[["Period", "Point Forecast", "Lower 80%", "Upper 80%", "Zone"]], use_container_width=True, hide_index=True)
        st.markdown('<div class="cite">Source: FPP3 Ch. 5.5 — Prediction intervals from residual standard deviation, assuming normally distributed errors.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-info">Run Tab 3 (Forward Forecast) to populate this section.</div>', unsafe_allow_html=True)

    # Section 4
    st.markdown("""<div style="font-size:0.85rem;font-weight:700;color:#0f172a;margin:1.25rem 0 0.5rem 0;text-transform:uppercase;letter-spacing:0.06em;">Section 4 — Risks and Alerts</div>""", unsafe_allow_html=True)
    if ac:
        if ac["red"] + ac["amber"] == 0:
            st.markdown('<div class="alert-success">No demand alerts in the forecast horizon.</div>', unsafe_allow_html=True)
        else:
            r4c1, r4c2, r4c3 = st.columns(3)
            with r4c1:
                st.markdown(f'<div class="card" style="border-left:3px solid {RED};"><div class="card-title">Stockout Risk Periods</div><div class="card-value mono">{ac["red"]}</div></div>', unsafe_allow_html=True)
            with r4c2:
                st.markdown(f'<div class="card" style="border-left:3px solid {AMBER};"><div class="card-title">Overstock Risk Periods</div><div class="card-value mono">{ac["amber"]}</div></div>', unsafe_allow_html=True)
            with r4c3:
                st.markdown(f'<div class="card" style="border-left:3px solid {GREEN};"><div class="card-title">Within Plan Periods</div><div class="card-value mono">{ac["green"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-info">Set alert thresholds in Tab 3 to populate this section.</div>', unsafe_allow_html=True)

    # Section 5
    st.markdown("""<div style="font-size:0.85rem;font-weight:700;color:#0f172a;margin:1.25rem 0 0.5rem 0;text-transform:uppercase;letter-spacing:0.06em;">Section 5 — Recommended Actions</div>""", unsafe_allow_html=True)

    actions_html = []
    cv_val = profile["cv"]
    mean_d = profile["mean"]
    fa_final = bm_metrics.get("fa_pct", np.nan) if bm_metrics else np.nan
    bias_final = bm_metrics.get("bias", 0) if bm_metrics else 0

    if cv_val > 0.5:
        ss_str = f"{ss_r['ss']:.0f}" if ss_r else "N/A"
        actions_html.append(
            f'<li style="margin-bottom:0.6rem;">Demand variability (CV={cv_val*100:.1f}%) is high. '
            f'Recommend maintaining safety stock of <b>{ss_str} units</b> and reviewing forecast weekly rather than monthly. '
            '<span class="cite">(Source: Vandeput, DFBP Ch. 13)</span></li>'
        )
    if mean_d > 0 and abs(bias_final) > 0.1 * mean_d:
        actions_html.append(
            f'<li style="margin-bottom:0.6rem;">Systematic bias of <b>{bias_final:+.1f} units</b> detected. '
            'Recommend reviewing baseline demand assumptions with the commercial team before the next S&OP cycle. '
            '<span class="cite">(Source: Jain, FDPF)</span></li>'
        )
    if not np.isnan(fa_final) and fa_final < 70:
        actions_html.append(
            f'<li style="margin-bottom:0.6rem;">Forecast accuracy of <b>{fa_final:.1f}%</b> is below the manufacturing '
            f'world-class threshold of 80%. Recommend switching to <b>{bm_key}</b> and running a forecast value-add review. '
            '<span class="cite">(Source: Vandeput, DFBP Ch. 12)</span></li>'
        )
    if not actions_html:
        actions_html.append('<li>Forecast performance is within acceptable bounds. Maintain current method and cadence.</li>')

    st.markdown(
        f'<div class="card"><ul style="margin:0;padding-left:1.25rem;font-size:0.85rem;color:#334155;line-height:1.8;">{"".join(actions_html)}</ul></div>',
        unsafe_allow_html=True
    )

    # ── Export ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Export</div>', unsafe_allow_html=True)

    summary_text = utils.generate_sopp_summary(
        profile, cv_class, season_info_s, trend_info,
        bm_key, bm_metrics, fc_tbl, ss_r, ac
    )
    st.markdown('<div class="summary-box">' + summary_text.replace("\n", "<br>") + '</div>', unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.download_button(
            label="Download Summary as Text",
            data=summary_text,
            file_name="demandiq_sopp_summary.txt",
            mime="text/plain",
            use_container_width=True
        )
    with exp_col2:
        # Build CSV export
        csv_parts = []
        csv_parts.append("=== FORWARD FORECAST TABLE ===")
        if not fc_tbl.empty:
            csv_parts.append(fc_tbl.to_csv(index=False))
        else:
            csv_parts.append("No forward forecast available. Run Tab 3.\n")

        csv_parts.append("\n=== ACCURACY METRICS COMPARISON ===")
        if "results_df" in dir() and not results_df.empty:
            display_export = results_df.drop(columns=["_metrics"], errors="ignore")
            csv_parts.append(display_export.to_csv(index=False))
        else:
            csv_parts.append("No accuracy metrics available. Run Tab 2.\n")

        csv_export = "\n".join(csv_parts)
        st.download_button(
            label="Download Analysis as CSV",
            data=csv_export,
            file_name="demandiq_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )


# ─── FOOTER ──────────────────────────────────────────────────────────────────

st.markdown("""
<div style="
  margin-top: 3rem;
  padding: 1rem 1.5rem;
  background: #0f1923;
  border-radius: 10px;
  font-size: 0.7rem;
  color: #64748b;
  line-height: 1.8;
  text-align: center;
">
  <span style="color:#94a3b8;">Forecasting methods:</span> Hyndman &amp; Athanasopoulos, <i>Forecasting: Principles and Practice</i>, 3rd ed. (OTexts, 2021) &nbsp;|&nbsp;
  <span style="color:#94a3b8;">Accuracy metrics &amp; KPI framework:</span> Vandeput, <i>Demand Forecasting Best Practices</i> (Manning, 2023) &nbsp;|&nbsp;
  <span style="color:#94a3b8;">Demand planning process:</span> Jain, <i>Fundamentals of Demand Planning &amp; Forecasting</i> (Graceway, 2020) &nbsp;|&nbsp;
  <span style="color:#94a3b8;">S&amp;OP framework:</span> Crum &amp; Palmatier, <i>Demand Management Best Practices</i> &nbsp;|&nbsp;
  Built by <span style="color:#e2e8f0;font-weight:600;">Rutwik Satish</span> — MS Engineering Management + Graduate Certificate in Supply Chain Engineering Management, Northeastern University
</div>
""", unsafe_allow_html=True)
