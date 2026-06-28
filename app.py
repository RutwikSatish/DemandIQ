# requirements:
#   streamlit>=1.32
#   pandas>=2.0
#   numpy>=1.26
#   plotly>=5.20
#   statsmodels>=0.14

"""
DemandIQ — Demand Sensing & Forecast Accuracy Engine
Sources:
  FPP3  — Hyndman & Athanasopoulos, Forecasting: Principles and Practice, 3rd ed. (OTexts, 2021)
  DFBP  — Vandeput, Demand Forecasting Best Practices (Manning, 2023)
  FDPF  — Jain, Fundamentals of Demand Planning & Forecasting (Graceway, 2020)
  DMBP  — Crum & Palmatier, Demand Management Best Practices (J. Ross Publishing)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from itertools import product
import utils

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DemandIQ — Demand Forecast Engine",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# ─── CSS — injected via components to avoid Streamlit markdown stripping ─────

def inject_css():
    css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #0a0f1a !important;
    color: #e2e8f0 !important;
}

[data-testid="stMain"], [data-testid="stAppViewContainer"],
[data-testid="block-container"], .main {
    background-color: #0a0f1a !important;
}

[data-testid="stSidebar"] {
    background-color: #060c16 !important;
    border-right: 1px solid #1e2d45 !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stFileUploader label {
    color: #64748b !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stSidebar"] hr { border-color: #1e2d45 !important; }
[data-testid="stSidebar"] .stButton button {
    background: #1B6EF3 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
}

.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
    background-color: #0a0f1a !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 2px solid #1e2d45 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.1rem !important;
    color: #475569 !important;
    background: transparent !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom: 2px solid #3b82f6 !important;
}

.stDataFrame { border-radius: 8px !important; overflow: hidden; }
.stDataFrame * { background-color: #111827 !important; color: #cbd5e1 !important; }

.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    background: #1e2d45 !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d4a6e !important;
}
.stButton > button:hover {
    background: #1B6EF3 !important;
    border-color: #1B6EF3 !important;
    color: white !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div { color: #e2e8f0 !important; }

div[data-baseweb="select"] > div {
    background: #111827 !important;
    border-color: #1e2d45 !important;
    color: #e2e8f0 !important;
}

.stDownloadButton button {
    background: #1e2d45 !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d4a6e !important;
    border-radius: 8px !important;
}
.stDownloadButton button:hover {
    background: #1B6EF3 !important;
    border-color: #1B6EF3 !important;
    color: white !important;
}

p, div, span, label, h1, h2, h3, h4 {
    color: #e2e8f0 !important;
}

[data-testid="stMarkdownContainer"] p { color: #cbd5e1 !important; }
</style>
"""
    st.html(css)

inject_css()

# ─── COLOUR TOKENS (dark theme) ──────────────────────────────────────────────

BLUE   = "#3b82f6"
GREEN  = "#22c55e"
RED    = "#f87171"
AMBER  = "#fbbf24"
ORANGE = "#fb923c"
NAVY   = "#060c16"

# Dark surface colors for inline HTML
BG_CARD    = "#111827"
BG_PAGE    = "#0a0f1a"
BORDER     = "#1e2d45"
TEXT_PRI   = "#f1f5f9"
TEXT_SEC   = "#94a3b8"
TEXT_MUTED = "#475569"

CHART = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
    font=dict(family="Inter", size=12, color="#94a3b8"),
    margin=dict(l=40, r=20, t=40, b=40), hovermode="x unified",
    xaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", color="#94a3b8"),
    yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", color="#94a3b8"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d45", borderwidth=1,
                font=dict(color="#94a3b8"))
)

# ─── HTML HELPERS ────────────────────────────────────────────────────────────

def card(title, value, sub="", border_color=None, value_color=None):
    bc = border_color or BLUE
    vc = value_color or TEXT_PRI
    return f"""<div style="background:{BG_CARD};border:1px solid {BORDER};border-left:4px solid {bc};
border-radius:10px;padding:1rem 1.25rem;margin-bottom:0.75rem;">
<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:{TEXT_MUTED};margin-bottom:0.2rem;">{title}</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:1.5rem;font-weight:600;color:{vc};line-height:1.2;">{value}</div>
{"<div style='font-size:0.75rem;color:"+TEXT_SEC+";margin-top:0.2rem;'>"+sub+"</div>" if sub else ""}
</div>"""

def section(text):
    st.markdown(f"""<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;
letter-spacing:0.1em;color:{TEXT_MUTED};margin:1.5rem 0 0.6rem 0;padding-bottom:0.35rem;
border-bottom:1px solid {BORDER};">{text}</div>""", unsafe_allow_html=True)

def insight(text):
    st.markdown(f"""<div style="background:#0f1e3a;border-left:3px solid {BLUE};
padding:0.5rem 0.85rem;border-radius:0 6px 6px 0;font-size:0.8rem;color:#93c5fd;
margin-top:0.4rem;">{text}</div>""", unsafe_allow_html=True)

def cite(text):
    st.markdown(f'<div style="font-size:0.68rem;color:{TEXT_MUTED};font-style:italic;margin-top:0.25rem;">{text}</div>', unsafe_allow_html=True)

def alert(text, kind="info"):
    colors = {
        "danger":  ("#2d0f0f","#7f1d1d","#fca5a5"),
        "warning": ("#2a1f07","#78350f","#fcd34d"),
        "success": ("#0a2218","#14532d","#86efac"),
        "info":    ("#0d1b3e","#1e3a8a","#93c5fd"),
    }
    bg, border, fg = colors.get(kind, colors["info"])
    st.markdown(f"""<div style="background:{bg};border:1px solid {border};border-radius:8px;
padding:0.7rem 1rem;color:{fg};font-weight:500;font-size:0.85rem;margin:0.5rem 0;">{text}</div>""",
                unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────

for k, v in [("df", None), ("forward_fc", None), ("forward_method", None),
             ("best_method_key", None), ("best_metrics", {}),
             ("ss_result", None), ("alert_counts", None), ("fc_table", pd.DataFrame())]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""<div style="padding:0.5rem 0 1rem;">
<div style="font-size:1.3rem;font-weight:700;color:#0f1724;letter-spacing:-0.02em;">DemandIQ</div>
<div style="font-size:0.72rem;color:#64748b;margin-top:0.15rem;">Demand Sensing & Forecast Engine</div>
</div>""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;margin-bottom:0.4rem;">Upload Demand Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV: Period, Demand (+ optional SKU)", type=["csv"], label_visibility="collapsed")

    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            raw.columns = [c.strip().lower() for c in raw.columns]
            if "demand" not in raw.columns:
                st.error("CSV must contain a 'Demand' column.")
            else:
                raw["demand"] = pd.to_numeric(raw["demand"], errors="coerce")
                neg = raw["demand"] < 0
                if neg.any():
                    st.warning(f"{neg.sum()} negative value(s) removed.")
                    raw = raw[~neg]
                raw = raw.dropna(subset=["demand"])
                if "period" not in raw.columns:
                    raw["period"] = range(1, len(raw)+1)
                if "sku" not in raw.columns:
                    raw["sku"] = "SKU-1"
                st.session_state["df"] = raw.rename(
                    columns={"period":"Period","demand":"Demand","sku":"SKU"}
                )[["Period","SKU","Demand"]].reset_index(drop=True)
                st.success(f"{len(raw)} periods loaded.")
        except Exception as ex:
            st.error(f"Could not read file: {ex}")

    if st.button("Load Sample Data (Component-A)", width='stretch'):
        st.session_state["df"] = utils.SAMPLE_DATA.copy()
        st.success("Sample data loaded.")

    st.markdown("---")
    st.markdown('<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;margin-bottom:0.4rem;">Forecast Settings</div>', unsafe_allow_html=True)
    sma_k = st.slider("SMA window k", 2, 8, 3)
    seasonal_periods = st.slider("Seasonal period", 4, 12, 12)

    st.markdown("---")
    st.markdown("""<div style="font-size:0.68rem;color:#94a3b8;line-height:1.7;">
Built by <span style="color:#1e2d45;font-weight:600;">Rutwik Satish</span><br>
MS Engineering Management<br>
Grad Certificate — Supply Chain<br>
Northeastern University
</div>""", unsafe_allow_html=True)

# ─── LANDING PAGE (no data loaded) ───────────────────────────────────────────

if st.session_state["df"] is None:

    # Hero
    st.markdown(f"""
<div style="background:linear-gradient(135deg,{NAVY} 0%,#1B3A6B 60%,#1B6EF3 100%);
border-radius:16px;padding:3rem 3.5rem;margin-bottom:2rem;color:#f1f5f9;">
  <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.15em;
  color:#93c5fd;margin-bottom:0.75rem;font-weight:600;">Enterprise Demand Planning</div>
  <div style="font-size:2.6rem;font-weight:700;letter-spacing:-0.03em;line-height:1.15;
  margin-bottom:1rem;">DemandIQ</div>
  <div style="font-size:1.15rem;color:#bfdbfe;font-weight:300;max-width:580px;
  line-height:1.6;margin-bottom:1.5rem;">
    The demand sensing and forecast accuracy engine built for supply chain planners
    who need more than a spreadsheet — and less than a six-figure planning platform.
  </div>
  <div style="display:flex;gap:1rem;flex-wrap:wrap;">
    <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
    border-radius:8px;padding:0.5rem 1rem;font-size:0.8rem;color:#93c5fd;">
      6 Forecasting Methods
    </div>
    <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
    border-radius:8px;padding:0.5rem 1rem;font-size:0.8rem;color:#93c5fd;">
      Vandeput Score Ranking
    </div>
    <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
    border-radius:8px;padding:0.5rem 1rem;font-size:0.8rem;color:#93c5fd;">
      Safety Stock Calculator
    </div>
    <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
    border-radius:8px;padding:0.5rem 1rem;font-size:0.8rem;color:#93c5fd;">
      S&OP Executive Summary
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # The problem
    st.markdown(f"""
<div style="background:#0d1a2e;border:1px solid #1e2d45;border-radius:12px;
padding:1.75rem 2rem;margin-bottom:1.5rem;">
  <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;
  color:#94a3b8;margin-bottom:0.5rem;font-weight:700;">The Problem DemandIQ Solves</div>
  <div style="font-size:1.05rem;font-weight:600;color:#f1f5f9;margin-bottom:0.75rem;">
    Every demand planner spends hours in Excel doing this manually.
  </div>
  <div style="font-size:0.88rem;color:#94a3b8;line-height:1.75;max-width:720px;">
    You upload historical demand data. DemandIQ runs six forecasting methods simultaneously —
    from Simple Moving Average to ARIMA — evaluates each one against a holdout period using
    academically validated accuracy metrics, and tells you which method fits your demand pattern
    best. Then it generates a forward forecast with confidence intervals, calculates your safety
    stock, flags stockout and overstock risk periods, and produces an S&OP executive summary you
    can present in your next planning meeting. In under five minutes.
  </div>
</div>""", unsafe_allow_html=True)

    # How it works steps
    col1, col2, col3, col4 = st.columns(4)
    steps = [
        ("01", "Upload or load sample", "Drop in a CSV with Period and Demand columns — or click Load Sample Data to see the app in action immediately.", BLUE),
        ("02", "Profile your demand", "DemandIQ classifies your demand pattern (Stable / Variable / Lumpy) using Coefficient of Variation and detects seasonality via additive decomposition.", "#7c3aed"),
        ("03", "Compare 6 methods", "Naive, SMA, SES, Holt Linear, Holt-Winters, and ARIMA are all fitted and ranked by the Vandeput Score — MAE plus absolute Bias — on a held-out evaluation window.", GREEN),
        ("04", "Get actionable outputs", "Forward forecast with 80% and 95% prediction intervals, safety stock calculation, alert zones, and a plain-English S&OP summary ready to export.", AMBER),
    ]
    for col, (num, title, desc, color) in zip([col1,col2,col3,col4], steps):
        with col:
            st.markdown(f"""
<div style="background:#0d1a2e;border:1px solid #1e2d45;border-radius:12px;
padding:1.25rem 1.25rem;height:100%;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:1.6rem;font-weight:700;
  color:{color};margin-bottom:0.5rem;">{num}</div>
  <div style="font-size:0.88rem;font-weight:600;color:#f1f5f9;margin-bottom:0.5rem;">{title}</div>
  <div style="font-size:0.78rem;color:#64748b;line-height:1.6;">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # Sample data preview
    st.markdown(f"""
<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;
color:#94a3b8;margin-bottom:0.75rem;padding-bottom:0.35rem;border-bottom:1px solid #1e2d45;">
Sample Dataset — Component-A (24 Monthly Periods)
</div>""", unsafe_allow_html=True)

    sd = utils.SAMPLE_DATA
    demand_arr_preview = sd["Demand"].values.astype(float)

    col_table, col_chart = st.columns([1, 2])
    with col_table:
        st.markdown("""<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:0.75rem;">
This is the pre-loaded dataset used when you click <b style="color:#cbd5e1;">Load Sample Data</b>. It represents
24 months of demand for a manufactured component with realistic variable demand
(mean ~478 units, CV ~18%) and mild seasonality.
</div>""", unsafe_allow_html=True)

        # Build HTML table — avoids Streamlit dataframe dark-theme rendering issues
        rows_html = ""
        for _, row in sd.iterrows():
            rows_html += f"""<tr style="border-bottom:1px solid #1e2d45;">
<td style="padding:0.35rem 0.75rem;color:#94a3b8;font-family:'IBM Plex Mono',monospace;font-size:0.8rem;">{int(row['Period'])}</td>
<td style="padding:0.35rem 0.75rem;color:#60a5fa;font-size:0.75rem;">{row['SKU']}</td>
<td style="padding:0.35rem 0.75rem;color:#f1f5f9;font-family:'IBM Plex Mono',monospace;font-size:0.8rem;text-align:right;">{int(row['Demand']):,}</td>
</tr>"""

        st.markdown(f"""
<div style="background:#111827;border:1px solid #1e2d45;border-radius:10px;
overflow:hidden;max-height:320px;overflow-y:auto;">
<table style="width:100%;border-collapse:collapse;">
<thead>
<tr style="background:#0d1724;border-bottom:2px solid #1e3a6e;">
  <th style="padding:0.45rem 0.75rem;text-align:left;font-size:0.68rem;text-transform:uppercase;
  letter-spacing:0.08em;color:#475569;font-weight:600;">Period</th>
  <th style="padding:0.45rem 0.75rem;text-align:left;font-size:0.68rem;text-transform:uppercase;
  letter-spacing:0.08em;color:#475569;font-weight:600;">SKU</th>
  <th style="padding:0.45rem 0.75rem;text-align:right;font-size:0.68rem;text-transform:uppercase;
  letter-spacing:0.08em;color:#475569;font-weight:600;">Demand</th>
</tr>
</thead>
<tbody>{rows_html}</tbody>
</table>
</div>""", unsafe_allow_html=True)

    with col_chart:
        sma6 = pd.Series(demand_arr_preview).rolling(6).mean().values
        fig_preview = go.Figure()
        fig_preview.add_trace(go.Bar(
            x=list(sd["Period"]), y=demand_arr_preview,
            name="Monthly Demand", marker_color=BLUE, opacity=0.7,
            marker_line_width=0
        ))
        fig_preview.add_trace(go.Scatter(
            x=list(sd["Period"]), y=sma6,
            name="6-Period SMA", mode="lines",
            line=dict(color=AMBER, width=2.5, dash="dash")
        ))
        fig_preview.update_layout(
            height=360, title=dict(text="Component-A — 24-Month Demand History",
                                   font=dict(size=13, color="#94a3b8")),
            xaxis_title="Period", yaxis_title="Demand (units)",
            **CHART)
        st.plotly_chart(fig_preview, width='stretch')

    # Framework citations
    st.markdown(f"""
<div style="background:{NAVY};border-radius:10px;padding:1.25rem 1.75rem;
margin-top:1.5rem;font-size:0.72rem;color:#64748b;line-height:1.9;">
  <span style="color:#93c5fd;font-weight:600;">Forecasting methods:</span>
  Hyndman & Athanasopoulos, <em>Forecasting: Principles and Practice</em>, 3rd ed. (OTexts, 2021) &nbsp;|&nbsp;
  <span style="color:#93c5fd;font-weight:600;">Accuracy metrics:</span>
  Vandeput, <em>Demand Forecasting Best Practices</em> (Manning, 2023) &nbsp;|&nbsp;
  <span style="color:#93c5fd;font-weight:600;">Planning process:</span>
  Jain, <em>Fundamentals of Demand Planning & Forecasting</em> (Graceway, 2020) &nbsp;|&nbsp;
  <span style="color:#93c5fd;font-weight:600;">S&OP:</span>
  Crum & Palmatier, <em>Demand Management Best Practices</em>
</div>""", unsafe_allow_html=True)

    st.stop()

# ─── DATA LOADED — MAIN APP ───────────────────────────────────────────────────

df = st.session_state["df"]
demand_arr = df["Demand"].values.astype(float)
periods = df["Period"].values
n = len(demand_arr)
n_train = max(int(n * 0.8), n - 4)
n_holdout = n - n_train
profile = utils.compute_profile(demand_arr)
cv_class = utils.classify_cv(profile["cv"])

# Page header
st.markdown(f"""
<div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;
padding-bottom:0.75rem;border-bottom:1px solid #1e2d45;">
  <div>
    <div style="font-size:1.4rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.02em;">DemandIQ</div>
    <div style="font-size:0.78rem;color:#94a3b8;">
      SKU: <b style="color:#cbd5e1;">{df['SKU'].iloc[0]}</b> &nbsp;|&nbsp;
      {n} periods &nbsp;|&nbsp;
      Mean: <b style="color:#cbd5e1;">{profile['mean']:.0f}</b> units &nbsp;|&nbsp;
      CV: <b style="color:{cv_class['color']};">{profile['cv']*100:.1f}%</b>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Profile & Segmentation",
    "Forecast Engine",
    "Forward Forecast & Safety Stock",
    "Forecast Accuracy Tracker",
    "S&OP Executive Summary"
])

# ═══════════════════════════════════════════════════════════
# TAB 1
# ═══════════════════════════════════════════════════════════

with tab1:
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    section("Demand Profile Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(card("Total Periods", f"{profile['n']}", "n"), unsafe_allow_html=True)
    with c2: st.markdown(card("Total Demand", f"{profile['total']:,.0f}", "units — sum(D_t)"), unsafe_allow_html=True)
    with c3: st.markdown(card("Mean Demand", f"{profile['mean']:,.1f}", "units/period — D-bar"), unsafe_allow_html=True)
    with c4: st.markdown(card("Std Deviation", f"{profile['std']:,.1f}", "units (sample std, ddof=1)"), unsafe_allow_html=True)

    section("Demand Pattern Classification — ABC-XYZ")
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown(f"""
<div style="background:#111827;border:1px solid #1e2d45;border-left:5px solid {cv_class['color']};
border-radius:10px;padding:1.25rem 1.5rem;">
  <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;margin-bottom:0.5rem;">Classification</div>
  <span style="background:{cv_class['color']};color:#111827;font-size:0.85rem;font-weight:600;
  padding:0.3rem 0.9rem;border-radius:999px;">{cv_class['label']}</span>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:1.8rem;font-weight:700;
  color:#f1f5f9;margin:0.6rem 0 0.25rem;">CV = {profile['cv']:.3f}</div>
  <div style="font-size:0.82rem;color:#94a3b8;">{cv_class['description']}</div>
  <div style="font-size:0.68rem;color:#94a3b8;font-style:italic;margin-top:0.5rem;">
    Source: Vandeput, DFBP Ch. 13 — ABC-XYZ segmentation<br>
    CV = std / mean (sample standard deviation)
  </div>
</div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown("""
<div style="background:#111827;border:1px solid #1e2d45;border-radius:10px;padding:1.25rem 1.5rem;">
<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;margin-bottom:0.75rem;">Classification Guide</div>
<table style="width:100%;font-size:0.8rem;border-collapse:collapse;">
<tr style="border-bottom:1px solid #e2e8f0;">
  <th style="text-align:left;padding:0.4rem 0.5rem;color:#94a3b8;font-weight:600;">Class</th>
  <th style="text-align:left;padding:0.4rem 0.5rem;color:#94a3b8;font-weight:600;">CV Range</th>
  <th style="text-align:left;padding:0.4rem 0.5rem;color:#94a3b8;font-weight:600;">Interpretation</th>
  <th style="text-align:left;padding:0.4rem 0.5rem;color:#94a3b8;font-weight:600;">Recommended Model</th>
</tr>
<tr style="border-bottom:1px solid #1e2d45;">
  <td style="padding:0.4rem 0.5rem;"><span style="color:#16a34a;font-weight:700;">X</span></td>
  <td style="padding:0.4rem 0.5rem;font-family:'IBM Plex Mono',monospace;">&lt; 0.20</td>
  <td style="padding:0.4rem 0.5rem;">Stable — predictable demand</td>
  <td style="padding:0.4rem 0.5rem;">SES or Holt Linear</td>
</tr>
<tr style="border-bottom:1px solid #1e2d45;">
  <td style="padding:0.4rem 0.5rem;"><span style="color:#d97706;font-weight:700;">Y</span></td>
  <td style="padding:0.4rem 0.5rem;font-family:'IBM Plex Mono',monospace;">0.20 – 0.50</td>
  <td style="padding:0.4rem 0.5rem;">Variable — adaptive methods needed</td>
  <td style="padding:0.4rem 0.5rem;">Holt-Winters or ARIMA</td>
</tr>
<tr>
  <td style="padding:0.4rem 0.5rem;"><span style="color:#dc2626;font-weight:700;">Z</span></td>
  <td style="padding:0.4rem 0.5rem;font-family:'IBM Plex Mono',monospace;">&gt; 0.50</td>
  <td style="padding:0.4rem 0.5rem;">Lumpy — statistical models may fail</td>
  <td style="padding:0.4rem 0.5rem;">Manual override + safety stock</td>
</tr>
</table>
</div>""", unsafe_allow_html=True)

    section("Seasonality Detection")
    decomp = utils.run_seasonal_decompose(tuple(demand_arr), period=seasonal_periods)
    season_info = utils.detect_seasonality(decomp)
    if season_info["detected"]:
        alert(f"Seasonal pattern detected (estimated period: {seasonal_periods}). Holt-Winters or seasonal ARIMA recommended.", "warning")
    else:
        alert(season_info["message"], "success")
    cite("Source: Hyndman & Athanasopoulos, FPP3 Ch. 3 — Seasonal decomposition (additive model). Seasonal flagged when seasonal std > 10% of trend std.")

    if decomp is not None:
        fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                    subplot_titles=["Observed","Trend","Seasonal","Residual"],
                                    vertical_spacing=0.06)
        panels = [("Observed", demand_arr, BLUE), ("Trend", decomp.trend, "#7c3aed"),
                  ("Seasonal", decomp.seasonal, "#0891b2"), ("Residual", decomp.resid, "#94a3b8")]
        for i, (name, vals, color) in enumerate(panels, 1):
            fig_decomp.add_trace(go.Scatter(x=list(range(n)), y=vals, name=name,
                                            line=dict(color=color, width=1.5), showlegend=False), row=i, col=1)
        for i in range(1, 5):
            fig_decomp.update_xaxes(gridcolor="#f1f5f9", linecolor="#e2e8f0", row=i, col=1)
            fig_decomp.update_yaxes(gridcolor="#f1f5f9", linecolor="#e2e8f0", row=i, col=1)
        fig_decomp.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="#fff", font=dict(family="Inter", size=11, color="#334155"),
                                  margin=dict(l=40, r=20, t=40, b=20))
        st.plotly_chart(fig_decomp, width='stretch')
        insight("Decomposition separates demand into trend (direction), seasonal (repeating pattern), and residual (noise). If the seasonal panel shows a consistent repeating wave, use Holt-Winters.")
    else:
        alert("Seasonal decomposition requires at least 24 periods. Upload more data to enable this view.", "info")

    section("Demand History")
    sma6 = pd.Series(demand_arr).rolling(6).mean().values
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(x=list(range(n)), y=demand_arr, name="Demand",
                               marker_color=BLUE, opacity=0.75, marker_line_width=0))
    fig_hist.add_trace(go.Scatter(x=list(range(n)), y=sma6, name="6-Period SMA",
                                   mode="lines", line=dict(color=AMBER, width=2.5, dash="dash")))
    fig_hist.update_layout(height=300, xaxis_title="Period", yaxis_title="Demand (units)", **CHART)
    st.plotly_chart(fig_hist, width='stretch')
    insight("The 6-period moving average removes noise to reveal the underlying trend. React to sustained moves in the dashed line, not individual spikes.")


# ═══════════════════════════════════════════════════════════
# TAB 2
# ═══════════════════════════════════════════════════════════

with tab2:
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    section("Cross-Validation Setup")
    st.markdown(f"""
<div style="background:#111827;border:1px solid #1e2d45;border-radius:10px;padding:1rem 1.25rem;">
<span style="font-size:0.85rem;color:#cbd5e1;">
<b>Training:</b> periods 1–{n_train} &nbsp;|&nbsp;
<b>Holdout:</b> periods {n_train+1}–{n} &nbsp;|&nbsp;
<b>Holdout size:</b> {n_holdout} period(s)
</span>
</div>""", unsafe_allow_html=True)
    cite("Source: FPP3 Ch. 5 — Rolling origin cross-validation. Any method that cannot beat Naive is not considered useful.")

    demand_tuple = tuple(demand_arr)
    with st.spinner("Fitting 6 models and computing holdout accuracy metrics..."):
        r_naive = utils.fit_naive(demand_tuple, n_train)
        r_sma   = utils.fit_sma(demand_tuple, n_train, k=sma_k)
        r_ses   = utils.fit_ses(demand_tuple, n_train)
        r_holt  = utils.fit_holt(demand_tuple, n_train)
        r_hw    = utils.fit_holtwinters(demand_tuple, n_train, seasonal_periods)
        r_arima = utils.fit_arima(demand_tuple, n_train)

    all_results = [r_naive, r_sma, r_ses, r_holt, r_hw, r_arima]

    section("Fitted Model Parameters")
    pc1, pc2, pc3 = st.columns(3)
    for col, r in zip([pc1, pc2, pc3], [r_ses, r_holt, r_hw]):
        with col:
            if r.get("error"):
                alert(f"{r['method']}: {r['error']}", "warning")
            else:
                st.markdown(f"""<div style="background:#111827;border:1px solid #1e2d45;border-radius:8px;
padding:0.85rem 1rem;"><div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;
letter-spacing:0.07em;">{r['method']}</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;color:#f1f5f9;margin-top:0.25rem;">
{r.get('params','')}</div></div>""", unsafe_allow_html=True)

    if r_arima.get("error"):
        alert(f"ARIMA: {r_arima['error']}", "warning")
    else:
        st.markdown(f"""<div style="background:#111827;border:1px solid #1e2d45;border-radius:8px;
padding:0.85rem 1rem;margin-top:0.5rem;">
<div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.07em;">ARIMA (AICc-selected)</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;color:#f1f5f9;margin-top:0.25rem;">{r_arima.get('params','')}</div>
<div style="font-size:0.68rem;color:#94a3b8;font-style:italic;margin-top:0.25rem;">AICc = AIC + (2k² + 2k) / (n − k − 1) — Source: FPP3 Ch. 9</div>
</div>""", unsafe_allow_html=True)

    section("Holdout Period — Forecast vs Actuals")
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=list(range(n)), y=demand_arr, name="Actual",
                                 mode="lines+markers", line=dict(color="#334155", width=2), marker=dict(size=4)))
    colors_fc = [BLUE, AMBER, GREEN, "#7c3aed", "#0891b2", RED]
    dashes_fc = ["solid","dash","dot","dashdot","longdash","dash"]
    for i, r in enumerate(all_results):
        if r.get("error") or r.get("fitted") is None:
            continue
        fitted = r["fitted"]
        hf = np.full(n, np.nan)
        if len(fitted) > n_train:
            hf[n_train:] = fitted[n_train:]
        fig_fc.add_trace(go.Scatter(x=list(range(n)), y=hf, name=r["method"], mode="lines",
                                     line=dict(color=colors_fc[i%len(colors_fc)], width=1.8,
                                               dash=dashes_fc[i%len(dashes_fc)])))
    fig_fc.add_vrect(x0=n_train-0.5, x1=n-0.5, fillcolor="rgba(27,110,243,0.06)",
                      line_width=0, annotation_text="Holdout", annotation_position="top left")
    fig_fc.update_layout(height=340, xaxis_title="Period", yaxis_title="Demand (units)", **CHART)
    st.plotly_chart(fig_fc, width='stretch')
    insight("Compare forecast lines (coloured) to the black actual line in the shaded holdout window — the line that tracks actuals most closely here will likely produce the lowest Vandeput Score.")

    section("Holdout Accuracy — Ranked by Vandeput Score")
    results_df = utils.build_results_table(all_results)

    if results_df.empty:
        alert("No models produced valid holdout forecasts. Upload more data (minimum 6 periods).", "warning")
    else:
        display_df = results_df.drop(columns=["_metrics"], errors="ignore").copy()

        def fa_style(val):
            if pd.isna(val): return ""
            if val >= 80: return "color: #16a34a; font-weight: 700;"
            if val >= 65: return "color: #d97706; font-weight: 700;"
            return "color: #dc2626; font-weight: 700;"

        # Build HTML table — dark-theme safe
        cols = [c for c in display_df.columns if c != "_metrics"]
        header_html = "".join(f'<th style="padding:0.45rem 0.85rem;text-align:left;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.07em;color:#475569;font-weight:600;white-space:nowrap;">{c}</th>' for c in cols)
        rows_html_t = ""
        for i, (_, row) in enumerate(display_df[cols].iterrows()):
            bg = "background:#0f2318;" if i == 0 else "background:#111827;"
            cells = ""
            for c in cols:
                val = row[c]
                # Color FA%
                if c == "FA%":
                    try:
                        fv = float(val)
                        fc_color = "#22c55e" if fv >= 80 else "#fbbf24" if fv >= 65 else "#f87171"
                        cell_val = f'<span style="color:{fc_color};font-weight:700;font-family:IBM Plex Mono,monospace;">{fv:.1f}%</span>'
                    except:
                        cell_val = str(val)
                elif c in ("Vandeput Score","MAE","Bias","RMSE"):
                    try: cell_val = f'<span style="font-family:IBM Plex Mono,monospace;">{float(val):.2f}</span>'
                    except: cell_val = str(val)
                elif i == 0 and c == "Method":
                    cell_val = f'<span style="color:#22c55e;font-weight:700;">{val} ★</span>'
                else:
                    cell_val = str(val)
                cells += f'<td style="padding:0.4rem 0.85rem;font-size:0.82rem;color:#cbd5e1;border-bottom:1px solid #1e2d45;">{cell_val}</td>'
            rows_html_t += f'<tr style="{bg}">{cells}</tr>'
        st.markdown(f'''<div style="background:#111827;border:1px solid #1e2d45;border-radius:10px;overflow:hidden;overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;min-width:600px;">
<thead><tr style="background:#0d1724;border-bottom:2px solid #1e3a6e;">{header_html}</tr></thead>
<tbody>{rows_html_t}</tbody>
</table></div>''', unsafe_allow_html=True)
        cite("Vandeput Score = MAE + |Bias| (primary ranking metric, DFBP Ch. 8–9) | WMAPE = weighted MAE / total demand (handles near-zero, DFBP Ch. 9) | MAPE* displayed for reference only — do not use for ranking (DFBP Ch. 8 caution) | RMSE for safety stock only (FPP3 Ch. 5.8)")

        best_row = results_df.iloc[0]
        bm_name = best_row["Method"]
        bm_metrics = best_row["_metrics"]
        st.session_state["best_method_key"] = bm_name
        st.session_state["best_metrics"] = bm_metrics

        section("Industry Benchmark Comparison")
        user_fa = bm_metrics.get("fa_pct", np.nan)
        benchmarks = [("Manufacturing", 80, "Discrete / component manufacturing"),
                      ("FMCG / Consumer Goods", 85, "High-volume consumer products"),
                      ("High-tech / Electronics", 70, "Short lifecycles, demand shocks")]
        bc1, bc2, bc3 = st.columns(3)
        for col, (industry, wc, desc) in zip([bc1, bc2, bc3], benchmarks):
            with col:
                gap = user_fa - wc if not np.isnan(user_fa) else None
                color = (GREEN if gap is not None and gap >= 0 else AMBER if gap is not None and gap >= -10 else RED)
                status = ("Above world-class" if gap is not None and gap >= 0
                          else f"{abs(gap):.1f}pp below world-class" if gap is not None else "N/A")
                fa_str = f"{user_fa:.1f}%" if not np.isnan(user_fa) else "N/A"
                st.markdown(f"""
<div style="background:#111827;border:1px solid #1e2d45;border-radius:10px;padding:1rem 1.25rem;">
<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.07em;color:#94a3b8;margin-bottom:0.5rem;">{industry}</div>
<div style="font-size:0.72rem;color:#64748b;margin-bottom:0.5rem;">{desc}</div>
<div style="display:flex;justify-content:space-between;align-items:flex-end;">
  <div><div style="font-size:0.65rem;color:#94a3b8;">World-class</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;font-weight:600;color:#cbd5e1;">{wc}%</div></div>
  <div style="text-align:right;"><div style="font-size:0.65rem;color:#94a3b8;">Your FA%</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;font-weight:700;color:{color};">{fa_str}</div></div>
</div>
<div style="font-size:0.72rem;color:{color};font-weight:600;margin-top:0.35rem;">{status}</div>
</div>""", unsafe_allow_html=True)
        cite("Source: Vandeput, DFBP Ch. 10 — Forecasting benchmarks by industry segment")

        section("Automated Method Recommendation")
        cat = cv_class["category"]
        if cat == "X":
            rationale = "Stable demand (CV < 0.2) is well-suited to statistical smoothing; SES or ARIMA efficiently track the level without overfitting noise (FPP3 Ch. 8.1)."
        elif cat == "Y":
            rationale = "Variable demand benefits from adaptive methods that update level and trend continuously. Monitor bias closely and retrain monthly (Vandeput, DFBP Ch. 13)."
        else:
            rationale = "Highly variable demand may exceed statistical model assumptions. Supplement with safety stock buffers and manual review (Vandeput, DFBP Ch. 13)."
        vs_val = bm_metrics.get("vandeput_score", 0)
        bias_dir = bm_metrics.get("bias_direction", "N/A")
        fa_str2 = f"{user_fa:.0f}%" if not np.isnan(user_fa) else "N/A"
        st.markdown(f"""
<div style="background:#111827;border:1px solid #1e2d45;border-left:4px solid {BLUE};
border-radius:10px;padding:1.25rem 1.5rem;">
<div style="font-size:0.88rem;color:#cbd5e1;line-height:1.75;">
Based on your demand pattern (<b>{cv_class['label']}</b>, CV={profile['cv']*100:.1f}%) and holdout
evaluation across <b>{n_holdout} periods</b>, <b>{bm_name}</b> performs best with a Vandeput Score
of <span style="font-family:'IBM Plex Mono',monospace;font-weight:600;">{vs_val:.2f} units</span>,
Forecast Accuracy of <b>{fa_str2}</b>, and a Bias of
<span style="font-family:'IBM Plex Mono',monospace;">{bm_metrics.get('bias',0):+.1f} units</span>
({bias_dir.lower()}). {rationale}
</div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 3
# ═══════════════════════════════════════════════════════════

with tab3:
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    section("Forward Forecast Configuration")

    method_map = {
        "Naive":"Naive", f"SMA (k={sma_k})":"SMA",
        "SES":"SES", "Holt Linear":"Holt Linear",
        "Holt-Winters":"Holt-Winters", "ARIMA (AICc-selected)":"ARIMA"
    }
    bk = st.session_state.get("best_method_key","")
    default_label = next((lbl for lbl, key in method_map.items() if bk and bk.startswith(key)), "SES")
    method_labels = list(method_map.keys())
    def_idx = method_labels.index(default_label) if default_label in method_labels else 0

    col_m, col_h = st.columns(2)
    with col_m: method_sel = st.selectbox("Forecast Method", method_labels, index=def_idx)
    with col_h: horizon = st.slider("Forecast Horizon (periods)", 1, 24, 6)

    method_key = method_map[method_sel]
    fc_result = utils.generate_forward_forecast(tuple(demand_arr), method_key, horizon,
                                                 k=sma_k, seasonal_periods=seasonal_periods)

    if fc_result.get("error"):
        alert(fc_result["error"], "danger")
    else:
        point = fc_result["point"]
        lo80, hi80 = fc_result["lower_80"], fc_result["upper_80"]
        lo95, hi95 = fc_result["lower_95"], fc_result["upper_95"]
        st.session_state["forward_fc"] = fc_result
        st.session_state["forward_method"] = method_sel

        section("Alert Zone Configuration")
        col_min, col_max = st.columns(2)
        with col_min:
            min_thresh = st.number_input("Min Demand Threshold (stockout risk below this)",
                                          min_value=0, value=int(profile["mean"]*0.7), step=10)
        with col_max:
            max_thresh = st.number_input("Max Demand Threshold (overstock risk above this)",
                                          min_value=0, value=int(profile["mean"]*1.3), step=10)

        zone_colors = ["red" if p < min_thresh else "amber" if p > max_thresh else "green" for p in point]
        ac = {"red": zone_colors.count("red"), "amber": zone_colors.count("amber"), "green": zone_colors.count("green")}
        st.session_state["alert_counts"] = ac

        future_x = list(range(n, n+horizon))
        fig_fwd = go.Figure()
        fig_fwd.add_trace(go.Scatter(x=list(range(n)), y=demand_arr, name="Historical",
                                      mode="lines+markers", line=dict(color="#334155", width=2), marker=dict(size=4)))
        fig_fwd.add_trace(go.Scatter(x=future_x+future_x[::-1], y=list(hi95)+list(lo95[::-1]),
                                      fill="toself", fillcolor="rgba(27,110,243,0.07)",
                                      line=dict(color="rgba(0,0,0,0)"), name="95% PI", hoverinfo="skip"))
        fig_fwd.add_trace(go.Scatter(x=future_x+future_x[::-1], y=list(hi80)+list(lo80[::-1]),
                                      fill="toself", fillcolor="rgba(27,110,243,0.14)",
                                      line=dict(color="rgba(0,0,0,0)"), name="80% PI", hoverinfo="skip"))
        for i, (fx, p) in enumerate(zip(future_x, point)):
            fill = {"red":"rgba(220,38,38,0.12)","amber":"rgba(217,119,6,0.12)","green":"rgba(22,163,74,0.08)"}[zone_colors[i]]
            fig_fwd.add_vrect(x0=fx-0.5, x1=fx+0.5, fillcolor=fill, line_width=0)
        fig_fwd.add_trace(go.Scatter(x=future_x, y=point, name="Point Forecast", mode="lines+markers",
                                      line=dict(color=BLUE, width=2.5),
                                      marker=dict(size=8, color=[{"red":RED,"amber":AMBER,"green":GREEN}[z] for z in zone_colors])))
        fig_fwd.add_hline(y=min_thresh, line_dash="dot", line_color=RED,
                           annotation_text="Min Threshold", annotation_position="bottom right")
        fig_fwd.add_hline(y=max_thresh, line_dash="dot", line_color=ORANGE,
                           annotation_text="Max Threshold", annotation_position="top right")
        fig_fwd.update_layout(height=380, xaxis_title="Period", yaxis_title="Demand (units)", **CHART)
        st.plotly_chart(fig_fwd, width='stretch')
        insight("Prediction intervals widen with horizon (sigma_h = RMSE × sqrt(h) — FPP3 Ch. 5.5). Red markers signal stockout risk — prioritise supply actions for those periods first.")
        cite("Note: Intervals assume normally distributed forecast errors. Treat as approximate for lumpy or intermittent demand.")

        az1, az2, az3 = st.columns(3)
        with az1: st.markdown(card("Stockout Risk Periods", str(ac["red"]), "Below min threshold", RED, RED), unsafe_allow_html=True)
        with az2: st.markdown(card("Overstock Risk Periods", str(ac["amber"]), "Above max threshold", AMBER, AMBER), unsafe_allow_html=True)
        with az3: st.markdown(card("Within Plan Periods", str(ac["green"]), "Between thresholds", GREEN, GREEN), unsafe_allow_html=True)

        section("Forward Forecast Table")
        fc_table = pd.DataFrame({
            "Period": [f"t+{i+1}" for i in range(horizon)],
            "Point Forecast": np.round(point, 0).astype(int),
            "Lower 80%": np.round(lo80, 0).astype(int),
            "Upper 80%": np.round(hi80, 0).astype(int),
            "Lower 95%": np.round(lo95, 0).astype(int),
            "Upper 95%": np.round(hi95, 0).astype(int),
            "Zone": [{"red":"Stockout Risk","amber":"Overstock Risk","green":"Within Plan"}[z] for z in zone_colors]
        })
        st.session_state["fc_table"] = fc_table
        # HTML forecast table
        fc_cols = ["Period","Point Forecast","Lower 80%","Upper 80%","Lower 95%","Upper 95%","Zone"]
        zone_colors_map = {"Stockout Risk": "#f87171", "Overstock Risk": "#fbbf24", "Within Plan": "#22c55e"}
        fc_header = "".join(f'<th style="padding:0.4rem 0.75rem;text-align:{"right" if i>0 else "left"};font-size:0.68rem;text-transform:uppercase;letter-spacing:0.07em;color:#475569;font-weight:600;">{c}</th>' for i,c in enumerate(fc_cols))
        fc_rows = ""
        for _, row in fc_table.iterrows():
            zc = zone_colors_map.get(str(row["Zone"]), "#94a3b8")
            cells = f'<td style="padding:0.38rem 0.75rem;color:#94a3b8;font-family:IBM Plex Mono,monospace;font-size:0.8rem;border-bottom:1px solid #1e2d45;">{row["Period"]}</td>'
            for col in ["Point Forecast","Lower 80%","Upper 80%","Lower 95%","Upper 95%"]:
                cells += f'<td style="padding:0.38rem 0.75rem;text-align:right;color:#f1f5f9;font-family:IBM Plex Mono,monospace;font-size:0.8rem;border-bottom:1px solid #1e2d45;">{int(row[col]):,}</td>'
            cells += f'<td style="padding:0.38rem 0.75rem;font-size:0.78rem;font-weight:600;color:{zc};border-bottom:1px solid #1e2d45;">{row["Zone"]}</td>'
            fc_rows += f"<tr>{cells}</tr>"
        st.markdown(f'''<div style="background:#111827;border:1px solid #1e2d45;border-radius:10px;overflow:hidden;overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;">
<thead><tr style="background:#0d1724;border-bottom:2px solid #1e3a6e;">{fc_header}</tr></thead>
<tbody>{fc_rows}</tbody>
</table></div>''', unsafe_allow_html=True)

    section("Safety Stock Calculator")
    bm = st.session_state.get("best_metrics", {})
    rmse_val = bm.get("rmse", profile["std"])
    if not rmse_val or np.isnan(rmse_val): rmse_val = profile["std"]

    ss1, ss2 = st.columns(2)
    with ss1: service_level = st.select_slider("Target Service Level", [90,95,99], value=95, format_func=lambda x: f"{x}%")
    with ss2: lead_time = st.slider("Lead Time (periods)", 1, 12, 2)

    ss_res = utils.compute_safety_stock(rmse_val, lead_time, service_level, profile["mean"])
    st.session_state["ss_result"] = ss_res
    z_labels = {90:"1.282", 95:"1.645", 99:"2.326"}
    z_str = z_labels.get(service_level, "1.645")

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"""
<div style="background:#111827;border:1px solid #1e2d45;border-left:4px solid {BLUE};border-radius:10px;padding:1.25rem 1.5rem;">
<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;margin-bottom:0.25rem;">Safety Stock</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:1.8rem;font-weight:700;color:#f1f5f9;">{ss_res['ss']:,.0f} <span style="font-size:0.9rem;color:#94a3b8;">units</span></div>
<div style="font-size:0.75rem;color:#64748b;margin:0.3rem 0 0.75rem;">{service_level}% service level | {lead_time}-period lead time</div>
<div style="background:#0d1724;border:1px solid #1e3a6e;border-radius:6px;padding:0.7rem 0.85rem;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#60a5fa;line-height:1.9;">
SS = Z × sigma_D × sqrt(L)<br>
SS = {z_str} × {rmse_val:.1f} × sqrt({lead_time})<br>
SS = {z_str} × {rmse_val:.1f} × {np.sqrt(lead_time):.3f}<br>
<b>SS = {ss_res['ss']:.1f} units</b>
</div>
<div style="font-size:0.68rem;color:#94a3b8;font-style:italic;margin-top:0.5rem;">Source: Jain, FDPF — Safety stock for normally distributed demand. Assumes fixed lead time.</div>
</div>""", unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
<div style="background:#111827;border:1px solid #1e2d45;border-left:4px solid {GREEN};border-radius:10px;padding:1.25rem 1.5rem;">
<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;margin-bottom:0.25rem;">Reorder Point</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:1.8rem;font-weight:700;color:#f1f5f9;">{ss_res['rop']:,.0f} <span style="font-size:0.9rem;color:#94a3b8;">units</span></div>
<div style="font-size:0.75rem;color:#64748b;margin:0.3rem 0 0.75rem;">Replenish when inventory falls to this level</div>
<div style="background:#0d1724;border:1px solid #1e3a6e;border-radius:6px;padding:0.7rem 0.85rem;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#60a5fa;line-height:1.9;">
ROP = D_bar × L + SS<br>
ROP = {profile['mean']:.1f} × {lead_time} + {ss_res['ss']:.1f}<br>
ROP = {profile['mean']*lead_time:.1f} + {ss_res['ss']:.1f}<br>
<b>ROP = {ss_res['rop']:.1f} units</b>
</div>
<div style="font-size:0.68rem;color:#94a3b8;font-style:italic;margin-top:0.5rem;">For variable lead times, consult your supply planner (Jain, FDPF).</div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 4
# ═══════════════════════════════════════════════════════════

with tab4:
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    section("Forecast vs Actual Input")

    input_mode = st.radio("Input method", ["Manual Entry","Upload CSV"], horizontal=True)
    tracker_df = None

    if input_mode == "Upload CSV":
        tf = st.file_uploader("CSV: Period, Forecast, Actual", type=["csv"], key="tracker")
        if tf:
            try:
                raw_t = pd.read_csv(tf)
                raw_t.columns = [c.strip().lower() for c in raw_t.columns]
                raw_t = raw_t.rename(columns={"period":"Period","forecast":"Forecast","actual":"Actual"})
                if not all(c in raw_t.columns for c in ["Forecast","Actual"]):
                    st.error("CSV must contain Forecast and Actual columns.")
                else:
                    tracker_df = raw_t[["Period","Forecast","Actual"]].dropna()
            except Exception as ex:
                st.error(f"Could not read: {ex}")
    else:
        # Manual entry using number inputs — dark-theme safe alternative to st.data_editor
        st.markdown(f"""<div style="background:#111827;border:1px solid #1e2d45;border-radius:10px;
padding:1rem 1.25rem;margin-bottom:0.75rem;">
<div style="display:grid;grid-template-columns:80px 1fr 1fr;gap:0.5rem;margin-bottom:0.5rem;">
  <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.07em;color:#475569;padding:0 0.25rem;">Period</div>
  <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.07em;color:#475569;padding:0 0.25rem;">Forecast</div>
  <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.07em;color:#475569;padding:0 0.25rem;">Actual</div>
</div></div>""", unsafe_allow_html=True)

        default_forecasts = [450, 470, 430, 490, 510, 480]
        default_actuals   = [420, 490, 410, 520, 495, 500]
        n_rows = st.slider("Number of periods", min_value=2, max_value=12, value=6, key="tracker_rows")

        periods_list, forecasts_list, actuals_list = [], [], []
        for i in range(n_rows):
            c_p, c_f, c_a = st.columns([1, 2, 2])
            with c_p:
                st.markdown(f'<div style="padding:0.6rem 0;font-family:IBM Plex Mono,monospace;font-size:0.85rem;color:#64748b;">P{i+1}</div>', unsafe_allow_html=True)
            with c_f:
                fc_val = st.number_input("", min_value=0, value=default_forecasts[i] if i < len(default_forecasts) else 500,
                                          key=f"fc_{i}", label_visibility="collapsed")
            with c_a:
                ac_val = st.number_input("", min_value=0, value=default_actuals[i] if i < len(default_actuals) else 480,
                                          key=f"ac_{i}", label_visibility="collapsed")
            periods_list.append(f"P{i+1}")
            forecasts_list.append(float(fc_val))
            actuals_list.append(float(ac_val))

        tracker_df = pd.DataFrame({"Period": periods_list, "Forecast": forecasts_list, "Actual": actuals_list})

    if tracker_df is not None and len(tracker_df) > 0:
        actual_t = tracker_df["Actual"].values.astype(float)
        forecast_t = tracker_df["Forecast"].values.astype(float)
        errors_t = actual_t - forecast_t
        metrics_t = utils.compute_metrics(actual_t, forecast_t)

        section("Accuracy Metrics")
        m1, m2, m3, m4 = st.columns(4)
        vs_val = metrics_t.get("vandeput_score", np.nan)
        fa_val = metrics_t.get("fa_pct", np.nan)
        bias_val = metrics_t.get("bias", np.nan)
        wmape_val = metrics_t.get("wmape", np.nan)
        fa_color = GREEN if not np.isnan(fa_val) and fa_val >= 80 else AMBER if not np.isnan(fa_val) and fa_val >= 65 else RED
        with m1: st.markdown(card("Vandeput Score", f"{vs_val:.2f}", "MAE + |Bias| — primary metric"), unsafe_allow_html=True)
        with m2: st.markdown(card("Forecast Accuracy", f"{fa_val:.1f}%", "(1 − WMAPE) × 100", fa_color, fa_color), unsafe_allow_html=True)
        with m3: st.markdown(card("Bias", f"{bias_val:+.2f}", metrics_t.get("bias_direction","")), unsafe_allow_html=True)
        with m4:
            wmape_str = f"{wmape_val*100:.1f}%" if not np.isnan(wmape_val) else "N/A"
            st.markdown(card("WMAPE", wmape_str, "Weighted — handles near-zero demand"), unsafe_allow_html=True)

        bt = utils.detect_bias_trend(errors_t)
        if bt["detected"]:
            alert(bt["message"], "danger")

        section("Forecast Error Waterfall")
        bar_cols = [BLUE if e >= 0 else ORANGE for e in errors_t]
        fig_wf = go.Figure()
        fig_wf.add_trace(go.Bar(x=list(tracker_df["Period"].astype(str)), y=errors_t,
                                 marker_color=bar_cols, name="Error (A − F)",
                                 text=[f"{e:+.0f}" for e in errors_t], textposition="outside"))
        fig_wf.add_hline(y=0, line_width=1.5, line_color="#334155")
        fig_wf.update_layout(height=300, xaxis_title="Period", yaxis_title="Error (Actual − Forecast)", **CHART)
        st.plotly_chart(fig_wf, width='stretch')
        insight("Blue bars (above zero) = under-forecast (stockout risk). Orange bars (below zero) = over-forecast (overstock risk). Persistent bars on one side indicate systematic bias requiring process correction.")

        section("Rolling Forecast Accuracy")
        cum_fa = [utils.compute_metrics(actual_t[:i], forecast_t[:i]).get("fa_pct", np.nan) for i in range(1, len(actual_t)+1)]
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=list(tracker_df["Period"].astype(str)), y=cum_fa,
                                      mode="lines+markers", line=dict(color=BLUE, width=2.5), marker=dict(size=7)))
        fig_cum.add_hline(y=80, line_dash="dot", line_color=GREEN, annotation_text="Manufacturing WC 80%")
        fig_cum.add_hline(y=70, line_dash="dot", line_color=AMBER, annotation_text="High-tech WC 70%")
        fig_cum.update_layout(height=280, xaxis_title="Period", yaxis_title="Cumulative FA%", **CHART)
        st.plotly_chart(fig_cum, width='stretch')
        insight("A declining trajectory means recent periods are being forecast less accurately — investigate whether demand patterns have shifted before next planning cycle.")

        section("Industry Benchmark Table")
        bmarks = [("Manufacturing",80),("FMCG / Consumer Goods",85),("High-tech / Electronics",70)]
        bmark_rows = []
        for ind, wc in bmarks:
            if not np.isnan(fa_val):
                gap = fa_val - wc
                status = "Above world-class" if gap >= 0 else ("Within 10pp" if gap >= -10 else "Below world-class")
            else:
                status = "N/A"
            bmark_rows.append({"Industry": ind, "World-Class FA%": f"{wc}%",
                                "Your FA%": f"{fa_val:.1f}%" if not np.isnan(fa_val) else "N/A",
                                "Status": status})
        bm_df = pd.DataFrame(bmark_rows)
        bm_header = "".join(f'<th style="padding:0.4rem 0.85rem;text-align:left;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.07em;color:#475569;font-weight:600;">{c}</th>' for c in bm_df.columns)
        bm_rows_html = ""
        status_colors = {"Above world-class":"#22c55e","Within 10pp":"#fbbf24","Below world-class":"#f87171","N/A":"#94a3b8"}
        for _, row in bm_df.iterrows():
            sc = status_colors.get(str(row["Status"]),"#94a3b8")
            bm_rows_html += f'''<tr style="border-bottom:1px solid #1e2d45;">
<td style="padding:0.4rem 0.85rem;color:#cbd5e1;font-size:0.82rem;">{row["Industry"]}</td>
<td style="padding:0.4rem 0.85rem;color:#94a3b8;font-family:IBM Plex Mono,monospace;font-size:0.82rem;">{row["World-Class FA%"]}</td>
<td style="padding:0.4rem 0.85rem;color:#f1f5f9;font-family:IBM Plex Mono,monospace;font-size:0.82rem;font-weight:700;">{row["Your FA%"]}</td>
<td style="padding:0.4rem 0.85rem;font-size:0.78rem;font-weight:600;color:{sc};">{row["Status"]}</td>
</tr>'''
        st.markdown(f'''<div style="background:#111827;border:1px solid #1e2d45;border-radius:10px;overflow:hidden;">
<table style="width:100%;border-collapse:collapse;">
<thead><tr style="background:#0d1724;border-bottom:2px solid #1e3a6e;">{bm_header}</tr></thead>
<tbody>{bm_rows_html}</tbody>
</table></div>''', unsafe_allow_html=True)
        cite("Source: Vandeput, DFBP Ch. 10 — Forecasting benchmarks by industry segment")
    else:
        alert("Enter forecast vs actual data above to compute accuracy metrics.", "info")


# ═══════════════════════════════════════════════════════════
# TAB 5
# ═══════════════════════════════════════════════════════════

with tab5:
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    cite("Source: Crum & Palmatier, DMBP — consensus demand planning and S&OP review structure | Jain, FDPF — How to report, present and sell forecasts to management")

    bm_key = st.session_state.get("best_method_key","—")
    bm_metrics = st.session_state.get("best_metrics",{})
    ss_r = st.session_state.get("ss_result")
    ac = st.session_state.get("alert_counts")
    fc_tbl = st.session_state.get("fc_table", pd.DataFrame())
    trend_info = utils.compute_trend_slope(demand_arr, periods=3)
    decomp_s = utils.run_seasonal_decompose(tuple(demand_arr), period=seasonal_periods)
    season_info_s = utils.detect_seasonality(decomp_s)

    section("Section 1 — Demand Signal")
    s1, s2, s3 = st.columns(3)
    with s1: st.markdown(card("Demand Pattern", cv_class["label"], f"CV = {profile['cv']*100:.1f}%", cv_class["color"], cv_class["color"]), unsafe_allow_html=True)
    with s2: st.markdown(card("Trend Direction", trend_info["direction"], "Last 3 periods"), unsafe_allow_html=True)
    with s3:
        scolor = AMBER if season_info_s["detected"] else GREEN
        slabel = "Detected" if season_info_s["detected"] else "Not Detected"
        st.markdown(card("Seasonality", slabel, f"Period: {seasonal_periods}", scolor, scolor), unsafe_allow_html=True)

    section("Section 2 — Recommended Forecast Method")
    if bm_key != "—" and bm_metrics:
        fa_bm = bm_metrics.get("fa_pct", np.nan)
        s2c1, s2c2, s2c3, s2c4 = st.columns(4)
        fa_col = GREEN if not np.isnan(fa_bm) and fa_bm >= 80 else AMBER if not np.isnan(fa_bm) and fa_bm >= 65 else RED
        with s2c1: st.markdown(card("Method", bm_key), unsafe_allow_html=True)
        with s2c2: st.markdown(card("Vandeput Score", f"{bm_metrics.get('vandeput_score',0):.2f}", "MAE + |Bias| units"), unsafe_allow_html=True)
        with s2c3: st.markdown(card("Forecast Accuracy", f"{fa_bm:.1f}%" if not np.isnan(fa_bm) else "N/A", "(1 − WMAPE) × 100", fa_col, fa_col), unsafe_allow_html=True)
        with s2c4: st.markdown(card("Bias", f"{bm_metrics.get('bias',0):+.1f}", bm_metrics.get("bias_direction","N/A")), unsafe_allow_html=True)
    else:
        alert("Run Tab 2 (Forecast Engine) to populate this section.", "info")

    section("Section 3 — Forward Outlook (Next 4 Periods)")
    if not fc_tbl.empty:
        sopp_cols = ["Period","Point Forecast","Lower 80%","Upper 80%","Zone"]
        zone_cm = {"Stockout Risk":"#f87171","Overstock Risk":"#fbbf24","Within Plan":"#22c55e"}
        sopp_head = "".join(f'<th style="padding:0.4rem 0.75rem;text-align:{"right" if i>0 and i<4 else "left"};font-size:0.68rem;text-transform:uppercase;letter-spacing:0.07em;color:#475569;font-weight:600;">{c}</th>' for i,c in enumerate(sopp_cols))
        sopp_rows = ""
        for _, row in fc_tbl.head(4).iterrows():
            zc = zone_cm.get(str(row["Zone"]),"#94a3b8")
            sopp_rows += f'''<tr style="border-bottom:1px solid #1e2d45;">
<td style="padding:0.38rem 0.75rem;color:#94a3b8;font-family:IBM Plex Mono,monospace;font-size:0.8rem;">{row["Period"]}</td>
<td style="padding:0.38rem 0.75rem;text-align:right;color:#f1f5f9;font-family:IBM Plex Mono,monospace;font-size:0.8rem;font-weight:600;">{int(row["Point Forecast"]):,}</td>
<td style="padding:0.38rem 0.75rem;text-align:right;color:#94a3b8;font-family:IBM Plex Mono,monospace;font-size:0.8rem;">{int(row["Lower 80%"]):,}</td>
<td style="padding:0.38rem 0.75rem;text-align:right;color:#94a3b8;font-family:IBM Plex Mono,monospace;font-size:0.8rem;">{int(row["Upper 80%"]):,}</td>
<td style="padding:0.38rem 0.75rem;font-size:0.78rem;font-weight:600;color:{zc};">{row["Zone"]}</td>
</tr>'''
        st.markdown(f'''<div style="background:#111827;border:1px solid #1e2d45;border-radius:10px;overflow:hidden;">
<table style="width:100%;border-collapse:collapse;">
<thead><tr style="background:#0d1724;border-bottom:2px solid #1e3a6e;">{sopp_head}</tr></thead>
<tbody>{sopp_rows}</tbody>
</table></div>''', unsafe_allow_html=True)
        cite("Source: FPP3 Ch. 5.5 — Prediction intervals from residual std deviation, assuming normally distributed errors.")
    else:
        alert("Run Tab 3 (Forward Forecast) to populate this section.", "info")

    section("Section 4 — Risks and Alerts")
    if ac:
        if ac["red"] + ac["amber"] == 0:
            alert("No demand alerts in the forecast horizon.", "success")
        else:
            a1, a2, a3 = st.columns(3)
            with a1: st.markdown(card("Stockout Risk Periods", str(ac["red"]), "Below min threshold", RED, RED), unsafe_allow_html=True)
            with a2: st.markdown(card("Overstock Risk Periods", str(ac["amber"]), "Above max threshold", AMBER, AMBER), unsafe_allow_html=True)
            with a3: st.markdown(card("Within Plan Periods", str(ac["green"]), "Between thresholds", GREEN, GREEN), unsafe_allow_html=True)
    else:
        alert("Set alert thresholds in Tab 3 to populate this section.", "info")

    section("Section 5 — Recommended Actions")
    actions = []
    fa_final = bm_metrics.get("fa_pct", np.nan) if bm_metrics else np.nan
    bias_final = bm_metrics.get("bias", 0) if bm_metrics else 0
    if profile["cv"] > 0.5:
        ss_str = f"{ss_r['ss']:.0f}" if ss_r else "N/A"
        actions.append(f"High demand variability (CV={profile['cv']*100:.1f}%). Maintain safety stock of <b>{ss_str} units</b> and review forecast weekly rather than monthly. <em>(Vandeput, DFBP Ch. 13)</em>")
    if profile["mean"] > 0 and abs(bias_final) > 0.1*profile["mean"]:
        actions.append(f"Systematic bias of <b>{bias_final:+.1f} units</b> detected. Review baseline demand assumptions with the commercial team before the next S&OP cycle. <em>(Jain, FDPF)</em>")
    if not np.isnan(fa_final) and fa_final < 70:
        actions.append(f"Forecast accuracy of <b>{fa_final:.1f}%</b> is below the manufacturing world-class threshold of 80%. Recommend running a forecast value-add review. <em>(Vandeput, DFBP Ch. 12)</em>")
    if not actions:
        actions.append("Forecast performance is within acceptable bounds. Maintain current method and cadence.")
    action_html = "".join(f'<li style="margin-bottom:0.6rem;font-size:0.85rem;color:#cbd5e1;">{a}</li>' for a in actions)
    st.markdown(f'<div style="background:#111827;border:1px solid #1e2d45;border-radius:10px;padding:1.25rem 1.5rem;"><ul style="margin:0;padding-left:1.25rem;line-height:1.8;">{action_html}</ul></div>', unsafe_allow_html=True)

    section("Export")
    summary_text = utils.generate_sopp_summary(
        profile, cv_class, season_info_s, trend_info,
        bm_key, bm_metrics, fc_tbl, ss_r, ac
    )
    st.markdown(f'<div style="background:{NAVY};color:#0f1724;font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;line-height:1.7;padding:1.25rem;border-radius:10px;white-space:pre-wrap;">{summary_text}</div>', unsafe_allow_html=True)
    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button("Download Summary (.txt)", data=summary_text,
                            file_name="demandiq_sopp.txt", mime="text/plain", width='stretch')
    with exp2:
        csv_parts = ["=== FORWARD FORECAST ==="]
        if not fc_tbl.empty: csv_parts.append(fc_tbl.to_csv(index=False))
        st.download_button("Download Analysis (.csv)", data="\n".join(csv_parts),
                            file_name="demandiq_analysis.csv", mime="text/csv", width='stretch')

# ─── FOOTER ──────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="margin-top:3rem;padding:1.25rem 1.75rem;background:{NAVY};border-radius:10px;
font-size:0.68rem;color:#94a3b8;line-height:1.9;text-align:center;">
  <span style="color:#93c5fd;font-weight:600;">Forecasting methods:</span>
  Hyndman & Athanasopoulos, Forecasting: Principles and Practice, 3rd ed. (OTexts, 2021) &nbsp;|&nbsp;
  <span style="color:#93c5fd;font-weight:600;">Accuracy metrics:</span>
  Vandeput, Demand Forecasting Best Practices (Manning, 2023) &nbsp;|&nbsp;
  <span style="color:#93c5fd;font-weight:600;">Planning process:</span>
  Jain, Fundamentals of Demand Planning & Forecasting (Graceway, 2020) &nbsp;|&nbsp;
  <span style="color:#93c5fd;font-weight:600;">S&OP:</span>
  Crum & Palmatier, Demand Management Best Practices &nbsp;|&nbsp;
  Built by <span style="color:#1e2d45;font-weight:600;">Rutwik Satish</span> —
  MS Engineering Management + Graduate Certificate in Supply Chain Engineering Management, Northeastern University
</div>""", unsafe_allow_html=True)
