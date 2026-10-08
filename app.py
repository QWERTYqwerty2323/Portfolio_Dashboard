import warnings
warnings.filterwarnings("ignore")

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ============================================================
# ALPHA — FULL PORTFOLIO RESEARCH DASHBOARD
# ============================================================
# Research universe: 15 equities + 3 debt instruments + Gold ETF
# Actual portfolio:      12 equities + 2 debt instruments + Gold ETF
# Phase 1:                first 6 weeks, with WEEKLY rebalancing
# Phase 2:                reselect from full research universe at week 6
# Phase 3:                continue with new 12 + 2 + Gold, weekly rebalancing
# Includes: personas, portfolio performance, weekly weights, company analysis,
# company comparison, risk, market pulse, watchlist-style research, goals,
# debt analysis, valuation and advanced analytics.
# ============================================================

st.set_page_config(
    page_title="ALPHA — Dynamic Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------- responsive UI -------------------------
st.markdown("""
<style>
.block-container {
    width: 100%;
    max-width: 1600px;
    padding-left: clamp(0.75rem, 2vw, 3rem);
    padding-right: clamp(0.75rem, 2vw, 3rem);
    padding-top: clamp(0.75rem, 2vw, 2rem);
}
h1, h2, h3, h4, p, label, [data-testid="stMetricLabel"],
[data-testid="stMetricValue"], [data-testid="stCaptionContainer"] {
    overflow-wrap: anywhere;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: auto !important;
}
[data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"] {
    width: 100% !important;
    max-width: 100% !important;
    overflow: hidden !important;
}
button, input, textarea, select { min-height: 42px; }
[data-baseweb="tab-list"] {
    overflow-x: auto !important;
    white-space: nowrap;
}
[data-baseweb="tab"] { flex: 0 0 auto !important; }

@media (max-width: 768px) {
    .block-container {
        padding-left: 0.65rem !important;
        padding-right: 0.65rem !important;
        padding-top: 0.6rem !important;
    }
    h1 { font-size: 1.65rem !important; line-height: 1.15 !important; }
    h2 { font-size: 1.35rem !important; line-height: 1.2 !important; }
    h3 { font-size: 1.15rem !important; line-height: 1.25 !important; }

    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.55rem !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
    }
    [data-testid="stMetric"] { padding: 0.65rem 0.75rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.35rem !important; }
    [data-testid="stDataFrame"] > div {
        max-width: 100% !important;
        overflow-x: auto !important;
    }
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [role="combobox"] {
        min-height: 44px !important;
    }
}
@media (max-width: 420px) {
    .block-container {
        padding-left: 0.45rem !important;
        padding-right: 0.45rem !important;
    }
    h1 { font-size: 1.45rem !important; }
    h2 { font-size: 1.2rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem !important; }
}
@media (min-width: 769px) and (max-width: 1024px) {
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 48% !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ------------------------- constants -------------------------
START_DATE = pd.Timestamp("2026-08-01")
BENCHMARK = "^NSEI"
GOLD = "GOLDBEES.NS"
CAPITAL = 1_00_00_000
RESELECT_WEEKS = 6
RISK_FREE = 0.07

# 15-company research universe. Only 12 are held in each phase.
EQUITY_META = {
    "MARUTI.NS": ("Maruti Suzuki India Ltd", "Automobiles"),
    "BAJAJ-AUTO.NS": ("Bajaj Auto Ltd", "Automobiles"),
    "M&M.NS": ("Mahindra & Mahindra Ltd", "Automobiles"),
    "HAL.NS": ("Hindustan Aeronautics Ltd", "Defense / Capital Goods"),
    "SIEMENS.NS": ("Siemens Ltd", "Capital Goods"),
    "POLYCAB.NS": ("Polycab India Ltd", "Capital Goods / Manufacturing"),
    "AXISBANK.NS": ("Axis Bank Ltd", "Financials"),
    "BSE.NS": ("BSE Ltd", "Capital Markets"),
    "HINDUNILVR.NS": ("Hindustan Unilever Ltd", "FMCG"),
    "BRITANNIA.NS": ("Britannia Industries Ltd", "FMCG"),
    "ITC.NS": ("ITC Ltd", "FMCG"),
    "PIDILITIND.NS": ("Pidilite Industries Ltd", "Chemicals"),
    "SOLARINDS.NS": ("Solar Industries India Ltd", "Chemicals / Defense"),
    "PIIND.NS": ("PI Industries Ltd", "Agrochemicals"),
    "ANGELONE.NS": ("Angel One Ltd", "Capital Markets"),
}

# 2 earlier debt instruments + 1 additional/new debt instrument.
DEBT_META = {
    "10Y_GSEC": {
        "name": "10Y Indian G-Sec",
        "yield": 0.070,
        "risk": 0.15,
        "sector": "Sovereign Debt",
    },
    "AAA_BOND": {
        "name": "AAA Corporate Bond / Debenture",
        "yield": 0.080,
        "risk": 0.30,
        "sector": "Corporate Debt",
    },
    "SDL_10Y": {
        "name": "10Y State Development Loan (NEW)",
        "yield": 0.0725,
        "risk": 0.22,
        "sector": "State Debt",
    },
}

PE_PEERS = {
    "Automobiles": 28.0,
    "Defense / Capital Goods": 40.0,
    "Capital Goods": 45.0,
    "Capital Goods / Manufacturing": 42.0,
    "Financials": 20.0,
    "Capital Markets": 35.0,
    "FMCG": 55.0,
    "Chemicals": 50.0,
    "Chemicals / Defense": 50.0,
    "Agrochemicals": 30.0,
}

PERSONAS = {
    "Conservative": {
        "equity": 60.0, "debt": 30.0, "gold": 10.0,
        "description": "Lower equity risk, larger fixed-income anchor and gold diversification.",
    },
    "Moderate": {
        "equity": 75.0, "debt": 15.0, "gold": 10.0,
        "description": "Balanced growth/risk profile with meaningful equity exposure.",
    },
    "Aggressive": {
        "equity": 85.0, "debt": 5.0, "gold": 10.0,
        "description": "Higher equity participation and greater sensitivity to market momentum.",
    },
}

YF_TICKERS = list(EQUITY_META) + [GOLD, BENCHMARK]

# ------------------------- helpers -------------------------
def norm_date(x):
    """Convert any Timestamp/date/string to timezone-naive normalized Timestamp."""
    t = pd.Timestamp(x)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t.normalize()


def normalize_index(df):
    if df is None or df.empty:
        return None
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def safe_float(v, default=np.nan):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


def fmt_pct(v):
    return "N/A" if pd.isna(v) else f"{v:+.2f}%"


def fmt_num(v):
    return "N/A" if pd.isna(v) else f"{v:,.2f}"


def fmt_inr(v):
    if pd.isna(v):
        return "N/A"
    if abs(v) >= 1e7:
        return f"₹{v/1e7:.2f} Cr"
    if abs(v) >= 1e5:
        return f"₹{v/1e5:.2f} L"
    return f"₹{v:,.0f}"


@st.cache_data(ttl=1800, show_spinner=False)
def get_history(ticker, period="2y"):
    try:
        h = yf.Ticker(ticker).history(period=period, auto_adjust=False, timeout=12)
        return normalize_index(h)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def get_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def load_histories():
    result = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(get_history, t, "2y"): t for t in YF_TICKERS}
        for f in as_completed(futures):
            t = futures[f]
            try:
                result[t] = f.result()
            except Exception:
                result[t] = None
    return result


HIST = load_histories()


def available_dates():
    h = HIST.get(BENCHMARK)
    return h.index if h is not None and not h.empty else pd.DatetimeIndex([])


def asof(ticker, date):
    h = HIST.get(ticker) if isinstance(ticker, str) else ticker
    if h is None or h.empty:
        return None
    d = norm_date(date)
    x = h[h.index <= d]
    return x if not x.empty else None


def close_asof(ticker, date):
    x = asof(ticker, date)
    if x is None or x.empty or "Close" not in x:
        return np.nan
    return safe_float(x["Close"].iloc[-1])


def first_trade_on_or_after(date):
    dates = available_dates()
    if len(dates) == 0:
        return norm_date(date)
    d = norm_date(date)
    x = dates[dates >= d]
    return x[0] if len(x) else dates[-1]


def last_trade_on_or_before(date):
    dates = available_dates()
    if len(dates) == 0:
        return norm_date(date)
    d = norm_date(date)
    x = dates[dates <= d]
    return x[-1] if len(x) else dates[0]


def daily_returns(ticker, end_date=None):
    h = asof(ticker, end_date) if end_date is not None else HIST.get(ticker)
    if h is None or h.empty:
        return pd.Series(dtype=float)
    return h["Close"].pct_change().dropna()


def return_pct(ticker, end_date, days):
    h = asof(ticker, end_date)
    if h is None or len(h) < 2:
        return np.nan
    end = safe_float(h["Close"].iloc[-1])
    target = norm_date(end_date) - pd.Timedelta(days=days)
    before = h[h.index <= target]
    if before.empty:
        # Use the earliest available point if the exact lookback is unavailable.
        start = safe_float(h["Close"].iloc[0])
    else:
        start = safe_float(before["Close"].iloc[-1])
    return (end / start - 1) * 100 if pd.notna(start) and start else np.nan


def volatility(ticker, end_date, days=42):
    r = daily_returns(ticker, end_date).tail(days)
    return safe_float(r.std() * np.sqrt(252) * 100) if len(r) >= 5 else np.nan


def beta(ticker, end_date):
    a = daily_returns(ticker, end_date).rename("a")
    b = daily_returns(BENCHMARK, end_date).rename("b")
    j = pd.concat([a, b], axis=1).dropna().tail(120)
    if len(j) < 20 or j["b"].var() == 0:
        return np.nan
    return safe_float(j["a"].cov(j["b"]) / j["b"].var())


def sharpe(ticker, end_date, days=252):
    r = daily_returns(ticker, end_date).tail(days)
    if len(r) < 20 or r.std() == 0:
        return np.nan
    return safe_float((r.mean() * 252 - RISK_FREE) / (r.std() * np.sqrt(252)))


def drawdown(ticker, end_date, days=252):
    h = asof(ticker, end_date)
    if h is None or h.empty:
        return np.nan
    s = h["Close"].tail(days)
    return safe_float((s / s.cummax() - 1).min() * 100)


def info_metrics(ticker, end_date):
    info = get_info(ticker)
    px = close_asof(ticker, end_date)
    eps = safe_float(info.get("trailingEps"))
    pe = safe_float(info.get("trailingPE"))
    if pd.isna(pe) and pd.notna(eps) and eps > 0 and pd.notna(px):
        pe = px / eps
    pb = safe_float(info.get("priceToBook"))
    roe = safe_float(info.get("returnOnEquity"))
    if pd.notna(roe) and abs(roe) < 2:
        roe *= 100
    debt_eq = safe_float(info.get("debtToEquity"))
    if pd.notna(debt_eq) and debt_eq > 10:
        debt_eq /= 100
    mcap = safe_float(info.get("marketCap"))
    div_yield = safe_float(info.get("dividendYield"))
    if pd.notna(div_yield) and div_yield < 1:
        div_yield *= 100
    return {
        "Price": px, "EPS": eps, "P/E": pe, "P/B": pb, "ROE %": roe,
        "Debt/Equity": debt_eq, "Market Cap": mcap,
        "Dividend Yield %": div_yield,
        "Beta": beta(ticker, end_date),
    }


def pe_peer(ticker, end_date):
    m = info_metrics(ticker, end_date)
    sector = EQUITY_META[ticker][1]
    peer = PE_PEERS.get(sector, np.nan)
    pe = m["P/E"]
    discount = ((peer - pe) / peer * 100) if pd.notna(pe) and pd.notna(peer) and peer else np.nan
    return pe, peer, discount

# ------------------------- factor scoring -------------------------
def minmax(s, inverse=False, neutral=0.5):
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        out = pd.Series(neutral, index=x.index)
    else:
        lo, hi = x.min(), x.max()
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            out = pd.Series(neutral, index=x.index)
        else:
            out = (x - lo) / (hi - lo)
    if inverse:
        out = 1 - out
    return out.fillna(neutral)


def equity_score_row(ticker, date, selection_window=False):
    r6 = return_pct(ticker, date, 42)
    r3 = return_pct(ticker, date, 21)
    vol = volatility(ticker, date, 42)
    b = beta(ticker, date)
    dd = drawdown(ticker, date, 42 if selection_window else 252)
    pe, peer, disc = pe_peer(ticker, date)
    return {
        "ticker": ticker,
        "Company": EQUITY_META[ticker][0],
        "Sector": EQUITY_META[ticker][1],
        "6W Return %": r6,
        "3W Return %": r3,
        "Volatility %": vol,
        "Beta": b,
        "Drawdown %": dd,
        "P/E": pe,
        "Peer P/E": peer,
        "P/E Discount %": disc,
    }


def rank_equities(date):
    rows = [equity_score_row(t, date, selection_window=True) for t in EQUITY_META]
    df = pd.DataFrame(rows)
    # For initial selection, the same framework is applied using information
    # available at the first trading date; no future data is used.
    mom = minmax(df["6W Return %"])
    short_mom = minmax(df["3W Return %"])
    value = minmax(df["P/E Discount %"])
    risk = minmax(df["Volatility %"], inverse=True)
    beta_score = minmax(df["Beta"], inverse=True)
    dd_score = minmax(df["Drawdown %"], inverse=False)  # less-negative is better
    df["Score"] = (
        0.35 * mom + 0.20 * short_mom + 0.18 * value +
        0.12 * risk + 0.08 * beta_score + 0.07 * dd_score
    ) * 100
    return df.sort_values(["Score", "6W Return %"], ascending=False, na_position="last").reset_index(drop=True)


def rank_debt():
    rows = []
    for t, m in DEBT_META.items():
        score = m["yield"] * 100 - m["risk"] * 2
        rows.append({
            "ticker": t, "Instrument": m["name"], "Yield %": m["yield"] * 100,
            "Model Risk": m["risk"], "Score": score,
        })
    return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

# ------------------------- persona / weekly weights -------------------------
def weekly_weights(selected_eq, selected_debt, date, persona):
    p = PERSONAS[persona]
    eq = pd.DataFrame([equity_score_row(t, date) for t in selected_eq])
    if eq.empty:
        eq_w = {}
    else:
        mom = minmax(eq["6W Return %"])
        val = minmax(eq["P/E Discount %"])
        risk = minmax(eq["Volatility %"], inverse=True)
        quality_proxy = minmax(eq["Beta"], inverse=True)
        score = (0.45 * mom + 0.25 * val + 0.20 * risk + 0.10 * quality_proxy).clip(lower=0.05)
        eq_w = dict(zip(eq["ticker"], score / score.sum() * p["equity"]))

    debt = rank_debt()
    debt = debt[debt["ticker"].isin(selected_debt)]
    if debt.empty:
        debt_w = {}
    else:
        score = debt["Score"].clip(lower=0.01)
        debt_w = dict(zip(debt["ticker"], score / score.sum() * p["debt"]))

    weights = {**eq_w, **debt_w, GOLD: p["gold"]}
    total = sum(weights.values())
    return {k: v * 100 / total for k, v in weights.items()}


def week_start_dates(start, end):
    dates = available_dates()
    if len(dates) == 0:
        return pd.DatetimeIndex([])
    d = dates[(dates >= norm_date(start)) & (dates <= norm_date(end))]
    if len(d) == 0:
        return d
    s = pd.Series(d, index=d)
    return pd.DatetimeIndex(s.groupby(d.to_period("W-FRI")).first().values)


def asset_daily_return(ticker, d0, d1):
    if ticker in DEBT_META:
        y = DEBT_META[ticker]["yield"]
        return (1 + y) ** (1 / 365.25) - 1
    p0 = close_asof(ticker, d0)
    p1 = close_asof(ticker, d1)
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return 0.0
    return p1 / p0 - 1


def simulate_phase(start, end, selected_eq, selected_debt, persona):
    dates = available_dates()
    dates = dates[(dates >= norm_date(start)) & (dates <= norm_date(end))]
    if len(dates) == 0:
        return pd.Series(dtype=float), pd.DataFrame()

    starts = week_start_dates(start, end)
    weight_map = {d: weekly_weights(selected_eq, selected_debt, d, persona) for d in starts}
    current = weight_map.get(dates[0])
    if current is None:
        current = weekly_weights(selected_eq, selected_debt, dates[0], persona)

    values = [CAPITAL]
    rows = [{"Date": dates[0], **current}]
    value = CAPITAL

    for i in range(1, len(dates)):
        d0, d1 = dates[i - 1], dates[i]
        if d1 in weight_map:
            current = weight_map[d1]
            rows.append({"Date": d1, **current})
        r = sum((w / 100) * asset_daily_return(t, d0, d1) for t, w in current.items())
        value *= 1 + r
        values.append(value)

    return pd.Series(values, index=dates), pd.DataFrame(rows)


def benchmark_series(start, end):
    h = asof(BENCHMARK, end)
    if h is None:
        return pd.Series(dtype=float)
    h = h[(h.index >= norm_date(start)) & (h.index <= norm_date(end))]
    if h.empty:
        return pd.Series(dtype=float)
    return CAPITAL * h["Close"] / h["Close"].iloc[0]


def combine_phases(phase1, phase2):
    if phase1.empty:
        return phase2.copy()
    if phase2.empty:
        return phase1.copy()
    # Scale phase 2 so its first point equals phase 1's ending value.
    p2 = phase2 / phase2.iloc[0] * phase1.iloc[-1]
    return pd.concat([phase1, p2.iloc[1:]])

# ------------------------- initial / reselected universe -------------------------
FIRST_DATE = first_trade_on_or_after(START_DATE)
RESELECT_DATE = last_trade_on_or_before(FIRST_DATE + pd.Timedelta(weeks=RESELECT_WEEKS) - pd.Timedelta(days=1))
LATEST_DATE = available_dates()[-1] if len(available_dates()) else pd.Timestamp.today().normalize()

initial_rank = rank_equities(FIRST_DATE)
initial_selected_eq = initial_rank.head(12)["ticker"].tolist()
debt_rank = rank_debt()
initial_selected_debt = debt_rank.head(2)["ticker"].tolist()

# Week-6 selection uses all 15 companies and all 3 debt instruments.
week6_rank = rank_equities(RESELECT_DATE)
selected_eq = week6_rank.head(12)["ticker"].tolist()
selected_debt = debt_rank.head(2)["ticker"].tolist()

# ------------------------- sidebar -------------------------
st.sidebar.title("⚙️ ALPHA Controls")
persona = st.sidebar.selectbox("Investor Persona", list(PERSONAS.keys()), index=1)
st.sidebar.caption(PERSONAS[persona]["description"])
capital = st.sidebar.number_input("Illustrative Capital (₹)", min_value=100000, value=CAPITAL, step=100000)

if st.sidebar.button("🔄 Clear Yahoo cache & reload"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Strategy timeline**")
st.sidebar.write(f"Start: {FIRST_DATE:%d %b %Y}")
st.sidebar.write(f"Week-6 review: {RESELECT_DATE:%d %b %Y}")
st.sidebar.write(f"Latest data: {LATEST_DATE:%d %b %Y}")

# ------------------------- header -------------------------
st.title("📈 ALPHA — Dynamic 6-Week Reselection Portfolio")
st.caption("15-company research universe + 3 debt instruments + Gold ETF | Actual holdings always = 12 equities + 2 debt + Gold")

c = st.columns(5)
c[0].metric("Research Equities", "15")
c[1].metric("Research Debt", "3")
c[2].metric("Actual Equity Holdings", "12")
c[3].metric("Actual Debt Holdings", "2")
c[4].metric("Gold", "1 ETF")

st.info(
    f"**Investment start:** {FIRST_DATE:%d %b %Y}  •  "
    f"**6-week review:** {RESELECT_DATE:%d %b %Y}  •  "
    f"**Persona:** {persona}  •  "
    f"**Weekly rebalancing:** ON in both phases"
)

# ------------------------- 1 personas -------------------------
st.header("1. 👤 Investor Personas")
pdf = pd.DataFrame(PERSONAS).T.reset_index().rename(columns={"index": "Persona"})
pdf.columns = ["Persona", "Equity %", "Debt %", "Gold %", "Description"]
st.dataframe(pdf, use_container_width=True, hide_index=True)
st.caption("Personas describe the portfolio's risk/allocation profile. They are not personalised investment advice.")

# ------------------------- 2 research universe -------------------------
st.header("2. 🔎 Research Universe")
research_df = pd.DataFrame([
    {"Type": "Equity", "Instrument": n, "Ticker": t, "Sector": s}
    for t, (n, s) in EQUITY_META.items()
] + [
    {"Type": "Debt", "Instrument": m["name"], "Ticker": t, "Sector": m["sector"]}
    for t, m in DEBT_META.items()
] + [{"Type": "Gold", "Instrument": "Nippon India ETF Gold BeES", "Ticker": GOLD, "Sector": "Commodity"}])
st.dataframe(research_df, use_container_width=True, hide_index=True)

# ------------------------- 3 starting selection -------------------------
st.header("3. 🎯 Starting Portfolio — First 6 Weeks")
st.write("The research universe is 15 + 3 + Gold, but the **actual portfolio is only 12 + 2 + Gold**. The same 15 instruments are never held simultaneously.")

init_show = initial_rank.copy()
init_show.insert(0, "Selected", init_show["ticker"].isin(initial_selected_eq).map({True: "✅", False: "—"}))
st.dataframe(init_show.round(2), use_container_width=True, hide_index=True)

debt_show = debt_rank.copy()
debt_show.insert(0, "Selected", debt_show["ticker"].isin(initial_selected_debt).map({True: "✅", False: "—"}))
st.dataframe(debt_show.round(3), use_container_width=True, hide_index=True)

st.success(
    "**Actual Phase-1 holdings:** "
    + ", ".join(EQUITY_META[t][0] for t in initial_selected_eq)
    + " + " + ", ".join(DEBT_META[t]["name"] for t in initial_selected_debt)
    + " + Gold ETF"
)

# ------------------------- 4 phase 1 weekly weights/performance -------------------------
phase1_end = min(RESELECT_DATE, LATEST_DATE)
phase1, weights1 = simulate_phase(FIRST_DATE, phase1_end, initial_selected_eq, initial_selected_debt, persona)
bench1 = benchmark_series(FIRST_DATE, phase1_end)

st.subheader("Weekly Weight Redistribution — FIRST 6 WEEKS")
if not weights1.empty:
    w1 = weights1.copy()
    w1["Date"] = pd.to_datetime(w1["Date"]).dt.strftime("%d %b %Y")
    w1 = w1.rename(columns={t: EQUITY_META[t][0] for t in EQUITY_META})
    w1 = w1.rename(columns={t: DEBT_META[t]["name"] for t in DEBT_META})
    w1 = w1.rename(columns={GOLD: "Gold ETF"})
    st.dataframe(w1.round(2), use_container_width=True, hide_index=True)

    long1 = weights1.melt(id_vars="Date", var_name="Ticker", value_name="Weight %")
    long1["Instrument"] = long1["Ticker"].map({t: EQUITY_META[t][0] for t in EQUITY_META} | {t: DEBT_META[t]["name"] for t in DEBT_META} | {GOLD: "Gold ETF"})
    figw = px.line(long1, x="Date", y="Weight %", color="Instrument", markers=True, title="Weekly portfolio weights — Phase 1")
    st.plotly_chart(figw, use_container_width=True, config={"responsive": True, "displaylogo": False})

if not phase1.empty:
    r1 = (phase1.iloc[-1] / phase1.iloc[0] - 1) * 100
    rb1 = (bench1.iloc[-1] / bench1.iloc[0] - 1) * 100 if not bench1.empty else np.nan
    a = st.columns(4)
    a[0].metric("6-Week Portfolio Return", fmt_pct(r1))
    a[1].metric("6-Week NIFTY Return", fmt_pct(rb1))
    a[2].metric("6-Week Outperformance", fmt_pct(r1 - rb1) if pd.notna(rb1) else "N/A")
    a[3].metric("Phase-1 End Value", fmt_inr(capital * (1 + r1 / 100)))

# ------------------------- 5 reselection -------------------------
st.header("5. 🔄 End-of-Week-6 Reselection")
st.write("At the end of week 6, all **15 companies** and all **3 debt instruments** are evaluated. Only the best 12 equities and 2 debt instruments enter Phase 2.")

rshow = week6_rank.copy()
rshow.insert(0, "Selected", rshow["ticker"].isin(selected_eq).map({True: "✅", False: "—"}))
st.dataframe(rshow.round(2), use_container_width=True, hide_index=True)

dshow = debt_rank.copy()
dshow.insert(0, "Selected", dshow["ticker"].isin(selected_debt).map({True: "✅", False: "—"}))
st.dataframe(dshow.round(3), use_container_width=True, hide_index=True)

st.success(
    "**Actual Phase-2 holdings:** "
    + ", ".join(EQUITY_META[t][0] for t in selected_eq)
    + " + " + ", ".join(DEBT_META[t]["name"] for t in selected_debt)
    + " + Gold ETF"
)

# ------------------------- 6 phase 2 weekly weights/performance -------------------------
phase2_start = first_trade_on_or_after(RESELECT_DATE + pd.Timedelta(days=1))
phase2_end = LATEST_DATE
phase2, weights2 = simulate_phase(phase2_start, phase2_end, selected_eq, selected_debt, persona)
bench2 = benchmark_series(phase2_start, phase2_end)

st.header("6. 📊 Continuing Period — Weekly Redistribution After Reselection")
if not weights2.empty:
    w2 = weights2.copy()
    w2["Date"] = pd.to_datetime(w2["Date"]).dt.strftime("%d %b %Y")
    w2 = w2.rename(columns={t: EQUITY_META[t][0] for t in EQUITY_META})
    w2 = w2.rename(columns={t: DEBT_META[t]["name"] for t in DEBT_META})
    w2 = w2.rename(columns={GOLD: "Gold ETF"})
    st.dataframe(w2.round(2), use_container_width=True, hide_index=True)

    long2 = weights2.melt(id_vars="Date", var_name="Ticker", value_name="Weight %")
    long2["Instrument"] = long2["Ticker"].map({t: EQUITY_META[t][0] for t in EQUITY_META} | {t: DEBT_META[t]["name"] for t in DEBT_META} | {GOLD: "Gold ETF"})
    figw2 = px.line(long2, x="Date", y="Weight %", color="Instrument", markers=True, title="Weekly portfolio weights — Phase 2")
    st.plotly_chart(figw2, use_container_width=True, config={"responsive": True, "displaylogo": False})

if not phase2.empty:
    r2 = (phase2.iloc[-1] / phase2.iloc[0] - 1) * 100
    rb2 = (bench2.iloc[-1] / bench2.iloc[0] - 1) * 100 if not bench2.empty else np.nan
    a = st.columns(4)
    a[0].metric("Post-Reselection Return", fmt_pct(r2))
    a[1].metric("Post-Reselection NIFTY", fmt_pct(rb2))
    a[2].metric("Post-Reselection Alpha", fmt_pct(r2 - rb2) if pd.notna(rb2) else "N/A")
    a[3].metric("Latest Phase-2 Value", fmt_inr(capital * (1 + r2 / 100)))
else:
    st.warning("No post-reselection trading data is currently available.")

# ------------------------- 7 overall performance -------------------------
st.header("7. 📈 Overall Portfolio Performance — 6 Weeks + Continuing Period")
overall = combine_phases(phase1, phase2)
bench_all = benchmark_series(FIRST_DATE, LATEST_DATE)

if not overall.empty:
    norm_port = overall / overall.iloc[0] * 100
    norm_bench = bench_all / bench_all.iloc[0] * 100 if not bench_all.empty else pd.Series(dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=norm_port.index, y=norm_port.values, name="ALPHA Portfolio", mode="lines"))
    if not norm_bench.empty:
        fig.add_trace(go.Scatter(x=norm_bench.index, y=norm_bench.values, name="NIFTY 50", mode="lines"))
    fig.add_vline(x=RESELECT_DATE.timestamp() * 1000, line_dash="dash", annotation_text="6-week reselection")
    fig.update_layout(title="Indexed growth — ₹100 starting value", xaxis_title="Date", yaxis_title="Indexed Value")
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displaylogo": False})

    overall_ret = (overall.iloc[-1] / overall.iloc[0] - 1) * 100
    bench_ret = (bench_all.iloc[-1] / bench_all.iloc[0] - 1) * 100 if not bench_all.empty else np.nan
    x = st.columns(5)
    x[0].metric("Overall Portfolio Return", fmt_pct(overall_ret))
    x[1].metric("Overall NIFTY Return", fmt_pct(bench_ret))
    x[2].metric("Overall Outperformance", fmt_pct(overall_ret - bench_ret) if pd.notna(bench_ret) else "N/A")
    x[3].metric("Starting Capital", fmt_inr(capital))
    x[4].metric("Illustrative Ending Value", fmt_inr(capital * (1 + overall_ret / 100)))

    # Drawdown
    pdd = overall / overall.cummax() - 1
    bdd = bench_all / bench_all.cummax() - 1 if not bench_all.empty else pd.Series(dtype=float)
    fd = go.Figure()
    fd.add_trace(go.Scatter(x=pdd.index, y=pdd * 100, name="ALPHA Drawdown", mode="lines"))
    if not bdd.empty:
        fd.add_trace(go.Scatter(x=bdd.index, y=bdd * 100, name="NIFTY Drawdown", mode="lines"))
    fd.update_layout(title="Portfolio vs NIFTY drawdown", yaxis_title="Drawdown %")
    st.plotly_chart(fd, use_container_width=True, config={"responsive": True, "displaylogo": False})

# ------------------------- 8 portfolio snapshot -------------------------
st.header("8. 💼 Current Portfolio Snapshot")
current_weights = weekly_weights(selected_eq, selected_debt, LATEST_DATE, persona)
rows = []
for t, w in current_weights.items():
    if t in EQUITY_META:
        name, sector = EQUITY_META[t]
        pxv = close_asof(t, LATEST_DATE)
        rows.append({"Instrument": name, "Ticker": t, "Type": "Equity", "Sector": sector, "Weight %": w, "Price": pxv})
    elif t in DEBT_META:
        rows.append({"Instrument": DEBT_META[t]["name"], "Ticker": t, "Type": "Debt", "Sector": DEBT_META[t]["sector"], "Weight %": w, "Price": np.nan})
    else:
        rows.append({"Instrument": "Nippon India ETF Gold BeES", "Ticker": GOLD, "Type": "Gold", "Sector": "Commodity", "Weight %": w, "Price": close_asof(GOLD, LATEST_DATE)})
current_df = pd.DataFrame(rows).sort_values("Weight %", ascending=False)
st.dataframe(current_df.round(2), use_container_width=True, hide_index=True)

# Sector allocation
sector_df = current_df.groupby("Sector", as_index=False)["Weight %"].sum().sort_values("Weight %", ascending=False)
fig_sector = px.bar(sector_df, x="Sector", y="Weight %", title="Current sector / asset allocation")
st.plotly_chart(fig_sector, use_container_width=True, config={"responsive": True, "displaylogo": False})

# ------------------------- 9 company analysis -------------------------
st.header("9. 🔎 Company Analysis")
analysis_ticker = st.selectbox("Select a company", list(EQUITY_META.keys()), format_func=lambda t: f"{EQUITY_META[t][0]} ({t})")

m = info_metrics(analysis_ticker, LATEST_DATE)
name, sector = EQUITY_META[analysis_ticker]
ret1m = return_pct(analysis_ticker, LATEST_DATE, 30)
ret3m = return_pct(analysis_ticker, LATEST_DATE, 90)
ret6m = return_pct(analysis_ticker, LATEST_DATE, 180)
ret1y = return_pct(analysis_ticker, LATEST_DATE, 365)
sh = sharpe(analysis_ticker, LATEST_DATE)
dd = drawdown(analysis_ticker, LATEST_DATE)
pe, peerpe, pedisc = pe_peer(analysis_ticker, LATEST_DATE)

cc = st.columns(6)
cc[0].metric("Current Price", fmt_inr(m["Price"]))
cc[1].metric("Market Cap", fmt_inr(m["Market Cap"]))
cc[2].metric("P/E", fmt_num(m["P/E"]))
cc[3].metric("P/B", fmt_num(m["P/B"]))
cc[4].metric("Beta", fmt_num(m["Beta"]))
cc[5].metric("ROE", fmt_pct(m["ROE %"]))

st.write(f"**{name}** · {analysis_ticker} · {sector}")

ret_df = pd.DataFrame({"Period": ["1M", "3M", "6M", "1Y"], "Return %": [ret1m, ret3m, ret6m, ret1y]})
fig_ret = px.bar(ret_df, x="Period", y="Return %", title=f"{name} — historical returns")
st.plotly_chart(fig_ret, use_container_width=True, config={"responsive": True, "displaylogo": False})

hist = HIST.get(analysis_ticker)
if hist is not None and not hist.empty:
    h = hist[hist.index >= norm_date(LATEST_DATE) - pd.Timedelta(days=365)].copy()
    if not h.empty:
        h["DMA50"] = h["Close"].rolling(50).mean()
        h["DMA200"] = h["Close"].rolling(200).mean()
        fp = go.Figure()
        fp.add_trace(go.Scatter(x=h.index, y=h["Close"], name="Price"))
        fp.add_trace(go.Scatter(x=h.index, y=h["DMA50"], name="50 DMA"))
        fp.add_trace(go.Scatter(x=h.index, y=h["DMA200"], name="200 DMA"))
        fp.update_layout(title=f"{name} — Price & Moving Averages", yaxis_title="₹")
        st.plotly_chart(fp, use_container_width=True, config={"responsive": True, "displaylogo": False})

fund = pd.DataFrame([
    ["EPS", m["EPS"]], ["P/E", m["P/E"]], ["Peer P/E", peerpe],
    ["P/E discount vs peer", pedisc], ["P/B", m["P/B"]], ["ROE %", m["ROE %"]],
    ["Debt/Equity", m["Debt/Equity"]], ["Dividend Yield %", m["Dividend Yield %"]],
    ["Volatility %", volatility(analysis_ticker, LATEST_DATE)], ["Sharpe", sh],
    ["Max Drawdown %", dd],
], columns=["Metric", "Value"])
st.subheader("Fundamental & risk snapshot")
st.dataframe(fund.round(2), use_container_width=True, hide_index=True)

if pd.notna(pedisc):
    if pedisc > 10:
        st.success("Relative valuation observation: P/E is below the model peer reference.")
    elif pedisc < -10:
        st.warning("Relative valuation observation: P/E is above the model peer reference.")
    else:
        st.info("Relative valuation observation: P/E is broadly around the model peer reference.")
st.caption("Company metrics are data-dependent; Yahoo Finance may not provide every fundamental field. Valuation observations are analytical, not buy/sell recommendations.")

# ------------------------- 10 company comparison -------------------------
st.header("10. 📊 Company Comparison")
comparison = st.multiselect(
    "Select 2–6 companies",
    list(EQUITY_META.keys()),
    default=initial_selected_eq[:3],
    max_selections=6,
    format_func=lambda t: EQUITY_META[t][0],
)

if len(comparison) >= 2:
    comp_rows = []
    for t in comparison:
        im = info_metrics(t, LATEST_DATE)
        comp_rows.append({
            "Company": EQUITY_META[t][0], "Ticker": t, "Sector": EQUITY_META[t][1],
            "Price": im["Price"], "Market Cap": im["Market Cap"], "P/E": im["P/E"],
            "P/B": im["P/B"], "EPS": im["EPS"], "ROE %": im["ROE %"],
            "Debt/Equity": im["Debt/Equity"], "Beta": im["Beta"],
            "6M Return %": return_pct(t, LATEST_DATE, 180),
            "Volatility %": volatility(t, LATEST_DATE), "Sharpe": sharpe(t, LATEST_DATE),
            "Drawdown %": drawdown(t, LATEST_DATE),
        })
    comp = pd.DataFrame(comp_rows)

    # Yahoo Finance can return missing/string/non-finite market-cap values.
    # Coerce comparison metrics before passing them to Plotly.
    numeric_cols = ["Market Cap", "P/E", "P/B", "EPS", "ROE %", "Debt/Equity",
                    "Beta", "6M Return %", "Volatility %", "Sharpe", "Drawdown %"]
    for c in numeric_cols:
        if c in comp.columns:
            comp[c] = pd.to_numeric(comp[c], errors="coerce")
    comp = comp.replace([np.inf, -np.inf], np.nan)
    st.dataframe(comp.round(2), use_container_width=True, hide_index=True)

    bar_ret = comp.dropna(subset=["6M Return %"]).copy()
    if not bar_ret.empty:
        fc = px.bar(bar_ret, x="Company", y="6M Return %", title="6M return comparison")
        st.plotly_chart(fc, use_container_width=True, config={"responsive": True, "displaylogo": False})

    bar_pe = comp.dropna(subset=["P/E"]).copy()
    if not bar_pe.empty:
        fv = px.bar(bar_pe, x="Company", y="P/E", title="Valuation comparison — P/E")
        st.plotly_chart(fv, use_container_width=True, config={"responsive": True, "displaylogo": False})

    # Plotly requires bubble sizes to be finite and strictly positive.
    scatter = comp.dropna(subset=["Volatility %", "6M Return %"]).copy()
    if not scatter.empty:
        scatter["Bubble Size"] = pd.to_numeric(scatter.get("Market Cap"), errors="coerce")
        valid = scatter["Bubble Size"].replace([np.inf, -np.inf], np.nan).dropna()
        positive = valid[valid > 0]
        fallback = float(positive.median()) if not positive.empty else 20.0
        scatter["Bubble Size"] = (
            scatter["Bubble Size"].replace([np.inf, -np.inf], np.nan)
            .where(scatter["Bubble Size"] > 0, fallback)
            .fillna(fallback)
            .clip(lower=1.0)
        )
        fs = px.scatter(
            scatter, x="Volatility %", y="6M Return %",
            size="Bubble Size", text="Company",
            title="Risk-return comparison", size_max=45
        )
        fs.update_traces(textposition="top center")
        st.plotly_chart(fs, use_container_width=True, config={"responsive": True, "displaylogo": False})
    else:
        st.info("Not enough valid return/volatility data to draw the risk-return comparison.")
else:
    st.info("Select at least two companies to compare.")

# ------------------------- 11 risk & performance analytics -------------------------
st.header("11. ⚠️ Risk & Performance Analytics")
if not overall.empty:
    pr = overall.pct_change().dropna()
    vol_ann = pr.std() * np.sqrt(252) * 100
    beta_port = np.nan
    br = benchmark_series(FIRST_DATE, LATEST_DATE).pct_change().dropna()
    j = pd.concat([pr.rename("p"), br.rename("b")], axis=1).dropna()
    if len(j) > 20 and j["b"].var() > 0:
        beta_port = j["p"].cov(j["b"]) / j["b"].var()
    sharpe_port = ((pr.mean() * 252) - RISK_FREE) / (pr.std() * np.sqrt(252)) if pr.std() else np.nan
    maxdd = (overall / overall.cummax() - 1).min() * 100
    downside = pr[pr < 0].std() * np.sqrt(252) * 100 if (pr < 0).sum() > 1 else np.nan
    rr = st.columns(5)
    rr[0].metric("Annualised Volatility", fmt_pct(vol_ann))
    rr[1].metric("Portfolio Beta", fmt_num(beta_port))
    rr[2].metric("Sharpe Ratio", fmt_num(sharpe_port))
    rr[3].metric("Max Drawdown", fmt_pct(maxdd))
    rr[4].metric("Downside Risk", fmt_pct(downside))

# ------------------------- 12 debt analysis -------------------------
st.header("12. 🏦 Debt / Fixed-Income Analysis")
debt_table = pd.DataFrame([
    {"Instrument": m["name"], "Ticker": t, "Yield %": m["yield"] * 100, "Model Risk": m["risk"], "Selected Phase 1": t in initial_selected_debt, "Selected Phase 2": t in selected_debt}
    for t, m in DEBT_META.items()
])
st.dataframe(debt_table.round(3), use_container_width=True, hide_index=True)

# ------------------------- 13 goals -------------------------
st.header("13. 🎯 Investment Goal Calculator")
g1, g2, g3, g4 = st.columns(4)
current = g1.number_input("Current investment (₹)", min_value=0.0, value=float(capital))
target = g2.number_input("Target amount (₹)", min_value=1.0, value=float(capital * 1.5))
years = g3.number_input("Time horizon (years)", min_value=0.5, value=5.0, step=0.5)
assumed = g4.number_input("Expected annual return %", min_value=-50.0, value=12.0, step=0.5)
projected = current * (1 + assumed / 100) ** years
req_cagr = ((target / current) ** (1 / years) - 1) * 100 if current > 0 else np.nan
st.metric("Projected corpus", fmt_inr(projected))
st.metric("Required CAGR to reach target", fmt_pct(req_cagr))
st.caption("Mathematical projection only; returns are not guaranteed.")

# ------------------------- 14 advanced analytics -------------------------
st.header("14. 🧮 Advanced Analytics")
tab1, tab2, tab3 = st.tabs(["CAPM", "Factor Scoring", "Risk–Return"])
with tab1:
    rf = st.number_input("Risk-free rate %", value=RISK_FREE * 100, step=0.25, key="rf")
    capm_rows = []
    for t in selected_eq:
        b = beta(t, LATEST_DATE)
        market_ret = return_pct(BENCHMARK, LATEST_DATE, 365)
        expected = rf + b * ((market_ret if pd.notna(market_ret) else 12.0) - rf) if pd.notna(b) else np.nan
        capm_rows.append({"Company": EQUITY_META[t][0], "Beta": b, "CAPM Expected Return %": expected})
    st.dataframe(pd.DataFrame(capm_rows).round(2), use_container_width=True, hide_index=True)
with tab2:
    st.write("Weekly allocation score = momentum + relative valuation + risk + beta factors. The score determines weights inside the selected 12-equity bucket.")
    st.code("45% momentum + 25% valuation + 20% inverse volatility + 10% inverse beta")
with tab3:
    if len(comp_rows) >= 2:
        st.dataframe(comp[["Company", "6M Return %", "Volatility %", "Beta", "Sharpe", "Drawdown %"]].round(2), use_container_width=True, hide_index=True)

# ------------------------- 15 investor insights -------------------------
st.header("15. 🧠 Investor Insights")
if not current_df.empty:
    largest = current_df.iloc[0]
    equity_weight = current_df.loc[current_df["Type"] == "Equity", "Weight %"].sum()
    st.write(f"• Largest current position: **{largest['Instrument']} ({largest['Weight %']:.2f}%)**.")
    st.write(f"• Current portfolio equity exposure: **{equity_weight:.2f}%** under the **{persona}** persona.")
    if not overall.empty:
        st.write(f"• Overall strategy return since start: **{overall_ret:+.2f}%** versus NIFTY **{bench_ret:+.2f}%** where benchmark data is available.")
    st.write("• Weekly redistribution is performed in both phases; the week-6 event changes the eligible equity/debt universe before Phase 2.")

# ------------------------- 16 methodology -------------------------
st.header("16. 📚 Methodology & Important Notes")
st.markdown(
    """
**Portfolio construction**
- Research universe = 15 equities + 3 debt instruments + Gold ETF.
- Actual holdings = 12 equities + 2 debt instruments + Gold ETF in both phases.
- Phase 1 runs from the first NSE trading session on/after 01 Aug 2026 through the week-6 review.
- Weights are recalculated at the beginning of **every week**, including the first six weeks.
- At the end of week 6, all 15 equities and all 3 debt instruments are re-ranked; the portfolio then switches to the new 12 + 2 + Gold set.
- The selected equity weights are driven by momentum, peer-relative valuation, inverse volatility and inverse beta.
- Persona controls the equity/debt/gold bucket weights.

**Performance**
- Equity/Gold returns use Yahoo Finance historical prices.
- Debt is modelled using the stated annual yield assumptions and daily accrual.
- The performance chart is a strategy simulation, not a broker statement.
- Taxes, brokerage, slippage and dividends are not separately modelled in the strategy backtest.
- Fundamental data can be missing or delayed depending on Yahoo Finance availability.
- All date comparisons are normalized to timezone-naive trading dates to avoid tz-aware/tz-naive errors.

**Interpretation**
This dashboard is designed as a research/academic portfolio experiment. Its scores and observations are not personalised investment recommendations.
"""
)

st.caption("ALPHA • Dynamic 6-week reselection + weekly weight redistribution • Research dashboard")
