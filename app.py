# =============================================================================
#  🇮🇳  3-YEAR INDIAN EQUITY PORTFOLIO DASHBOARD — STREAMLIT EDITION
#
#  Test locally / in Colab:   streamlit run app.py
#  Deploy free:               https://share.streamlit.io  (point it at this repo,
#                              main file = app.py)
#
#  Why this version is hardened against the "app isn't running" problem:
#  Yahoo Finance often silently throttles or drops connections from cloud
#  data-center IP ranges (AWS/GCP — which is what Streamlit Cloud runs on).
#  Locally this usually errors out fast; on cloud infra it can instead just
#  HANG with no error, which looks exactly like a stuck/dead app. Every
#  network call below is wrapped in a hard wall-clock timeout via
#  concurrent.futures, and all 15 tickers are fetched in PARALLEL at startup
#  instead of one-by-one — so the absolute worst case is a few seconds slow,
#  never an indefinite hang, and the app always finishes loading and renders
#  something (live data, or the offline fallback figures) either way.
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import concurrent.futures as cf
from datetime import datetime, timedelta, date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# -----------------------------------------------------------------------------
# 0. PAGE CONFIG  (must be the very first Streamlit command in the script)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Indian Equity Portfolio Dashboard",
    page_icon="🇮🇳",
    layout="wide",
)

# -----------------------------------------------------------------------------
# 1. CORE CONSTANTS
# -----------------------------------------------------------------------------
TOTAL_CAPITAL    = 1_00_00_000     # ₹1 Crore
BROKERAGE_RATE   = 0.0025          # 0.25% on total volume (invested + current)
STCG_RATE        = 0.20            # Modelled 20% tax if positions were sold today
BENCHMARK        = "^NSEI"
HOLDING_YEARS    = 3
INVESTMENT_DATE  = pd.Timestamp("2026-09-01")
TARGET_DATE      = INVESTMENT_DATE + pd.DateOffset(years=HOLDING_YEARS)
NETWORK_TIMEOUT  = 8                # hard cap, seconds, per external call
PREFETCH_WORKERS = 8                # parallel fetch workers at startup


def inr(x):
    """Format a number in Indian currency style: ₹#,##,###.##"""
    try:
        x = float(x)
    except Exception:
        return "₹0.00"
    if x is None or pd.isna(x):
        return "-"
    neg = x < 0
    x = abs(x)
    s = f"{x:,.2f}"
    int_part, dec_part = s.split(".")
    int_part = int_part.replace(",", "")
    if len(int_part) <= 3:
        grouped = int_part
    else:
        last3, rest = int_part[-3:], int_part[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3
    out = f"₹{grouped}.{dec_part}"
    return f"-{out}" if neg else out


# -----------------------------------------------------------------------------
# 2. ASSET UNIVERSE, SECTORS, THESES
# -----------------------------------------------------------------------------
ASSET_META = {
    "10Y_GSEC": {"name": "10Y Indian G-Sec", "class": "Debt",
                 "sector": "Sovereign Debt", "yield": 0.07,
                 "thesis": "Risk-free sovereign anchor providing fixed 7% p.a. accrual, uncorrelated with equity drawdowns."},
    "AAA_BOND": {"name": "AAA Corporate Bond / Debenture", "class": "Debt",
                 "sector": "Corporate Debt", "yield": 0.08,
                 "thesis": "High-grade corporate debenture offering a modest credit-spread pick-up of 100bps over the G-Sec."},
    "GOLDBEES.NS": {"name": "Nippon India ETF Gold BeES", "class": "Gold",
                     "sector": "Commodity", "yield": 0.09,
                     "thesis": "Sovereign-backed gold ETF used as an inflation hedge and portfolio diversifier against equity/currency shocks."},

    "MARUTI.NS":     {"name": "Maruti Suzuki India Ltd", "class": "Equity", "sector": "Automobiles",
                       "thesis": "India's largest passenger-vehicle maker; a defensive, low-beta compounder riding rural demand recovery and SUV mix upgrades."},
    "BAJAJ-AUTO.NS": {"name": "Bajaj Auto Ltd", "class": "Equity", "sector": "Automobiles",
                       "thesis": "Two/three-wheeler leader with a strong export franchise and EV optionality via Chetak; quality-growth compounder with high payout."},
    "HAL.NS":        {"name": "Hindustan Aeronautics Ltd", "class": "Equity", "sector": "Defense / Capital Goods",
                       "thesis": "Near-monopoly domestic defense-aircraft manufacturer benefiting from indigenization push and a record order backlog; high-beta momentum name."},
    "SIEMENS.NS":    {"name": "Siemens Ltd", "class": "Equity", "sector": "Capital Goods",
                       "thesis": "Diversified industrial automation and electrification major positioned on India's capex and power-infrastructure upcycle."},
    "POLYCAB.NS":    {"name": "Polycab India Ltd", "class": "Equity", "sector": "Capital Goods / Manufacturing",
                       "thesis": "Market-leading cables & wires manufacturer; steady compounder with an expanding FMEG (fans, appliances) distribution business."},
    "AXISBANK.NS":   {"name": "Axis Bank Ltd", "class": "Equity", "sector": "Financials",
                       "thesis": "Private-sector bank in a valuation re-rating phase, supported by improving asset quality and retail-loan growth."},
    "BSE.NS":        {"name": "BSE Ltd", "class": "Equity", "sector": "Capital Markets",
                       "thesis": "India's oldest stock exchange; a high-beta play on retail derivatives-trading volumes and market-infrastructure growth."},
    "HINDUNILVR.NS": {"name": "Hindustan Unilever Ltd", "class": "Equity", "sector": "FMCG",
                       "thesis": "India's largest FMCG company; a low-beta defensive anchor with pricing power across home & personal-care categories."},
    "BRITANNIA.NS":  {"name": "Britannia Industries Ltd", "class": "Equity", "sector": "FMCG",
                       "thesis": "Dominant biscuits & bakery player expanding into adjacent categories; defensive earnings visibility."},
    "ITC.NS":        {"name": "ITC Ltd", "class": "Equity", "sector": "FMCG",
                       "thesis": "Diversified FMCG-cigarettes-agri-paper conglomerate; a defensive anchor with one of the highest dividend yields in the index."},
    "PIDILITIND.NS": {"name": "Pidilite Industries Ltd", "class": "Equity", "sector": "Chemicals",
                       "thesis": "Dominant adhesives (Fevicol) franchise with pricing power over raw-material cycles; a steady structural compounder."},
    "SOLARINDS.NS":  {"name": "Solar Industries India Ltd", "class": "Equity", "sector": "Chemicals / Defense",
                       "thesis": "Leading industrial-explosives maker with a fast-scaling defense ordnance order book; a high-beta aggressive-growth name."},
}

EQUITY_TICKERS = [t for t, m in ASSET_META.items() if m["class"] == "Equity"]
BOND_TICKERS   = ["10Y_GSEC", "AAA_BOND"]
ALL_TICKERS    = list(ASSET_META.keys())
YF_TICKERS     = [t for t in ALL_TICKERS if t not in BOND_TICKERS]

INDUSTRY_AVG_PE = {
    "Automobiles": 28.0, "Defense / Capital Goods": 40.0, "Capital Goods": 45.0,
    "Capital Goods / Manufacturing": 42.0, "Financials": 20.0, "Capital Markets": 35.0,
    "FMCG": 55.0, "Chemicals": 50.0, "Chemicals / Defense": 50.0,
}

SCENARIOS = {
    "Conservative (₹1.75 Cr)": {
        "target": 1_75_00_000,
        "weights": {"10Y_GSEC": 10, "AAA_BOND": 7, "GOLDBEES.NS": 8,
                    "MARUTI.NS": 7, "BAJAJ-AUTO.NS": 6, "HAL.NS": 9, "SIEMENS.NS": 8,
                    "POLYCAB.NS": 11, "AXISBANK.NS": 5, "BSE.NS": 6, "HINDUNILVR.NS": 7,
                    "BRITANNIA.NS": 4, "ITC.NS": 4, "PIDILITIND.NS": 3, "SOLARINDS.NS": 5},
    },
    "Moderate (₹2.00 Cr)": {
        "target": 2_00_00_000,
        "weights": {"10Y_GSEC": 5, "AAA_BOND": 4, "GOLDBEES.NS": 6,
                    "MARUTI.NS": 4, "BAJAJ-AUTO.NS": 4, "HAL.NS": 15, "SIEMENS.NS": 9,
                    "POLYCAB.NS": 11, "AXISBANK.NS": 6, "BSE.NS": 13, "HINDUNILVR.NS": 4,
                    "BRITANNIA.NS": 3, "ITC.NS": 3, "PIDILITIND.NS": 3, "SOLARINDS.NS": 10},
    },
    "Aggressive (₹2.25 Cr)": {
        "target": 2_25_00_000,
        "weights": {"10Y_GSEC": 2, "AAA_BOND": 2, "GOLDBEES.NS": 3,
                    "MARUTI.NS": 2, "BAJAJ-AUTO.NS": 2, "HAL.NS": 24, "SIEMENS.NS": 6,
                    "POLYCAB.NS": 6, "AXISBANK.NS": 6, "BSE.NS": 20, "HINDUNILVR.NS": 3,
                    "BRITANNIA.NS": 2, "ITC.NS": 3, "PIDILITIND.NS": 3, "SOLARINDS.NS": 16},
    },
}

FALLBACK_DATA = {
    "MARUTI.NS":     {"cmp": 12800.0, "eps": 440.0, "pe": 29.0, "beta": 0.85, "mcap_cr": 402000, "low52": 10500.0, "high52": 13700.0, "buy_price": 9200.0},
    "BAJAJ-AUTO.NS": {"cmp": 9500.0,  "eps": 250.0, "pe": 38.0, "beta": 0.75, "mcap_cr": 265000, "low52": 7000.0,  "high52": 12800.0, "buy_price": 4300.0},
    "HAL.NS":        {"cmp": 4800.0,  "eps": 105.0, "pe": 45.7, "beta": 1.20, "mcap_cr": 320000, "low52": 3200.0,  "high52": 5675.0,  "buy_price": 1450.0},
    "SIEMENS.NS":    {"cmp": 6500.0,  "eps": 95.0,  "pe": 68.0, "beta": 1.05, "mcap_cr": 231000, "low52": 5200.0,  "high52": 8400.0,  "buy_price": 2600.0},
    "POLYCAB.NS":    {"cmp": 6900.0,  "eps": 150.0, "pe": 46.0, "beta": 0.95, "mcap_cr": 104000, "low52": 4800.0,  "high52": 7600.0,  "buy_price": 2200.0},
    "AXISBANK.NS":   {"cmp": 1180.0,  "eps": 78.0,  "pe": 15.1, "beta": 1.30, "mcap_cr": 365000, "low52": 950.0,   "high52": 1340.0,  "buy_price": 700.0},
    "BSE.NS":        {"cmp": 5200.0,  "eps": 95.0,  "pe": 54.7, "beta": 1.60, "mcap_cr": 70000,  "low52": 1800.0,  "high52": 6100.0,  "buy_price": 350.0},
    "HINDUNILVR.NS": {"cmp": 2500.0,  "eps": 43.0,  "pe": 58.1, "beta": 0.55, "mcap_cr": 588000, "low52": 2100.0,  "high52": 3035.0,  "buy_price": 2300.0},
    "BRITANNIA.NS":  {"cmp": 5700.0,  "eps": 90.0,  "pe": 63.3, "beta": 0.60, "mcap_cr": 137000, "low52": 4500.0,  "high52": 6470.0,  "buy_price": 3400.0},
    "ITC.NS":        {"cmp": 460.0,   "eps": 16.0,  "pe": 28.8, "beta": 0.65, "mcap_cr": 575000, "low52": 390.0,   "high52": 500.0,   "buy_price": 210.0},
    "PIDILITIND.NS": {"cmp": 3100.0,  "eps": 45.0,  "pe": 68.9, "beta": 0.70, "mcap_cr": 157000, "low52": 2500.0,  "high52": 3400.0,  "buy_price": 1750.0},
    "SOLARINDS.NS":  {"cmp": 12500.0, "eps": 180.0, "pe": 69.4, "beta": 1.55, "mcap_cr": 116000, "low52": 5900.0,  "high52": 13100.0, "buy_price": 1500.0},
    "GOLDBEES.NS":   {"cmp": 72.0,    "eps": None,  "pe": None, "beta": 0.10, "mcap_cr": None,   "low52": 50.0,    "high52": 78.0,    "buy_price": 45.0},
}


def validate_scenarios():
    problems = []
    for name, sc in SCENARIOS.items():
        total = sum(sc["weights"].values())
        if abs(total - 100) > 1e-6:
            problems.append(f"'{name}' weights sum to {total}, expected 100")
    return problems


_config_problems = validate_scenarios()
if _config_problems:
    st.error("⚠️ Configuration error — fix SCENARIOS before this app can run:\n\n" +
              "\n".join(f"- {p}" for p in _config_problems))
    st.stop()

# -----------------------------------------------------------------------------
# 3. HARD-TIMEOUT NETWORK LAYER
#    Every yfinance call is wrapped so it can NEVER hang the app: if Yahoo
#    Finance is slow, throttled, or blocked, the call gives up after
#    NETWORK_TIMEOUT seconds and the app falls back automatically.
# -----------------------------------------------------------------------------
def _bounded(fn, timeout=NETWORK_TIMEOUT):
    """Run fn() with a hard wall-clock timeout. Returns None on any failure or timeout."""
    try:
        with cf.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn)
            return fut.result(timeout=timeout)
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker, period="5y"):
    def _do():
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True, timeout=NETWORK_TIMEOUT)
        return df if df is not None and not df.empty else None
    return _bounded(_do)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_info(ticker):
    def _do():
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    return _bounded(_do, timeout=6) or {}


@st.cache_resource(show_spinner=False)
def prefetch_market_data():
    """Warm the cache for every ticker IN PARALLEL, once per hour, so a
    single slow/blocked ticker can never delay the others. Sequential
    fetching of 15 tickers at up to 8s each could take 2 minutes; this
    bounds the whole batch to roughly one timeout window."""
    tickers = YF_TICKERS + [BENCHMARK]
    with cf.ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as ex:
        list(ex.map(fetch_history, tickers))
        list(ex.map(fetch_info, YF_TICKERS))
    return True


with st.spinner("Fetching live NSE market data (falls back automatically if Yahoo Finance is unreachable)..."):
    prefetch_market_data()

# -----------------------------------------------------------------------------
# 4. PRICE / BETA / ENTRY-DATE HELPERS
# -----------------------------------------------------------------------------
def get_entry_date(hist):
    """Return the first available trading session on/after the fixed investment date."""
    if hist is None or hist.empty:
        return None

    idx = hist.index
    target = pd.Timestamp(INVESTMENT_DATE)

    if getattr(idx, "tz", None) is not None and target.tzinfo is None:
        target = target.tz_localize(idx.tz)

    eligible = idx[idx >= target]
    return eligible[0] if len(eligible) else None


def compute_beta_vs_nifty(hist):
    try:
        nifty_hist = fetch_history(BENCHMARK)
        if nifty_hist is None or nifty_hist.empty or hist is None:
            return None
        s_ret = hist["Close"].pct_change().dropna()
        n_ret = nifty_hist["Close"].pct_change().dropna()
        joined = pd.concat([s_ret, n_ret], axis=1, join="inner")
        joined.columns = ["s", "n"]
        if len(joined) < 30:
            return None
        var = joined["n"].var()
        return float(joined["s"].cov(joined["n"]) / var) if var > 0 else None
    except Exception:
        return None


def get_price_series(t):
    hist, info = fetch_history(t), fetch_info(t)
    fb = FALLBACK_DATA.get(t, {})
    live = hist is not None and len(hist) > 5

    if live:
        cmp_ = float(hist["Close"].iloc[-1])
        entry_date = get_entry_date(hist)
        buy_price = (
            float(hist.loc[entry_date, "Close"])
            if entry_date is not None
            else None
        )
        beta = info.get("beta") or compute_beta_vs_nifty(hist) or fb.get("beta", 1.0)
        low52 = float(hist["Close"].tail(252).min())
        high52 = float(hist["Close"].tail(252).max())
    else:
        # Do NOT use the old hard-coded historical buy prices as the
        # 01-Sep-2026 entry price. If Yahoo is unavailable, the true
        # investment-date price cannot be known.
        cmp_ = fb.get("cmp", 100.0)
        buy_price = None
        beta = fb.get("beta", 1.0)
        low52 = fb.get("low52", cmp_ * 0.8)
        high52 = fb.get("high52", cmp_ * 1.2)

    return {
        "cmp": cmp_, "buy_price": buy_price, "beta": beta,
        "low52": low52, "high52": high52, "live": live, "info": info
    }


# -----------------------------------------------------------------------------
# 5. SCENARIO PORTFOLIO ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def compute_scenario(scenario_name):
    sc = SCENARIOS[scenario_name]
    rows = []
    today = pd.Timestamp.now().normalize()
    elapsed_years = max((today - INVESTMENT_DATE).days / 365.25, 0.0)

    for t, w_pct in sc["weights"].items():
        meta = ASSET_META[t]
        allocated = TOTAL_CAPITAL * w_pct / 100.0

        if meta["class"] == "Debt":
            invested = allocated
            current_value = invested * ((1 + meta["yield"]) ** elapsed_years)
            qty = buy_price = cmp_ = None
            beta = 0.0
            entry_available = True
        else:
            p = get_price_series(t)
            buy_price, cmp_, beta = p["buy_price"], p["cmp"], p["beta"]
            entry_available = buy_price is not None and buy_price > 0

            if entry_available:
                qty = int(allocated / buy_price)
                invested = qty * buy_price
                leftover = allocated - invested
                invested += leftover
                current_value = qty * cmp_ + leftover
            else:
                # Offline mode: keep capital at cost rather than inventing
                # a 01-Sep-2026 entry price.
                qty = 0
                invested = allocated
                current_value = allocated

        gross_profit = current_value - invested
        brokerage = BROKERAGE_RATE * (invested + current_value)
        taxable_gain = max(gross_profit - brokerage, 0)
        tax = STCG_RATE * taxable_gain
        net_profit = gross_profit - brokerage - tax
        cagr = (
            (current_value / invested) ** (1 / elapsed_years) - 1
            if invested > 0 and elapsed_years > 0 and current_value > 0
            else 0.0
        )

        rows.append({
            "ticker": t, "name": meta["name"], "class": meta["class"],
            "sector": meta.get("sector", ""), "weight_pct": w_pct,
            "allocated": allocated, "qty": qty, "buy_price": buy_price,
            "cmp": cmp_, "invested": invested, "current_value": current_value,
            "gross_profit": gross_profit, "brokerage": brokerage, "stcg": tax,
            "net_profit": net_profit, "beta": beta, "cagr": cagr,
            "entry_available": entry_available,
        })

    df = pd.DataFrame(rows)
    inv_sum, cur_sum = df["invested"].sum(), df["current_value"].sum()
    kpis = {
        "current_value": cur_sum,
        "invested": inv_sum,
        "gross_profit": df["gross_profit"].sum(),
        "brokerage": df["brokerage"].sum(),
        "stcg": df["stcg"].sum(),
        "net_profit": df["net_profit"].sum(),
        "net_return_pct": (cur_sum / inv_sum - 1) * 100 if inv_sum else 0.0,
        "weighted_beta": (df["weight_pct"] / 100 * df["beta"]).sum(),
        "target": sc["target"],
        "elapsed_years": elapsed_years,
        "target_date": TARGET_DATE,
        "days_to_target": max((TARGET_DATE - today).days, 0),
        "entry_date": INVESTMENT_DATE,
    }

    return df, kpis


@st.cache_data(ttl=900, show_spinner=False)
def get_nifty_norm():
    close_hist = fetch_history(BENCHMARK)
    if close_hist is None or close_hist.empty:
        return None

    entry_date = get_entry_date(close_hist)
    if entry_date is None:
        return None

    close = close_hist.loc[entry_date:, "Close"]
    return close / close.iloc[0] * 100


@st.cache_data(ttl=900, show_spinner=False)
def build_portfolio_index_series(scenario_name):
    """Actual portfolio performance from 01-Sep-2026, not a historical 3-year lookback."""
    weights = SCENARIOS[scenario_name]["weights"]
    nifty_hist = fetch_history(BENCHMARK)

    if nifty_hist is None or nifty_hist.empty:
        return None

    entry_date = get_entry_date(nifty_hist)
    if entry_date is None:
        return None

    master_idx = nifty_hist.loc[entry_date:].index
    elapsed_days = (master_idx - master_idx[0]).days.values
    portfolio_rel = pd.Series(0.0, index=master_idx)

    for t, w_pct in weights.items():
        meta = ASSET_META[t]
        w = w_pct / 100.0

        if meta["class"] == "Debt":
            rel = pd.Series(
                (1 + meta["yield"]) ** (elapsed_days / 365.25),
                index=master_idx
            )
        else:
            hist = fetch_history(t)
            if hist is not None and not hist.empty:
                stock_entry = get_entry_date(hist)
                if stock_entry is not None:
                    s = hist.loc[stock_entry:, "Close"].reindex(master_idx, method="ffill")
                    s = s.dropna()
                    rel = s / s.iloc[0]
                    rel = rel.reindex(master_idx).ffill().bfill()
                else:
                    rel = pd.Series(1.0, index=master_idx)
            else:
                # Missing live data is kept flat rather than fabricating a
                # return from an unrelated historical fallback price.
                rel = pd.Series(1.0, index=master_idx)

        portfolio_rel = portfolio_rel.add(rel * w, fill_value=0)

    return portfolio_rel * 100


def valuation_status(pe, sector):
    avg = INDUSTRY_AVG_PE.get(sector)
    if pe is None or avg is None:
        return "N/A"
    diff = (pe - avg) / avg
    if diff > 0.15:
        return f"Overvalued vs {sector} avg ({avg:.1f}x)"
    if diff < -0.15:
        return f"Undervalued vs {sector} avg ({avg:.1f}x)"
    return f"Fairly valued vs {sector} avg ({avg:.1f}x)"


# -----------------------------------------------------------------------------
# 6. CHART BUILDERS
# -----------------------------------------------------------------------------
CLASS_ORDER = ["Equity", "Debt", "Gold"]
CLASS_COLORS = ["#1a73e8", "#5f6368", "#fbbc04"]


def build_donut(df, scenario_name):
    grp = df.groupby("class")["current_value"].sum().reindex(CLASS_ORDER).fillna(0)
    fig = go.Figure(go.Pie(labels=grp.index, values=grp.values, hole=0.55,
                            marker=dict(colors=CLASS_COLORS)))
    fig.update_layout(title=f"Asset Class Allocation — {scenario_name}", height=380,
                       template="plotly_white", legend=dict(orientation="h", y=-0.15),
                       margin=dict(t=60, b=40))
    return fig


def build_bar(df, scenario_name):
    d = df.sort_values("net_profit")
    colors = ["#188038" if v >= 0 else "#d93025" for v in d["net_profit"]]
    fig = go.Figure(go.Bar(x=d["net_profit"], y=d["name"], orientation="h", marker_color=colors,
                            text=[inr(v) for v in d["net_profit"]], textposition="outside"))
    fig.update_layout(title=f"Asset-wise Net Profit / Loss — {scenario_name}", height=520,
                       template="plotly_white", margin=dict(l=190, t=60))
    return fig


def build_line(scenario_name):
    port_idx = build_portfolio_index_series(scenario_name)
    nifty_norm = get_nifty_norm()

    fig = go.Figure()

    if port_idx is not None:
        fig.add_trace(go.Scatter(
            x=port_idx.index, y=port_idx.values, mode="lines",
            name="Portfolio", line=dict(color="#1a73e8", width=2)
        ))

        sc = SCENARIOS[scenario_name]
        target_index = sc["target"] / TOTAL_CAPITAL * 100
        target_x = [INVESTMENT_DATE, TARGET_DATE]
        fig.add_trace(go.Scatter(
            x=target_x, y=[target_index, target_index],
            mode="lines", name=f"3-Year Target ({inr(sc['target'])})",
            line=dict(color="#188038", width=2, dash="dash")
        ))

    if nifty_norm is not None:
        fig.add_trace(go.Scatter(
            x=nifty_norm.index, y=nifty_norm.values, mode="lines",
            name="Nifty 50", line=dict(color="#e37400", width=2, dash="dot")
        ))
    else:
        fig.add_annotation(
            text="Benchmark history unavailable — Nifty 50 data could not be fetched",
            showarrow=False
        )

    fig.add_vline(
        x=TARGET_DATE,
        line_dash="dash",
        annotation_text=f"Target date: {TARGET_DATE.strftime('%d %b %Y')}",
        annotation_position="top right"
    )

    fig.update_layout(
        title=f"Performance Since {INVESTMENT_DATE.strftime('%d %b %Y')}: Portfolio vs Nifty 50",
        template="plotly_white", height=420,
        xaxis_title="Date", yaxis_title="Indexed Value (₹100 at investment)",
        xaxis=dict(range=[INVESTMENT_DATE, TARGET_DATE])
    )
    return fig


def build_dma_chart(ticker):
    meta = ASSET_META[ticker]
    hist = fetch_history(ticker)
    fig = go.Figure()
    if hist is not None and not hist.empty:
        dma50, dma200 = hist["Close"].rolling(50).mean(), hist["Close"].rolling(200).mean()
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Close", line=dict(color="#1a73e8")))
        fig.add_trace(go.Scatter(x=hist.index, y=dma50, name="50 DMA", line=dict(color="#e37400", width=1.3)))
        fig.add_trace(go.Scatter(x=hist.index, y=dma200, name="200 DMA", line=dict(color="#d93025", width=1.3)))
    else:
        fig.add_annotation(text="5-year chart unavailable — offline fallback mode", showarrow=False)
    fig.update_layout(title=f"{meta['name']} — 5-Year Price with 50/200 DMA", template="plotly_white",
                       height=430, xaxis=dict(rangeslider=dict(visible=True)))
    return fig


# -----------------------------------------------------------------------------
# 7. UI — HEADER + DATA-SOURCE STATUS
# -----------------------------------------------------------------------------
st.title("🇮🇳 3-Year Indian Equity Portfolio Dashboard")
st.caption(
    f"₹1 Crore capital · 15-asset universe · Investment date: {INVESTMENT_DATE.strftime('%d %b %Y')} "
    f"· 3-year target date: {TARGET_DATE.strftime('%d %b %Y')}"
)

with st.expander("📡 Data source status (tap to check live vs. offline-fallback)"):
    live_flags = {t: (fetch_history(t) is not None) for t in YF_TICKERS}
    live_count = sum(live_flags.values())
    st.caption(
        f"{live_count} / {len(live_flags)} tickers on live Yahoo Finance data. "
        f"Entry prices are taken from the first trading session on/after "
        f"{INVESTMENT_DATE.strftime('%d %b %Y')}. If Yahoo Finance is unavailable, "
        f"the dashboard will not invent an entry price from older fallback data."
    )
    status_cols = st.columns(4)
    for i, (t, ok) in enumerate(live_flags.items()):
        status_cols[i % 4].write(f"{'🟢' if ok else '🟡'} {ASSET_META[t]['name']}")

if st.button("🔄 Refresh market data now"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# -----------------------------------------------------------------------------
# 8. UI — SCENARIO BUTTONS  (st.session_state keeps the selection across reruns)
# -----------------------------------------------------------------------------
if "scenario" not in st.session_state:
    st.session_state.scenario = list(SCENARIOS.keys())[0]

btn_cols = st.columns(3)
for col, name in zip(btn_cols, SCENARIOS.keys()):
    with col:
        if st.button(
            name,
            width="stretch",
            type="primary" if st.session_state.scenario == name else "secondary",
            key=f"btn_{name}",
        ):
            st.session_state.scenario = name
            st.rerun()

scenario_name = st.session_state.scenario

try:
    df, kpis = compute_scenario(scenario_name)

    st.markdown(f"**Showing:** {scenario_name}")

    # ---- KPI cards -----------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Portfolio Value Today", inr(kpis["current_value"]))
    k2.metric(f"Invested on {INVESTMENT_DATE.strftime('%d %b %Y')}", inr(kpis["invested"]))
    k3.metric("Gross Profit", inr(kpis["gross_profit"]))
    k4.metric("Total Brokerage", inr(kpis["brokerage"]))

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Estimated Tax (Model)", inr(kpis["stcg"]))
    k6.metric("Net Take-Home Profit", inr(kpis["net_profit"]))
    k7.metric("Net Portfolio Return", f"{kpis['net_return_pct']:.2f}%")
    k8.metric("Weighted Portfolio Beta", f"{kpis['weighted_beta']:.2f}")

    gap = kpis["target"] - kpis["current_value"]
    st.info(
        f"📅 **Investment date:** {INVESTMENT_DATE.strftime('%d %b %Y')}  ·  "
        f"**3-year target date:** {TARGET_DATE.strftime('%d %b %Y')}  ·  "
        f"**Days remaining:** {kpis['days_to_target']:,}"
    )
    if gap <= 0:
        st.success(
            f"Current value is already above the model target of {inr(kpis['target'])}. "
            f"The actual 3-year result will be measured on {TARGET_DATE.strftime('%d %b %Y')}."
        )
    else:
        st.warning(
            f"Current value is {inr(gap)} below the model target of {inr(kpis['target'])}. "
            f"This is a progress check, not a completed 3-year result."
        )

    # ---- Charts ----------------------------------------------------------
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.plotly_chart(build_donut(df, scenario_name), width="stretch")
    with c2:
        st.plotly_chart(build_bar(df, scenario_name), width="stretch")

    st.plotly_chart(build_line(scenario_name), width="stretch")

    # ---- Holdings table ----------------------------------------------------
    st.subheader("Holdings Detail")
    show = df.copy()
    show["Weight %"] = show["weight_pct"].map(lambda v: f"{v:.1f}%")
    show["Allocated"] = show["allocated"].map(inr)
    show["Qty"] = show["qty"].map(lambda v: "-" if v is None or pd.isna(v) else f"{int(v):,}")
    show["Buy Price"] = show["buy_price"].map(lambda v: "-" if v is None or pd.isna(v) else inr(v))
    show["CMP"] = show["cmp"].map(lambda v: "-" if v is None or pd.isna(v) else inr(v))
    show["Current Value"] = show["current_value"].map(inr)
    show["Net Profit"] = show["net_profit"].map(inr)
    show["Return Since Entry"] = show.apply(lambda r: f"{((r["cmp"] / r["buy_price"] - 1) * 100):.2f}%" if pd.notna(r["buy_price"]) and r["buy_price"] else "-", axis=1)
    show["Beta"] = show["beta"].map(lambda v: f"{v:.2f}")
    cols = ["name", "class", "Weight %", "Allocated", "Qty", "Buy Price", "CMP",
            "Current Value", "Net Profit", "Return Since Entry", "Beta"]
    st.dataframe(show[cols].rename(columns={"name": "Asset", "class": "Class"}),
                 width="stretch", hide_index=True)

except Exception as e:
    st.error("Something went wrong rendering the scenario dashboard. Details below:")
    st.exception(e)

# -----------------------------------------------------------------------------
# 9. UI — COMPANY DEEP-DIVE SCREENER
# -----------------------------------------------------------------------------
st.divider()
st.header("🔎 Company Deep-Dive Screener")

try:
    ticker = st.selectbox(
        "Select Company",
        options=EQUITY_TICKERS,
        format_func=lambda t: ASSET_META[t]["name"],
    )

    meta = ASSET_META[ticker]
    p = get_price_series(ticker)
    info, fb = p["info"], FALLBACK_DATA.get(ticker, {})
    cmp_, hist = p["cmp"], fetch_history(ticker)

    pe = info.get("trailingPE") or fb.get("pe")
    eps = info.get("trailingEps") or fb.get("eps")
    mcap = info.get("marketCap")
    mcap_cr = (mcap / 1e7) if mcap else fb.get("mcap_cr")
    beta = p["beta"]
    low52 = info.get("fiftyTwoWeekLow") or p["low52"]
    high52 = info.get("fiftyTwoWeekHigh") or p["high52"]

    if hist is not None and not hist.empty and len(hist) > 1:
        day_change_pct = (cmp_ - float(hist["Close"].iloc[-2])) / float(hist["Close"].iloc[-2]) * 100
    else:
        day_change_pct = 0.0

    st.subheader(f"{meta['name']} ({ticker})")
    st.caption(f"{meta['sector']} — {'Live Data' if p['live'] else '⚠ Offline Fallback Data'}")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("CMP", inr(cmp_), f"{day_change_pct:+.2f}% today")
    d2.metric("Entry Price (01-Sep-2026)", inr(p["buy_price"]) if p["buy_price"] else "N/A")
    d3.metric("Trailing P/E", f"{pe:.1f}x" if pe else "N/A")
    d4.metric("Beta", f"{beta:.2f}")

    d5, d6 = st.columns(2)
    d5.metric("Trailing EPS", f"₹{eps:.2f}" if eps else "N/A")
    d6.metric("52W Range", f"{inr(low52)} – {inr(high52)}")
    if p["buy_price"]:
        entry_return = (cmp_ / p["buy_price"] - 1) * 100
        st.metric(
            f"Return Since {INVESTMENT_DATE.strftime('%d %b %Y')}",
            f"{entry_return:+.2f}%"
        )

    st.markdown(f"**Valuation Status:** {valuation_status(pe, meta['sector'])}")
    st.markdown(meta["thesis"])

    st.plotly_chart(build_dma_chart(ticker), width="stretch")

except Exception as e:
    st.error("Something went wrong rendering the company screener. Details below:")
    st.exception(e)
