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
from datetime import datetime, timedelta

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
STCG_RATE        = 0.20            # 20% on net positive gains after brokerage
BENCHMARK        = "^NSEI"
HOLDING_YEARS    = 3
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker, period="5y"):
    def _do():
        df = yf.Ticker(ticker).history(period=period, auto_adjust=False, timeout=NETWORK_TIMEOUT)
        return df if df is not None and not df.empty else None
    return _bounded(_do)


@st.cache_data(ttl=3600, show_spinner=False)
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
# 4. PRICE / REPORT-DATE HELPERS
# -----------------------------------------------------------------------------
INVESTMENT_DATE = pd.Timestamp("2026-09-01")
TARGET_DATE = pd.Timestamp("2029-09-01")


def _date_index(hist):
    """Return a timezone-naive normalized DatetimeIndex for safe date comparisons."""
    idx = hist.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return pd.DatetimeIndex(idx).normalize()


def slice_as_of(hist, as_of_date):
    if hist is None or hist.empty:
        return None
    idx = _date_index(hist)
    target = pd.Timestamp(as_of_date).normalize()
    mask = idx <= target
    if not mask.any():
        return None
    out = hist.loc[mask].copy()
    out.index = idx[mask]
    return out


def slice_from_date(hist, start_date):
    if hist is None or hist.empty:
        return None
    idx = _date_index(hist)
    target = pd.Timestamp(start_date).normalize()
    mask = idx >= target
    if not mask.any():
        return None
    out = hist.loc[mask].copy()
    out.index = idx[mask]
    return out


@st.cache_data(ttl=900, show_spinner=False)
def get_report_dates():
    """Return the actual last trading day of each week from 1-Sep-2026 onward."""
    hist = fetch_history(BENCHMARK)
    if hist is None or hist.empty:
        return []
    h = slice_from_date(hist, INVESTMENT_DATE)
    if h is None or h.empty:
        return []
    dates = pd.Series(h.index, index=h.index)
    # W-FRI groups Monday-Friday. max() gives the actual last trading day,
    # so exchange holidays are handled correctly (e.g. Thursday may be the
    # last working/trading day of a holiday-shortened week).
    weekly = dates.groupby(dates.index.to_period("W-FRI")).max()
    return [d.date() for d in weekly.tolist()]


def format_report_date(d):
    return pd.Timestamp(d).strftime("%d %b %Y")


def elapsed_years(as_of_date=None):
    # None means live mode: use today's calendar date for time-based assets.
    target = pd.Timestamp.today().normalize() if as_of_date is None else pd.Timestamp(as_of_date).normalize()
    days = max((target - INVESTMENT_DATE).days, 0)
    return days / 365.25


def find_entry_row(hist):
    """Find the first available trading day on/after 1-Sep-2026."""
    h = slice_from_date(hist, INVESTMENT_DATE)
    if h is None or h.empty:
        return None
    return h.iloc[0]


def compute_beta_vs_nifty(hist, as_of_date=None):
    try:
        nifty_hist = fetch_history(BENCHMARK)
        if nifty_hist is None or nifty_hist.empty or hist is None:
            return None
        if as_of_date is not None:
            hist = slice_as_of(hist, as_of_date)
            nifty_hist = slice_as_of(nifty_hist, as_of_date)
        if hist is None or nifty_hist is None:
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


def get_price_series(t, as_of_date=None):
    """Return market data strictly as it stood on the selected report date."""
    hist, info = fetch_history(t), fetch_info(t)
    fb = FALLBACK_DATA.get(t, {})
    full_live = hist is not None and len(hist) > 5
    h_asof = slice_as_of(hist, as_of_date) if full_live and as_of_date is not None else hist

    if h_asof is not None and not h_asof.empty:
        cmp_ = float(h_asof["Close"].iloc[-1])
        entry_row = find_entry_row(hist)
        buy_price = float(entry_row["Close"]) if entry_row is not None else None
        entry_idx = slice_from_date(hist, INVESTMENT_DATE)
        if entry_idx is None or entry_idx.empty or pd.Timestamp(h_asof.index[-1]) < pd.Timestamp(entry_idx.index[0]):
            buy_price = None
        beta = info.get("beta") or compute_beta_vs_nifty(hist, as_of_date) or fb.get("beta", 1.0)
        low52 = float(h_asof["Close"].tail(252).min())
        high52 = float(h_asof["Close"].tail(252).max())
        prev_close = float(h_asof["Close"].iloc[-2]) if len(h_asof) > 1 else cmp_
        day_change_pct = ((cmp_ / prev_close) - 1) * 100 if prev_close else 0.0
        return {"cmp": cmp_, "buy_price": buy_price, "beta": beta,
                "low52": low52, "high52": high52, "live": True,
                "info": info, "price_date": h_asof.index[-1],
                "day_change_pct": day_change_pct}

    # Offline fallback: useful only when historical Yahoo data is unavailable.
    cmp_ = fb.get("cmp", 100.0)
    buy_price = fb.get("buy_price", cmp_ * 0.6)
    beta = fb.get("beta", 1.0)
    low52 = fb.get("low52", cmp_ * 0.8)
    high52 = fb.get("high52", cmp_ * 1.2)
    return {"cmp": cmp_, "buy_price": buy_price, "beta": beta,
            "low52": low52, "high52": high52, "live": False,
            "info": info, "price_date": None, "day_change_pct": 0.0}


# -----------------------------------------------------------------------------
# 5. SCENARIO PORTFOLIO ENGINE — AS OF SELECTED WEEKLY REPORT DATE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def compute_scenario(scenario_name, as_of_date):
    sc = SCENARIOS[scenario_name]
    as_of = pd.Timestamp.today().normalize() if as_of_date is None else pd.Timestamp(as_of_date).normalize()
    years = elapsed_years(as_of)
    rows = []

    for t, w_pct in sc["weights"].items():
        meta = ASSET_META[t]
        allocated = TOTAL_CAPITAL * w_pct / 100.0

        if meta["class"] == "Debt":
            invested = allocated
            current_value = invested * ((1 + meta["yield"]) ** years)
            qty = buy_price = cmp_ = None
            beta = 0.0
            leftover = 0.0
            price_date = None
            live = True
        else:
            p = get_price_series(t, as_of)
            buy_price, cmp_, beta = p["buy_price"], p["cmp"], p["beta"]
            if buy_price and buy_price > 0 and cmp_ is not None:
                qty = int(allocated / buy_price)
                invested_equity = qty * buy_price
                leftover = allocated - invested_equity
                # Leftover cash remains cash; it does not magically earn stock returns.
                current_value = qty * cmp_ + leftover
            else:
                qty = 0
                leftover = allocated
                current_value = allocated
            invested = allocated
            price_date = p.get("price_date")
            live = p.get("live", False)

        gross_profit = current_value - invested
        brokerage = BROKERAGE_RATE * (invested + current_value)
        taxable_gain = max(gross_profit - brokerage, 0.0)
        stcg = STCG_RATE * taxable_gain
        net_profit = gross_profit - brokerage - stcg
        cagr = ((current_value / invested) ** (1 / years) - 1) if invested > 0 and years > 0 else 0.0

        rows.append({
            "ticker": t, "name": meta["name"], "class": meta["class"], "sector": meta.get("sector", ""),
            "weight_pct": w_pct, "allocated": allocated, "qty": qty, "buy_price": buy_price, "cmp": cmp_,
            "invested": invested, "leftover": leftover, "current_value": current_value,
            "gross_profit": gross_profit, "brokerage": brokerage, "stcg": stcg,
            "net_profit": net_profit, "beta": beta, "cagr": cagr,
            "price_date": price_date, "live": live,
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
        "gross_return_pct": (cur_sum / inv_sum - 1) * 100 if inv_sum else 0.0,
        "net_return_pct": (df["net_profit"].sum() / inv_sum) * 100 if inv_sum else 0.0,
        "weighted_beta": (df["weight_pct"] / 100 * df["beta"]).sum(),
        "target": sc["target"],
        "years_elapsed": years,
        "days_elapsed": max((as_of - INVESTMENT_DATE).days, 0),
    }
    return df, kpis


@st.cache_data(ttl=900, show_spinner=False)
def build_portfolio_index_series(scenario_name, as_of_date):
    """Portfolio growth from the fixed 1-Sep-2026 investment date to report date."""
    nifty_hist = fetch_history(BENCHMARK)
    if nifty_hist is None or nifty_hist.empty:
        return None, None
    if as_of_date is not None:
        nifty_hist = slice_as_of(nifty_hist, as_of_date)
    nifty_hist = slice_from_date(nifty_hist, INVESTMENT_DATE)
    if nifty_hist is None or nifty_hist.empty:
        return None, None

    weights = SCENARIOS[scenario_name]["weights"]
    master_idx = nifty_hist.index
    days = (master_idx - master_idx[0]).days.values
    portfolio_rel = pd.Series(0.0, index=master_idx)

    for t, w_pct in weights.items():
        meta = ASSET_META[t]
        w = w_pct / 100.0
        if meta["class"] == "Debt":
            rel = pd.Series((1 + meta["yield"]) ** (days / 365.25), index=master_idx)
        else:
            hist = fetch_history(t)
            h = slice_as_of(hist, as_of_date) if as_of_date is not None else hist
            h = slice_from_date(h, INVESTMENT_DATE) if h is not None else None
            if h is not None and not h.empty:
                s = h["Close"].reindex(master_idx).ffill().bfill()
                # Align to the master index and start at the first available price.
                if not s.empty:
                    rel = s / s.iloc[0]
                else:
                    rel = pd.Series(1.0, index=master_idx)
            else:
                # Do not invent a historical stock return if Yahoo data is unavailable.
                rel = pd.Series(1.0, index=master_idx)
        portfolio_rel = portfolio_rel.add(rel * w, fill_value=0)

    portfolio_index = portfolio_rel * 100
    nifty_close = nifty_hist["Close"]
    nifty_norm = nifty_close / nifty_close.iloc[0] * 100
    return portfolio_index, nifty_norm


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


def build_donut(df, scenario_name, report_label):
    grp = df.groupby("class")["current_value"].sum().reindex(CLASS_ORDER).fillna(0)
    fig = go.Figure(go.Pie(labels=grp.index, values=grp.values, hole=0.55,
                            marker=dict(colors=CLASS_COLORS)))
    fig.update_layout(title=f"Asset Class Allocation — {scenario_name} — {report_label}", height=380,
                      template="plotly_white", legend=dict(orientation="h", y=-0.15),
                      margin=dict(t=60, b=40))
    return fig


def build_bar(df, scenario_name, report_label):
    d = df.sort_values("net_profit")
    colors = ["#188038" if v >= 0 else "#d93025" for v in d["net_profit"]]
    fig = go.Figure(go.Bar(x=d["net_profit"], y=d["name"], orientation="h", marker_color=colors,
                           text=[inr(v) for v in d["net_profit"]], textposition="outside"))
    fig.update_layout(title=f"Asset-wise Net Profit / Loss — {scenario_name} — {report_label}", height=520,
                      template="plotly_white", margin=dict(l=190, t=60))
    return fig


def build_line(scenario_name, as_of_date):
    port_idx, nifty_norm = build_portfolio_index_series(scenario_name, as_of_date)
    fig = go.Figure()
    if port_idx is not None and nifty_norm is not None:
        fig.add_trace(go.Scatter(x=port_idx.index, y=port_idx.values, mode="lines",
                                 name="Portfolio", line=dict(color="#1a73e8", width=2)))
        fig.add_trace(go.Scatter(x=nifty_norm.index, y=nifty_norm.values, mode="lines",
                                 name="Nifty 50", line=dict(color="#e37400", width=2, dash="dot")))
    else:
        fig.add_annotation(text="Historical benchmark data unavailable", showarrow=False)
    end_label = "Latest Available" if as_of_date is None else format_report_date(as_of_date)
    fig.update_layout(title=f"Portfolio vs Nifty 50: 1-Sep-2026 to {end_label} (Base = 100) — {scenario_name}",
                      template="plotly_white", height=420, xaxis_title="Date", yaxis_title="Indexed Value")
    return fig


def build_dma_chart(ticker, as_of_date):
    meta = ASSET_META[ticker]
    hist = fetch_history(ticker)
    if as_of_date is not None:
        hist = slice_as_of(hist, as_of_date)
    fig = go.Figure()
    if hist is not None and not hist.empty:
        dma50, dma200 = hist["Close"].rolling(50).mean(), hist["Close"].rolling(200).mean()
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Close", line=dict(color="#1a73e8")))
        fig.add_trace(go.Scatter(x=hist.index, y=dma50, name="50 DMA", line=dict(color="#e37400", width=1.3)))
        fig.add_trace(go.Scatter(x=hist.index, y=dma200, name="200 DMA", line=dict(color="#d93025", width=1.3)))
    else:
        fig.add_annotation(text="Historical chart unavailable", showarrow=False)
    end_label = "Latest Available" if as_of_date is None else format_report_date(as_of_date)
    fig.update_layout(title=f"{meta['name']} — Price & 50/200 DMA as of {end_label}",
                      template="plotly_white", height=430, xaxis=dict(rangeslider=dict(visible=True)))
    return fig


# -----------------------------------------------------------------------------
# 7. UI — HEADER + REPORT DATE SELECTOR
# -----------------------------------------------------------------------------
st.title("🇮🇳 3-Year Indian Equity Portfolio Dashboard")
st.caption("₹1 Crore capital · 15-asset universe · Fixed investment date: 1 September 2026 · 3-year target: 1 September 2029")

with st.expander("📡 Data source status (tap to check live vs. offline-fallback)"):
    live_flags = {t: (fetch_history(t) is not None) for t in YF_TICKERS}
    live_count = sum(live_flags.values())
    st.caption(f"{live_count} / {len(live_flags)} tickers have Yahoo Finance history available. "
               f"Historical report dates use the market data available up to the selected date.")
    status_cols = st.columns(4)
    for i, (t, ok) in enumerate(live_flags.items()):
        status_cols[i % 4].write(f"{'🟢' if ok else '🟡'} {ASSET_META[t]['name']}")

# Build the report-date dropdown from actual NSE/Nifty trading sessions.
# Fresh start defaults to LIVE. Selecting a date switches the whole dashboard
# into historical/as-of mode. The selector and live/status controls are kept
# side-by-side in a 3:2 layout.
report_dates = get_report_dates()
if not report_dates:
    st.error("Unable to load weekly NSE trading dates. Please refresh after Yahoo Finance data becomes available.")
    st.stop()

LIVE_OPTION = "🟢 LIVE — Latest Available Market Data"
report_options = [LIVE_OPTION] + report_dates

if "report_selection" not in st.session_state:
    st.session_state.report_selection = LIVE_OPTION

# 3:2 layout: report-date dropdown on the left, live/status controls on the right.
selector_col, status_col = st.columns([3, 2], gap="large")

with selector_col:
    selected_option = st.selectbox(
        "📅 Weekly Performance Report — Select a Last Working / Trading Day",
        options=report_options,
        index=report_options.index(st.session_state.report_selection) if st.session_state.report_selection in report_options else 0,
        format_func=lambda d: d if isinstance(d, str) else format_report_date(d),
        help="Fresh start shows live/latest available market data. Select any historical weekly date to see the complete dashboard exactly as of that date."
    )

st.session_state.report_selection = selected_option

# Live mode is represented internally by None. Historical mode uses the selected date.
is_live_mode = selected_option == LIVE_OPTION
selected_date = None if is_live_mode else selected_option

# Keep the historical-report summary inside the 3x selector column.
with selector_col:
    if not is_live_mode:
        report_label_preview = format_report_date(selected_date)
        if pd.Timestamp(selected_date) < INVESTMENT_DATE:
            st.error("The selected report date is before the 1 September 2026 investment date.")
            st.stop()
        st.info(
            f"📌 **Historical report as on {report_label_preview}** | "
            f"Investment date: **01 Sep 2026** | Target date: **01 Sep 2029** | "
            f"Elapsed: **{max((pd.Timestamp(selected_date) - INVESTMENT_DATE).days, 0)} days**"
        )

with status_col:
    if is_live_mode:
        st.markdown("**🟢 LIVE DATA**")
        st.caption("Showing the latest available market data. Select a weekly date on the left to generate a historical report.")
        st.button(
            "🟢 Live Data (Current)",
            type="primary",
            width="stretch",
            disabled=True,
            help="The dashboard is already showing the latest available market data."
        )
    else:
        report_label_preview = format_report_date(selected_date)
        st.markdown(f"**📊 Historical Report: {report_label_preview}**")
        st.caption("The entire dashboard is calculated as of the selected last working/trading day.")
        if st.button(
            "↩️ Back to Live Data",
            type="primary",
            width="stretch",
            help="Return the entire dashboard to the latest available market data."
        ):
            st.session_state.report_selection = LIVE_OPTION
            st.rerun()

if is_live_mode:
    report_label = "LIVE — Latest Available"
else:
    report_label = format_report_date(selected_date)

# -----------------------------------------------------------------------------
# 8. UI — SCENARIO BUTTONS
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
    df, kpis = compute_scenario(scenario_name, selected_date)

    st.markdown(f"**Showing:** {scenario_name}  |  **As on:** {report_label}")

    # ---- KPI cards -----------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Portfolio Value", inr(kpis["current_value"]))
    k2.metric("Capital Invested", inr(kpis["invested"]))
    k3.metric("Gross Profit / Loss", inr(kpis["gross_profit"]))
    k4.metric("Hypothetical Brokerage", inr(kpis["brokerage"]))

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Estimated STCG Tax (20%)", inr(kpis["stcg"]))
    k6.metric("Net Profit After Costs", inr(kpis["net_profit"]))
    k7.metric("Net Return Since 01-Sep-2026", f"{kpis['net_return_pct']:.2f}%")
    k8.metric("Weighted Portfolio Beta", f"{kpis['weighted_beta']:.2f}")

    # Target is a 3-year terminal objective, not an achievement today.
    target_gap = kpis["target"] - kpis["current_value"]
    target_progress = (kpis["current_value"] / kpis["target"] * 100) if kpis["target"] else 0
    if not is_live_mode and pd.Timestamp(selected_date) >= TARGET_DATE:
        if target_gap <= 0:
            st.success(f"3-Year Target {inr(kpis['target'])} — achieved / exceeded ✅")
        else:
            st.warning(f"3-Year Target {inr(kpis['target'])} — shortfall of {inr(target_gap)}")
    else:
        st.info(f"3-Year Target: **{inr(kpis['target'])}** | Current value is **{target_progress:.1f}%** of target | "
                f"Current gap: **{inr(abs(target_gap))}** {'below' if target_gap > 0 else 'above'} target.")

    # ---- Charts ----------------------------------------------------------
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.plotly_chart(build_donut(df, scenario_name, report_label), width="stretch")
    with c2:
        st.plotly_chart(build_bar(df, scenario_name, report_label), width="stretch")

    st.plotly_chart(build_line(scenario_name, selected_date), width="stretch")

    # ---- Holdings table ----------------------------------------------------
    st.subheader(f"Holdings Detail — As on {report_label}")
    show = df.copy()
    show["Weight %"] = show["weight_pct"].map(lambda v: f"{v:.1f}%")
    show["Allocated"] = show["allocated"].map(inr)
    show["Qty"] = show["qty"].map(lambda v: "-" if v is None or pd.isna(v) else f"{int(v):,}")
    show["Buy Price (01-Sep-2026)"] = show["buy_price"].map(lambda v: "-" if v is None or pd.isna(v) else inr(v))
    show["Price on Report Date"] = show["cmp"].map(lambda v: "-" if v is None or pd.isna(v) else inr(v))
    show["Current Value"] = show["current_value"].map(inr)
    show["Gross P/L"] = show["gross_profit"].map(inr)
    show["Net Profit"] = show["net_profit"].map(inr)
    show["CAGR"] = show["cagr"].map(lambda v: f"{v * 100:.2f}%")
    show["Beta"] = show["beta"].map(lambda v: f"{v:.2f}")
    cols = ["name", "class", "Weight %", "Allocated", "Qty", "Buy Price (01-Sep-2026)",
            "Price on Report Date", "Current Value", "Gross P/L", "Net Profit", "CAGR", "Beta"]
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
    p = get_price_series(ticker, selected_date)
    info, fb = p["info"], FALLBACK_DATA.get(ticker, {})
    cmp_, hist = p["cmp"], fetch_history(ticker)

    pe = info.get("trailingPE") or fb.get("pe")
    eps = info.get("trailingEps") or fb.get("eps")
    mcap = info.get("marketCap")
    mcap_cr = (mcap / 1e7) if mcap else fb.get("mcap_cr")
    beta = p["beta"]
    low52 = info.get("fiftyTwoWeekLow") or p["low52"]
    high52 = info.get("fiftyTwoWeekHigh") or p["high52"]

    st.subheader(f"{meta['name']} ({ticker})")
    st.caption(f"{meta['sector']} — Price data as of **{report_label}** — "
               f"{'Live Yahoo history' if p['live'] else 'Offline fallback'}")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Price on Report Date", inr(cmp_), f"{p.get('day_change_pct', 0):+.2f}% vs previous trading day")
    d2.metric("Market Cap", ("₹" + format(mcap_cr, ",.0f") + " Cr") if mcap_cr else "N/A")
    d3.metric("Trailing P/E", f"{pe:.1f}x" if pe else "N/A")
    d4.metric("Beta", f"{beta:.2f}")

    d5, d6 = st.columns(2)
    d5.metric("Trailing EPS", f"₹{eps:.2f}" if eps else "N/A")
    d6.metric("52W Range", f"{inr(low52)} – {inr(high52)}")

    st.caption("⚠️ P/E, EPS, market cap and valuation are the latest fundamentals supplied by Yahoo Finance; "
               "Yahoo does not reliably provide point-in-time historical fundamentals. Price, return and chart data are as of the selected report date.")
    st.markdown(f"**Valuation Status (latest fundamentals):** {valuation_status(pe, meta['sector'])}")
    st.markdown(meta["thesis"])

    st.plotly_chart(build_dma_chart(ticker, selected_date), width="stretch")

except Exception as e:
    st.error("Something went wrong rendering the company screener. Details below:")
    st.exception(e)
