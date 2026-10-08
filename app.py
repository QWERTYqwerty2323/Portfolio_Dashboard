import warnings
warnings.filterwarnings('ignore')

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ============================================================
# ALPHA 6-WEEK RESELECTION + WEEKLY REBALANCING DASHBOARD
# ============================================================
# Initial universe: 15 equities + 3 debt instruments + Gold ETF
# After 6 weeks: select 12 equities + 2 debt instruments + Gold ETF
# Then: redistribute weights every week using a transparent factor score.
# Start: 01-Aug-2026. Since 01-Aug-2026 is Saturday, first trading day is
# the first available NSE session on/after that date.
# ============================================================

st.set_page_config(page_title='ALPHA — 6-Week Reselection Portfolio', page_icon='📈', layout='wide')

START_DATE = pd.Timestamp('2026-08-01')
RESELECT_AFTER_WEEKS = 6
RESELECT_DATE = pd.Timestamp('2026-09-11')  # last trading day of week 6
CAPITAL = 1_00_00_000
BENCHMARK = '^NSEI'
GOLD = 'GOLDBEES.NS'

# Existing two debt instruments + one new debt instrument.
DEBT_META = {
    '10Y_GSEC': {
        'name': '10Y Indian G-Sec', 'yield': 0.0700,
        'risk': 0.15, 'class': 'Debt', 'sector': 'Sovereign Debt'
    },
    'AAA_BOND': {
        'name': 'AAA Corporate Bond / Debenture', 'yield': 0.0800,
        'risk': 0.30, 'class': 'Debt', 'sector': 'Corporate Debt'
    },
    'SDL_10Y': {
        'name': '10Y State Development Loan (NEW)', 'yield': 0.0725,
        'risk': 0.22, 'class': 'Debt', 'sector': 'State Debt'
    },
}

EQUITY_META = {
    'MARUTI.NS': ('Maruti Suzuki India Ltd', 'Automobiles'),
    'BAJAJ-AUTO.NS': ('Bajaj Auto Ltd', 'Automobiles'),
    'M&M.NS': ('Mahindra & Mahindra Ltd', 'Automobiles'),
    'HAL.NS': ('Hindustan Aeronautics Ltd', 'Defense / Capital Goods'),
    'SIEMENS.NS': ('Siemens Ltd', 'Capital Goods'),
    'POLYCAB.NS': ('Polycab India Ltd', 'Capital Goods / Manufacturing'),
    'AXISBANK.NS': ('Axis Bank Ltd', 'Financials'),
    'BSE.NS': ('BSE Ltd', 'Capital Markets'),
    'HINDUNILVR.NS': ('Hindustan Unilever Ltd', 'FMCG'),
    'BRITANNIA.NS': ('Britannia Industries Ltd', 'FMCG'),
    'ITC.NS': ('ITC Ltd', 'FMCG'),
    'PIDILITIND.NS': ('Pidilite Industries Ltd', 'Chemicals'),
    'SOLARINDS.NS': ('Solar Industries India Ltd', 'Chemicals / Defense'),
    'PIIND.NS': ('PI Industries Ltd', 'Agrochemicals'),
    'ANGELONE.NS': ('Angel One Ltd', 'Capital Markets'),
}

ALL_EQUITIES = list(EQUITY_META.keys())
ALL_DEBT = list(DEBT_META.keys())
YF_TICKERS = ALL_EQUITIES + [GOLD, BENCHMARK]

# Peer-relative P/E reference values. These are deliberately simple model inputs,
# not claims about today's live sector averages.
PE_PEERS = {
    'Automobiles': 28.0,
    'Defense / Capital Goods': 40.0,
    'Capital Goods': 45.0,
    'Capital Goods / Manufacturing': 42.0,
    'Financials': 20.0,
    'Capital Markets': 35.0,
    'FMCG': 55.0,
    'Chemicals / Defense': 50.0,
    'Chemicals': 50.0,
    'Agrochemicals': 30.0,
}

# ------------------------- data layer -------------------------

def normalize_index(df):
    if df is None or df.empty:
        return None
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep='last')].sort_index()
    return out


def safe_history(ticker, period='6mo'):
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=False, timeout=10)
        return normalize_index(df)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def get_history(ticker, period='6mo'):
    return safe_history(ticker, period)


@st.cache_data(ttl=1800, show_spinner=False)
def get_info(ticker):
    try:
        x = yf.Ticker(ticker).info
        return x if isinstance(x, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def load_market_data():
    data = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(get_history, t, '6mo'): t for t in YF_TICKERS}
        for f in as_completed(futures):
            t = futures[f]
            try:
                data[t] = f.result()
            except Exception:
                data[t] = None
    return data


market = load_market_data()


def first_on_or_after(hist, date):
    if hist is None or hist.empty:
        return None
    h = hist[hist.index >= pd.Timestamp(date).normalize()]
    return h.iloc[0] if not h.empty else None


def asof(hist, date):
    if hist is None or hist.empty:
        return None
    h = hist[hist.index <= pd.Timestamp(date).normalize()]
    return h if not h.empty else None


def close_on_or_before(ticker, date):
    h = asof(market.get(ticker), date)
    if h is None or h.empty:
        return np.nan
    return float(h['Close'].iloc[-1])


def last_available_date():
    dates = []
    for t in YF_TICKERS:
        h = market.get(t)
        if h is not None and not h.empty:
            dates.append(h.index[-1])
    return max(dates) if dates else pd.Timestamp.today().normalize()


LATEST_DATE = last_available_date()

# ------------------------- scoring -------------------------

def return_pct(hist, end_date, lookback_days):
    h = asof(hist, end_date)
    if h is None or h.empty:
        return np.nan
    end = float(h['Close'].iloc[-1])
    target = pd.Timestamp(end_date).normalize() - pd.Timedelta(days=lookback_days)
    before = h[h.index <= target]
    if before.empty or end <= 0:
        return np.nan
    start = float(before['Close'].iloc[-1])
    return (end / start - 1) * 100 if start else np.nan


def volatility_pct(hist, end_date, lookback=30):
    h = asof(hist, end_date)
    if h is None or len(h) < 10:
        return np.nan
    r = h['Close'].pct_change().dropna().tail(lookback)
    return float(r.std() * np.sqrt(252) * 100) if len(r) >= 5 else np.nan


def beta_vs_nifty(ticker, end_date):
    h = asof(market.get(ticker), end_date)
    n = asof(market.get(BENCHMARK), end_date)
    if h is None or n is None:
        return np.nan
    a = h['Close'].pct_change().rename('asset')
    b = n['Close'].pct_change().rename('nifty')
    j = pd.concat([a, b], axis=1).dropna().tail(90)
    if len(j) < 20 or j['nifty'].var() == 0:
        return np.nan
    return float(j['asset'].cov(j['nifty']) / j['nifty'].var())


def max_drawdown(hist, end_date, lookback=42):
    h = asof(hist, end_date)
    if h is None or h.empty:
        return np.nan
    s = h['Close'].tail(lookback)
    if s.empty:
        return np.nan
    dd = s / s.cummax() - 1
    return float(dd.min() * 100)


def fundamentals(ticker, end_date):
    info = get_info(ticker)
    pe = info.get('trailingPE')
    if pe is None:
        eps = info.get('trailingEps')
        px = close_on_or_before(ticker, end_date)
        pe = px / eps if eps and eps > 0 else np.nan
    try:
        pe = float(pe)
    except Exception:
        pe = np.nan
    sector = EQUITY_META[ticker][1]
    peer = PE_PEERS.get(sector, np.nan)
    pe_discount = ((peer - pe) / peer * 100) if pd.notna(pe) and pd.notna(peer) and peer else np.nan
    return pe, peer, pe_discount


def equity_score(ticker, end_date):
    h = market.get(ticker)
    r6 = return_pct(h, end_date, 42)
    r3 = return_pct(h, end_date, 21)
    vol = volatility_pct(h, end_date, 30)
    beta = beta_vs_nifty(ticker, end_date)
    dd = max_drawdown(h, end_date, 42)
    pe, peer, pe_discount = fundamentals(ticker, end_date)

    # Higher is better: momentum + valuation discount, while penalising volatility,
    # beta and drawdown. Missing metrics receive neutral values.
    def z(v, neutral=0):
        return float(v) if pd.notna(v) else neutral

    raw = (
        0.35 * z(r6) +
        0.20 * z(r3) +
        0.20 * z(pe_discount) -
        0.10 * z(vol) -
        0.10 * z(beta - 1) * 10 +
        0.05 * z(dd)
    )
    return {
        'ticker': ticker,
        'company': EQUITY_META[ticker][0],
        'sector': EQUITY_META[ticker][1],
        '6W Return %': r6,
        '3W Return %': r3,
        'Volatility %': vol,
        'Beta': beta,
        '6W Drawdown %': dd,
        'P/E': pe,
        'Peer P/E': peer,
        'P/E Discount %': pe_discount,
        'Score': raw,
    }


def debt_score(ticker):
    m = DEBT_META[ticker]
    # Yield is the primary driver; lower model risk is a secondary driver.
    score = m['yield'] * 100 - m['risk'] * 2
    return {'ticker': ticker, 'Instrument': m['name'], 'Yield %': m['yield'] * 100,
            'Risk Score': m['risk'], 'Score': score}


# ------------------------- selection -------------------------

def select_rebalanced_universe(reselection_date):
    equity_rows = [equity_score(t, reselection_date) for t in ALL_EQUITIES]
    eq = pd.DataFrame(equity_rows)
    # If live Yahoo metrics are unavailable, ranking still works from the available
    # columns because neutral missing values are used.
    eq = eq.sort_values(['Score', '6W Return %'], ascending=False, na_position='last').reset_index(drop=True)
    selected_eq = eq.head(12)['ticker'].tolist()

    debt_rows = pd.DataFrame([debt_score(t) for t in ALL_DEBT])
    debt_rows = debt_rows.sort_values('Score', ascending=False)
    selected_debt = debt_rows.head(2)['ticker'].tolist()

    return eq, debt_rows, selected_eq, selected_debt


# ------------------------- weekly weights -------------------------

def minmax(series, default=0.5):
    s = pd.to_numeric(series, errors='coerce')
    if s.notna().sum() == 0:
        return pd.Series(default, index=series.index)
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(default, index=series.index)
    out = (s - lo) / (hi - lo)
    return out.fillna(default)


def weekly_weights(selected_eq, selected_debt, rebalance_date):
    # Equity target 78%, debt 12%, gold 10%. Within each bucket, weights move
    # weekly according to rolling momentum/risk/valuation score.
    rows = []
    eq_rows = [equity_score(t, rebalance_date) for t in selected_eq]
    eq = pd.DataFrame(eq_rows)
    if eq.empty:
        eq_weights = {}
    else:
        mom = minmax(eq['6W Return %'])
        val = minmax(eq['P/E Discount %'])
        risk = 1 - minmax(eq['Volatility %'])
        beta = 1 - minmax(eq['Beta'])
        composite = 0.45 * mom + 0.25 * val + 0.20 * risk + 0.10 * beta
        composite = composite.clip(lower=0.05)
        eq_weights = dict(zip(eq['ticker'], (composite / composite.sum() * 78.0)))

    # Debt weights: relative yield/risk score inside the selected debt bucket.
    drows = [debt_score(t) for t in selected_debt]
    ds = pd.DataFrame(drows)
    if not ds.empty:
        dscore = ds['Score'].clip(lower=0.01)
        dweights = dict(zip(ds['ticker'], dscore / dscore.sum() * 12.0))
    else:
        dweights = {}

    weights = {**eq_weights, **dweights, GOLD: 10.0}
    # Floating point safety.
    total = sum(weights.values())
    weights = {k: v * 100 / total for k, v in weights.items()}
    return weights


# ------------------------- portfolio simulation -------------------------

def trading_dates(start, end):
    n = asof(market.get(BENCHMARK), end)
    if n is None:
        return pd.DatetimeIndex([])
    return n[(n.index >= pd.Timestamp(start)) & (n.index <= pd.Timestamp(end))].index


def portfolio_value_path(start_date, end_date, selected_eq, selected_debt, weekly=True):
    dates = trading_dates(start_date, end_date)
    if len(dates) == 0:
        return pd.Series(dtype=float), pd.DataFrame()

    # Weekly rebalance occurs on each week's first available trading day.
    week_keys = pd.Series(dates, index=dates).groupby(dates.to_period('W-FRI')).first()
    rebalance_map = {d: weekly_weights(selected_eq, selected_debt, d) for d in week_keys.tolist()} if weekly else {}

    value = CAPITAL
    values = []
    weight_rows = []
    prev_date = dates[0]

    # Start with weights at first trading day.
    current_weights = rebalance_map.get(dates[0], {t: 100 / len(selected_eq + selected_debt + [GOLD]) for t in selected_eq + selected_debt + [GOLD]})

    for d in dates:
        if d in rebalance_map:
            current_weights = rebalance_map[d]
            weight_rows.append({'Date': d, **current_weights})

        # Apply one-day return from previous session. Debt accrues continuously.
        if d != dates[0]:
            daily_return = 0.0
            for t, w in current_weights.items():
                if t in DEBT_META:
                    r = (1 + DEBT_META[t]['yield']) ** (1 / 365.25) - 1
                else:
                    h = market.get(t)
                    if h is None:
                        r = 0.0
                    else:
                        try:
                            p0 = float(h.loc[prev_date, 'Close']) if prev_date in h.index else close_on_or_before(t, prev_date)
                            p1 = float(h.loc[d, 'Close']) if d in h.index else close_on_or_before(t, d)
                            r = p1 / p0 - 1 if p0 and pd.notna(p1) else 0.0
                        except Exception:
                            r = 0.0
                daily_return += (w / 100.0) * r
            value *= (1 + daily_return)
        values.append((d, value))
        prev_date = d

    series = pd.Series(dict(values)).sort_index()
    weights_df = pd.DataFrame(weight_rows).fillna(0)
    return series, weights_df


def benchmark_path(start_date, end_date):
    h = market.get(BENCHMARK)
    h = asof(h, end_date)
    if h is None:
        return pd.Series(dtype=float)
    h = h[h.index >= pd.Timestamp(start_date)]
    if h.empty:
        return pd.Series(dtype=float)
    return CAPITAL * h['Close'] / h['Close'].iloc[0]


# ============================================================
# UI
# ============================================================
st.title('📈 ALPHA — 6-Week Reselection & Weekly Rebalancing')
st.caption('Research universe: 15 equities + 3 debt + Gold ETF → starting portfolio: 12 equities + 2 debt + Gold → after 6 weeks: reselect 12 + 2 + Gold → weekly weight redistribution throughout')

st.info(
    f'**Investment start:** {START_DATE:%d %b %Y}  |  '
    f'**6-week review:** {RESELECT_DATE:%d %b %Y}  |  '
    f'**Latest available data:** {LATEST_DATE:%d %b %Y}'
)

# ------------------------------------------------------------
# 1. INITIAL UNIVERSE + INITIAL SELECTION
# ------------------------------------------------------------
st.header('1. Initial Universe and Starting Selection')
st.write('The strategy begins with a **research universe of 15 companies + 3 debt instruments + Gold ETF**. Before the first investment period, the model selects **12 companies + 2 debt instruments + Gold ETF**. Only these 15 selected instruments are held during the first six weeks.')

initial_cols = st.columns(4)
initial_cols[0].metric('Research equities', '15')
initial_cols[1].metric('Research debt', '3')
initial_cols[2].metric('Selected equities', '12')
initial_cols[3].metric('Selected debt', '2 + Gold')

initial_assets = pd.DataFrame([
    {'Type': 'Equity', 'Instrument': EQUITY_META[t][0], 'Ticker': t, 'Sector': EQUITY_META[t][1]}
    for t in ALL_EQUITIES
] + [
    {'Type': 'Debt', 'Instrument': DEBT_META[t]['name'], 'Ticker': t, 'Sector': DEBT_META[t]['sector']}
    for t in ALL_DEBT
] + [{'Type': 'Gold', 'Instrument': 'Nippon India ETF Gold BeES', 'Ticker': GOLD, 'Sector': 'Commodity'}])
st.dataframe(initial_assets, use_container_width=True, hide_index=True)

# Select the starting 12 + 2 + Gold using information available at the
# first trading session on/after 01-Aug-2026. This is the actual portfolio
# held during the first six weeks — NOT all 19 research instruments.
initial_start_row = first_on_or_after(market.get(BENCHMARK), START_DATE)
initial_first_date = initial_start_row.name if initial_start_row is not None else START_DATE
initial_eq_scores, initial_debt_scores, initial_selected_eq, initial_selected_debt = select_rebalanced_universe(initial_first_date)

st.subheader('Starting selection — before the first six-week holding period')
initial_eq_display = initial_eq_scores.copy()
initial_eq_display['Selected'] = initial_eq_display['ticker'].isin(initial_selected_eq).map({True: '✅ Yes', False: '—'})
for c in ['6W Return %', '3W Return %', 'Volatility %', 'Beta', '6W Drawdown %', 'P/E', 'Peer P/E', 'P/E Discount %', 'Score']:
    if c in initial_eq_display:
        initial_eq_display[c] = pd.to_numeric(initial_eq_display[c], errors='coerce').round(2)
st.dataframe(initial_eq_display[['Selected','company','ticker','sector','6W Return %','3W Return %','Volatility %','Beta','6W Drawdown %','P/E','Peer P/E','P/E Discount %','Score']], use_container_width=True, hide_index=True)

initial_debt_display = initial_debt_scores.copy()
initial_debt_display['Selected'] = initial_debt_display['ticker'].isin(initial_selected_debt).map({True: '✅ Yes', False: '—'})
st.dataframe(initial_debt_display[['Selected','Instrument','ticker','Yield %','Risk Score','Score']].round(3), use_container_width=True, hide_index=True)

st.success('**First 6-week holding universe:** ' + ', '.join(EQUITY_META[t][0] for t in initial_selected_eq) + ' + ' + ', '.join(DEBT_META[t]['name'] for t in initial_selected_debt) + ' + Gold ETF')

# First six-week performance uses ONLY the selected 12 + 2 + Gold.
initial_end = min(RESELECT_DATE, LATEST_DATE)
initial_series, initial_weights_df = portfolio_value_path(
    initial_first_date, initial_end, initial_selected_eq, initial_selected_debt, weekly=True
)
initial_bench = benchmark_path(initial_first_date, initial_end)

if not initial_series.empty:
    c1, c2, c3 = st.columns(3)
    initial_ret = (initial_series.iloc[-1] / initial_series.iloc[0] - 1) * 100
    bench_ret = (initial_bench.iloc[-1] / initial_bench.iloc[0] - 1) * 100 if not initial_bench.empty else np.nan
    c1.metric('First 6-Week Portfolio Return', f'{initial_ret:+.2f}%')
    c2.metric('NIFTY 50 Return', f'{bench_ret:+.2f}%')
    c3.metric('Outperformance', f'{initial_ret - bench_ret:+.2f} pp' if pd.notna(bench_ret) else 'N/A')

    if not initial_weights_df.empty:
        st.subheader('Weekly weights during the FIRST 6 weeks')
        iw = initial_weights_df.copy()
        iw['Date'] = pd.to_datetime(iw['Date']).dt.strftime('%d %b %Y')
        iw = iw.rename(columns={t: EQUITY_META[t][0] for t in ALL_EQUITIES})
        iw = iw.rename(columns={t: DEBT_META[t]['name'] for t in ALL_DEBT})
        iw = iw.rename(columns={GOLD: 'Gold ETF'})
        num = [c for c in iw.columns if c != 'Date']
        iw[num] = iw[num].round(2)
        st.dataframe(iw, use_container_width=True, hide_index=True)

# ------------------------------------------------------------

# ------------------------------------------------------------
st.header('2. Reselection at the End of Week 6')
st.write('At the end of week 6, the model re-scores every company using the information available up to **11 Sep 2026**: 6-week momentum, 3-week momentum, volatility, beta, drawdown and peer-relative P/E.')

selection_date = min(RESELECT_DATE, LATEST_DATE)
eq_scores, debt_scores, selected_eq, selected_debt = select_rebalanced_universe(selection_date)

st.subheader('Equity ranking — end of week 6')
show_eq = eq_scores.copy()
show_eq['Selected'] = show_eq['ticker'].isin(selected_eq).map({True: '✅ Yes', False: '—'})
for c in ['6W Return %', '3W Return %', 'Volatility %', 'Beta', '6W Drawdown %', 'P/E', 'Peer P/E', 'P/E Discount %', 'Score']:
    if c in show_eq:
        show_eq[c] = pd.to_numeric(show_eq[c], errors='coerce').round(2)
st.dataframe(show_eq[['Selected','company','ticker','sector','6W Return %','3W Return %','Volatility %','Beta','6W Drawdown %','P/E','Peer P/E','P/E Discount %','Score']], use_container_width=True, hide_index=True)

st.subheader('Debt ranking — end of week 6')
debt_display = debt_scores.copy()
debt_display['Selected'] = debt_display['ticker'].isin(selected_debt).map({True: '✅ Yes', False: '—'})
st.dataframe(debt_display[['Selected','Instrument','ticker','Yield %','Risk Score','Score']].round(3), use_container_width=True, hide_index=True)

sel_cols = st.columns(3)
sel_cols[0].metric('Selected equities', len(selected_eq))
sel_cols[1].metric('Selected debt', len(selected_debt))
sel_cols[2].metric('Gold', 'GOLDBEES')

st.success('**Post-6-week portfolio:** ' + ', '.join(EQUITY_META[t][0] for t in selected_eq) + ' + ' + ', '.join(DEBT_META[t]['name'] for t in selected_debt) + ' + Gold ETF')

# ------------------------------------------------------------
# 3. CONTINUING PERIOD + WEEKLY REBALANCING
# ------------------------------------------------------------
st.header('3. Continuing Period — Weekly Redistribution of Weights')
continue_start = selection_date + pd.Timedelta(days=1)
continue_end = LATEST_DATE

post_series, weekly_weights_df = portfolio_value_path(continue_start, continue_end, selected_eq, selected_debt, weekly=True)
post_bench = benchmark_path(continue_start, continue_end)

if post_series.empty:
    st.warning('No post-reselection market sessions are available yet.')
else:
    post_ret = (post_series.iloc[-1] / post_series.iloc[0] - 1) * 100
    post_bench_ret = (post_bench.iloc[-1] / post_bench.iloc[0] - 1) * 100 if not post_bench.empty else np.nan
    k1, k2, k3 = st.columns(3)
    k1.metric('Post-reselection Return', f'{post_ret:+.2f}%')
    k2.metric('Post-reselection NIFTY', f'{post_bench_ret:+.2f}%' if pd.notna(post_bench_ret) else 'N/A')
    k3.metric('Post-reselection Alpha', f'{post_ret - post_bench_ret:+.2f} pp' if pd.notna(post_bench_ret) else 'N/A')

    # Weight chart
    wdf = weekly_weights_df.copy()
    if not wdf.empty:
        long = wdf.melt(id_vars='Date', var_name='Ticker', value_name='Weight %')
        long['Instrument'] = long['Ticker'].map({**{t:EQUITY_META[t][0] for t in ALL_EQUITIES}, **{t:DEBT_META[t]['name'] for t in ALL_DEBT}, GOLD:'Gold ETF'})
        figw = go.Figure()
        for name in long['Instrument'].unique():
            z = long[long['Instrument'] == name]
            figw.add_trace(go.Scatter(x=z['Date'], y=z['Weight %'], mode='lines+markers', name=name))
        figw.update_layout(title='Weekly Portfolio Weight Redistribution', xaxis_title='Rebalance Date', yaxis_title='Weight %', template='plotly_white', height=520)
        st.plotly_chart(figw, use_container_width=True)

        st.subheader('Weekly weight table')
        pretty = wdf.copy()
        pretty['Date'] = pd.to_datetime(pretty['Date']).dt.strftime('%d %b %Y')
        pretty = pretty.rename(columns={t: EQUITY_META[t][0] for t in ALL_EQUITIES})
        pretty = pretty.rename(columns={t: DEBT_META[t]['name'] for t in ALL_DEBT})
        pretty = pretty.rename(columns={GOLD:'Gold ETF'})
        numeric_cols = [c for c in pretty.columns if c != 'Date']
        pretty[numeric_cols] = pretty[numeric_cols].round(2)
        st.dataframe(pretty, use_container_width=True, hide_index=True)

        # Weight-change table
        changes = wdf.set_index('Date').diff().fillna(0).reset_index()
        changes['Date'] = pd.to_datetime(changes['Date']).dt.strftime('%d %b %Y')
        changes = changes.rename(columns={t: EQUITY_META[t][0] for t in ALL_EQUITIES})
        changes = changes.rename(columns={t: DEBT_META[t]['name'] for t in ALL_DEBT})
        changes = changes.rename(columns={GOLD:'Gold ETF'})
        st.subheader('Week-over-week weight changes (percentage points)')
        st.dataframe(changes.round(2), use_container_width=True, hide_index=True)

    # Performance chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=post_series.index, y=post_series.values, mode='lines', name='ALPHA Portfolio'))
    if not post_bench.empty:
        fig.add_trace(go.Scatter(x=post_bench.index, y=post_bench.values, mode='lines', name='NIFTY 50', line=dict(dash='dot')))
    fig.update_layout(title='Continuing Period: Portfolio vs NIFTY 50', xaxis_title='Date', yaxis_title='Portfolio Value (₹)', template='plotly_white', height=450)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# 4. FULL JOURNEY — 6 WEEKS + CONTINUING PERIOD
# ------------------------------------------------------------
st.header('4. Overall Performance — Initial 6 Weeks + Continuing Period')

# Combine initial and post-reselection paths, resetting post series to the value
# actually achieved at the end of the first phase.
if not initial_series.empty:
    phase1_end_value = float(initial_series.iloc[-1])
    if not post_series.empty:
        scale = phase1_end_value / float(post_series.iloc[0])
        full_post = post_series * scale
        full_series = pd.concat([initial_series, full_post.iloc[1:]])
    else:
        full_series = initial_series
else:
    full_series = post_series

full_bench = benchmark_path(initial_first_date, LATEST_DATE)

if not full_series.empty:
    overall_ret = (full_series.iloc[-1] / CAPITAL - 1) * 100
    overall_bench = (full_bench.iloc[-1] / CAPITAL - 1) * 100 if not full_bench.empty else np.nan
    o1, o2, o3, o4 = st.columns(4)
    o1.metric('Starting Capital', f'₹{CAPITAL:,.0f}')
    o2.metric('Current Portfolio Value', f'₹{full_series.iloc[-1]:,.0f}')
    o3.metric('Overall Return', f'{overall_ret:+.2f}%')
    o4.metric('Overall vs NIFTY', f'{overall_ret-overall_bench:+.2f} pp' if pd.notna(overall_bench) else 'N/A')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=full_series.index, y=full_series.values, mode='lines', name='ALPHA Portfolio', line=dict(width=3)))
    if not full_bench.empty:
        fig.add_trace(go.Scatter(x=full_bench.index, y=full_bench.values, mode='lines', name='NIFTY 50', line=dict(dash='dot')))
    fig.add_vline(x=selection_date, line_dash='dash', annotation_text='6-week reselection', annotation_position='top')
    fig.update_layout(title='Complete Investment Journey: Start → 6-Week Reselection → Continuing Weekly Rebalancing', template='plotly_white', height=500, yaxis_title='Portfolio Value (₹)')
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# 5. REBALANCING AUDIT / METHODOLOGY
# ------------------------------------------------------------
st.header('5. What Changed and Why?')
method = pd.DataFrame([
    ['Phase 1', '01 Aug 2026 → 11 Sep 2026', 'Selected 12 equities + 2 debt + Gold', 'Hold the starting selection and redistribute weights weekly'],
    ['Reselection', '11 Sep 2026', 'Re-select 12 equities + 2 debt + Gold from the full research universe', 'Re-rank all 15 companies and 3 debt instruments using information available at week 6'],
    ['Phase 2', 'Next trading day onward', 'Newly selected 12 equities + 2 debt + Gold', 'Recalculate weights every week using rolling factor scores'],
    ['Gold', 'Both phases', 'GOLDBEES.NS', 'Kept as the diversification sleeve'],
], columns=['Stage','Date','Universe','Rule'])
st.dataframe(method, use_container_width=True, hide_index=True)

st.caption('Model note: debt returns are simulated using stated annual yields; equity and Gold returns use Yahoo Finance market prices. This is a backtest/model demonstration, not investment advice. Weekly weights are model-generated and are not historical actual trades.')

# ------------------------- sidebar -------------------------
st.sidebar.header('ALPHA Controls')
st.sidebar.write('**Research universe:** 15 Equity + 3 Debt + Gold')
st.sidebar.write('**Held in each phase:** 12 Equity + 2 Debt + Gold')
st.sidebar.write(f'**Start:** {START_DATE:%d %b %Y}')
st.sidebar.write(f'**Review:** {RESELECT_DATE:%d %b %Y}')
st.sidebar.write(f'**Latest data:** {LATEST_DATE:%d %b %Y}')
