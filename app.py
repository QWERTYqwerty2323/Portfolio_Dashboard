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
import numpy as np
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
