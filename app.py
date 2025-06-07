import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import traceback
import concurrent.futures
import threading
from functools import partial
import matplotlib.pyplot as plt
import io
import base64
import os

# Import the exchange ticker lists
from exchange_tickers import get_exchange_tickers, get_exchange_info

# Constants
MAX_TICKERS = 950
# Realistic Default Fundamental Filters
DEFAULT_MIN_NI = 0.0  # Default minimum Net Income in trillion IDR
DEFAULT_MAX_PE = 25.0  # Default maximum P/E ratio
DEFAULT_MAX_PB = 3.0  # Default maximum P/B ratio
DEFAULT_MIN_GROWTH = -20.0 # Default minimum YoY growth

RSI_PERIOD = 25  # Period for RSI calculation
OVERSOLD_THRESHOLD = 30
OVERBOUGHT_THRESHOLD = 70
MAX_WORKERS = 10
BATCH_SIZE = 50

# --- Helper function for Wilder rejuvenated --- 
def calculate_rsi_wilder(prices, period=RSI_PERIOD, ticker="N/A"):
    print(f"[{ticker}] Calculating RSI Wilder: Input prices length = {len(prices)}")
    delta = prices.diff()
    delta = delta[1:]
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    if len(gain) < period:
        print(f"[{ticker}] RSI Wilder Error: Not enough gain/loss data (need {period}, got {len(gain)}).")
        return pd.Series(dtype=float)

    try:
        avg_gain_series = gain.rolling(window=period, min_periods=period).mean()
        avg_loss_series = loss.rolling(window=period, min_periods=period).mean()
        
        first_valid_index = period - 1
        while first_valid_index < len(avg_gain_series) and pd.isna(avg_gain_series.iloc[first_valid_index]):
            first_valid_index += 1
            
        if first_valid_index >= len(avg_gain_series):
             print(f"[{ticker}] RSI Wilder Error: Not enough valid data points after rolling SMA.")
             return pd.Series(dtype=float)
             
        avg_gain = avg_gain_series.iloc[first_valid_index]
        avg_loss = avg_loss_series.iloc[first_valid_index]
        
        if pd.isna(avg_gain) or pd.isna(avg_loss):
             print(f"[{ticker}] RSI Wilder Error: Initial SMA calculation resulted in NaN (AvgGain: {avg_gain}, AvgLoss: {avg_loss}).")
             return pd.Series(dtype=float)

        print(f"[{ticker}] RSI Wilder Initial AvgGain: {avg_gain:.4f}, AvgLoss: {avg_loss:.4f}")

    except Exception as e:
        print(f"[{ticker}] RSI Wilder Error during initial SMA: {e}")
        return pd.Series(dtype=float)

    wilder_avg_gain = np.array([avg_gain])
    wilder_avg_loss = np.array([avg_loss])

    start_calc_index = first_valid_index + 1
    for i in range(start_calc_index, len(gain)):
        current_gain = gain.iloc[i] if not pd.isna(gain.iloc[i]) else 0
        current_loss = loss.iloc[i] if not pd.isna(loss.iloc[i]) else 0
        
        avg_gain = (wilder_avg_gain[-1] * (period - 1) + current_gain) / period
        avg_loss = (wilder_avg_loss[-1] * (period - 1) + current_loss) / period
        wilder_avg_gain = np.append(wilder_avg_gain, avg_gain)
        wilder_avg_loss = np.append(wilder_avg_loss, avg_loss)

    rs = np.divide(wilder_avg_gain, wilder_avg_loss, out=np.full_like(wilder_avg_gain, np.inf), where=wilder_avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))

    rsi_index = prices.index[start_calc_index + 1 : start_calc_index + 1 + len(rsi)]
    if len(rsi) != len(rsi_index):
         print(f"[{ticker}] RSI Wilder Warning: Index alignment mismatch (RSI len {len(rsi)}, Index len {len(rsi_index)}). Returning values only.")
         rsi_series = pd.Series(rsi)
    else:
        rsi_series = pd.Series(rsi, index=rsi_index)
        
    print(f"[{ticker}] RSI Wilder Calculation successful. Output series length: {len(rsi_series)}")
    return rsi_series


@st.cache_data(ttl=300)
def get_rsi_yfinance(ticker):
    print(f"[{ticker}] --- Starting get_rsi_yfinance --- ")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo", interval="1d")
        print(f"[{ticker}] History fetched. Shape: {hist.shape}")

        if hist.empty or "Close" not in hist.columns:
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: History empty or no Close column."
            return None
            
        close_prices = hist["Close"].dropna()
        if len(close_prices) < RSI_PERIOD + 1:
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: Not enough data (need {RSI_PERIOD + 1}, got {len(close_prices)})."
            return None

        rsi_series = calculate_rsi_wilder(close_prices, period=RSI_PERIOD, ticker=ticker)
        if rsi_series.empty or rsi_series.isna().all():
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: Wilder calc empty/NaN series."
            return None

        latest_rsi = rsi_series.iloc[-1]
        if pd.isna(latest_rsi):
             st.session_state.setdefault("errors", {})
             st.session_state.errors[ticker] = f"RSI Error: Latest RSI is NaN."
             return None

        signal = "Neutral"
        if latest_rsi < OVERSOLD_THRESHOLD: signal = "Oversold"
        elif latest_rsi > OVERBOUGHT_THRESHOLD: signal = "Overbought"

        rsi_history = rsi_series.dropna().tail(RSI_PERIOD).values
        if len(rsi_history) == 0:
             st.session_state.setdefault("errors", {})
             st.session_state.errors[ticker] = f"RSI Error: No valid RSI history for chart."
             return None

        print(f"[{ticker}] --- Successfully completed get_rsi_yfinance --- ")
        return (latest_rsi, signal, rsi_history)

    except Exception as e:
        error_msg = f"RSI yfinance Error: {e}\n{traceback.format_exc()}"
        print(f"[{ticker}] !!! EXCEPTION in get_rsi_yfinance: {error_msg}")
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = error_msg
        return None

@st.cache_data(ttl=86400)
def get_fundamentals(ticker):
    '''
    Retrieve fundamental financial data for a given ticker using yfinance.
    Returns: (net_income, prev_net_income, pe_ratio, pb_ratio, raw_data) or None if essential data unavailable/invalid
    '''
    print(f"[{ticker}] --- Starting get_fundamentals --- ")
    raw_data = {}
    net_income, prev_net_income, pe_ratio, pb_ratio = None, 0, None, None # prev_net_income defaults to 0

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials
        print(f"[{ticker}] yfinance info and financials fetched.")

        # Log raw info and financials for debugging
        raw_data["info_data"] = {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool, type(None)))} # Keep it serializable
        if not financials.empty:
            raw_data["financials_head"] = financials.head().to_dict()
            raw_data["financials_index"] = list(financials.index)
        else:
            raw_data["financials_head"] = "Financials DataFrame is empty"
            raw_data["financials_index"] = []

        raw_data["currency"] = info.get("currency", "Unknown")
        raw_data["marketCap"] = info.get("marketCap", "Not Available")
        
        # --- Net Income ---
        if not financials.empty and "Net Income" in financials.index:
            ni_series = financials.loc["Net Income"]
            raw_data["net_income_series_raw"] = ni_series.to_dict() # Log the raw series
            if not ni_series.empty and pd.api.types.is_numeric_dtype(ni_series):
                try:
                    # Handle potential MultiIndex or different structures
                    if isinstance(ni_series, pd.Series):
                        net_income = ni_series.iloc[0] / 1e12 # Current NI in Trillion IDR
                        raw_data["net_income_calculated_trillions"] = float(net_income) if not pd.isna(net_income) else "NaN"
                        if len(ni_series) > 1:
                            prev_net_income = ni_series.iloc[1] / 1e12
                            raw_data["prev_net_income_calculated_trillions"] = float(prev_net_income) if not pd.isna(prev_net_income) else "NaN"
                        else:
                            raw_data["prev_net_income_missing_in_series"] = True
                    else: # Handle DataFrame case if structure changes
                         net_income = ni_series.iloc[0, 0] / 1e12
                         raw_data["net_income_calculated_trillions"] = float(net_income) if not pd.isna(net_income) else "NaN"
                         if ni_series.shape[1] > 1:
                             prev_net_income = ni_series.iloc[0, 1] / 1e12
                             raw_data["prev_net_income_calculated_trillions"] = float(prev_net_income) if not pd.isna(prev_net_income) else "NaN"
                         else:
                             raw_data["prev_net_income_missing_in_series"] = True
                except (IndexError, TypeError, ValueError, KeyError) as e:
                    raw_data["net_income_extraction_error"] = str(e)
                    net_income = None # Mark as invalid if extraction failed
            else:
                raw_data["net_income_series_empty_or_not_numeric"] = True
        else:
            raw_data["net_income_not_in_financials_or_empty"] = True
        
        if net_income is None or pd.isna(net_income):
            print(f"[{ticker}] Fund. Warning: Net Income is missing or invalid ({net_income}).")
            st.session_state.setdefault("warnings", {})
            st.session_state.warnings[ticker] = f"Fund. Warning: Net Income missing/invalid ({net_income})."
            raw_data["net_income_final_value_is_None_or_NaN"] = True
            net_income = None # Ensure it is None if invalid

        # --- P/E Ratio ---
        pe_value = info.get("trailingPE")
        raw_data["trailingPE_raw"] = pe_value
        if pe_value is not None and isinstance(pe_value, (int, float)) and not np.isnan(pe_value):
            pe_ratio = float(pe_value)
            raw_data["pe_ratio_final_value"] = pe_ratio
        else:
            print(f"[{ticker}] Fund. Warning: Trailing P/E missing or invalid ({pe_value}).")
            st.session_state.setdefault("warnings", {})
            st.session_state.warnings[ticker] = f"Fund. Warning: P/E missing/invalid ({pe_value})."
            raw_data["pe_ratio_missing_or_invalid"] = True

        # --- P/B Ratio ---
        pb_value = info.get("priceToBook")
        raw_data["priceToBook_raw"] = pb_value
        if pb_value is not None and isinstance(pb_value, (int, float)) and not np.isnan(pb_value):
            pb_ratio = float(pb_value)
            raw_data["pb_ratio_final_value"] = pb_ratio
        else:
            print(f"[{ticker}] Fund. Warning: P/B ratio missing or invalid ({pb_value}).")
            st.session_state.setdefault("warnings", {})
            st.session_state.warnings[ticker] = f"Fund. Warning: P/B missing/invalid ({pb_value})."
            raw_data["pb_ratio_missing_or_invalid"] = True

        print(f"[{ticker}] --- Successfully completed get_fundamentals (NI: {net_income}, PE: {pe_ratio}, PB: {pb_ratio}) --- ")
        return (net_income, prev_net_income, pe_ratio, pb_ratio, raw_data)

    except Exception as e:
        error_msg = f"Fund. yfinance Error: {e}\n{traceback.format_exc()}"
        print(f"[{ticker}] !!! EXCEPTION in get_fundamentals: {error_msg}")
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = error_msg
        # Return None for all metrics in case of a major error, along with raw_data if populated
        raw_data["get_fundamentals_exception"] = error_msg
        return (None, 0, None, None, raw_data)


def process_ticker_technical_first(ticker, rsi_min, rsi_max, show_oversold, show_overbought, show_neutral, exchange):
    print(f"[{ticker}] Processing technical filters...")
    try:
        rsi_data = get_rsi_yfinance(ticker)
        if not rsi_data: return None
        rsi, signal, rsi_history = rsi_data

        if not ((rsi_min <= rsi <= rsi_max) and 
                ((signal == "Oversold" and show_oversold) or 
                 (signal == "Overbought" and show_overbought) or 
                 (signal == "Neutral" and show_neutral))):
            st.session_state.setdefault("filtered_out_technical", {})
            st.session_state.filtered_out_technical[ticker] = f"RSI {rsi:.1f} ({signal}) outside criteria."
            return None
        return [ticker, rsi, signal, rsi_history]
    except Exception as e:
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = f"Tech Processing Error: {e}"
        return None

def process_ticker_fundamental(ticker, min_net_income_filter, max_pe_filter, max_pb_filter, min_growth_filter, skip_missing=True):
    print(f"[{ticker}] Processing fundamental filters...")
    try:
        fund_data_tuple = get_fundamentals(ticker)
        # fund_data_tuple will always be (net_income, prev_net_income, pe_ratio, pb_ratio, raw_data)
        # Individual metrics within can be None.
        net_income, prev_net_income, pe_ratio, pb_ratio, raw_data = fund_data_tuple

        # Calculate YoY growth
        growth = None
        if net_income is not None and prev_net_income != 0 and prev_net_income is not None:
            try:
                growth = ((net_income - prev_net_income) / abs(prev_net_income)) * 100
                raw_data["growth_calculated_value"] = float(growth) if not pd.isna(growth) else "NaN"
            except TypeError: # Handle if net_income or prev_net_income became non-numeric unexpectedly
                growth = None
                raw_data["growth_calculation_type_error"] = True
        else:
            raw_data["growth_calculation_skipped_due_to_missing_ni_or_prev_ni"] = True

        # Apply Net Income filter
        if net_income is not None:
            if net_income < min_net_income_filter:
                st.session_state.setdefault("filtered_out_fundamental", {})
                st.session_state.filtered_out_fundamental[ticker] = f"Net Income {net_income:.3f}T < min {min_net_income_filter:.3f}T"
                return None
        elif not skip_missing: # net_income is None and we are NOT skipping missing data
            st.session_state.setdefault("filtered_out_fundamental", {})
            st.session_state.filtered_out_fundamental[ticker] = "Net Income missing and skip_missing is False"
            return None
        else: # net_income is None and skip_missing is True
            raw_data["ni_filter_skipped_missing_data"] = True

        # Apply P/E filter
        if pe_ratio is not None:
            if max_pe_filter < 100 and pe_ratio > max_pe_filter: # Filter active
                st.session_state.setdefault("filtered_out_fundamental", {})
                st.session_state.filtered_out_fundamental[ticker] = f"P/E {pe_ratio:.2f} > max {max_pe_filter:.2f}"
                return None
        elif not skip_missing: # pe_ratio is None and we are NOT skipping missing data
            st.session_state.setdefault("filtered_out_fundamental", {})
            st.session_state.filtered_out_fundamental[ticker] = "P/E ratio missing and skip_missing is False"
            return None
        else: # pe_ratio is None and skip_missing is True
            raw_data["pe_filter_skipped_missing_data"] = True

        # Apply P/B filter
        if pb_ratio is not None:
            if max_pb_filter < 20 and pb_ratio > max_pb_filter: # Filter active
                st.session_state.setdefault("filtered_out_fundamental", {})
                st.session_state.filtered_out_fundamental[ticker] = f"P/B {pb_ratio:.2f} > max {max_pb_filter:.2f}"
                return None
        elif not skip_missing: # pb_ratio is None and we are NOT skipping missing data
            st.session_state.setdefault("filtered_out_fundamental", {})
            st.session_state.filtered_out_fundamental[ticker] = "P/B ratio missing and skip_missing is False"
            return None
        else: # pb_ratio is None and skip_missing is True
            raw_data["pb_filter_skipped_missing_data"] = True

        # Apply Growth filter
        if growth is not None:
            if min_growth_filter > -100 and growth < min_growth_filter: # Filter active
                st.session_state.setdefault("filtered_out_fundamental", {})
                st.session_state.filtered_out_fundamental[ticker] = f"Growth {growth:.2f}% < min {min_growth_filter:.2f}%"
                return None
        elif not skip_missing: # growth is None and we are NOT skipping missing data
            st.session_state.setdefault("filtered_out_fundamental", {})
            st.session_state.filtered_out_fundamental[ticker] = "Growth missing and skip_missing is False"
            return None
        else: # growth is None and skip_missing is True
            raw_data["growth_filter_skipped_missing_data"] = True
        
        print(f"[{ticker}] Passed fundamental filters.")
        return [ticker, net_income, growth, pe_ratio, pb_ratio, raw_data]

    except Exception as e:
        error_msg = f"Fundamental Processing Error: {e}\n{traceback.format_exc()}"
        print(f"[{ticker}] !!! EXCEPTION in process_ticker_fundamental: {error_msg}")
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = error_msg
        # Ensure raw_data is returned if available, even on exception
        current_raw_data = locals().get("raw_data", {})
        current_raw_data["process_ticker_fundamental_exception"] = error_msg
        return [ticker, None, None, None, None, current_raw_data] # Return with None metrics but with raw_data


def create_rsi_chart(ticker, rsi_history):
    try:
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.plot(range(len(rsi_history)), rsi_history, color="blue", linewidth=2)
        ax.axhline(y=OVERBOUGHT_THRESHOLD, color="red", linestyle="--", alpha=0.5)
        ax.axhline(y=OVERSOLD_THRESHOLD, color="green", linestyle="--", alpha=0.5)
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        ax.set_ylabel("RSI")
        ax.set_title(f"{ticker} RSI ({RSI_PERIOD}-day)")
        ax.fill_between(range(len(rsi_history)), rsi_history, OVERBOUGHT_THRESHOLD, 
                        where=(rsi_history >= OVERBOUGHT_THRESHOLD), 
                        color="red", alpha=0.2)
        ax.fill_between(range(len(rsi_history)), rsi_history, OVERSOLD_THRESHOLD, 
                        where=(rsi_history <= OVERSOLD_THRESHOLD), 
                        color="green", alpha=0.2)
        buf = io.BytesIO()
        fig.tight_layout()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        return f'<img src="data:image/png;base64,{img_str}" alt="RSI Chart for {ticker}">'
    except Exception as e:
        return f"<div style='color:red'>Chart failed: {e}</div>"


def get_exchange_suffix(exchange):
    if exchange == 'IDX': return '.JK'
    return ''

def main():
    st.set_page_config(page_title="Multi-Exchange Stock Screener", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #e6f0ff; border-bottom: 2px solid #4c78a8; }
    .dataframe { font-size: 12px; }
    .warning-box { 
        background-color: #fff3cd; 
        border-left: 6px solid #ffc107; 
        padding: 10px; 
        margin-bottom: 15px; 
        border-radius: 4px;
    }
    .info-box { 
        background-color: #cfe2ff; 
        border-left: 6px solid #0d6efd; 
        padding: 10px; 
        margin-bottom: 15px; 
        border-radius: 4px;
    }
    .success-box { 
        background-color: #d1e7dd; 
        border-left: 6px solid #198754; 
        padding: 10px; 
        margin-bottom: 15px; 
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    if 'selected_exchange' not in st.session_state: st.session_state.selected_exchange = 'IDX'
    exchange_info = get_exchange_info()
    
    col1_title, col2_select = st.columns([2, 3])
    with col1_title: st.title("Multi-Exchange Stock Screener")
    with col2_select:
        selected_exchange = st.selectbox("Select Exchange:", options=list(exchange_info.keys()), 
                                         format_func=lambda x: f"{exchange_info[x]['name']} ({exchange_info[x]['count']} stocks)", 
                                         index=list(exchange_info.keys()).index(st.session_state.selected_exchange))
    
    if selected_exchange != st.session_state.selected_exchange:
        st.session_state.selected_exchange = selected_exchange
        for key in ['tech_passed_tickers', 'fund_passed_tickers', 'errors', 'warnings', 'filtered_out_technical', 'filtered_out_fundamental', 'raw_fundamental_data']:
            if key in st.session_state: del st.session_state[key]
    
    all_tickers = get_exchange_tickers(selected_exchange)
    if len(all_tickers) > MAX_TICKERS: all_tickers = all_tickers[:MAX_TICKERS]
    
    tab_results, tab_settings, tab_logs, tab_fund_data, tab_custom_data = st.tabs(["📊 Screener Results", "⚙️ Settings", "ℹ️ About & Logs", "📈 Fundamental Data", "📥 Custom Data"])
    
    with st.sidebar:
        st.header("Screening Filters")
        
        # Technical Filters Section
        st.subheader("Technical Filters")
        rsi_min, rsi_max = st.slider("RSI Range", 0, 100, (0, 100), 1)
        cols_tech = st.columns(3)
        show_oversold = cols_tech[0].checkbox("Oversold", value=True)
        show_overbought = cols_tech[1].checkbox("Overbought", value=True)
        show_neutral = cols_tech[2].checkbox("Neutral", value=True)
        
        # Fundamental Filters Section with Enable/Disable option
        st.subheader("Fundamental Filters")
        enable_fundamental = st.checkbox("Enable Fundamental Screening", value=True, 
                                        help="Warning: Yahoo Finance has limited fundamental data for many exchanges. Disable for technical-only screening.")
        
        # Only show fundamental filters if enabled
        if enable_fundamental:
            skip_missing = st.checkbox("Skip Missing Data", value=True, 
                                      help="If checked, stocks will not be filtered out when data is missing")
            min_net_income_filter = st.slider("Min Net Income (T)", 0.0, 5.0, DEFAULT_MIN_NI, 0.1)
            max_pe_filter = st.slider("Max P/E Ratio", 1.0, 100.0, DEFAULT_MAX_PE, 1.0)
            max_pb_filter = st.slider("Max P/B Ratio", 0.1, 20.0, DEFAULT_MAX_PB, 0.1)
            min_growth_filter = st.slider("Min YoY Growth (%)", -100.0, 200.0, DEFAULT_MIN_GROWTH, 5.0)
        else:
            # Default values when fundamental screening is disabled
            skip_missing = True
            min_net_income_filter = 0.0
            max_pe_filter = 100.0
            max_pb_filter = 20.0
            min_growth_filter = -100.0
            
            # Show info about disabled fundamental screening
            st.markdown("""
            <div class="info-box">
            <strong>Fundamental Screening Disabled</strong><br>
            Only technical filters will be applied. All stocks passing technical screening will be shown.
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("Run Screener Now", type="primary"):
            for key in ['tech_passed_tickers', 'fund_passed_tickers', 'errors', 'warnings', 'filtered_out_technical', 'filtered_out_fundamental', 'raw_fundamental_data']:
                 st.session_state[key] = [] if 'passed_tickers' in key else {}
            
            with tab_results:
                progress_bar = st.progress(0)
                status_text = st.empty()
                if not all_tickers: status_text.error("No tickers available."); progress_bar.empty(); return

                # Technical Screening Phase
                tech_passed_all = []
                total_batches = (len(all_tickers) + BATCH_SIZE - 1) // BATCH_SIZE
                for batch_idx in range(total_batches):
                    status_text.text(f"Tech screening: batch {batch_idx+1}/{total_batches}...")
                    batch_tickers_list = all_tickers[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures_tech = {executor.submit(process_ticker_technical_first, t, rsi_min, rsi_max, show_oversold, show_overbought, show_neutral, selected_exchange): t for t in batch_tickers_list}
                        for future in concurrent.futures.as_completed(futures_tech):
                            result = future.result()
                            if result: tech_passed_all.append(result)
                    progress_bar.progress((batch_idx + 1) / total_batches * 0.5)
                st.session_state.tech_passed_tickers = tech_passed_all

                if not tech_passed_all: 
                    status_text.text("No stocks passed technical screening."); 
                    progress_bar.empty(); 
                    return
                
                # If fundamental screening is disabled, skip to results
                if not enable_fundamental:
                    st.session_state.fund_passed_tickers = []  # Empty list, no fundamental screening
                    status_text.text(f"Screening complete! {len(tech_passed_all)} stocks passed technical screening. Fundamental screening disabled.")
                    progress_bar.empty()
                else:
                    # Fundamental Screening Phase (only if enabled)
                    fund_passed_all = []
                    tech_passed_symbols_list = [res[0] for res in tech_passed_all]
                    total_fund_batches = (len(tech_passed_symbols_list) + BATCH_SIZE - 1) // BATCH_SIZE
                    raw_fundamental_data_dict = {}

                    for batch_idx in range(total_fund_batches):
                        status_text.text(f"Fundamental screening: batch {batch_idx+1}/{total_fund_batches} for {len(tech_passed_symbols_list)} stocks...")
                        batch_fund_tickers = tech_passed_symbols_list[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]
                        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                            futures_fund = {executor.submit(process_ticker_fundamental, t, min_net_income_filter, max_pe_filter, max_pb_filter, min_growth_filter, skip_missing): t for t in batch_fund_tickers}
                            for i, future in enumerate(concurrent.futures.as_completed(futures_fund)):
                                ticker_processed = futures_fund[future]
                                result_fund = future.result()
                                if result_fund and len(result_fund) == 6: # Ensure result is not None and has 6 elements
                                    if result_fund[0] is not None: # Check if ticker is part of the result (passed filters)
                                         fund_passed_all.append(result_fund)
                                    raw_fundamental_data_dict[ticker_processed] = result_fund[5] # Store raw data for all processed
                                elif result_fund and len(result_fund) == 6 and result_fund[0] is None: # Ticker failed fundamental but raw data exists
                                    raw_fundamental_data_dict[ticker_processed] = result_fund[5]
                                progress_bar.progress(0.5 + ( (batch_idx * BATCH_SIZE + i + 1) / len(tech_passed_symbols_list) * 0.5) )
                    
                    st.session_state.fund_passed_tickers = fund_passed_all
                    st.session_state.raw_fundamental_data = raw_fundamental_data_dict
                    status_text.text(f"Screening complete! {len(tech_passed_all)} passed technical, {len(fund_passed_all)} passed fundamental.")
                    progress_bar.empty()

    with tab_settings: 
        st.header("Settings")
        show_debug = st.checkbox("Show Debug Logs", value=True)
        
        st.subheader("Data Source Settings")
        st.markdown("""
        <div class="info-box">
        <strong>About Yahoo Finance Data</strong><br>
        Yahoo Finance provides reliable technical data (price history, RSI) for most exchanges, but fundamental data coverage varies:
        <ul>
        <li>IDX (Indonesia): Good fundamental data for many stocks</li>
        <li>NYSE, NASDAQ, AMEX: Inconsistent fundamental data through API</li>
        </ul>
        For best results with non-IDX exchanges, use technical screening only or upload custom fundamental data.
        </div>
        """, unsafe_allow_html=True)
        
    with tab_logs:
        st.header("About & Logs")
        
        st.markdown("""
        <div class="info-box">
        <strong>Multi-Exchange Stock Screener</strong><br>
        This application allows you to screen stocks from multiple exchanges using technical and fundamental criteria.
        <ul>
        <li><strong>Technical Screening:</strong> Based on RSI (Relative Strength Index) values and signals (Oversold < {OVERSOLD_THRESHOLD}, Overbought > {OVERBOUGHT_THRESHOLD}). Uses Wilder's Smoothing.</li>
        <li><strong>Fundamental Screening:</strong> Based on Net Income, P/E Ratio, P/B Ratio, and YoY Growth to the stocks that passed the technical screen.</li>
        <li><strong>Data:</strong> Technical data cached for 5 mins, Fundamental data for 24 hours.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if show_debug:
            for log_name, log_dict_key in [("Errors", "errors"), ("Warnings", "warnings"), ("Filtered (Technical)", "filtered_out_technical"), ("Filtered (Fundamental)", "filtered_out_fundamental")]:
                with st.expander(log_name, expanded=False):
                    log_data = st.session_state.get(log_dict_key, {})
                    if log_data: [st.write(f"**{k}**: {v}") for k, v in log_data.items()]
                    else: st.write("None")
                    
    with tab_fund_data:
        st.header("Fundamental Data Availability")
        
        st.markdown("""
        <div class="warning-box">
        <strong>Fundamental Data Coverage</strong><br>
        Yahoo Finance provides good fundamental data for IDX (Indonesian) stocks, but coverage for other exchanges varies.
        This tab shows which fundamental metrics are available for each stock that passed technical screening.
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.get('raw_fundamental_data'):
            fund_data_rows = []
            for ticker, data in st.session_state.raw_fundamental_data.items():
                fund_data_rows.append({"Ticker": ticker, "NI Avail": "✅" if data.get("net_income_calculated_trillions") not in [None, "NaN"] else "❌", 
                                     "P/E Avail": "✅" if data.get("pe_ratio_final_value") is not None else "❌", 
                                     "P/B Avail": "✅" if data.get("pb_ratio_final_value") is not None else "❌",
                                     "Growth Calc": "✅" if data.get("growth_calculated_value") not in [None, "NaN"] else "❌" })
            if fund_data_rows: 
                st.dataframe(pd.DataFrame(fund_data_rows))
                
                # Summary statistics
                total_stocks = len(fund_data_rows)
                ni_available = sum(1 for row in fund_data_rows if row["NI Avail"] == "✅")
                pe_available = sum(1 for row in fund_data_rows if row["P/E Avail"] == "✅")
                pb_available = sum(1 for row in fund_data_rows if row["P/B Avail"] == "✅")
                growth_available = sum(1 for row in fund_data_rows if row["Growth Calc"] == "✅")
                
                st.markdown(f"""
                <div class="info-box">
                <strong>Data Availability Summary</strong><br>
                Out of {total_stocks} stocks that passed technical screening:
                <ul>
                <li>Net Income available: {ni_available} ({ni_available/total_stocks*100:.1f}%)</li>
                <li>P/E Ratio available: {pe_available} ({pe_available/total_stocks*100:.1f}%)</li>
                <li>P/B Ratio available: {pb_available} ({pb_available/total_stocks*100:.1f}%)</li>
                <li>Growth calculable: {growth_available} ({growth_available/total_stocks*100:.1f}%)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                selected_ticker_details = st.selectbox("Select Ticker for Raw Details", options=list(st.session_state.raw_fundamental_data.keys()))
                if selected_ticker_details: st.json(st.session_state.raw_fundamental_data[selected_ticker_details])
        else: st.info("Run screener to see data.")
        
    with tab_custom_data:
        st.header("Custom Fundamental Data")
        
        st.markdown("""
        <div class="info-box">
        <strong>Upload Your Own Fundamental Data</strong><br>
        Since Yahoo Finance has limited fundamental data for many exchanges, you can upload your own data here.
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Upload CSV File")
        st.markdown("""
        Prepare a CSV file with the following columns:
        - **Ticker**: Stock symbol (required)
        - **NetIncome**: Net Income value (optional)
        - **PE**: Price to Earnings ratio (optional)
        - **PB**: Price to Book ratio (optional)
        - **Growth**: Year-over-Year growth percentage (optional)
        
        Example:
        ```
        Ticker,NetIncome,PE,PB,Growth
        AAPL,100.5,15.2,6.7,12.3
        MSFT,85.3,28.1,10.2,8.5
        ```
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                custom_data = pd.read_csv(uploaded_file)
                if 'Ticker' not in custom_data.columns:
                    st.error("CSV file must contain a 'Ticker' column")
                else:
                    st.success(f"Successfully loaded data for {len(custom_data)} stocks")
                    st.dataframe(custom_data.head(10))
                    
                    # Store in session state for future use
                    st.session_state['custom_fundamental_data'] = custom_data
                    
                    # Option to use this data
                    if st.button("Use Custom Data for Screening"):
                        st.session_state['use_custom_data'] = True
                        st.info("Custom data will be used for the next screening run. Go back to the Screener Results tab and click 'Run Screener Now'.")
            except Exception as e:
                st.error(f"Error loading CSV file: {e}")

    with tab_results:
        if not st.session_state.get('tech_passed_tickers'): 
            st.info("👈 Set your filters and click 'Run Screener Now' to start screening.")
            return
            
        # Display warning about fundamental data if appropriate
        if enable_fundamental and not st.session_state.get('fund_passed_tickers') and 'tech_passed_tickers' in st.session_state:
            st.markdown("""
            <div class="warning-box">
            <strong>No Stocks Passed Fundamental Screening</strong><br>
            This is likely due to limited fundamental data availability from Yahoo Finance. Consider:
            <ul>
            <li>Disabling fundamental screening to see all stocks that pass technical criteria</li>
            <li>Uploading custom fundamental data in the "Custom Data" tab</li>
            <li>Using "Skip Missing Data" option with very relaxed fundamental filters</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Technical Results Section
        st.subheader("Technical Screening Results")
        tech_df = pd.DataFrame([t[:3] for t in st.session_state.tech_passed_tickers], columns=["Ticker", "RSI", "Signal"])
        if not tech_df.empty: 
            st.dataframe(tech_df, height=400)
            st.download_button("Download Technical Results CSV", tech_df.to_csv(index=False), "technical_results.csv", "text/csv")
            
            # RSI Charts Section
            st.subheader("RSI Charts (Max 12)")
            chart_cols_grid = 3
            display_limit_charts = min(12, len(st.session_state.tech_passed_tickers))
            for i in range(0, display_limit_charts, chart_cols_grid):
                cols_charts = st.columns(chart_cols_grid)
                for j in range(chart_cols_grid):
                    if i + j < display_limit_charts:
                        ticker_chart, rsi_chart, signal_chart, rsi_hist_chart = st.session_state.tech_passed_tickers[i+j]
                        with cols_charts[j]:
                            st.markdown(f"**{ticker_chart}** - RSI: {rsi_chart:.2f} ({signal_chart})")
                            st.markdown(create_rsi_chart(ticker_chart, rsi_hist_chart), unsafe_allow_html=True)
        
        # Fundamental Results Section (only if enabled and stocks passed)
        if enable_fundamental and st.session_state.get('fund_passed_tickers'):
            st.subheader("Fundamental Screening Results")
            fund_df_data = []
            for res_fund in st.session_state.fund_passed_tickers:
                ticker, ni, growth, pe, pb, _ = res_fund
                tech_match = next((t for t in st.session_state.tech_passed_tickers if t[0] == ticker), [None]*3)
                fund_df_data.append({"Ticker": ticker, "RSI": f"{tech_match[1]:.2f}" if tech_match[1] is not None else "N/A", "Signal": tech_match[2],
                                     "Net Income (T)": f"{ni:.3f}" if ni is not None else "N/A", 
                                     "Growth (%)": f"{growth:.2f}" if growth is not None else "N/A", 
                                     "P/E": f"{pe:.2f}" if pe is not None else "N/A", 
                                     "P/B": f"{pb:.2f}" if pb is not None else "N/A"})
            if fund_df_data:
                fund_df = pd.DataFrame(fund_df_data)
                st.dataframe(fund_df, height=400)
                st.download_button("Download Fundamental Results CSV", fund_df.to_csv(index=False), "fundamental_results.csv", "text/csv")
            else: 
                st.warning("No stocks passed fundamental screening. Try disabling fundamental screening or using custom data.")

if __name__ == "__main__":
    main()
