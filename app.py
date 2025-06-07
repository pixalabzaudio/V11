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

# --- Helper function for Wilder's RSI ---
def calculate_rsi_wilder(prices, period=RSI_PERIOD, ticker="N/A"):
    '''Calculate RSI using Wilder's smoothing method.'''
    print(f"[{ticker}] Calculating RSI Wilder: Input prices length = {len(prices)}")
    delta = prices.diff()
    delta = delta[1:]
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Check if enough data for initial SMA
    if len(gain) < period:
        print(f"[{ticker}] RSI Wilder Error: Not enough gain/loss data (need {period}, got {len(gain)}).")
        return pd.Series(dtype=float)

    # Calculate initial average gain and loss using SMA
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

    # Initialize arrays for Wilder's averages
    wilder_avg_gain = np.array([avg_gain])
    wilder_avg_loss = np.array([avg_loss])

    # Calculate subsequent averages using Wilder's smoothing
    start_calc_index = first_valid_index + 1
    for i in range(start_calc_index, len(gain)):
        current_gain = gain.iloc[i] if not pd.isna(gain.iloc[i]) else 0
        current_loss = loss.iloc[i] if not pd.isna(loss.iloc[i]) else 0
        
        avg_gain = (wilder_avg_gain[-1] * (period - 1) + current_gain) / period
        avg_loss = (wilder_avg_loss[-1] * (period - 1) + current_loss) / period
        wilder_avg_gain = np.append(wilder_avg_gain, avg_gain)
        wilder_avg_loss = np.append(wilder_avg_loss, avg_loss)

    # Handle division by zero for avg_loss
    rs = np.divide(wilder_avg_gain, wilder_avg_loss, out=np.full_like(wilder_avg_gain, np.inf), where=wilder_avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))

    # Return the full RSI series aligned with the original price index
    rsi_index = prices.index[start_calc_index + 1 : start_calc_index + 1 + len(rsi)]
    if len(rsi) != len(rsi_index):
         print(f"[{ticker}] RSI Wilder Warning: Index alignment mismatch (RSI len {len(rsi)}, Index len {len(rsi_index)}). Returning values only.")
         rsi_series = pd.Series(rsi)
    else:
        rsi_series = pd.Series(rsi, index=rsi_index)
        
    print(f"[{ticker}] RSI Wilder Calculation successful. Output series length: {len(rsi_series)}")
    return rsi_series


# Cache technical data for 5 minutes (300 seconds)
@st.cache_data(ttl=300)
def get_rsi_yfinance(ticker):
    '''
    Calculate RSI for a given ticker using Wilder's smoothing (yfinance version with enhanced logging).
    Returns: (rsi_value, signal, rsi_history) or None if data unavailable or calculation fails
    '''
    print(f"[{ticker}] --- Starting get_rsi_yfinance --- ")
    try:
        print(f"[{ticker}] Initializing yf.Ticker...")
        stock = yf.Ticker(ticker)
        if not stock:
             print(f"[{ticker}] yf.Ticker initialization failed.")
             st.session_state.setdefault("errors", {})
             st.session_state.errors[ticker] = f"RSI Error: yf.Ticker initialization failed."
             return None
             
        # Fetch enough history for RSI calculation
        fetch_period = "6mo"
        fetch_interval = "1d"
        print(f"[{ticker}] Fetching history: period='{fetch_period}', interval='{fetch_interval}'...")
        hist = stock.history(period=fetch_period, interval=fetch_interval)
        print(f"[{ticker}] History fetched. Shape: {hist.shape}")

        if hist.empty:
            print(f"[{ticker}] RSI Error: Fetched history is empty.")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: Fetched history is empty."
            return None
            
        if "Close" not in hist.columns:
            print(f"[{ticker}] RSI Error: 'Close' column not found in fetched history.")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: 'Close' column not found in history."
            return None
            
        close_prices = hist["Close"].dropna() # Drop NaNs from close prices
        print(f"[{ticker}] Close prices extracted. Length after dropna: {len(close_prices)}")

        if len(close_prices) < RSI_PERIOD + 1:
            print(f"[{ticker}] RSI Error: Not enough valid historical data points (need {RSI_PERIOD + 1}, got {len(close_prices)}).")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: Not enough valid historical data (need {RSI_PERIOD + 1}, got {len(close_prices)})."
            return None

        # Calculate RSI using Wilder's method
        print(f"[{ticker}] Calling calculate_rsi_wilder...")
        rsi_series = calculate_rsi_wilder(close_prices, period=RSI_PERIOD, ticker=ticker)

        if rsi_series.empty or rsi_series.isna().all():
            print(f"[{ticker}] RSI Error: Wilder calculation resulted in empty or all-NaN series.")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: Wilder calculation resulted in empty or NaN series."
            return None

        # Get the latest RSI value
        try:
            latest_rsi = rsi_series.iloc[-1]
            print(f"[{ticker}] Latest RSI value extracted: {latest_rsi:.2f}")
        except IndexError:
            print(f"[{ticker}] RSI Error: Could not get latest RSI value from series (IndexError). Series length: {len(rsi_series)}")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: Could not get latest RSI value (IndexError)."
            return None

        # Check if latest RSI is valid
        if pd.isna(latest_rsi):
             print(f"[{ticker}] RSI Error: Latest RSI value is NaN.")
             st.session_state.setdefault("errors", {})
             st.session_state.errors[ticker] = f"RSI Error: Latest RSI value is NaN."
             return None

        # Determine signal based on RSI value
        if latest_rsi < OVERSOLD_THRESHOLD:
            signal = "Oversold"
        elif latest_rsi > OVERBOUGHT_THRESHOLD:
            signal = "Overbought"
        else:
            signal = "Neutral"
        print(f"[{ticker}] RSI Signal determined: {signal}")

        # Return latest RSI, signal, and the last RSI_PERIOD values for the chart
        rsi_history = rsi_series.dropna().tail(RSI_PERIOD).values
        if len(rsi_history) == 0:
             print(f"[{ticker}] RSI Error: No valid RSI history values found for chart after dropna/tail.")
             st.session_state.setdefault("errors", {})
             st.session_state.errors[ticker] = f"RSI Error: No valid RSI history values found for chart."
             return None # Cannot create chart without history
        print(f"[{ticker}] RSI History extracted for chart. Length: {len(rsi_history)}")

        print(f"[{ticker}] --- Successfully completed get_rsi_yfinance --- ")
        return (latest_rsi, signal, rsi_history)

    except Exception as e:
        error_msg = f"RSI yfinance Error: {e}\n{traceback.format_exc()}"
        print(f"[{ticker}] !!! EXCEPTION in get_rsi_yfinance: {error_msg}")
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = error_msg
        return None

# Cache fundamentals data for 24 hours (86400 seconds) - KEEP USING YFINANCE
@st.cache_data(ttl=86400)
def get_fundamentals(ticker):
    '''
    Retrieve fundamental financial data for a given ticker using yfinance.
    Returns: (net_income, prev_net_income, pe_ratio, pb_ratio, raw_data) or None if essential data unavailable/invalid
    raw_data is a dictionary containing all retrieved data for debugging
    '''
    print(f"[{ticker}] --- Starting get_fundamentals --- ")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials
        print(f"[{ticker}] yfinance info and financials fetched.")
        
        # Create a raw data dictionary for debugging
        raw_data = {
            "info_keys": list(info.keys()) if info else [],
            "financials_index": list(financials.index) if not financials.empty else [],
            "has_net_income": "Net Income" in financials.index if not financials.empty else False,
            "trailing_pe": info.get("trailingPE", "Not Available"),
            "price_to_book": info.get("priceToBook", "Not Available"),
            "market_cap": info.get("marketCap", "Not Available"),
            "currency": info.get("currency", "Unknown")
        }

        # Initialize metrics
        net_income, prev_net_income, pe_ratio, pb_ratio = None, 0, None, None # Default prev_ni to 0

        # --- Net Income ---
        if not financials.empty and "Net Income" in financials.index:
            ni_series = financials.loc["Net Income"]
            if not ni_series.empty:
                try:
                    # Handle potential MultiIndex or different structures
                    if isinstance(ni_series, pd.Series):
                        net_income = ni_series.iloc[0] / 1e12 # Current NI in Trillion IDR
                        raw_data["net_income_raw"] = float(ni_series.iloc[0]) if not pd.isna(ni_series.iloc[0]) else "NaN"
                        
                        if len(ni_series) > 1:
                            prev_net_income = ni_series.iloc[1] / 1e12 # Previous NI in Trillion IDR
                            raw_data["prev_net_income_raw"] = float(ni_series.iloc[1]) if not pd.isna(ni_series.iloc[1]) else "NaN"
                        else:
                            print(f"[{ticker}] Fund. Warning: Previous Net Income missing (Series format). Growth may be inaccurate.")
                            st.session_state.setdefault("warnings", {})
                            st.session_state.warnings[ticker] = f"Fund. Warning: Previous Net Income missing, growth calculation may be inaccurate."
                            raw_data["prev_net_income_missing"] = True
                    else: # Handle DataFrame case if structure changes
                         net_income = ni_series.iloc[0, 0] / 1e12
                         raw_data["net_income_raw"] = float(ni_series.iloc[0, 0]) if not pd.isna(ni_series.iloc[0, 0]) else "NaN"
                         
                         if ni_series.shape[1] > 1:
                             prev_net_income = ni_series.iloc[0, 1] / 1e12
                             raw_data["prev_net_income_raw"] = float(ni_series.iloc[0, 1]) if not pd.isna(ni_series.iloc[0, 1]) else "NaN"
                         else:
                             print(f"[{ticker}] Fund. Warning: Previous Net Income missing (DataFrame format). Growth may be inaccurate.")
                             st.session_state.setdefault("warnings", {})
                             st.session_state.warnings[ticker] = f"Fund. Warning: Previous Net Income missing (DataFrame format)."
                             raw_data["prev_net_income_missing"] = True
                    print(f"[{ticker}] Net Income extracted: Current={net_income:.3f}T, Previous={prev_net_income:.3f}T")

                except (IndexError, TypeError, ValueError, KeyError) as e:
                     error_msg = f"Fund. Error extracting Net Income: {e}"
                     print(f"[{ticker}] {error_msg}")
                     st.session_state.setdefault("errors", {})
                     st.session_state.errors[ticker] = error_msg
                     raw_data["net_income_error"] = str(e)
                     net_income = None # Mark as invalid if extraction failed
            else:
                print(f"[{ticker}] Fund. Warning: 'Net Income' series is empty in financials.")
                st.session_state.setdefault("warnings", {})
                st.session_state.warnings[ticker] = f"Fund. Warning: 'Net Income' series is empty in financials."
                raw_data["net_income_series_empty"] = True
        else:
            print(f"[{ticker}] Fund. Warning: 'Net Income' not found or financials empty.")
            st.session_state.setdefault("warnings", {})
            st.session_state.warnings[ticker] = f"Fund. Warning: 'Net Income' not found or empty in yfinance financials."
            raw_data["net_income_not_found"] = True
            
            # Try to use a default value for Net Income if it's missing
            # This is a fallback to allow screening to continue
            if info.get("marketCap") is not None:
                # Estimate Net Income as 5% of market cap (very rough estimate)
                estimated_ni = info.get("marketCap", 0) * 0.05 / 1e12
                print(f"[{ticker}] Using estimated Net Income based on market cap: {estimated_ni:.3f}T")
                net_income = estimated_ni
                raw_data["net_income_estimated"] = True
                raw_data["net_income_estimated_value"] = float(estimated_ni)

        # --- P/E Ratio ---
        pe_ratio = info.get("trailingPE", None)
        if pe_ratio is None or not isinstance(pe_ratio, (int, float)) or np.isnan(pe_ratio):
            warning_msg = f"Fund. Warning: Trailing P/E missing or invalid ({pe_ratio}). Will not filter by P/E."
            print(f"[{ticker}] {warning_msg}")
            st.session_state.setdefault("warnings", {})
            st.session_state.warnings[ticker] = warning_msg
            pe_ratio = None # Ensure it's None if invalid
            raw_data["pe_ratio_missing"] = True
        else:
            print(f"[{ticker}] P/E Ratio extracted: {pe_ratio:.2f}")
            raw_data["pe_ratio_value"] = float(pe_ratio)

        # --- P/B Ratio ---
        pb_ratio = info.get("priceToBook", None)
        if pb_ratio is None or not isinstance(pb_ratio, (int, float)) or np.isnan(pb_ratio):
            warning_msg = f"Fund. Warning: P/B ratio missing or invalid ({pb_ratio}). Will not filter by P/B."
            print(f"[{ticker}] {warning_msg}")
            st.session_state.setdefault("warnings", {})
            st.session_state.warnings[ticker] = warning_msg
            pb_ratio = None # Ensure it's None if invalid
            raw_data["pb_ratio_missing"] = True
        else:
             print(f"[{ticker}] P/B Ratio extracted: {pb_ratio:.2f}")
             raw_data["pb_ratio_value"] = float(pb_ratio)

        # --- Check Essential Data for Filtering ---
        if net_income is None or not isinstance(net_income, (int, float)) or np.isnan(net_income):
            error_msg = f"Fund. Error: Net Income is missing or invalid ({net_income}). Cannot apply fundamental filters."
            print(f"[{ticker}] {error_msg}")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = error_msg
            raw_data["net_income_invalid"] = True
            return None # Cannot proceed without Net Income

        # Return collected data (prev_net_income defaults to 0 if missing)
        print(f"[{ticker}] --- Successfully completed get_fundamentals --- ")
        return (net_income, prev_net_income, pe_ratio, pb_ratio, raw_data)

    except Exception as e:
        error_msg = f"Fund. yfinance Error: {e}\n{traceback.format_exc()}"
        print(f"[{ticker}] !!! EXCEPTION in get_fundamentals: {error_msg}")
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = error_msg
        return None


def process_ticker_technical_first(ticker, rsi_min, rsi_max, show_oversold, show_overbought, show_neutral, exchange):
    '''
    Process a single ticker with technical filters first (yfinance version).
    Returns: [ticker_symbol, rsi, signal, rsi_history] or None if not matching criteria
    '''
    print(f"[{ticker}] Processing technical filters...")
    try:
        # Use the yfinance-based function with enhanced logging
        rsi_data = get_rsi_yfinance(ticker)
        if not rsi_data:
            # Error/Warning logged in get_rsi_yfinance
            print(f"[{ticker}] Skipping technical processing due to RSI fetch/calc failure.")
            return None

        rsi, signal, rsi_history = rsi_data

        # Apply RSI range filter
        if (rsi_min > 0 and rsi < rsi_min) or (rsi_max < 100 and rsi > rsi_max):
            print(f"[{ticker}] Filtering out: RSI {rsi:.1f} outside range {rsi_min}-{rsi_max}.")
            st.session_state.setdefault("filtered_out_technical", {})
            st.session_state.filtered_out_technical[ticker] = f"Filtered out: RSI {rsi:.1f} outside range {rsi_min}-{rsi_max}."
            return None

        # Apply RSI signal filters - FIXED to properly handle signal filtering
        if ((signal == "Oversold" and not show_oversold) or 
            (signal == "Overbought" and not show_overbought) or 
            (signal == "Neutral" and not show_neutral)):
            print(f"[{ticker}] Filtering out: RSI signal '{signal}' not selected for display.")
            st.session_state.setdefault("filtered_out_technical", {})
            st.session_state.filtered_out_technical[ticker] = f"Filtered out: RSI signal '{signal}' not selected for display."
            return None

        # If we get here, the ticker passed all technical filters
        print(f"[{ticker}] Passed technical filters: RSI={rsi:.1f}, Signal={signal}")
        return [ticker, rsi, signal, rsi_history]

    except Exception as e:
        error_msg = f"Technical Processing Error: {e}\n{traceback.format_exc()}"
        print(f"[{ticker}] !!! EXCEPTION in process_ticker_technical_first: {error_msg}")
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = error_msg
        return None


def process_ticker_fundamental(ticker, min_net_income, max_pe, max_pb, min_growth, skip_missing=True):
    '''
    Process a single ticker with fundamental filters.
    Returns: [ticker_symbol, net_income, growth, pe, pb, raw_data] or None if not matching criteria
    If skip_missing is True, filters will be skipped if the corresponding data is missing
    '''
    print(f"[{ticker}] Processing fundamental filters...")
    try:
        # Get fundamental data
        fund_data = get_fundamentals(ticker)
        if not fund_data:
            # Error/Warning logged in get_fundamentals
            print(f"[{ticker}] Skipping fundamental processing due to data fetch/calc failure.")
            return None

        net_income, prev_net_income, pe_ratio, pb_ratio, raw_data = fund_data

        # Calculate YoY growth if previous net income is available and not zero
        if prev_net_income != 0:
            growth = ((net_income - prev_net_income) / abs(prev_net_income)) * 100
            raw_data["growth_calculated"] = float(growth)
        else:
            # If previous NI is zero, we can't calculate percentage growth
            growth = None
            print(f"[{ticker}] Fund. Warning: Cannot calculate growth (previous NI is zero or missing).")
            st.session_state.setdefault("warnings", {})
            st.session_state.warnings[ticker] = f"Fund. Warning: Cannot calculate growth (previous NI is zero or missing)."
            raw_data["growth_calculation_failed"] = True

        # Apply Net Income filter
        if net_income < min_net_income:
            print(f"[{ticker}] Filtering out: Net Income {net_income:.3f}T < minimum {min_net_income:.3f}T")
            st.session_state.setdefault("filtered_out_fundamental", {})
            st.session_state.filtered_out_fundamental[ticker] = f"Filtered out: Net Income {net_income:.3f}T < minimum {min_net_income:.3f}T"
            return None

        # Apply P/E filter if available
        if pe_ratio is not None and max_pe < 100:  # Only apply if we have valid PE and filter is active
            if pe_ratio > max_pe:
                print(f"[{ticker}] Filtering out: P/E {pe_ratio:.2f} > maximum {max_pe:.2f}")
                st.session_state.setdefault("filtered_out_fundamental", {})
                st.session_state.filtered_out_fundamental[ticker] = f"Filtered out: P/E {pe_ratio:.2f} > maximum {max_pe:.2f}"
                return None
        else:
            print(f"[{ticker}] Skipping P/E filter (P/E={pe_ratio}, max_pe={max_pe})")
            if skip_missing:
                raw_data["pe_filter_skipped"] = True

        # Apply P/B filter if available
        if pb_ratio is not None and max_pb < 20:  # Only apply if we have valid PB and filter is active
            if pb_ratio > max_pb:
                print(f"[{ticker}] Filtering out: P/B {pb_ratio:.2f} > maximum {max_pb:.2f}")
                st.session_state.setdefault("filtered_out_fundamental", {})
                st.session_state.filtered_out_fundamental[ticker] = f"Filtered out: P/B {pb_ratio:.2f} > maximum {max_pb:.2f}"
                return None
        else:
            print(f"[{ticker}] Skipping P/B filter (P/B={pb_ratio}, max_pb={max_pb})")
            if skip_missing:
                raw_data["pb_filter_skipped"] = True

        # Apply Growth filter if available
        if growth is not None and min_growth > -100:  # Only apply if we have valid growth and filter is active
            if growth < min_growth:
                print(f"[{ticker}] Filtering out: Growth {growth:.2f}% < minimum {min_growth:.2f}%")
                st.session_state.setdefault("filtered_out_fundamental", {})
                st.session_state.filtered_out_fundamental[ticker] = f"Filtering out: Growth {growth:.2f}% < minimum {min_growth:.2f}%"
                return None
        else:
            print(f"[{ticker}] Skipping Growth filter (Growth={growth}, min_growth={min_growth})")
            if skip_missing:
                raw_data["growth_filter_skipped"] = True

        # If we get here, the ticker passed all fundamental filters
        print(f"[{ticker}] Passed fundamental filters: NI={net_income:.3f}T, Growth={growth:.2f if growth is not None else None}%, P/E={pe_ratio:.2f if pe_ratio is not None else None}, P/B={pb_ratio:.2f if pb_ratio is not None else None}")
        return [ticker, net_income, growth, pe_ratio, pb_ratio, raw_data]

    except Exception as e:
        error_msg = f"Fundamental Processing Error: {e}\n{traceback.format_exc()}"
        print(f"[{ticker}] !!! EXCEPTION in process_ticker_fundamental: {error_msg}")
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = error_msg
        return None


def create_rsi_chart(ticker, rsi_history):
    '''Create a matplotlib chart for RSI history'''
    try:
        fig, ax = plt.subplots(figsize=(6, 1.5))  # Smaller chart size
        ax.plot(range(len(rsi_history)), rsi_history, color='blue', linewidth=2)
        
        # Add overbought/oversold lines
        ax.axhline(y=OVERBOUGHT_THRESHOLD, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=OVERSOLD_THRESHOLD, color='green', linestyle='--', alpha=0.5)
        
        # Set y-axis limits and remove x-axis ticks
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        
        # Add labels
        ax.set_ylabel('RSI')
        ax.set_title(f'{ticker} RSI ({RSI_PERIOD}-day)')
        
        # Fill areas
        ax.fill_between(range(len(rsi_history)), rsi_history, OVERBOUGHT_THRESHOLD, 
                        where=(rsi_history >= OVERBOUGHT_THRESHOLD), 
                        color='red', alpha=0.2)
        ax.fill_between(range(len(rsi_history)), rsi_history, OVERSOLD_THRESHOLD, 
                        where=(rsi_history <= OVERSOLD_THRESHOLD), 
                        color='green', alpha=0.2)
        
        # Convert plot to base64 image for embedding in HTML
        buf = io.BytesIO()
        fig.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f'<img src="data:image/png;base64,{img_str}" alt="RSI Chart for {ticker}">'
    except Exception as e:
        print(f"Error creating RSI chart for {ticker}: {e}")
        return f"<div style='color:red'>Chart generation failed: {e}</div>"


def get_exchange_suffix(exchange):
    """
    Get the correct suffix for tickers based on the exchange.
    """
    if exchange == 'IDX':
        return '.JK'
    elif exchange == 'NYSE':
        return ''  # NYSE tickers don't need a suffix in yfinance
    elif exchange == 'NASDAQ':
        return ''  # NASDAQ tickers don't need a suffix in yfinance
    elif exchange == 'AMEX':
        return ''  # AMEX tickers don't need a suffix in yfinance
    else:
        return ''


def main():
    # Set page config
    st.set_page_config(
        page_title="Multi-Exchange Stock Screener",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom CSS for a more professional look
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
        border-bottom: 2px solid #4c78a8;
    }
    .dataframe {
        font-size: 12px;
    }
    .css-1aumxhk {
        max-width: 1200px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state for selected exchange
    if 'selected_exchange' not in st.session_state:
        st.session_state.selected_exchange = 'IDX'

    # Get exchange info for display
    exchange_info = get_exchange_info()
    
    # Title and exchange selection in a more compact layout
    col1, col2 = st.columns([2, 3])
    with col1:
        st.title("Multi-Exchange Stock Screener")
    with col2:
        # Exchange selection dropdown
        selected_exchange = st.selectbox(
            "Select Exchange:",
            options=list(exchange_info.keys()),
            format_func=lambda x: f"{exchange_info[x]['name']} ({exchange_info[x]['count']} stocks)",
            index=list(exchange_info.keys()).index(st.session_state.selected_exchange)
        )
    
    # Update session state if exchange changed
    if selected_exchange != st.session_state.selected_exchange:
        st.session_state.selected_exchange = selected_exchange
        # Clear previous results when changing exchanges
        if 'tech_passed_tickers' in st.session_state:
            del st.session_state.tech_passed_tickers
        if 'fund_passed_tickers' in st.session_state:
            del st.session_state.fund_passed_tickers
        if 'errors' in st.session_state:
            del st.session_state.errors
        if 'warnings' in st.session_state:
            del st.session_state.warnings
        if 'filtered_out_technical' in st.session_state:
            del st.session_state.filtered_out_technical
        if 'filtered_out_fundamental' in st.session_state:
            del st.session_state.filtered_out_fundamental
    
    # Get the current exchange suffix
    exchange_suffix = get_exchange_suffix(selected_exchange)
    
    # Get tickers for the selected exchange
    all_tickers = get_exchange_tickers(selected_exchange)
    
    # Display diagnostic info about CSV files for debugging
    if selected_exchange != 'IDX' and len(all_tickers) == 0:
        st.error(f"No tickers found for {exchange_info[selected_exchange]['name']}. Checking CSV files...")
        
        # Check if CSV files exist in various locations
        if selected_exchange == 'NYSE':
            csv_name = 'NYSEtickers.csv'
        elif selected_exchange == 'NASDAQ':
            csv_name = 'NASDAQtickers.csv'
        elif selected_exchange == 'AMEX':
            csv_name = 'AMEXtickers.csv'
        else:
            csv_name = ''
            
        if csv_name:
            # Check current directory
            if os.path.exists(csv_name):
                st.info(f"Found {csv_name} in current directory")
            else:
                st.warning(f"Could not find {csv_name} in current directory")
                
            # Check repository root
            repo_path = os.path.join('/mount/src/v11', csv_name)
            if os.path.exists(repo_path):
                st.info(f"Found {csv_name} at {repo_path}")
            else:
                st.warning(f"Could not find {csv_name} at {repo_path}")
                
            # List files in repository root
            try:
                repo_dir = '/mount/src/v11'
                if os.path.exists(repo_dir):
                    files = os.listdir(repo_dir)
                    st.info(f"Files in repository root: {files}")
                else:
                    st.warning(f"Repository directory {repo_dir} does not exist")
            except Exception as e:
                st.error(f"Error listing repository directory: {e}")
    
    # Limit to MAX_TICKERS to avoid overloading
    if len(all_tickers) > MAX_TICKERS:
        st.warning(f"⚠️ Limited to first {MAX_TICKERS} tickers to avoid overloading.")
        all_tickers = all_tickers[:MAX_TICKERS]
    
    # Create tabs with a more professional layout
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Screener Results", "⚙️ Settings", "ℹ️ About & Logs", "📈 Fundamental Data"])
    
    # Sidebar for filters - more compact layout
    with st.sidebar:
        st.header("Screening Filters")
        
        # Technical Filters
        st.subheader("Technical Filters")
        
        # RSI Range Slider
        rsi_min, rsi_max = st.slider(
            "RSI Range",
            min_value=0,
            max_value=100,
            value=(0, 100),  # Default to full range
            step=1
        )
        
        # RSI Signal Checkboxes in a more compact layout
        cols = st.columns(3)
        with cols[0]:
            show_oversold = st.checkbox("Oversold", value=True)
        with cols[1]:
            show_overbought = st.checkbox("Overbought", value=True)
        with cols[2]:
            show_neutral = st.checkbox("Neutral", value=True)
        
        # Fundamental Filters
        st.subheader("Fundamental Filters")
        
        # Skip missing data option
        skip_missing = st.checkbox("Skip Missing Data", value=True, 
                                  help="If checked, stocks will not be filtered out when data is missing")
        
        # Net Income Slider (in Trillion IDR)
        min_net_income = st.slider(
            "Min Net Income (T)",
            min_value=0.0,
            max_value=5.0,  # More realistic maximum
            value=DEFAULT_MIN_NI,
            step=0.1
        )
        
        # P/E Ratio Slider
        max_pe = st.slider(
            "Max P/E Ratio",
            min_value=1.0,
            max_value=100.0,  # More realistic maximum
            value=DEFAULT_MAX_PE,
            step=1.0
        )
        
        # P/B Ratio Slider
        max_pb = st.slider(
            "Max P/B Ratio",
            min_value=0.1,
            max_value=20.0,  # More realistic maximum
            value=DEFAULT_MAX_PB,
            step=0.1
        )
        
        # YoY Growth Slider
        min_growth = st.slider(
            "Min YoY Growth (%)",
            min_value=-100.0,  # More realistic minimum
            max_value=200.0,   # More realistic maximum
            value=DEFAULT_MIN_GROWTH,
            step=5.0
        )
        
        # Run Button
        if st.button("Run Screener Now", type="primary"):
            # Clear previous results
            st.session_state.tech_passed_tickers = []
            st.session_state.fund_passed_tickers = []
            st.session_state.errors = {}
            st.session_state.warnings = {}
            st.session_state.filtered_out_technical = {}
            st.session_state.filtered_out_fundamental = {}
            st.session_state.raw_fundamental_data = {}
            
            # Show progress bar in the main area
            with tab1:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Check if we have tickers to process
                if len(all_tickers) == 0:
                    status_text.error(f"No tickers available for {exchange_info[selected_exchange]['name']}. Please check CSV files and try again.")
                    progress_bar.empty()
                    return
                
                # Process tickers in batches to avoid memory issues
                total_batches = (len(all_tickers) + BATCH_SIZE - 1) // BATCH_SIZE
                tech_passed_all = []
                
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = min(start_idx + BATCH_SIZE, len(all_tickers))
                    batch_tickers = all_tickers[start_idx:end_idx]
                    
                    status_text.text(f"Processing technical screening: {batch_idx+1}/{total_batches} batches...")
                    
                    # Process technical filters in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        # Create a list to store futures
                        futures = []
                        
                        # Submit tasks to the executor
                        for ticker in batch_tickers:
                            # Ensure ticker has correct suffix for yfinance
                            # For IDX tickers, they already have .JK suffix in the list
                            # For other exchanges, we need to ensure no double suffix
                            if selected_exchange == 'IDX':
                                # IDX tickers already have .JK suffix in the list
                                yf_ticker = ticker
                            else:
                                # For other exchanges, add suffix only if needed
                                yf_ticker = ticker
                            
                            future = executor.submit(
                                process_ticker_technical_first,
                                yf_ticker,
                                rsi_min,
                                rsi_max,
                                show_oversold,
                                show_overbought,
                                show_neutral,
                                selected_exchange
                            )
                            futures.append((ticker, future))
                        
                        print(f"Submitted {len(futures)} technical jobs to executor.")
                        
                        # Process results as they complete
                        for i, (ticker, future) in enumerate(futures):
                            result = future.result()
                            if result:
                                tech_passed_all.append(result)
                    
                    # Update progress
                    progress = (batch_idx + 1) / total_batches * 0.5  # Technical screening is 50% of total
                    progress_bar.progress(progress)
                
                # Store technical results in session state
                st.session_state.tech_passed_tickers = tech_passed_all
                
                # If no stocks passed technical screening, show message and stop
                if not tech_passed_all:
                    status_text.text("No stocks passed the technical screening. Try relaxing your filters.")
                    progress_bar.empty()
                    return
                
                # Process fundamental filters for stocks that passed technical screening
                status_text.text(f"Processing fundamental screening for {len(tech_passed_all)} stocks...")
                
                # Extract just the ticker symbols from technical results
                tech_passed_symbols = [result[0] for result in tech_passed_all]
                
                # Process fundamental filters in parallel with no timeout
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # Create a list to store futures
                    futures = []
                    future_to_ticker = {}  # Dictionary to map futures to tickers
                    
                    # Submit tasks to the executor
                    for ticker in tech_passed_symbols:
                        future = executor.submit(
                            process_ticker_fundamental,
                            ticker,
                            min_net_income,
                            max_pe,
                            max_pb,
                            min_growth,
                            skip_missing
                        )
                        futures.append(future)
                        future_to_ticker[future] = ticker  # Store mapping of future to ticker
                    
                    print(f"Submitted {len(futures)} fundamental jobs to executor.")
                    
                    # Process results as they complete (no timeout)
                    fund_passed = []
                    completed = 0
                    raw_fundamental_data = {}
                    
                    # Use as_completed to process futures as they finish
                    for future in concurrent.futures.as_completed(futures):
                        ticker = future_to_ticker[future]  # Look up the ticker for this future
                        result = future.result()
                        completed += 1
                        
                        if result:
                            fund_passed.append(result)
                            # Store raw data for debugging
                            raw_fundamental_data[ticker] = result[5]
                        
                        # Update progress for fundamental processing
                        progress = 0.5 + (0.5 * completed / len(futures))
                        progress_bar.progress(progress)
                        
                        # Update status every 10 stocks
                        if completed % 10 == 0 or completed == len(futures):
                            status_text.text(f"Processed fundamental data: {completed}/{len(futures)} stocks...")
                
                # Store fundamental results in session state
                st.session_state.fund_passed_tickers = fund_passed
                st.session_state.raw_fundamental_data = raw_fundamental_data
                
                # Final status
                if fund_passed:
                    status_text.text(f"Screening complete! {len(fund_passed)} stocks passed all filters.")
                else:
                    status_text.text(f"{len(tech_passed_all)} stocks passed technical screening, but none passed fundamental screening. Try relaxing your fundamental filters.")
                progress_bar.empty()
    
    # Settings Tab
    with tab2:
        st.header("Settings")
        
        # Debug Logs Toggle
        show_debug = st.checkbox("Show Debug Logs", value=False)
        
        # About the App
        st.header("About This App")
        st.write("""
        This multi-exchange stock screener helps you find stocks based on technical and fundamental criteria.
        
        **Technical Screening:**
        - RSI (Relative Strength Index) with customizable range
        - Signal classification (Oversold, Overbought, Neutral)
        
        **Fundamental Screening:**
        - Net Income filter
        - P/E Ratio filter
        - P/B Ratio filter
        - Year-over-Year Growth filter
        
        **Supported Exchanges:**
        - IDX (Indonesia Stock Exchange)
        - NYSE (New York Stock Exchange)
        - NASDAQ
        - AMEX (American Stock Exchange)
        
        Data is sourced from Yahoo Finance via the yfinance Python library.
        
        **Note on Fundamental Data:**
        Yahoo Finance may not have complete fundamental data for all stocks, especially for international exchanges like IDX. 
        When data is missing, you can check the "Skip Missing Data" option to prevent stocks from being filtered out based on missing metrics.
        """)
    
    # About & Logs Tab
    with tab3:
        st.header("About & Logs")
        
        # App Information
        st.subheader("App Information")
        st.write(f"""
        - Current Exchange: {exchange_info[selected_exchange]['name']}
        - Available Tickers: {exchange_info[selected_exchange]['count']}
        - Last Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """)
        
        # Debug Logs (only shown if enabled in Settings)
        if show_debug and hasattr(st.session_state, 'errors') and hasattr(st.session_state, 'warnings'):
            st.subheader("Debug Logs")
            
            # Errors
            with st.expander("Errors (Failed Operations)", expanded=False):
                if st.session_state.errors:
                    for ticker, error in st.session_state.errors.items():
                        st.write(f"**{ticker}**: {error}")
                else:
                    st.write("No errors reported.")
            
            # Warnings
            with st.expander("Warnings (Data Issues)", expanded=False):
                if st.session_state.warnings:
                    for ticker, warning in st.session_state.warnings.items():
                        st.write(f"**{ticker}**: {warning}")
                else:
                    st.write("No warnings reported.")
            
            # Filtered Out - Technical
            with st.expander("Filtered Out (Technical)", expanded=False):
                if hasattr(st.session_state, 'filtered_out_technical') and st.session_state.filtered_out_technical:
                    for ticker, reason in st.session_state.filtered_out_technical.items():
                        st.write(f"**{ticker}**: {reason}")
                else:
                    st.write("No stocks filtered out by technical criteria.")
            
            # Filtered Out - Fundamental
            with st.expander("Filtered Out (Fundamental)", expanded=False):
                if hasattr(st.session_state, 'filtered_out_fundamental') and st.session_state.filtered_out_fundamental:
                    for ticker, reason in st.session_state.filtered_out_fundamental.items():
                        st.write(f"**{ticker}**: {reason}")
                else:
                    st.write("No stocks filtered out by fundamental criteria.")
    
    # Fundamental Data Tab (for debugging)
    with tab4:
        st.header("Fundamental Data Availability")
        st.write("""
        This tab shows what fundamental data is available from Yahoo Finance for each stock that passed technical screening.
        Use this information to understand why stocks might not be passing fundamental screening.
        """)
        
        if hasattr(st.session_state, 'raw_fundamental_data') and st.session_state.raw_fundamental_data:
            # Create a table of available data
            data_rows = []
            for ticker, raw_data in st.session_state.raw_fundamental_data.items():
                row = {
                    "Ticker": ticker,
                    "Net Income": "✅" if "net_income_raw" in raw_data else "❌",
                    "Previous Net Income": "✅" if "prev_net_income_raw" in raw_data and not raw_data.get("prev_net_income_missing", False) else "❌",
                    "P/E Ratio": "✅" if "pe_ratio_value" in raw_data else "❌",
                    "P/B Ratio": "✅" if "pb_ratio_value" in raw_data else "❌",
                    "Growth Calculation": "✅" if "growth_calculated" in raw_data else "❌",
                    "Filters Skipped": ", ".join([k.replace("_filter_skipped", "") for k in raw_data.keys() if k.endswith("_filter_skipped")])
                }
                data_rows.append(row)
            
            if data_rows:
                st.dataframe(pd.DataFrame(data_rows))
                
                # Show detailed data for a selected ticker
                st.subheader("Detailed Fundamental Data")
                selected_ticker = st.selectbox("Select Ticker for Details", 
                                              options=[row["Ticker"] for row in data_rows])
                
                if selected_ticker:
                    raw_data = st.session_state.raw_fundamental_data.get(selected_ticker, {})
                    st.json(raw_data)
            else:
                st.info("No fundamental data available. Run the screener first.")
        else:
            st.info("No fundamental data available. Run the screener first.")
    
    # Results Tab - Improved layout
    with tab1:
        # Check if screening has been run
        if not hasattr(st.session_state, 'tech_passed_tickers'):
            st.info("👈 Set your filters and click 'Run Screener Now' in the sidebar to start screening.")
            return
        
        # Create two columns for results
        col1, col2 = st.columns([1, 1])
        
        # Technical Results in first column
        with col1:
            # Always display technical screening results
            if hasattr(st.session_state, 'tech_passed_tickers') and st.session_state.tech_passed_tickers:
                st.subheader("Technical Screening Results")
                tech_count = len(st.session_state.tech_passed_tickers)
                st.write(f"**{tech_count} stocks passed technical screening**")
                
                # Create a DataFrame for technical results
                tech_results = []
                for tech_result in st.session_state.tech_passed_tickers:
                    ticker, rsi, signal, rsi_history = tech_result
                    
                    # Add to results without chart for table display
                    tech_results.append({
                        "Ticker": ticker,
                        "RSI": f"{rsi:.2f}",
                        "Signal": signal
                    })
                
                # Convert to DataFrame
                if tech_results:
                    tech_df = pd.DataFrame(tech_results)
                    
                    # Display as a sortable dataframe
                    st.dataframe(tech_df, height=400)
                    
                    # Add download button for CSV
                    csv = tech_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    st.download_button(
                        label="Download Technical Results as CSV",
                        data=csv,
                        file_name="technical_results.csv",
                        mime="text/csv"
                    )
        
        # Fundamental Results in second column
        with col2:
            # Display fundamental screening results if available
            if hasattr(st.session_state, 'fund_passed_tickers') and st.session_state.fund_passed_tickers:
                st.subheader("Fundamental Screening Results")
                fund_count = len(st.session_state.fund_passed_tickers)
                st.write(f"**{fund_count} stocks passed all filters**")
                
                # Create a DataFrame for display
                fund_results = []
                for fund_result in st.session_state.fund_passed_tickers:
                    ticker, net_income, growth, pe, pb, _ = fund_result
                    
                    # Find the corresponding technical result
                    tech_result = next((t for t in st.session_state.tech_passed_tickers if t[0] == ticker), None)
                    if tech_result:
                        _, rsi, signal, _ = tech_result
                        
                        # Add to results
                        fund_results.append({
                            "Ticker": ticker,
                            "RSI": f"{rsi:.2f}",
                            "Signal": signal,
                            "Net Income (T)": f"{net_income:.3f}",
                            "Growth (%)": f"{growth:.2f}" if growth is not None else "N/A",
                            "P/E": f"{pe:.2f}" if pe is not None else "N/A",
                            "P/B": f"{pb:.2f}" if pb is not None else "N/A"
                        })
                
                # Convert to DataFrame
                if fund_results:
                    fund_df = pd.DataFrame(fund_results)
                    
                    # Display as a sortable dataframe
                    st.dataframe(fund_df, height=400)
                    
                    # Add download button for CSV
                    csv = fund_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    st.download_button(
                        label="Download Fundamental Results as CSV",
                        data=csv,
                        file_name="fundamental_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Something went wrong with the fundamental results processing.")
            elif hasattr(st.session_state, 'tech_passed_tickers') and st.session_state.tech_passed_tickers:
                st.subheader("Fundamental Screening Results")
                st.warning("⚠️ No stocks passed the fundamental screening. Try these solutions:")
                st.markdown("""
                1. **Check the 'Fundamental Data' tab** to see what data is available
                2. **Enable 'Skip Missing Data'** in the sidebar
                3. **Relax your fundamental filters**:
                   - Lower the 'Min Net Income' value
                   - Increase the 'Max P/E Ratio' value
                   - Increase the 'Max P/B Ratio' value
                   - Lower the 'Min YoY Growth' value
                """)
        
        # Display RSI charts in a more compact grid layout
        if hasattr(st.session_state, 'tech_passed_tickers') and st.session_state.tech_passed_tickers:
            st.subheader("RSI Charts")
            
            # Create a grid of charts
            chart_cols = 3  # Number of charts per row
            
            # Limit to first 12 charts to avoid excessive scrolling
            display_limit = min(12, len(st.session_state.tech_passed_tickers))
            
            # Create rows of charts
            for i in range(0, display_limit, chart_cols):
                cols = st.columns(chart_cols)
                for j in range(chart_cols):
                    if i + j < display_limit:
                        ticker, rsi, signal, rsi_history = st.session_state.tech_passed_tickers[i + j]
                        with cols[j]:
                            st.markdown(f"**{ticker}** - RSI: {rsi:.2f} ({signal})")
                            st.markdown(create_rsi_chart(ticker, rsi_history), unsafe_allow_html=True)
            
            # Show message if there are more charts
            if len(st.session_state.tech_passed_tickers) > display_limit:
                st.info(f"Showing first {display_limit} of {len(st.session_state.tech_passed_tickers)} charts. Download the CSV for complete results.")


if __name__ == "__main__":
    main()
