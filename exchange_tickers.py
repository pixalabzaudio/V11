'''
Exchange ticker lists for stock screening application.
This file contains ticker lists for multiple exchanges: IDX, NYSE, NASDAQ, and AMEX.
'''

import pandas as pd
import os
import streamlit as st

# Import the IDX tickers from the original file
from idx_all_tickers import IDX_ALL_TICKERS_YF

# Function to load tickers from CSV files with better error handling and diagnostics
def load_tickers_from_csv(csv_path):
    """Load tickers from CSV file and return as a list."""
    try:
        # First check if file exists
        if not os.path.exists(csv_path):
            error_msg = f"CSV file not found at path: {csv_path}"
            print(error_msg)
            st.error(error_msg)
            # Try to list files in directory to help diagnose
            try:
                dir_path = os.path.dirname(csv_path)
                if os.path.exists(dir_path):
                    files = os.listdir(dir_path)
                    print(f"Files in directory {dir_path}: {files}")
                else:
                    print(f"Directory does not exist: {dir_path}")
            except Exception as dir_e:
                print(f"Error listing directory: {dir_e}")
            return []
            
        # File exists, try to read it
        print(f"Loading tickers from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Check if 'Symbol' column exists
        if 'Symbol' not in df.columns:
            columns = list(df.columns)
            error_msg = f"'Symbol' column not found in CSV. Available columns: {columns}"
            print(error_msg)
            st.error(error_msg)
            return []
            
        # Extract just the Symbol column and convert to list
        tickers = df['Symbol'].tolist()
        
        # Filter out any non-string values or empty strings
        tickers = [str(ticker) for ticker in tickers if ticker and isinstance(ticker, (str, int, float))]
        
        print(f"Successfully loaded {len(tickers)} tickers from {csv_path}")
        return tickers
        
    except Exception as e:
        error_msg = f"Error loading tickers from {csv_path}: {e}"
        print(error_msg)
        st.error(error_msg)
        return []

# Try multiple possible paths for CSV files to handle different deployment environments
def get_csv_path(filename):
    """Try multiple possible paths for CSV files and return the first one that exists."""
    possible_paths = [
        filename,  # Current directory (relative path)
        os.path.join(os.getcwd(), filename),  # Absolute path in current working directory
        f"/mount/src/v11/{filename}",  # Streamlit Cloud path (v11 repo)
        f"/mount/src/v10/{filename}",  # Streamlit Cloud path (v10 repo)
        f"/app/{filename}"  # Another possible Streamlit Cloud path
    ]
    
    # Print current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    
    # Try each path and return the first one that exists
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found CSV file at: {path}")
            return path
    
    # If no path exists, return the relative path and let the load function handle the error
    print(f"No valid path found for {filename}, using relative path")
    return filename

# Define CSV filenames
NYSE_CSV_FILENAME = 'NYSEtickers.csv'
NASDAQ_CSV_FILENAME = 'NASDAQtickers.csv'
AMEX_CSV_FILENAME = 'AMEXtickers.csv'

# Get paths to CSV files
NYSE_CSV_PATH = get_csv_path(NYSE_CSV_FILENAME)
NASDAQ_CSV_PATH = get_csv_path(NASDAQ_CSV_FILENAME)
AMEX_CSV_PATH = get_csv_path(AMEX_CSV_FILENAME)

# Load tickers from CSV files
NYSE_TICKERS = load_tickers_from_csv(NYSE_CSV_PATH)
NASDAQ_TICKERS = load_tickers_from_csv(NASDAQ_CSV_PATH)
AMEX_TICKERS = load_tickers_from_csv(AMEX_CSV_PATH)

# Create a dictionary mapping exchange names to ticker lists
EXCHANGE_TICKERS = {
    'IDX': IDX_ALL_TICKERS_YF,
    'NYSE': NYSE_TICKERS,
    'NASDAQ': NASDAQ_TICKERS,
    'AMEX': AMEX_TICKERS
}

# Function to get tickers for a specific exchange
def get_exchange_tickers(exchange_name):
    """Get the list of tickers for the specified exchange."""
    tickers = EXCHANGE_TICKERS.get(exchange_name, [])
    print(f"Returning {len(tickers)} tickers for {exchange_name}")
    return tickers

# Function to get exchange information (name and ticker count)
def get_exchange_info():
    """Get information about available exchanges."""
    info = {
        'IDX': {'name': 'Indonesia Stock Exchange', 'count': len(IDX_ALL_TICKERS_YF)},
        'NYSE': {'name': 'New York Stock Exchange', 'count': len(NYSE_TICKERS)},
        'NASDAQ': {'name': 'NASDAQ', 'count': len(NASDAQ_TICKERS)},
        'AMEX': {'name': 'American Stock Exchange', 'count': len(AMEX_TICKERS)}
    }
    print(f"Exchange info: {info}")
    return info

# Print ticker counts for debugging
if __name__ == "__main__":
    print(f"IDX Tickers: {len(IDX_ALL_TICKERS_YF)}")
    print(f"NYSE Tickers: {len(NYSE_TICKERS)}")
    print(f"NASDAQ Tickers: {len(NASDAQ_TICKERS)}")
    print(f"AMEX Tickers: {len(AMEX_TICKERS)}")
