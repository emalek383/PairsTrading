import yfinance as yf
from contextlib import contextmanager
import sys
import io

@contextmanager
def suppress_stdout():
    new_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield
    finally:
        sys.stdout = old_stdout

def get_data(stocks, start, end):
    """
    Download stock data from yahoo finance.
    
    Args:
        stocks: list of stock tickers
        start: start date in datetime format
        end: end date in datetime format
    
    Returns:
        stockData: pd.DataFrame with close of stock data
    """
    # Disable progress bar
    yf.pdr_override()
    yf.set_tz_cache_location(None)
        
    with suppress_stdout():
        stockData = yf.download(stocks, start=start, end=end, progress=False)
        
    stockData = stockData['Close']
    return stockData
