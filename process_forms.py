""" Module to process the streamlit forms. """

import streamlit as st
from StockUniverse import StockUniverse
from PairsTrading import PairsTrading
from cointegration_analysis import find_coint_pairs
from helper import is_in_universe
from data_loader import load_default_stocks

DEFAULT_SIGNIFICANCE = 0.05
MIN_LOOKBACK_WINDOW = 5

state = st.session_state

def process_stock_form(universe = None, start_date = None, end_date = None, significance = DEFAULT_SIGNIFICANCE):
    """
    Process the stock selection form by downloading stock data and setting up the stock universe.
    Updates streamlit session_state automatically.

    Parameters
    ----------
    stocks_form : st.form
        Form that corresponds to the stock selection form..
    universe : str, optional
        Stock universe that was chosen, passed as a comma-separated string. The default is DEFAULT_STOCKS.
    start_date : datetime, optional
        Start date of stocks to be considered, passed as datetime. The default is DEFAULT_START.
    end_date : datetime, optional
        End date of stocks to be considered, passed as datetime. The default is DEFAULT_END.

    Returns
    -------
    universe : StockUniverse
        StockUniverse for the chosen stocks (or those which could be downloaded).

    """
    errors = ""
    # If stock_list, start_date or end_date are missing, use saved
    if not universe or not start_date or not end_date:
        stock_data = load_default_stocks()
        stocks = list(stock_data.columns)
        start_date, end_date = stock_data.index[0], stock_data.index[-1]
        universe = StockUniverse(stocks, start_date, end_date)
        universe.stock_data = stock_data
        
    else:
        # Extract stock list
        stocks = universe.split(",")
        cleaned_stocks = []
        for stock in stocks:
            stock = stock.strip()
            cleaned_stocks.append(stock.upper())
            
        # If stocks, start and end date are the same as loaded, just use the loaded data
        
        if (state.universe and state.universe.stocks and set(state.universe.stocks) == set(cleaned_stocks) and 
            state.universe.start_date and state.universe.start_date == start_date and
            state.universe.end_date and state.universe.end_date == end_date):
            
            # If significance is also the same, just return
            if significance == state.significance:
                return errors
            
            universe = state.universe
        
        else: # process form
            if start_date >= end_date:
                errors += "You must pick a start date before the end date."
                return errors
        
            if len(cleaned_stocks) < 2:
                errors += "Less than two assets entered. Need at least two assets to look for viable pairs."
                return errors
    
            universe = StockUniverse(cleaned_stocks, start_date, end_date)
            ignored = universe.get_data()
    
            if len(ignored) > 0:
                if len(ignored) == 1:
                    ignored_str = ignored[0]
                else:
                    ignored_str = ", ".join(ignored)
                
                errors += f"Failed to download {ignored_str}. Check the tickers. Will try to continue without them.\n"
                if len(ignored) == len(cleaned_stocks):
                    errors += "Failed to download any assets. There may be an issue with the Yahoo Finance connection."
                    return errors
            
                if len(universe.stocks) < 2:
                    errors += "Less than two assets downloaded. Need at least two assets to look for viable pairs."
                    return errors
            
            universe.calc_mean_returns_cov()
                    
    state.universe = universe
    
    state.significance = significance
    
    # Look for valid pairs in stock universe
    state.pairs, state.pvalue_df = find_coint_pairs(state.universe.stock_data, state.significance)
    
    return errors

def process_simple_pairs_selection_form(pair):
    """
    Process the simple pairs selection form by initiating a PairsTrading object and calculating its attributes.
    Set the pair as "selected_pair" in session_state.
    Directly switch pages to "pages/pairs_strategy.py" to display analysis of trading strategy.

    Parameters
    ----------
    pair : tuple(str)
        Tuple of tickers of the two assets that form the pair.

    Returns
    -------
    None.

    """
    state.selected_pair = PairsTrading(state.universe.stock_data[pair[0]],
                                       state.universe.stock_data[pair[1]])
    
    state.selected_pair.setup_lookback_window()
    state.selected_pair.calc_spread()
    state.selected_pair.calc_rolling_spread()
    state.selected_pair.calc_half_life()
    state.selected_pair.create_trading_signals()
    state.selected_pair.calc_PnL()
    st.switch_page("pages/pairs_strategy.py")
    
    return

def process_pairs_selection_form(pair, start_date, end_date, method, model, lookback_window, upper_entry, lower_entry):
    """
    Process the pairs selection form by initiating a PairsTrading object and calculating its attributes.
    If the two assets are not part of already loaded stock universe, set them as the stock universe and download data.
    Set the pair as "selected_pair" in session_state.
    Directly switch pages to "pages/pairs_strategy.py" to display analysis of trading strategy.

    Parameters
    ----------
    form : st.form
        Pairs selection form used. Needed to print out error statements.
    pair : tuple(str)
        Tuple of tickers of the two assets that form the pair.
    start_date : datetime
        Start date of stocks to be considered, passed as datetime.
    end_date : datetime
        End date of stocks to be considered, passed as datetime.
    method : str
        Estimation method for rolling hedge ratio, to be passed as attribute to PairsTrading object.
    model : str
        Spread model for the pair, to be passed as attribute to PairsTrading object.
    lookback_window : int
        Lookback window for the pair, to be passed as attribute to PairsTrading object.
    upper_entry : float
        Number of standard deviations above rolling mean spread to be used as entry point for short position.
    lower_entry : float
        Number of standard deviations below rolling mean spread to be used as entry point for long position.

    Returns
    -------
    errors : str
        Error message.

    """
    
    # Check if part of universe. If not, create new universe with just them.
    errors = ""
    
    if not is_in_universe(pair, state.universe, start_date, end_date):
        
        assets = list(pair)
        universe = StockUniverse(assets, start_date, end_date)
        ignored = universe.get_data()
        
        if len(ignored) > 0:
            if len(ignored) == 1:
                ignored_str = ignored[0]
            else:
                ignored_str = ", ".join(ignored)
                
            errors += f"Failed to download {ignored_str}. Check the tickers."
            return errors
                        
        state.universe = universe
        
    pair_data = (state.universe.stock_data[pair[0]], state.universe.stock_data[pair[1]])
            
    if method == 'lookback' and lookback_window < MIN_LOOKBACK_WINDOW:
        lookback_window = None
    
    state.selected_pair = PairsTrading(pair_data[0],
                                       pair_data[1],
                                       method = method,
                                       model = model,
                                       lookback_window = lookback_window,
                                       upper_entry = upper_entry,
                                       lower_entry = lower_entry)
    
    print(f"Method: {method}, Model: {model}, lookback_window: {lookback_window}")
    
    state.selected_pair.setup_lookback_window()
    state.selected_pair.calc_spread()
    state.selected_pair.calc_rolling_spread()
    state.selected_pair.calc_half_life()
    state.selected_pair.create_trading_signals()
    state.selected_pair.calc_PnL()
    
    return errors