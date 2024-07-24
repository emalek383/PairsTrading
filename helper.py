import datetime as dt
from dateutil.relativedelta import relativedelta
now = dt.datetime.today()
DEFAULT_START = now + relativedelta(years = -1)
DEFAULT_END = now
DEFAULT_STOCKS = "AAPL, NFLX, GOOG, MSFT, NVDA, AMD, MRVL, TSEM, AMZN, TSM, INTC, ASML, QCOM"


def get_default_values(state):
    """
    If stocks have not yet been loaded into state, then get default values for stocks, start and end dates.
    If stocks have been loaded into state, then get default value for stocks but the already set start and end date.

    Parameters
    ----------
    state : dict (st.session_state)
        Streamlit session_state dictionary containing boolean of whether stocks have been loaded, as well as loaded stocks.

    Returns
    -------
    def_stocks : str
        String of comma-separated list of ticker for default stocks.
    start_date : datetime
        Start date.
    end_date : datetime
        End date.

    """
    
    if state.loaded_stocks:
        start_date = state.universe.start_date
        end_date = state.universe.end_date
        def_stocks = DEFAULT_STOCKS
    else:
        start_date = DEFAULT_START
        end_date = DEFAULT_END
        def_stocks = DEFAULT_STOCKS
        
    return def_stocks, start_date, end_date

def get_pair_choices(state, incl_other = True):
    """
    Create a list of possible pairs to choose for the pairs trading strategy.
    Include the current pair, if it exists and is part of the chosen universe, at the top of list.
    Include pairs saved in state. Include an "Other" option.

    Parameters
    ----------
    state : dict (st.session_state)
        Dictionary containing the current selected_pair and the stock universe.
    incl_other : bool, optional
        Whether to include an "Other" option. The default is True.

    Returns
    -------
    pair_choices : list(tuple(str))
        List of choices of pairs.

    """
    
    if state.selected_pair:
        if is_in_universe((state.selected_pair.ticker1, state.selected_pair.ticker2), state.universe):
            pair_choices = [(state.selected_pair.ticker1, state.selected_pair.ticker2)]
        else:
            pair_choices = []
    else:
        pair_choices = []
    
    for ticker1, ticker2, _ in state.pairs:
        if state.selected_pair:
            if ticker1 == state.selected_pair.ticker1 and ticker2 == state.selected_pair.ticker2:
                continue
        pair_choices.append((ticker1, ticker2))     
    
    if incl_other:
        pair_choices.append("Other")
    
    return pair_choices

def format_pairs(pair):
    """
    Format a tuple of assets to be printed out nicely.

    Parameters
    ----------
    pair : tuple(str)
        Tuple of tickers of the two assets.

    Returns
    -------
    str
        Formatted string of the two asset tickers.

    """
    
    if pair == 'Other':
        return pair
    
    return f"{pair[0]} - {pair[1]}"

def is_in_universe(pair, universe, start_date = None, end_date = None):
    """
    Check whether the current pair is in the chosen stock universe.

    Parameters
    ----------
    pair : tuple(str)
        Tuple of the tickers of the two assets of the pair.
    universe : StockUniverse
        Chosen stock universe.
    start_date : datetime, optional
        Start date of the pair for analysis. The default is None.
    end_date : datetime, optional
        End date of the pair for analysis. The default is None.

    Returns
    -------
    bool
        True if current pair is in chosen stock universe, False otherwise.

    """
    
    if pair[0] in universe.stocks and pair[1] in universe.stocks:
        if not start_date or not end_date:
            return True
        elif start_date == universe.start_date and end_date == universe.end_date:
            return True
    else:
        return False