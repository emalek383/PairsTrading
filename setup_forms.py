""" Module for setting up the UI forms in streamlit. """

import streamlit as st
import datetime as dt
from dateutil.relativedelta import relativedelta
from process_forms import process_stock_form, process_simple_pairs_selection_form, process_pairs_selection_form
from helper import get_default_values, get_pair_choices, format_pairs

DEFAULT_STOCKS = "AAPL, NFLX, GOOG, MSFT, NVDA, AMD, MRVL, TSEM, AMZN, TSM, INTC, ASML, QCOM"
now = dt.datetime.today()
DEFAULT_START = now + relativedelta(years = -1)
DEFAULT_END = now
DEFAULT_SIGNIFICANCE = 0.05
MODEL_OPTIONS = ['Linear', 'Log', 'Ratio']
MIN_LOOKBACK_WINDOW = 5
DATE_FORMAT = "DD/MM/YYYY"

state = st.session_state

def setup_stock_selection_form(form):
    """
    Setup the stock selection form.

    Parameters
    ----------
    form : st.form
        Will become the stock selection form.

    Returns
    -------
    None.

    """
    
    def_stocks, def_start_date, def_end_date = get_default_values(state)
    
    universe = form.text_input("Enter your stocks, separated by commas",
                               help = "Enter the stock tickers separated by commas",
                               value = def_stocks)

    start_date = form.date_input("Choose the start date for the analysis", 
                                 help = "Latest start date is 1 month ago",
                                 value = def_start_date,
                                 max_value = DEFAULT_START,
                                 format = DATE_FORMAT)
    end_date = form.date_input("Choose the end date for the analysis",
                               help = "Latest end date is today", 
                               value = def_end_date,
                               max_value = now,
                               format = DATE_FORMAT)
    
    significance = form.number_input("Enter the desired significance for the cointegration test",
                                help = "Desired significance for cointegration test",
                                value = DEFAULT_SIGNIFICANCE,
                                max_value = 0.25)

    submit_button = form.form_submit_button(label = "Analyse")
    
    if submit_button:
        state.loaded_stocks = True
        errors = process_stock_form(universe, start_date, end_date, significance)
        if errors:
            form.error(errors)
    
    return

def setup_simple_pairs_selection_form(form):
    """
    Setup a simple pairs selection form, including only a pair of assets.

    Parameters
    ----------
    form : st.container
        Will become the pairs selection form.

    Returns
    -------
    None.

    """
    
    if len(state.pairs) == 0:
        form.error("No cointegrating pairs were found! There are no good candidate pairs.") 
        return
    
    pair_choices = get_pair_choices(state, incl_other = False)
    
    selected_pair = form.selectbox("Choose from pairs that performed well on the cointegration analysis",
                   help = "You can only choose from pairs that performed well in cointegration analysis",
                   options = pair_choices,
                   format_func = format_pairs)
        
    submit_button = form.button(label = "Analyse pair")
    if submit_button:
        process_simple_pairs_selection_form(selected_pair)
        

def setup_pairs_selection_form(form):
    """
    Setup a pairs selection form.

    Parameters
    ----------
    form : st.container
        Will become the pairs selection form.

    Returns
    -------
    None.

    """
    
    pair_choices = get_pair_choices(state)
    
    form.header("Select pairs to analyse")
    if len(state.pairs) > 0:
        selected_pair = form.selectbox("Choose from pairs that performed well on the cointegration analysis or choose 'Other':",
                                       help = "Choose 'Other' to enter any pair.",
                                       options = pair_choices,
                                       format_func = format_pairs)
    
    if len(state.pairs) == 0 or selected_pair == 'Other':
        custom_ticker1 = form.text_input("Choose asset 1")
        custom_ticker2 = form.text_input("Choose asset 2")
        selected_pair = (custom_ticker1, custom_ticker2)
    
    model = form.selectbox("Choose your pairs model",
                           options = MODEL_OPTIONS)
    
    def_stocks, def_start_date, def_end_date = get_default_values(state)
    
    start_date = form.date_input("Choose the start date for the analysis", 
                                 help = "Latest start date is 1 month ago",
                                 value = def_start_date,
                                 max_value = DEFAULT_START,
                                 format = DATE_FORMAT)
    end_date = form.date_input("Choose the end date for the analysis",
                               help = "Latest end date is today", 
                               value = def_end_date,
                               max_value = DEFAULT_END,
                               format = DATE_FORMAT)
    
    start_end_date_difference = (end_date - start_date).days
    max_window = start_end_date_difference // 4
    
    lookback_window = form.number_input(f"""Choose the lookback window for computing the rolling spread. 
                                        Pick values less than {MIN_LOOKBACK_WINDOW} for automatic suggestion.
                                        Max possible: {max_window}.""",
                                        help = f"Pick values less than {MIN_LOOKBACK_WINDOW} to have the lookback window estimated using Ohrenstein-Uhlenbeck process",
                                        min_value = 0,
                                        max_value = max_window)
    
    upper_entry = form.number_input("Choose how many STDs above spread to enter short trade",
                                    value = 1.0)
    
    lower_entry = form.number_input("Choose how many STDs below spread to enter long trade",
                                    value = 1.0)

        
    submit_button = form.button(label = "Analyse pair")
    if submit_button:
        errors = process_pairs_selection_form(selected_pair, start_date, end_date, model.lower(), lookback_window, upper_entry, -lower_entry)
        if errors:
            form.error(errors)