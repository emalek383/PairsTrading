import streamlit as st
from setup_forms import setup_stock_selection_form
from process_forms import process_stock_form
from setup_displays import setup_cointegration_display

state = st.session_state    
if 'loaded_stocks' not in state:
    state.loaded_stocks = False
    
if 'universe' not in state:
    state.universe = None
     
if 'pvalue_df' not in state:
    state.pvalue_matrix = None
     
if 'pairs' not in state:
    state.pairs = None
    
if 'selected_pair' not in state:
    state.selected_pair = None
    
if 'significance' not in state:
    state.significance = None

with st.sidebar:
    st.header("Select stocks for your analysis")
    stock_selection_form = st.form(border = True, key = "stock_form")
    
cointegration_display = st.container(border = False)
cointegration_display.header("Cointegration test of pairs")
if not state.loaded_stocks:
    errors = process_stock_form()
    if errors:
        stock_selection_form.error(errors)
    state.loaded_stocks = True
    
setup_stock_selection_form(stock_selection_form)

setup_cointegration_display(cointegration_display)
