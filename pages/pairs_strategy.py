import streamlit as st
from setup_forms import setup_pairs_selection_form
from setup_displays import setup_pairs_display, setup_trading_display

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
    pairs_selection_form = st.container(border = True)

pairs_display = st.container(border = False)
trading_display = st.container(border = False)    
    
setup_pairs_selection_form(pairs_selection_form)

setup_pairs_display(pairs_display)
setup_trading_display(trading_display)