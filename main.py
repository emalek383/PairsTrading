import streamlit as st

st.set_page_config(layout="wide")
analyse_stocks_page = st.Page("pages/analyse_stocks.py", title = "Search for cointegrating pairs")
pairs_strategy_page = st.Page("pages/pairs_strategy.py", title = "Analyse a pairs strategy")

pg = st.navigation([analyse_stocks_page, pairs_strategy_page])
pg.run()
