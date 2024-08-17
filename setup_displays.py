""" Module for setting up the streamlit displays. """

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from plot_functions import plot_spread, plot_assets, plot_regression, plot_all
from setup_forms import setup_simple_pairs_selection_form

state = st.session_state

def setup_cointegration_display(display):
    """
    Display the a list of cointegrating pairs from state.universe, the cointegration matrix,
    and a simple selection form for cointegrating pairs to be analysed.

    Parameters
    ----------
    display : st.container
        Container where the cointegration results will be displayed.

    Returns
    -------
    display : st.container
        Container with the cointegration results displayed.

    """
    
    stocks = ", ".join(state.universe.stocks)
    display.write(f"Analysing asset basket {stocks} from {state.universe.start_date.strftime('%d/%m/%Y')} to {state.universe.end_date.strftime('%d/%m/%Y')}.")
    
    if len(state.pairs) > 0:
        display.write(f"Found {len(state.pairs)} cointegrating pairs:")
        for ticker1, ticker2, pvalue in state.pairs:
            display.write(f"({ticker1}, {ticker2}), p-value: {pvalue:.4f}")
    
    pairs_selection_form = display.container(border = False)
    setup_simple_pairs_selection_form(pairs_selection_form)
    
    # Plotting of pairs
    cointegration_matrix_display = display.expander("Cointegration matrix")
    plt.rcParams.update({'font.size': 7})
    
    fig, ax = plt.subplots(figsize = (5, 5))
    ax = sns.heatmap(state.pvalue_df,
                     xticklabels = state.pvalue_df.columns, 
                     yticklabels = state.pvalue_df.columns,
                     cmap = 'RdYlGn_r',
                     cbar=True,
                     cbar_kws={'label': 'p-value'})
    
    plt.title("Cointegration Test p-values")
    #plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    cointegration_matrix_display.pyplot(fig)
    
    return display

def setup_pairs_display(display):
    """
    Display an analysis of the selected pairs, i.e. their spread, the assets and the OLS regression result.

    Parameters
    ----------
    display : st.container
        Container in which the results will be displayed.

    Returns
    -------
    None.

    """
    
    if not state.selected_pair:
        return
    
    display.header(f"Analysis of {state.selected_pair.ticker1} - {state.selected_pair.ticker2}")
    
    ADF_results = state.selected_pair.run_ADF()
    score, pvalue = ADF_results[2][1:]
    if state.selected_pair.model == 'linear':
        spread_str = 'Spread'
    elif state.selected_pair.model == 'ratio':
        spread_str = 'Spread of ratio'
    else:
        spread_str = 'Spread of log levels'
    
    if pvalue > state.significance:
        display.write(f"{spread_str} is not stationary: ADF p-value {pvalue:.4f}")
    else:
        display.write(f"{spread_str} is stationary: ADF p-value {pvalue:.4f}")
    
    display.write(f"Ornstein-Uhlenbeck half-life estimate: {state.selected_pair.half_life:.1f} days")
    
    # Plot the spread and the two assets
    spread_tab, both_assets_tab, regression_tab = display.tabs(["Spread", "Individual Assets", "Regression of Assets"])
        
    setup_assets_tab(both_assets_tab)
    
    setup_spread_tab(spread_tab)
    
    setup_regression_tab(regression_tab)
    
def setup_trading_display(display):
    """
    Display the analysis of the trading strategy, i.e. plot the pnl of trades, plot the strategy entry/exit points,
    and the cumulative returns. Show a summary of results on the side.

    Parameters
    ----------
    display : st.container
        Container that will house the display.

    Returns
    -------
    None.

    """
    
    if not state.selected_pair:
        display.write("Choose a pair on the left to begin analysis.")
        return
    
    display.header(f"{state.selected_pair.ticker1} - {state.selected_pair.ticker2} pairs strategy results")
    display.write(f"Parameters: spread model = {state.selected_pair.model}, lookback window = {state.selected_pair.lookback_window}")
    
    plots_col, results_col = display.columns([0.8, 0.2])
    
    # Plot the trades, strategy entry/exit points and cumulative returns
    fig = plot_all(state.selected_pair)
    plots_col.pyplot(fig)
    
    
    results_col.markdown("### Results ###")
    winning_trades = sum(state.selected_pair.trades['pnl'] >= 0)
    total_trades = state.selected_pair.trades.shape[0]
    win_rate = winning_trades / total_trades
    
    results_col.write(f"Total trades: {total_trades}")
    results_col.write(f"Winning trades: {winning_trades}")
    results_col.write(f"Win rate: {win_rate:.2%}")
    results_col.write(f"Total returns: {state.selected_pair.portfolio.cum_returns.iloc[-1]:.2%}")
    results_col.write(f"APR: {state.selected_pair.APR:.2%}")
    results_col.write(f"Sharpe Ratio: {state.selected_pair.Sharpe:.2}")
    results_col.write(f"Max Drawdown: {state.selected_pair.max_drawdown:.2%}")
    
def setup_spread_tab(tab):
    """
    Setup tab showing the spread.

    Parameters
    ----------
    tab : st.tab
        Tab that will contain the spread plot.

    Returns
    -------
    None.

    """
    
    tab.pyplot(plot_spread(state.selected_pair))

def setup_assets_tab(tab):
    """
    Setup tab showing the two assets.

    Parameters
    ----------
    tab : st.tab
        Tab that will contain the plot of the two assets.

    Returns
    -------
    None.

    """
    
    tab.pyplot(plot_assets(state.selected_pair))

def setup_regression_tab(tab):
    """
    Setup tab showing the OLS regression of the two assets.

    Parameters
    ----------
    tab : st.tab
        Tab that will show the OLS regression of the two assets.

    Returns
    -------
    None.

    """
    
    tab.pyplot(plot_regression(state.selected_pair))