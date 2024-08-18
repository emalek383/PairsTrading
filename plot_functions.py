""" Module to plot graphs for Pairs Trading. """

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_spread(pair):
    """
    Plot the spread between our two assets.

    Parameters
    ----------
    pair : PairsTrading object.
        PairsTrading object with the information on the two assets.

    Returns
    -------
    fig : matplotlib.figure
        Plot of the spread.

    """
    
    fig, ax = plt.subplots(figsize = (15, 5))
    fig.suptitle(f'Spread of {pair.ticker1} and {pair.ticker2}')
    
    ax.plot(pair.spread)
    ax.axhline(pair.spread.mean(), color = 'black')
    ax.set(ylabel = 'Spread')
    ax.set(xlabel = 'Date')
    ax.legend(['Spread', 'Mean'])
    
    return fig

def plot_assets(pair):
    """
    Plot the two assets shifted so that their means align.

    Parameters
    ----------
    pair : PairsTrading object
        PairsTrading object with the information on the two assets.

    Returns
    -------
    fig : matplotlib.figure
        Plot of the two assets.

    """
    
    fig, ax = plt.subplots(figsize = (15, 5))
    fig.suptitle(f'Comparison of {pair.ticker1} and {pair.ticker2}. Levels are shifted so means align.')
    
    # Shift data so that their averages align
    asset1 = pair.levels1 - pair.levels1.mean()
    asset2 = pair.levels2 - pair.levels2.mean()
    ax.plot(asset1)
    ax.plot(asset2)
    ax.legend([pair.ticker1, pair.ticker2], loc = 'right')
    if pair.model == 'log':
        y_label = 'Log Price Shifted'
    else:
        y_label = 'Price Shifted'
    ax.set(ylabel = y_label)
    ax.set(xlabel = 'Date')
    
    return fig
 
def plot_regression(pair):
    """
    Plot the OLS regression of the two assets against each other.

    Parameters
    ----------
    pair : PairsTrading object
        PairsTrading object with the information on the two assets.

    Returns
    -------
    fig : matplotlib.figure
        Plot of the OLS regression.

    """

    fig, ax = plt.subplots(figsize = (15, 5))
    fig.suptitle(pair.ticker2 + ' vs ' + pair.ticker1)
    
    ax.scatter(pair.levels2, pair.levels1, label = '_')
    if pair.model == 'log':
        log_pre = 'Log '
    else:
        log_pre = ''
    ax.set(ylabel = log_pre + 'Prices ' + pair.ticker2)
    ax.set(xlabel = log_pre + 'Prices ' + pair.ticker1)
    ax.plot(pair.lin_reg_intercept + pair.lin_reg_hedge_ratio * pair.levels1, pair.levels1, color = 'black', label = 'Regression Line')
    ax.legend()
    
    return fig

def plot_all(pair):
    """
    Plot the trades, strategy entry and exit points (with rolling spread, MAVE and bollinger bands) and cumulative returns.

    Parameters
    ----------
    pair : PairsTrading object
        PairsTrading object with the information on the two assets.

    Returns
    -------
    fig : matplotlib.figure
        Figure with the three plots in different grids.

    """
    
    # Create figure and grid
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 3, 2])

    # Trades subplot
    ax_trades = fig.add_subplot(gs[0])
    trades = pair.trades
    winning_trades = trades[trades['pnl'] >= 0]
    losing_trades = trades[trades['pnl'] < 0]
    
    ax_trades.scatter(winning_trades['exit_date'], winning_trades['pnl'], 
                      color='green', s=50, zorder=5, label='Winning Trade')
    ax_trades.scatter(losing_trades['exit_date'], losing_trades['pnl'], 
                      color='red', s=50, zorder=5, label='Losing Trade')
    
    ax_trades.set_ylabel('Trade PnL')
    ax_trades.legend(loc='upper left')
    ax_trades.set_title('Trades')

    # Main strategy subplot
    ax_strategy = fig.add_subplot(gs[1], sharex=ax_trades)
    ax_strategy.plot(pair.rolling_spread, label='Rolling Spread')
    
    if pair.method == 'lookback':
        mave_label = f'{pair.lookback_window}-Day Moving Average (MAVE)'
    else:
        mave_label = 'Kalman-Filter moving average (MAVE)'
        
    ax_strategy.plot(pair.spread_mave, label = mave_label, color = 'black')
    ax_strategy.plot(pair.spread_mave + pair.upper_entry * pair.std_rolling_spread, 
                     color='red', linestyle='--', label=f'MAVE + {pair.upper_entry} * Vol')
    ax_strategy.plot(pair.spread_mave + pair.lower_entry * pair.std_rolling_spread, 
                     color='red', linestyle='--', label=f'MAVE - {abs(pair.lower_entry)} * Vol')
    ax_strategy.plot(pair.signals[pair.signals == 1].index, pair.rolling_spread[pair.signals == 1], 
                     '^', markersize=10, color='blue', label='Long Position')
    ax_strategy.plot(pair.signals[pair.signals == -1].index, pair.rolling_spread[pair.signals == -1], 
                     'v', markersize=10, color='orange', label='Short Position')
    
    ax_strategy.set_ylabel("Spread")
    ax_strategy.set_xlabel("Date")
    ax_strategy.legend(loc='upper left')
    ax_strategy.set_title('Strategy')

    # Cummulative returns
    ax_cum_returns = fig.add_subplot(gs[2], sharex=ax_trades)
    ax_cum_returns.plot(pair.portfolio['cum_returns'], color = 'black')
    ax_cum_returns.set(ylabel = 'Cummulative Returns')
    ax_cum_returns.set(xlabel = 'Date')
    ax_cum_returns.set_title('Cummulative Returns')

    # Set x-axis limits
    first_valid_date = pair.spread_mave.first_valid_index()
    ax_strategy.set_xlim(left=first_valid_date)

    ax_trades.set_xlabel('')

    plt.tight_layout()
    
    return fig