import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

class PairsTrading():
    def __init__(self, asset1, asset2, model = 'linear', lookback_window = None, upper_entry = 1, lower_entry = -1):
        self.asset1 = asset1
        self.asset2 = asset2
        self.ticker1 = asset1.name
        self.ticker2 = asset2.name
        self.model = model
        if model == 'log':
            self.levels1 = np.log(asset1)
            self.levels2 = np.log(asset2)
        else:
            self.levels1 = asset1
            self.levels2 = asset2
        
        self.hedge_ratio = None
        self.lin_reg_hedge_ratio = None
        self.lin_reg_intercept = None
        self.rolling_hedge_ratio = pd.Series()
        self.spread = pd.Series()            
        self.lookback_window = lookback_window
        self.rolling_spread = pd.Series()
        self.spread_mave = pd.Series()
        self.std_rolling_spread = pd.Series()
        self.stand_rolling_spread = pd.Series()
        self.half_life = None
        self.upper_entry = upper_entry
        self.lower_entry = lower_entry
        self.signals = pd.Series()
        self.portfolio = pd.DataFrame()
        self.Sharpe = None
    
    def setup_lookback_window(self):
        if not self.lookback_window:
            if not self.half_life:
                self.calc_half_life()
                
            self.lookback_window = int(2 * self.half_life)

    def calc_hedge_ratio(self):
        '''Calculate the hedge ratio on the whole data set
        NOT the rolling hedge ratio (see calc_rolling_hedge_ratio() for that)'''
        X = sm.add_constant(self.levels1)
        lin_model = sm.OLS(self.levels2, X).fit()
        hedge_ratio = lin_model.params[self.ticker1]
        self.hedge_ratio = hedge_ratio
        self.lin_reg_hedge_ratio = hedge_ratio
        self.lin_reg_intercept = lin_model.params['const']
        
        if self.model == 'ratio':
            self.hedge_ratio = 1

        return self.hedge_ratio
    
    def calc_rolling_hedge_ratio(self, lookback_window = None):
        '''Calculate the rolling hedge ratio using lookback window'''
 
        if lookback_window:
            self.lookback_window = lookback_window
            
        if self.model == 'ratio':
            hedge_ratio = pd.Series(1, index = self.asset2.index)
        
        else:
            hedge_ratio = [np.nan] * self.lookback_window
            for date in range(self.lookback_window, len(self.asset1)):
                y = self.levels2[(date - self.lookback_window): date]
                X = self.levels1[(date - self.lookback_window): date]
                lin_model_params = sm.OLS(y, sm.add_constant(X)).fit().params
                hedge_ratio.append(lin_model_params.iloc[1])
        
            hedge_ratio = pd.Series(hedge_ratio, index = self.asset2.index)
        
        self.rolling_hedge_ratio = hedge_ratio
        
        return hedge_ratio

    def calc_spread(self):
        '''Calculate the spread on the whole data, not the rolling spread'''
        if not self.hedge_ratio:
            self.calc_hedge_ratio()

        if self.model == 'ratio':
            spread = self.levels2 / self.levels1
        else:
            spread = self.levels2 - self.hedge_ratio * self.levels1
        spread.name = 'Spread'
        self.spread = spread

        return spread

    def calc_rolling_spread(self, lookback_window = None):
        '''Calulcate the rolling spread using a lookback window'''
        
        if lookback_window and self.lookback_window != lookback_window:
            print(f'Resetting lookback window from {self.lookback_window} to {lookback_window}')
            self.lookback_window = lookback_window
            self.calc_rolling_hedge_ratio()

        if self.rolling_hedge_ratio.empty:
            self.calc_rolling_hedge_ratio()

        if self.model == 'ratio':
            rolling_spread = self.levels2 / self.levels1
        else:
            rolling_spread = self.levels2 - self.rolling_hedge_ratio * self.levels1
        rolling_spread.name = 'Rolling Spread'
        self.rolling_spread = rolling_spread
        self.spread_mave = self.rolling_spread.rolling(window = self.lookback_window).mean()
        self.spread_mave.name = 'Spread Moving Average'
        std_rolling_spread = self.rolling_spread.rolling(window = self.lookback_window).std()
        self.std_rolling_spread = std_rolling_spread
        self.std_rolling_spread.name = 'Spread Moving Average Standard Deviation'
        self.stand_rolling_spread = (self.rolling_spread - self.spread_mave) / std_rolling_spread
        self.stand_rolling_spread.name = 'Standardised Rolling Spread'

        return rolling_spread

    def calc_half_life(self):
        '''Estimate half-life of mean reversion by fitting to Ohrenstein-Uhlenbeck process'''
        if self.spread.empty:
            self.calc_spread()
            
        delta_spread = (self.spread - self.spread.shift(1)).dropna()
        lin_model = sm.OLS(delta_spread, sm.add_constant(self.spread.shift(1)).dropna()).fit()
        lambda_hat = - lin_model.params['Spread']
        self.half_life = math.log(2) / lambda_hat

        return self.half_life

    def create_trading_signals(self, lookback_window = None, upper_entry = None, lower_entry = None):
        '''Create the trading signals based on a lookback window and upper/lower entry points'''
        # ADJUST TO NEW PARAMETERS NEEDS TO BE WRITTEN IN #
        
        if lookback_window:
            self.lookback_window = lookback_window
        
        if upper_entry:
            self.upper_entry = upper_entry

        if lower_entry:
            self.lower_entry = lower_entry

        if self.rolling_spread.empty:
            self.calc_rolling_spread()

        long_entries = self.stand_rolling_spread < self.lower_entry
        long_exits = self.stand_rolling_spread >= 0
        short_entries = self.stand_rolling_spread > self.upper_entry
        short_exits = self.stand_rolling_spread <= 0

        # Create signals from long/short entry and exit
        ones = pd.Series(1, index = self.stand_rolling_spread.index)
        zeros = pd.Series(0, index = self.stand_rolling_spread.index)
        minus_ones = pd.Series(-1, index = self.stand_rolling_spread.index)

        long_signals = ones.where(long_entries).fillna(zeros.where(long_exits))
        long_signals.iloc[0] = 0
        long_signals = long_signals.ffill()
        short_signals = minus_ones.where(short_entries).fillna(zeros.where(short_exits))
        short_signals.iloc[0] = 0
        short_signals = short_signals.ffill()
        self.signals = long_signals + short_signals
        self.signals[-1] = 0
        
        # Keep track of trades
        trade_starts = self.signals.diff().ne(0) & self.signals.ne(0)
        trade_ends = self.signals.diff().dropna().ne(0) & self.signals.shift().ne(0)
        
        # Create trade summary
        trades = pd.DataFrame(index=self.signals.index[trade_ends])
        trades['trade_id'] = range(1, len(trades) + 1)
        trades['entry_date'] = self.signals.index[trade_starts][trades['trade_id'] - 1]
        trades['exit_date'] = trades.index
        trades['duration'] = trades['exit_date'] - trades['entry_date']
        
        self.trades = trades
        
        return self.signals, self.trades
    
    # Functions computing the strategy performance
    def calc_PnL(self, fixed_position_value=10000):
        if self.signals.empty:
            self.create_trading_signals()

        spread = self.rolling_spread
        hedge_ratio = self.rolling_hedge_ratio
        signals = self.signals
        trades = self.trades

        # Calculate positions for trades
        # Entering trades
        position2 = signals[trades['entry_date']]
        position1 = - position2 * hedge_ratio
        
        normalisation = abs(position1 * self.asset1) + abs(position2 * self.asset2)
        position1 = fixed_position_value * position1 / normalisation
        position2 = fixed_position_value * position2 / normalisation
            
        # Exiting trades
        position2[trades['exit_date']] = position2[trades['exit_date']].fillna(0)
        position1[trades['exit_date']] = position1[trades['exit_date']].fillna(0)

        # Fill forward and afterwards fillna to 0 to keep constant positions during active trades
        position2.ffill(inplace = True)
        position1.ffill(inplace = True)
        position2.fillna(0, inplace = True)
        position1.fillna(0, inplace = True)
        
        # Initialize portfolio DataFrame
        self.portfolio = pd.DataFrame(index=self.signals.index)
        self.portfolio['signals'] = self.signals
        self.portfolio['spread'] = spread
        self.portfolio['zscore'] = self.stand_rolling_spread
        self.portfolio['hedge_ratio'] = self.rolling_hedge_ratio
        
        self.portfolio['position_asset1'] = position1
        self.portfolio['position_asset2'] = position2

        # Calculate asset values
        self.portfolio['value_asset1'] = self.portfolio['position_asset1'] * self.asset1
        self.portfolio['value_asset2'] = self.portfolio['position_asset2'] * self.asset2
        
        # Calculate Gross Asset Value (GAV)
        self.portfolio['GAV'] = abs(self.portfolio['value_asset1']) + abs(self.portfolio['value_asset2'])
        
        # Calculate PnL
        self.portfolio['pnl_asset1'] = self.portfolio['position_asset1'].shift(1) * (self.asset1 - self.asset1.shift(1))
        self.portfolio['pnl_asset2'] = self.portfolio['position_asset2'].shift(1) * (self.asset2 - self.asset2.shift(1))
        self.portfolio['pnl'] = self.portfolio['pnl_asset1'] + self.portfolio['pnl_asset2']
        self.portfolio['cum_pnl'] = self.portfolio['pnl'].cumsum()
        
        # Calculate returns
        self.portfolio['returns'] = self.portfolio['pnl'] / self.portfolio['GAV'].shift(1)
        self.portfolio['returns'].fillna(0, inplace = True)
        self.portfolio['cum_returns'] = (1 + self.portfolio['returns']).cumprod() - 1
        
        
        # Add info to trades
        for col in ['signals', 'spread', 'zscore', 'hedge_ratio',
                    'position_asset1', 'position_asset2', 'GAV']:
            self.trades[f'entry_{col}'] = self.portfolio.loc[self.trades['entry_date'], col].values
            self.trades[f'exit_{col}'] = self.portfolio.loc[self.trades['exit_date'], col].values

        self.trades['entry_price1'] = self.asset1.loc[self.trades['entry_date']].values
        self.trades['entry_price2'] = self.asset2.loc[self.trades['entry_date']].values
        self.trades['exit_price1'] = self.asset1.loc[self.trades['exit_date']].values
        self.trades['exit_price2'] = self.asset2.loc[self.trades['exit_date']].values

        self.trades['pnl'] = self.portfolio.loc[self.trades['exit_date'], 'cum_pnl'].values - \
                               self.portfolio.loc[self.trades['entry_date'], 'cum_pnl'].values
        self.trades['return'] = self.trades['pnl'] / fixed_position_value
        
        self.Sharpe = self.calc_Sharpe(self.portfolio['returns'])
        self.APR = self.calc_APR(self.portfolio['returns'])
        cum_returns_after_first_trade = self.portfolio['cum_returns'][self.trades.iloc[0]['exit_date']:]
        self.max_drawdown, self.max_drawdown_end = self.calc_max_drawdown(cum_returns_after_first_trade)

        return self.portfolio, self.trades

    def calc_APR(self, returns):
        '''Calculate the annualised percentage return'''
        returns = returns.dropna()
        if len(returns) == 0:
            return 0.0
        return ((1 + returns).product())**(252 / len(returns)) - 1
    
    def calc_Sharpe(self, returns, annualised = False):
        '''Compute the Sharpe Ratio'''
        returns = returns.dropna()
        if len(returns) == 0:
            return 0.0
        return math.sqrt(252) * returns.mean()/returns.std()
    
    def calc_max_drawdown(self, cum_returns):
        cum_returns = cum_returns[cum_returns != 0]
        
        if len(cum_returns) == 0:
            return None
        
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        drawdown_end = drawdown.idxmin()
        
        return max_drawdown, drawdown_end
    
    
    # Cointegration tests

    def check_cointegration(self, asset_list):
        '''Check cointegration of each asset in the asset_list'''
        results = []
        for asset in asset_list:
            score, pvalue, *_ = adfuller(asset)
            results.append((asset.name, score, pvalue))
            
        return results

    def run_ADF(self):
        '''Run the ADF test on each asset in the pair and on the spread'''
        if self.spread.empty:
            self.calc_spread()

        asset_list = [self.levels1, self.levels2, self.spread]
        return self.check_cointegration(asset_list)