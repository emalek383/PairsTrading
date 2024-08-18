"""Module creating the PairsTrading class for pairs trading."""

import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from helper import calc_APR, calc_Sharpe, calc_max_drawdown, check_stationarity
from pykalman import KalmanFilter

class PairsTrading():
    """
    Class to capture a Pairs Trading Strategy. Contains info on the two assets forming the pair, their hedge ratio and spread (both overall and rolling
    inside a lookbackack window). Also contains the trading strategy and PnL.
    
    Attributes
    ----------
    asset1 : pd.Series
        Price data of the first asset.
    asset2 : pd.Series
        Price data of the second asset.
    model : str
        What form of spread ('linear', 'log', 'ratio') will be used for the trading signal.
    lookback_window : int
        Number of days in the lookback window for calculating rolling quantities (rolling hedge ratio, spread, ...)
    upper_entry : float
        Upper entry point for the trading strategy, expressed as how many std's away from rolling average spread.
    lower_entry : float
        Lower entry point for the trading strategy, expressed as how many std's away from rolling average spread.
    levels1 : pd.Series
        Price or log price (if model == 'log') data of the first asset used for calculating the spread.
    levels2 : pd.Series
        Price or log price (if model == 'log') data of the first asset used for calculating the spread.
    hedge_ratio : float
        The hedge ratio (regression coefficient) of levels2 vs levels1 used for calculating the spread. Is set to 1 for ratio method.
    lin_reg_hedge_ratio : float
        Hedge ratio as computed from the OLS of levels2 vs levels1 (since for ratio method hedge_ratio = 1)
    lin_reg_intercept : float
        Intercept of OLS of levels2 vs levels1.
    method : 
        Estimation method for the rolling hedge ratio ('lookback' or 'kalman').
    rolling_hedge_ratio : pd.Series
        The rolling hedge ratio (regression coefficient) of levels2 vs levels1 calculating only using data in lookback window.
    rolling_spread : pd.Series
        The rolling spread between asset2 and asset1, calculated only using data in lookback window.
    spread_mave : pd.Series
        Moving average of the rolling spread in the lookback window.
    std_rolling_spread : pd.Series
        Standard deviation of the rolling spread in the lookback window.
    stand_rolling_spread : pd.Series
        Standardised rolling spread in the lookback window, i.e. mean = 0 and std = 1.
    half_life : float
        Half-life of the spread over whole time period calculated by fitting an Ornstein-Uhlenbeck process.
    self.kf : pykalman.KalmanFilter
        KalmanFilter object to run KalmanFilter analysis.
    self.kalman_hedge_ratio :
        Rolling hedge ratio computed using a Kalman Filter.
    self.kalman_intercept :
        Intercept of the regression equation predicted by a Kalman Filter.
    signals : pd.Series
        Long/short trading signals.
    portfolio : pd.DataFrame
        DataFrame containig the trading positions held in the two assets, PnL, etc.
    trades : pd.DataFrame
        DataFrame containing all the info (position, asset prices, spread, etc.) at the entry and exit times of the trades as well as the PnL.
    Sharpe : float
        Sharpe Ratio of the pairs trading strategy.
    APR : float
        Annualised percentage return of the pairs trading strategy.
    max_drawdown : float
        The maximum drawdown of the pair strading strategy.
        
    Methods
    -------
    setup_lookback_window():
        Setup the lookback window by calculating the half-life from the Ornstein-Uhlenbeck process.
    calc_hedge_ratio():
        Calculate the hedge ratio from the OLS regression of levels2 vs levels1.
    calc_lookback_rolling_hedge_ratio():
        Calculate the rolling hedge ratio for the spread from the OLS regression of levels2 vs levels1 inside the lookback window.
    calc_spread():
        Calculate the spread.
    calc_lookback_rolling_spread():
        Calculate the rolling spread in the lookback window, as well as its moving average, standard deviation and standardised form.
    calc_rolling_spread():
        Calculate the rolling spread either using a lookback window or a Kalman filter, including its moving average, standard deviation and standardised form.
    setup_kalman_filter():
        Setup a Kalman Filter to compute the rolling hedge ratio.
    calc_kalman_hedge_ratio():
        Calculate the rolling hedge ratio using a Kalman Filter.
    calc_kalman_spread():
        Calculate the spread using a Kalman Filter, as well as the intercept predicted by the Kalman Filter (proxy for the moving average), standard deviation and standardised form..
    calc_half_life():
        Calculate the half-life by fitting the spread to an Ornstein-Uhlenbeck process.
    create_trading_signals():
        Create the trading signals for the pairs trading strategy.
    calc_PnL():
        Calculate the strategy's performance: PnL, Sharpe Ratio, APR and max drawdown.
    
    """
    
    def __init__(self, asset1, asset2, model = 'linear', method = 'lookback', lookback_window = None, upper_entry = 1, lower_entry = -1):
        """
        Construct the attributes of the PairsTrading object.

        Parameters
        ----------
        asset1 : pd.Series
            Prices of asset1 over the time period we're interested in.
        asset2 : pd.Series
            Prices of asset2 over the time period we're interested in
        model : str, optional
            What form of spread ('linear', 'log', 'ratio') will be used for the trading signal. The default is 'linear'.
        method : str, optional
            Estimation method for the rolling hedge ratio ('lookback' or 'kalman'). The default is 'lookback'.
        lookback_window : int, optional
            Number of days in the lookback window for calculating rolling quantities (rolling hedge ratio, spread, ...). The default is None.
        upper_entry : float, optional
            Upper entry point for the trading strategy, expressed as how many std's away from rolling average spread. The default is 1.
        lower_entry : float, optional
            Lower entry point for the trading strategy, expressed as how many std's away from rolling average spread. The default is -1.

        Returns
        -------
        None.

        """
        # General analysis of pairs attributes
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
        
        self.method = method
        # Rolling attributes
        self.rolling_hedge_ratio = pd.Series()
        self.spread = pd.Series()            
        self.lookback_window = lookback_window
        self.rolling_spread = pd.Series()
        self.spread_mave = pd.Series()
        self.std_rolling_spread = pd.Series()
        self.stand_rolling_spread = pd.Series()
        self.half_life = None
        # Kalman attributes
        self.kf = None
        self.kalman_hedge_ratio = None
        self.kalman_intercept = None
        # Trades and signals attributes
        self.upper_entry = upper_entry
        self.lower_entry = lower_entry
        self.signals = pd.Series()
        self.portfolio = pd.DataFrame()
        self.trades = pd.DataFrame()
        self.Sharpe = None
        self.APR = None
        self.max_drawdown = None
    
    def setup_kalman_filter(self, delta = 1e-5, obs_cov = 1e-3):
        """
        Setup the kalman filter for estimating the hedge ratio and spread.

        Returns
        -------
        None.

        """
        
        # Setup initial transition covariance and initial state mean and cov as 0 & 1
        obs_cov = obs_cov
        trans_cov = delta / (1 - delta) * np.eye(2)
        obs_mat = sm.add_constant(self.levels1.values, prepend=False)[:, np.newaxis]
        initial_state_mean = np.zeros(2)
        initial_state_cov = np.eye(2)
        
        self.kf = KalmanFilter( 
            n_dim_obs = 1, 
            n_dim_state = 2,
            initial_state_mean = initial_state_mean,
            initial_state_covariance = initial_state_cov,
            transition_matrices = np.eye(2),
            observation_matrices = obs_mat,
            observation_covariance = obs_cov,
            transition_covariance = trans_cov
            )

        self.kalman_state_means, self.kalman_state_covs = self.kf.filter(self.levels2.values)
        
    def calc_kalman_hedge_ratio(self):
        """
        Calculate the rolling hedge ratio using a Kalman Filter.

        Returns
        -------
        None.

        """
        
        if not self.kf:
            self.setup_kalman_filter()
            
        self.kalman_hedge_ratio = pd.Series(self.kalman_state_means[:, 0], index = self.levels1.index)
        self.kalman_intercept = pd.Series(self.kalman_state_means[:, 1], index = self.levels1.index)
    
    def setup_lookback_window(self):
        """
        Setup the lookback window by calculating the half-life from the Ornstein-Uhlenbeck process. Update the attribute directly.

        Returns
        -------
        None.

        """
        
        if not self.lookback_window:
            if not self.half_life:
                self.calc_half_life()
                
            self.lookback_window = int(2 * self.half_life)

    def calc_hedge_ratio(self):
        """
        Calculate the hedge ratio using OLS of levels2 vs levels1 on the whole data set. This will not be used in the trading strategy, since
        it requires knowledge of future levels. Instead, we will us calc_rolling_hedge_ratio with the lookback window.

        Returns
        -------
        float
            Hedge ratio.

        """
        
        X = sm.add_constant(self.levels1)
        lin_model = sm.OLS(self.levels2, X).fit()
        hedge_ratio = lin_model.params[self.ticker1]
        self.hedge_ratio = hedge_ratio
        self.lin_reg_hedge_ratio = hedge_ratio
        self.lin_reg_intercept = lin_model.params['const']
        
        if self.model == 'ratio':
            self.hedge_ratio = 1

        return self.hedge_ratio
    
    def calc_lookback_rolling_hedge_ratio(self, lookback_window = None):
        """
        Calculate the hedge ratio using OLS of levels2 vs levels1 during the lookback window. This will be used in the trading strategy.

        Parameters
        ----------
        lookback_window : int, optional
            Lookback window to be used. The default is None.

        Returns
        -------
        hedge_ratio : pd.Series
            Rolling hedge ratio.

        """
 
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
        """
        Calculate the spread between levels2 and levels 1 using the whole data set. This is not used for the trading strategy as it requires knowledge
        of future levels. Instead, we will use calc_rolling_spread for the trading strategy.

        Returns
        -------
        spread : pd.Series
            Spread between levels2 and levels1.

        """

        if not self.hedge_ratio:
            self.calc_hedge_ratio()

        if self.model == 'ratio':
            spread = self.levels2 / self.levels1
        else:
            spread = self.levels2 - self.hedge_ratio * self.levels1
        spread.name = 'Spread'
        self.spread = spread

        return spread

    def calc_lookback_rolling_spread(self, lookback_window = None):
        """
        Calculate the rolling spread between levels2 and levels1 using only the data in the lookback window. This is used for the trading strategy.
        Also calculate the moving average of the rolling spread, its mean and standard deviation and its standardised form (mean = 0 and std = 1).

        Parameters
        ----------
        lookback_window : int, optional
            Number of days to be used for the lookback window. The default is None.

        Returns
        -------
        stand_rolling_spread : pd.Series
            Rolling spread standardised to mean 0 and std 1.

        """
        
        if lookback_window and self.lookback_window != lookback_window:
            self.lookback_window = lookback_window
            self.calc_lookback_rolling_hedge_ratio()

        if self.rolling_hedge_ratio.empty:
            self.calc_lookback_rolling_hedge_ratio()

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
        stand_rolling_spread = (self.rolling_spread - self.spread_mave) / std_rolling_spread
        stand_rolling_spread.name = 'Standardised Rolling Spread'

        return stand_rolling_spread
    
    def calc_kalman_spread(self):
        """
        Calculate the spread between levels2 and levels1 using a Kalman Filter. This is used for the trading strategy.
        Also use the intercept predicted by the Kalman Filter as the "moving average" of the spread, but compute the standard 
        deviation of the spread using a lookback window. Calculate the spread in standardised form (mean = 0, std = 1),

        Returns
        -------
        stand_rolling_spread : pd.Series
            Rolling spread standardised to mean 0 and std 1.

        """
        
        if self.kalman_hedge_ratio is None:
            self.calc_kalman_hedge_ratio()
            
        self.rolling_spread = self.levels2 - self.kalman_hedge_ratio.shift(1) * self.levels1
        
        self.spread_mave = self.kalman_intercept.shift(1)
        self.spread_mave.name = 'Kalman Filter Spread Expectation'
        
        self.std_rolling_spread = self.rolling_spread.rolling(window = self.lookback_window).std()
        self.std_rolling_spread.name = 'Kalman Fiter Spread Standard Deviation'
        
        stand_rolling_spread = (self.rolling_spread - self.spread_mave) / self.std_rolling_spread
        
        stand_rolling_spread.name = 'Standardised Kalman Filter Spread'

        return stand_rolling_spread
    
    def calc_rolling_spread(self, lookback_window = None):
        """
        Calculate the rolling spread either using a lookback window or a Kalman filter.

        Parameters
        ----------
        lookback_window : int, optional
            Size of the lookback window to be used in case of using the lookbakc method. The default is None.

        Returns
        -------
        None.

        """
        
        if self.method == 'lookback' or self.model == 'ratio':
            self.stand_rolling_spread = self.calc_lookback_rolling_spread(lookback_window)
        
        else: # compute using Kalman filter
            self.stand_rolling_spread = self.calc_kalman_spread()
            self.rolling_hedge_ratio = self.kalman_hedge_ratio

    def calc_half_life(self):
        """
        Estimate the half-life of mean reversion by fitting the spread to an Ornstein-Uhlenbeck process.

        Returns
        -------
        float
            Half-life of mean reversion of the spread.

        """

        if self.spread.empty:
            self.calc_spread()
            
        delta_spread = (self.spread - self.spread.shift(1)).dropna()
        lin_model = sm.OLS(delta_spread, sm.add_constant(self.spread.shift(1)).dropna()).fit()
        lambda_hat = - lin_model.params['Spread']
        self.half_life = math.log(2) / lambda_hat

        return self.half_life

    def create_trading_signals(self, lookback_window = None, upper_entry = None, lower_entry = None):
        """
        Create the trading signals for the pairs trading strategy, by going long when the rolling spread is lower than lower_entry * its std from
        the rolling spread mave and going short when it is upper_entry * its std from the rolling spread mave.
        Create the trades Series which keeps track of the entry and exit dates of the trades and their duration.

        Parameters
        ----------
        lookback_window : int, optional
            Number of days to use as lookback window, if the already set lookback window is to be overwritten. The default is None.
        upper_entry : float, optional
            Number of std's above the rolling spread mave we use as the entry point for a short position for the spread.
            If none is passed, use the already set upper_entry. The default is None.
        lower_entry : float, optional
            Number of std's below the rolling spread mave we use as the entry point for a long position for the spread.
            If none is passed, use the already set upper_entry. The default is None.

        Returns
        -------
        self.signals : pd.Series
            Series of the long/short positions in the spread: 1 = long position, -1 = short position, 0 = no position.
        self.trades : pd.Series
            Series of the trades.

        """
        
        if self.method == 'lookback' and lookback_window:
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
        self.signals.iloc[-1] = 0
        
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
    
    def calc_PnL(self, fixed_position_value=10_000):
        """
        Calculate the strategy's performance: PnL, Sharpe Ratio, APR and max drawdown.
        Enter each long/short trade with a fixed position value, determined by fixed_position_value. The position size is split between asset1 and 
        asset2 as determined by the hege ratio. Create a portfolio dataframe containing the relevant data of the portfolio created by the strategy
        and a trades dataframe containing information on each trade with data at the entry and exit points.

        Parameters
        ----------
        fixed_position_value : float, optional
            Fixed position value that determines the size of each trade at entry. The default is 10_000.

        Returns
        -------
        self.portfolio : pd.DataFrame
            DataFrame of the portfolio that the strategy is holding each day. Includes the position in asset1, asset2, spread, pnl, returns, cumulative returns.
        self.trades : pd.DataFrame
            DataFrame of the trades that the strategy is executing. Includes the position in asset1, asset2, spread at entry and exit dates and pnl of trade.

        """
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
        
        # Initialise portfolio DataFrame
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
        
        self.Sharpe = calc_Sharpe(self.portfolio['returns'])
        self.APR = calc_APR(self.portfolio['returns'])
        #cum_returns_after_first_trade = self.portfolio['cum_returns'][self.trades.iloc[0]['exit_date']:]
        #self.max_drawdown = calc_max_drawdown(cum_returns_after_first_trade)
        self.max_drawdown = calc_max_drawdown(self.portfolio['cum_returns'])

        return self.portfolio, self.trades

    def run_ADF(self):
        """
        Run the ADF test on each asset in the pair and on the spread.

        Returns
        -------
        tuple(str, float, float)
            Results of the ADF test in form (ticker, score, p-value) for each asset and the spread all as a list.

        """
        
        if self.spread.empty:
            self.calc_spread()

        asset_list = [self.levels1, self.levels2, self.spread]
        return check_stationarity(asset_list)