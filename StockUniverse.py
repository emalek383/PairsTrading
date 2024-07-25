import datetime as dt
from dateutil.relativedelta import relativedelta
from data_loader import download_data

TRADING_DAYS = 252

class StockUniverse():
    """
    Class to capture a Stock Universe. Contains important info, such as the stocks' mean returns, cov matrix,
    the risk free rate, as well as the max Sharpe Ratio and min vol portfolios.
    
    Attributes
    ----------
    stocks : list(str)
        Tickers of stocks in the stock universe
    start_date : datetime
        Start date of stock universe (important for getting data)
    end_date : datetime
        End date of stock universe (important for getting data)
    stock_data: pd.DataFrame
        Stock data
    bonds_data: pd.DataFrame
        3-months T-Bill data
    mean_returns : np.array
        Mean returns of stocks
    cov_matrix : np.array
        Covariance matrix
    max_SR_portfolio : portfolio
        Max Sharpe Ratio Portfolio
    min_vol_portfolio : portfolio
        Min Vol Portfolio
    min_returns : float
        Minimum annualised returns achievable by a portfolio in the universe
    max_returns : float
        Maximum annualised returns achievable by a portfolio in the universe
    min_vol : float
        Minimum annualised volatility achievable by a portfolio in the universe
    max_vol : float
        Maximum annualised volatility achievable by a portfolio in the universe
        
    Methods
    -------
    get_data():
        Downloads stock data.
        Returns the tickers of stocks that could not be downloaded.
    
    calc_mean_returns_cov():
        Calculates the mean returns and covariance matrix from stock data.
        Updates mean_returns and cov_matrix.
    
    """
    def __init__(self, 
                 stocks, 
                 start_date = dt.datetime.today() + relativedelta(years = -1), 
                 end_date = dt.datetime.today(), 
                 mean_returns = [], 
                 cov_matrix = []
                 ):
        """
        Constructs the attributes of the stock universe object.

        Parameters
        ----------
        stocks : list(str)
            List of stock tickers for stocks in universe.
        start_date : datetime, optional
            Start date of universe to be considered. The default is 1 year ago.
        end_date : datetime, optional
            End date of universe to be considered. The default is today.
        mean_returns : np.array, optional
            Mean returns of the stock universe. Typically will be downloaded and computed in-class, but can 
            optionally be passed directly. The default is None.
        cov_matrix : np.array, optional
            Covariance matrix of the stock universe. Typically will be downloaded and computed in-class, but can 
            optionally be passed directly. The default is None.

        Returns
        -------
        None.

        """
        
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = None
        self.bonds_data = None
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        
    def get_data(self):
        """
        Download the stock and bonds data and save it in relevant attributes.

        Returns
        -------
        ignored : list(str)
            List of stock tickers for stocks that could not be donloaded.

        """
        
        stock_data = download_data(self.stocks, self.start_date, self.end_date)
        ignored = []
        
        for ticker in self.stocks:
            if stock_data[ticker].isnull().all():
                ignored.append(ticker)
                
        for ticker in ignored:    
            self.stocks.remove(ticker)

        stock_data = stock_data[self.stocks]
                
        self.stock_data = stock_data
       
        return ignored
        
    def calc_mean_returns_cov(self):
        """
        Compute mean returns and covariance matrix of stock universe from the (downloaded) stock data.
        Will set the risk-free-rate from downloaded bonds data if it has not yet been set.
        Calculate the max Sharpe Ratio and min vol portfolios and upadte the min/max excess returns/vol.
        Update all these attributes.

        Returns
        -------
        None.

        """
        
        returns = self.stock_data.pct_change()
        #mean_returns = get_mean_returns(returns)
        cov_matrix = returns.cov()
        
        #self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix