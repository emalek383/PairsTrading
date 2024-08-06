""" Module creating the StockUniverse class. """

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
        
    Methods
    -------
    get_data():
        Download stock data.
        Return the tickers of stocks that could not be downloaded.
    
    calc_mean_returns_cov():
        Calculate the mean returns and covariance matrix from stock data.
        Update `mean_returns` and `cov_matrix`.
    
    """
    
    def __init__(self, 
                 stocks, 
                 start_date = dt.datetime.today() + relativedelta(years = -1), 
                 end_date = dt.datetime.today(), 
                 mean_returns = [], 
                 cov_matrix = []
                 ):
        """
        Construct the attributes of the stock universe object.

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
        Set the risk-free-rate from downloaded bonds data if it has not yet been set.
        Calculate the max Sharpe Ratio and min vol portfolios and upadte the min/max excess returns/vol.
        Update all these attributes.

        Returns
        -------
        None.

        """
        
        returns = self.stock_data.pct_change()
        cov_matrix = returns.cov()
        
        self.cov_matrix = cov_matrix