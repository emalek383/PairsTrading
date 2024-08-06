""" Module to run the cointegration analysis between two pairs of stocks. """

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

DEFAULT_SIGNIFICANCE = 0.05

def find_coint_pairs(data, significance = DEFAULT_SIGNIFICANCE):
    """
    Run the Augmented Engle-Granger cointegration test on the data and find viable pairs (with p-value below significance).

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with the stock data.
    significance : float, optional
        Significance level of the AEG hypothesis test. The default is DEFAULT_SIGNIFICANCE.

    Returns
    -------
    pairs : list(tuple(str))
        List of possible pairs (given as tuple of their tickers).
    pvalue_df : pd.DataFrame
        Dataframe containing the p-values of all the pairwise AEG tests.

    """
    num_stocks = data.shape[1]
    pvalue_matrix = np.ones((num_stocks, num_stocks))
    keys = data.keys()
    pairs = []
    for i in range(num_stocks):
        stock1 = data[keys[i]]
        for j in range(i + 1, num_stocks):
            stock2 = data[keys[j]]
            score, pvalue,  *_= coint(stock1, stock2, maxlag = 1)
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance:
                pairs.append((keys[i], keys[j], pvalue))
                
    pvalue_df = pd.DataFrame(pvalue_matrix, index=keys, columns=keys)

    
    return pairs, pvalue_df