# Pairs-Trading
Web app for finding cointegrated stock pairs to use in pairs trading, i.e. exploits the spread between two assets which mean revert.
Link to the  [web app](https://emanuelmalek.com/quant_projects/pairs_trading.html).

## Functionalities:
- Test for cointegration of assets in a stock basket (via EG/ADF test)
- Compute hedge ratio for spread (using simple moving average)
- Estimate half-life by fitting Ornstein-Uhlenbeck Process
- Use spread of asset prices, log(asset prices) or ratio of asset prices for signal.
- Vectorised evaluation of trading strategy based on bollinger bands entry criteria

## How to run
The web app is built using streamlit. After pip installing streamlit, you can launch the web app by running
```
streamlit run main.py
```
