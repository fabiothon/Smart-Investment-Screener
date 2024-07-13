# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

# STATUS: OBSOLETE

# This script is a seperate document that was used to develop a monte carlo simulation 
# that was implemented into the main.py (Smart Investment Screener). The code is documented
# and commented in the main.py file and additional information is found in the README File

import pandas as pd
import yfinance as yf
import statistics
import numpy as np
import plotly.express as px

stock_symbol = 'AAPL'

price_data_raw = yf.download(stock_symbol, period="5y")
price_data = price_data_raw['Close']
log_returns_raw = np.log(price_data / price_data.shift(1))
log_returns = log_returns_raw.dropna()

mean_log_returns = log_returns.mean()
var_log_returns = statistics.variance(log_returns)
sdv_log_returns = statistics.stdev(log_returns)

drift = mean_log_returns - 0.5 * var_log_returns
random_value = sdv_log_returns * np.random.normal()


mc_sims = 5     # number of simulations
T = 365         # timeframe in days


simulated_prices = np.zeros((mc_sims, T))
initial_price = price_data.iloc[-1]

# Run Monte Carlo simulations
for i in range(mc_sims):
    today_price = initial_price

    for day in range(T):
        random_value = sdv_log_returns * np.random.normal()
        today_price = today_price * np.exp(drift + random_value)
        simulated_prices[i, day] = today_price

simulated_prices_df = pd.DataFrame(simulated_prices.T, columns=[f'Simulation_{i+1}' for i in range(mc_sims)])
simulated_prices_df.info()

# Plotly
fig = px.line(simulated_prices_df, 
              x=simulated_prices_df.index, 
              y=simulated_prices_df.columns, 
              title='Title')
fig.show()


# =============================================================================
# # Plot the Monte Carlo simulation using hvplot
# plot = simulated_prices_df.hvplot.line(title=f'Monte Carlo Simulation - {stock_symbol}', xlabel='Days', ylabel='Simulated Prices')
# plot
# =============================================================================








