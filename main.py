# =============================================================================
# SMART INVESTMENT SCREENER
# =============================================================================

# DESCRIPTION:
# This Python code creates an interactive dashboard for smart investment screening, 
# leveraging data fetched from a financial data API. The goal is to provide a 
# comprehensive analysis of a single company based on its stock symbol. The dashboard 
# includes various charts such as revenue history, earnings history, operating margin 
# trends, historical stock prices, Monte Carlo simulation of stock prices, and MACD chart 
# for technical analysis. Additionally, key financial indicators and scores like total 
# investment score, debt-to-equity ratio, return on invested capital (ROIC), enterprise 
# value over EBIT, and minimum growth rate are displayed. Users can also apply structured 
# and unstructured filters using check buttons for further analysis. Company information 
# including ticker symbol, exchange, currency, current price, industry, CEO, description, 
# location, and contact details are displayed in a sidebar. The dashboard layout is organized 
# using the BootstrapTemplate from Panel library, providing an intuitive and interactive 
# interface for investment decision-making. 
# Additional information can be found in the README File.

# NOTE: 
# Please be aware that the current script operates under the constraints of the free plan for 
# the utilized API. This means there's a limit on the number of API calls that can be made 
# per day, and it's possible that this limit could change in the future. Additionally, you may 
# notice some commented-out code within the script, which remains unused at present. However, 
# it could become relevant if a paid plan is opted for in the future.

# =============================================================================
# Libraries
# =============================================================================
import keyring # For save storage and access to API-Key
import pandas as pd # To handle dataframe
from warnings import simplefilter # Filters out unecessary advices
import json # Library to work with json
from urllib.request import urlopen # Library to work with URL requests
import panel as pn # Library to create a dashboard
pn.extension('tabulator') # Extension for a better graphic
import holoviews as hv # Holoviews creates interactive graphs
hv.extension('bokeh') # Extension for better graphic
from PIL import Image # Library to import .png pictures
from io import BytesIO # Library to handle pictures
import plotly.graph_objects as go # Creates graphs
import plotly.express as px # Creates graphs
import hvplot.pandas # Creates graphs
import yfinance as yf # Library to access Yahoo Finance API
import statistics # For easier calculation
import numpy as np # For easier calculation
from plotly.subplots import make_subplots # Creates graphs
from ta.utils import dropna # Creates graphs
from ta.volatility import BollingerBands # Creates graphs
from ta.trend import MACD # Creates graphs

# =============================================================================
# Global default setting: This section sets global variables
# =============================================================================
simplefilter(action="ignore", category=pd.errors.PerformanceWarning) # Supresses performance warnings
pd.options.mode.chained_assignment = None
accent = "#BB2649" # Sets a color
upwards_pyramid_symbol = "triangle-up" # Sets symbol
downwards_pyramid_symbol = "triangle-down" # Sets symbol

# =============================================================================
# Variables
# =============================================================================
stock_symbol = input("INPUT - Fill in the Stock Symbol: ") # Input of the stock symbol
api_key = keyring.get_password("Financial_Modeling_Prep", "Financial_Modeling_Prep") # Uses the library keyring to safely get the API-Key

# Column orders for all the loaded datasets
column_order_1 = ['symbol', 'price', 'range', 'changes', 'currency', 'exchangeShortName', 
                  'industry', 'website', 'description', 'ceo', 'sector', 'country', 'companyName',
                  'fullTimeEmployees', 'address', 'city', 'zip', 'state', 'phone']

column_order_2 = ['date','weightedaveragenumberofsharesoutstandingbasic', 'netincomeloss'
                  , 'incometaxexpensebenefit']

column_order_3 = ['date','marketCap','roic','enterpriseValue','debtToEquity']

column_order_4 = ['year','weightedaveragenumberofsharesoutstandingbasic', 
                  'netincomeloss', 'incometaxexpensebenefit', 
                  'marketCap','roic','enterpriseValue', 'debtToEquity', 'revenue', 'grossProfit', 
                  'netIncome', 'operatingIncome', 'totalOtherIncomeExpensesNet']

# =============================================================================
# DATA NOT ACCESSIBLE WITH THE FREE API PLAN
# column_order_5 = ['year','totalBought']
# =============================================================================

column_order_6 = ['date','revenue', 'grossProfit', 'netIncome', 'operatingIncome', 'totalOtherIncomeExpensesNet']

column_order_7 = ['date', 'open', 'high', 'low', 'close']

# =============================================================================
# FUNCTIONS
# =============================================================================

# BOLLINGER BANDS: This function generates bollinger bands for the technical analysis
def calculate_bb_indicators(technical_analysis_df):
    indicator_bb = BollingerBands(close=technical_analysis_df["Close"], window=20, window_dev=2)
    technical_analysis_df['bb_bbm'] = indicator_bb.bollinger_mavg() # Moveing average SMA20
    technical_analysis_df['bb_bbh'] = indicator_bb.bollinger_hband() # High band
    technical_analysis_df['bb_bbl'] = indicator_bb.bollinger_lband() # Low band
    
    return technical_analysis_df # Return of dataframe

# MACD INDICATORS: This function generates MACD Indicators for the technical analysis
def calculate_MACD_indicators(technical_analysis_df):
    macd_object = MACD(technical_analysis_df['Close']) # Gets the Closing Price
    technical_analysis_df['MACD'] = macd_object.macd() # Generates the MACD Line
    technical_analysis_df['MACD_Signal'] = macd_object.macd_signal() # Generates the MACD Signal Line
    technical_analysis_df['MACD_Diff'] = macd_object.macd_diff() # Generates the MACD Difference (unused in the plot)
    technical_analysis_df['SMA50'] = technical_analysis_df['Close'].rolling(window=50).mean() # Generates the SMA50
    technical_analysis_df['SMA200'] = technical_analysis_df['Close'].rolling(window=200).mean() # Generates the SMA200
    
    return technical_analysis_df # Return of dataframe
    
# UPDATE FUNCTION: This functions represents the backend logic for the buttons in the technical analysis
def update_traces(button_idx):
    visibility = [
        [True, True, False, False, False, False, False, False, False, True, True, True, True],    # None
        [True, True, True, True, True, False, False, False, False, True, True, True, True],       # MA
        [True, True, True, True, True, True, True, False, False, True, True, True, True],         # Buy/Sell
        [True, True, False, False, False, False, False, True, True, True, True, True, True],      # BB
        [True, True, True, True, True, True, True, True, True, True, True, True, True]]           # All
    
    return visibility[button_idx] # Return of the logic

# =============================================================================
# API-CALLS AND TRANSFORMATION TO DATAFRAMES
# =============================================================================
# API-CALL FETCHING: Access to the FMP-API with exception handling and manual logging
def get_jsonparsed_data(url):
    try:
        res = urlopen(url)
        data = res.read().decode("utf-8")
        return json.loads(data)
        print("SUCCESS: API-call successfully executed.")
    except Exception as e:
        print("ERROR: Failure to retrieve Statement Analysis.","\n", e)

print('*********** START OF SCRIPT ***********') # Info log to mark the start of the main loop

# API-Call for financial statement
try:
    financial_statement = get_jsonparsed_data(f"https://financialmodelingprep.com/api/v3/financial-statement-full-as-reported/{str(stock_symbol)}?period=annual&limit=50&apikey={str(api_key)}")
    financial_statement_df_raw = pd.DataFrame(financial_statement)
    financial_statement_df_raw = financial_statement_df_raw[column_order_2]
    print("SUCCESS: Financial statement loaded.")
except Exception as e:
    print("ERROR: Failure to retrieve Financial Statement.","\n", e)

# API-Call for income statement
try:
    income_statement = get_jsonparsed_data(f"https://financialmodelingprep.com/api/v3/income-statement/{str(stock_symbol)}?period=annual&limit=50&apikey={str(api_key)}")
    income_statement_df_raw = pd.DataFrame(income_statement) # Transformation to a pandas dataframe
    income_statement_df_raw = income_statement_df_raw[column_order_6] # Reorder of the dataframe
    print("SUCCESS: Income statement loaded.")
except Exception as e:
    print("ERROR: Failure to retrieve Income Statement.","\n", e)

# API-Call for analysis of statement
try:
    statement_analysis = get_jsonparsed_data(f"https://financialmodelingprep.com/api/v3/key-metrics/{str(stock_symbol)}?period&limit=50&apikey={str(api_key)}")
    statement_analysis_df_raw = pd.DataFrame(statement_analysis) # Transformation to a pandas dataframe
    statement_analysis_df_raw = statement_analysis_df_raw[column_order_3] # Reorder of the dataframe
    print("SUCCESS: Statement analysis loaded.")
except Exception as e:
    print("\n","ERROR: Failure to retrieve Statement Analysis.","\n", e)

# API-Call for general company information
try:
    company_information = get_jsonparsed_data(f"https://financialmodelingprep.com/api/v3/profile/{str(stock_symbol)}?period&limit=1&apikey={str(api_key)}")
    company_information_df_raw = pd.DataFrame(company_information) # Transformation to a pandas dataframe
    company_information_df = company_information_df_raw[column_order_1] # Reorder of the dataframe
    print("SUCCESS: Company Information loaded.")
except Exception as e:
    print("\n","ERROR: Failure to retrieve Company Information.","\n", e)

# API-Call for chart information
try:
    chart_information = get_jsonparsed_data(f"https://financialmodelingprep.com/api/v3/historical-price-full/{str(stock_symbol)}?apikey={str(api_key)}")
    chart_information_df_raw = pd.DataFrame(chart_information['historical']) # Transformation to a pandas dataframe
    chart_information_df = chart_information_df_raw[column_order_7] # Reorder of the dataframe
    print("SUCCESS: Chart Information loaded.")
except Exception as e:
    print("\n","ERROR: Failure to retrieve Chart Information.","\n", e)

# =============================================================================
# DATA NOT ACCESSIBLE WITH THE FREE API PLAN
# try:
#     insider_information = get_jsonparsed_data(f"https://financialmodelingprep.com/api/v4/insider-roaster-statistic/{str(stock_symbol)}?apikey={str(api_key)}")
#     insider_information_df_raw = pd.DataFrame(insider_information)
#     insider_information_df = insider_information_df_raw[column_order_5]
#     print("SUCCESS: Insider Information loaded.")
# except Exception as e:
#     print("\n","ERROR: Failure to retrieve Insider Information.","\n", e)
# =============================================================================

# Creation of a temp storage for the raw data set -> Save to a SQLite DB
financial_statement_df = financial_statement_df_raw
income_statement_df = income_statement_df_raw
statement_analysis_df = statement_analysis_df_raw

# =============================================================================
# BASIC DATA MANIPULATION
# =============================================================================
# Transformation of date column and creation of year column
try:
    financial_statement_df['date'] = pd.to_datetime(financial_statement_df['date'])
    financial_statement_df['year'] = financial_statement_df['date'].dt.year
    financial_statement_df.drop(columns=['date'], inplace=True)
    
    income_statement_df['date'] = pd.to_datetime(income_statement_df['date'])
    income_statement_df['year'] = income_statement_df['date'].dt.year
    income_statement_df.drop(columns=['date'], inplace=True)
    
    statement_analysis_df['date'] = pd.to_datetime(statement_analysis_df['date'])
    statement_analysis_df['year'] = statement_analysis_df['date'].dt.year
    statement_analysis_df.drop(columns=['date'], inplace=True)
    print("SUCCESS: Dataframe converted to new format.")
except Exception as e:
    print("\n","ERROR: Failure to convert dataframe to new format.","\n", e)

# Merging of three financial dataframes to one: financial_df
try:
    finance_df = pd.merge(financial_statement_df, statement_analysis_df, on=['year'])
    finance_df = pd.merge(finance_df, income_statement_df, on=['year'])
    finance_df = finance_df[column_order_4]
    print("SUCCESS: Dataframes successfully merged and columns re-ordered.")
except Exception as e:
    print("\n","ERROR: Failure to merge dataframes.","\n", e)

# =============================================================================
# CALCULATIONS BASED ON FINANCE DATA
# =============================================================================

# Calculation of TTM values (.loc enabled): TTM = Trailing Twelve Months
try:
    latest_date_idx = finance_df['year'].idxmax()
    ttm_values = finance_df.loc[latest_date_idx].copy()
    print("SUCCESS: TTM values successfully generated.")
except Exception as e:
    print("\n","ERROR: Failure to generate TTM values.","\n", e)
    
# Calculation of historic (20XX) values (.loc enabled): The oldest date is taken - in our case 5 years
try:
    historic_date_idx = finance_df['year'].idxmin()
    historic_values = finance_df.loc[historic_date_idx].copy()
    print("SUCCESS: Historic values successfully generated.")
except Exception as e:
    print("\n","ERROR: Failure to generate historic values.","\n", e)
    
# Calculation of TTM and historic financial ratios
try:
    finance_df['operating_margin_historic'] = historic_values['operatingIncome'] / historic_values['revenue'] # Calculation of histroic operating margin
    finance_df['operating_margin'] = (finance_df['operatingIncome'] / finance_df['revenue'])*100 # Calculation of current operating margin
    finance_df['revenue_MM'] = (finance_df['revenue']) / (1000000) # Calculation of revenue expressed in Millions
    finance_df['operating_margin_ttm'] = ttm_values['operatingIncome'] / ttm_values['revenue'] # Calculation of TTM operating margin
    finance_df['revenue_per_share_historic'] = historic_values['revenue'] / historic_values['weightedaveragenumberofsharesoutstandingbasic'] # Calculation of historic revenue per share
    finance_df['revenue_per_share_ttm'] = ttm_values['revenue'] / ttm_values['weightedaveragenumberofsharesoutstandingbasic'] # Calculation of TTM revenue per share
    finance_df['enterprisevalue_over_ebit'] = (ttm_values['enterpriseValue']) / (ttm_values['netIncome'] + ttm_values['incometaxexpensebenefit'] + ttm_values['totalOtherIncomeExpensesNet']) # Calculation of EV/EBIT
    finance_df['roic'] = finance_df['roic'] * 100 # Calculation of Return on invested capital
    finance_df['netIncome_MM'] = (finance_df['netIncome']) / (1000000) # Calculation of net income expressed in Millions
    print("SUCCESS: TTM and historic ratios successfully calculated.")
except Exception as e:
    print("\n","ERROR: Failure to calculate TTM and histroic ratios.","\n", e)

# =============================================================================
# DATA NOT ACCESSIBLE WITH THE FREE API PLAN
# # Calculation of insider trading ratio
# try:
#     finance_df['insiderratio'] = (insider_information_df['rototalBoughtic']) / (finance_df['marketCap'])
#     print("SUCCESS: Insider trading ratio successfully calculated.")
# except Exception as e:
#     print("\n","ERROR: Failure to calculate insider trading ratio.","\n", e)
# =============================================================================
    
# Calculation of growth CAGR (Compound Annual Growth Rate) and evaluation of the lowest growth variable
try:
    time_period = (finance_df.loc[latest_date_idx, 'year'] - finance_df.loc[historic_date_idx, 'year']) # Calculation of time period between now and first vaule (should be more or less 5 years)
    finance_df['revenue_growth_cagr'] = (ttm_values['revenue']) / (historic_values['revenue'])**(1/time_period) - 1 # Calculation of revenue growth (CAGR)
    finance_df['operating_income_growth_cagr'] = (ttm_values['operatingIncome']) / (historic_values['operatingIncome'])**(1/time_period) - 1 # Calculation of operating income growth (CAGR)
    finance_df['revenue_per_share_growth_cagr'] = (finance_df['revenue_per_share_ttm']) / (finance_df['revenue_per_share_historic'])**(1/time_period) - 1 # Calculation of revenue per share growth (CAGR)
    finance_df['operating_income_per_share_growth_cagr'] = (finance_df['revenue_per_share_ttm'] * finance_df['operating_margin_ttm']) / (finance_df['revenue_per_share_historic'] * finance_df['operating_margin_historic'])**(1/time_period) - 1 # Calculation of operating income per share growth (CAGR)
    finance_df['lowest_growth'] = finance_df[['revenue_growth_cagr', 'operating_income_growth_cagr', 'revenue_per_share_growth_cagr', 'operating_income_per_share_growth_cagr']].min(axis=1) # Calculation of the lowest (weakest) growth (CAGR) of all listed growths in this code block
    print("SUCCESS: Growth CAGR successfully calculated and new variable evaluated.")
except Exception as e:
    print("\n","ERROR: Failure to calculate growth CAGR or evaluate new variable","\n", e)

# =============================================================================
# NORMALIZATION AND WEIGHTED SCORING OF KEY DATA
# =============================================================================

# Calculation of the normalized values of key parameters in order to create a weighted score
try:
    finance_df['growth_normalized'] = (finance_df['lowest_growth']) / (2) # Normalization
    finance_df['roic_normalized'] = (finance_df['roic']) - (5 / 40 - 5) # Normalization
    finance_df['enterprisevalue_over_ebit_normalized'] = (finance_df['enterprisevalue_over_ebit'] - 5) / (10 - 5) # Normalization
    # finance_df['insiderratio_normalized'] = (finance_df['insiderratio'] - 0.08) / (0.1 - 0.08) # Normalization (No data available due to free API-Plan)
    finance_df['score'] = (finance_df['growth_normalized'] * 0.25) + (finance_df['roic_normalized'] * 0.25) + (finance_df['enterprisevalue_over_ebit_normalized'] * 0.5) # Calculation of weighted score
    finance_df['debtToEquity'] = (finance_df['debtToEquity']) * 100 # Calculation of debt to equity ratio
    print("SUCCESS: Key data normalized and scored successfully.")
except Exception as e:
    print("\n","ERROR: Failure to normalize and score key data.","\n", e)

# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

# Fetching financial data from Yahoo Finance
price_data_raw = yf.download(stock_symbol, period="5y")
price_data = price_data_raw['Close']
log_returns_raw = np.log(price_data / price_data.shift(1))
log_returns = log_returns_raw.dropna()

mean_log_returns = log_returns.mean()
var_log_returns = statistics.variance(log_returns)
sdv_log_returns = statistics.stdev(log_returns)

drift = mean_log_returns - 0.5 * var_log_returns
random_value = sdv_log_returns * np.random.normal()


mc_sims = 225  # number of simulations
T = 360  # timeframe in days


simulated_prices = np.zeros((mc_sims, T))
initial_price = price_data.iloc[-1]

for i in range(mc_sims):
    today_price = initial_price

    for day in range(T):
        random_value = sdv_log_returns * np.random.normal()
        today_price = today_price * np.exp(drift + random_value)
        simulated_prices[i, day] = today_price

simulated_prices_df = pd.DataFrame(simulated_prices.T, columns=[f'Simulation_{i+1}' for i in range(mc_sims)])

# =============================================================================
# TECHNICAL CHART ANALYSIS
# =============================================================================

# Grab information form YahooFinance
technical_analysis_df = yf.download(stock_symbol, period="5y")
technical_analysis_df = dropna(technical_analysis_df)
technical_analysis_df = calculate_bb_indicators(technical_analysis_df)
technical_analysis_df = calculate_MACD_indicators(technical_analysis_df)

# Crossings of SMA to stock price
technical_analysis_df['Crossings'] = 0
crossing_indices_below = (technical_analysis_df['Close'] > technical_analysis_df['SMA50']) & (technical_analysis_df['Close'].shift(-1) < technical_analysis_df['SMA50'].shift(-1))
crossing_indices_above = (technical_analysis_df['Close'] < technical_analysis_df['SMA50']) & (technical_analysis_df['Close'].shift(-1) > technical_analysis_df['SMA50'].shift(-1))
technical_analysis_df.loc[crossing_indices_below, 'Crossings'] = -1
technical_analysis_df.loc[crossing_indices_above, 'Crossings'] = 1

# Add markers for crossings "SELL"
cross_below = go.Scatter(x=technical_analysis_df[technical_analysis_df['Crossings'] == -1].index, y=technical_analysis_df[technical_analysis_df['Crossings'] == -1]['SMA50'],
    mode='markers', name='Price Crosses Below SMA50', marker=dict(symbol=downwards_pyramid_symbol, color='red', size=7.5))

# Add markers for crossings "BUY"
cross_above = go.Scatter(x=technical_analysis_df[technical_analysis_df['Crossings'] == 1].index, y=technical_analysis_df[technical_analysis_df['Crossings'] == 1]['SMA50'],
    mode='markers', name='Price Crosses Above SMA50', marker=dict(symbol=upwards_pyramid_symbol, color='green', size=7.5))

# Identify bullish and bearish crossover points
technical_analysis_df['Crossings_1'] = 0
crossing_indices_below_1 = (technical_analysis_df['MACD'] > technical_analysis_df['MACD_Signal']) & (technical_analysis_df['MACD'].shift(-1) <= technical_analysis_df['MACD_Signal'].shift(-1))
crossing_indices_above_1 = (technical_analysis_df['MACD'] < technical_analysis_df['MACD_Signal']) & (technical_analysis_df['MACD'].shift(-1) >= technical_analysis_df['MACD_Signal'].shift(-1))
technical_analysis_df.loc[crossing_indices_below_1, 'Crossings_1'] = -1
technical_analysis_df.loc[crossing_indices_above_1, 'Crossings_1'] = 1

# =============================================================================
# CREATION OF CHARTS FOR DASHBOARD
# =============================================================================
try:
    revenue_chart = finance_df.hvplot.line(
        x='year', 
        y='revenue_MM', 
        height=300, 
        width=700,
        color="#E54871",
        title="History of Total Revenues",
        xlabel="Time (Years)",
        ylabel=f"Total Revenues ({company_information_df.loc[0, 'currency']} M.)"
    )
    
    earning_chart = finance_df.hvplot.line(
        x='year', 
        y='netIncome_MM', 
        height=300, 
        width=700,
        color="#E54871",
        title="History of Total Earnings",
        xlabel="Time (Years)",
        ylabel=f"Total Earnings ({company_information_df.loc[0, 'currency']} M.)"
    )
    
    operating_margin_chart = finance_df.hvplot.line(
        x='year', 
        y='operating_margin', 
        height=300, 
        width=700,
        color="#E54871",
        title="History of Operating Margin",
        xlabel="Time (Years)",
        ylabel="Operating Margin(%)"
    )
    
    montecarlo_chart = px.line(simulated_prices_df, 
        x=simulated_prices_df.index, 
        y=simulated_prices_df.columns, 
        title= f"Monte Carlo Simulation of {company_information_df.loc[0, 'companyName']} Stock",
        labels={'index': 'Days', 'value': f"Stock Price in {company_information_df.loc[0, 'currency']}"},
        line_shape='linear'
    )
    montecarlo_chart.update_layout(showlegend=False)
    montecarlo_chart.update_traces(line={'width':0.5})
    
    total_distribution_raw = simulated_prices_df.melt(var_name='Simulation', value_name='Stock Prices')
    total_distribution = total_distribution_raw.drop(columns='Simulation')
    montecarlo_hist_chart = px.histogram(total_distribution,
        x='Stock Prices',
        title= f"Distribution of simulated {company_information_df.loc[0, 'companyName']} Stock Prices through Monte Carlo Simulation",
        labels={'value': f"Stock Price in {company_information_df.loc[0, 'currency']}"}
    )
    montecarlo_hist_chart.add_vline(x=company_information_df.loc[0, 'price'],
                                    line_dash="dot",
                                    annotation_text="Current price", 
                                    annotation_position="bottom right")
    
    montecarlo_box_chart = px.box(total_distribution, 
        x='Stock Prices',
        points= False,
        title= f"Distribution of simulated {company_information_df.loc[0, 'companyName']} Stock Prices through Monte Carlo Simulation",
        labels={'value': f"Stock Price in {company_information_df.loc[0, 'currency']}"}
    )
    montecarlo_box_chart.add_vline(x=company_information_df.loc[0, 'price'],
                                   line_dash="dot",
                                   annotation_text="Current price", 
                                   annotation_position="bottom right")
    
    daily_chart_eod = go.Figure(data=[go.Candlestick(x=chart_information_df['date'],
        open=chart_information_df['open'],
        high=chart_information_df['high'],
        low=chart_information_df['low'],
        close=chart_information_df['close'])])
    daily_chart_eod.update_layout(
    title=f"Historical prices of {company_information_df.loc[0, 'companyName']}",
    yaxis_title=f"{company_information_df.loc[0, 'symbol']} Stock Price ({company_information_df.loc[0, 'currency']})",
    xaxis_title="Time (Date)",
    shapes = [dict(
    x0='2020-02-20', x1='2020-02-20', y0=0, y1=1, xref='x', yref='paper',
    line_width=1)],
    annotations=[dict(
        x='2020-02-20', y=0, xref='x', yref='paper',
        showarrow=False, xanchor='left', text='Covid stock market crash')]
    )
    
    # Initialize tech_charture
    tech_chart = make_subplots(rows=6, cols=1, subplot_titles=("Interactive Chart","MACD Chart"), 
                        specs=[[{"rowspan": 3}], [None], [None], [None], [{"rowspan": 2}], [None]])

    bullish_crossover = go.Scatter(x=technical_analysis_df[technical_analysis_df['Crossings_1'] == -1].index, y=technical_analysis_df[technical_analysis_df['Crossings_1'] == -1]['MACD'],
        mode='markers', name='Bearish Crossover', marker=dict(symbol='diamond', color='red', size=7.5))
    bearish_crossover = go.Scatter(x=technical_analysis_df[technical_analysis_df['Crossings_1'] == 1].index, y=technical_analysis_df[technical_analysis_df['Crossings_1'] == 1]['MACD'],
        mode='markers', name='Bullish Crossover', marker=dict(symbol='diamond', color='green', size=7.5))

    # Add Traces
    tech_chart.add_trace(go.Candlestick(x=technical_analysis_df.index, name="Stock Price", open=technical_analysis_df['Open'], high=technical_analysis_df['High'], low=technical_analysis_df['Low'],
                                 close=technical_analysis_df['Close'], increasing_line_color= 'skyblue', decreasing_line_color= 'gray'), row=1, col=1)
    tech_chart.add_trace(go.Scatter(x=technical_analysis_df.index, y=technical_analysis_df['Close'], name="Close", line=dict(color="#BB2649", width=1)), row=1, col=1)
    tech_chart.add_trace(go.Scatter(x=technical_analysis_df.index, y=technical_analysis_df['bb_bbm'], name="SMA-20", visible=False, line=dict(color="#006400", width=1)), row=1, col=1)
    tech_chart.add_trace(go.Scatter(x=technical_analysis_df.index, y=technical_analysis_df['SMA50'], name="SMA-50", visible=False, line=dict(color="#009000", width=1)), row=1, col=1)
    tech_chart.add_trace(go.Scatter(x=technical_analysis_df.index, y=technical_analysis_df['SMA200'], name="SMA-200", visible=False, line=dict(color="#090000", width=1)), row=1, col=1)
    tech_chart.add_trace(cross_below)
    tech_chart.add_trace(cross_above)
    tech_chart.add_trace(go.Scatter(x=technical_analysis_df.index, y=technical_analysis_df['bb_bbh'], fill='tonexty', fillcolor='rgba(186, 38, 73, 0.025)', 
                             name="Bollinger Bands", visible=False, line=dict(color="#000000", width=0.5)), row=1, col=1)
    tech_chart.add_trace(go.Scatter(x=technical_analysis_df.index, y=technical_analysis_df['bb_bbl'], fill='tonexty', fillcolor='rgba(186, 38, 73, 0.05)', 
                             visible=False, showlegend=False, line=dict(color="#000000", width=0.5)), row=1, col=1)
    # Subplot Traces
    tech_chart.add_trace(go.Scatter(x=technical_analysis_df.index, y=technical_analysis_df['MACD'], name="MACD Line", line=dict(color="#3633FF", width=1)), row=5, col=1)
    tech_chart.add_trace(go.Scatter(x=technical_analysis_df.index, y=technical_analysis_df['MACD_Signal'], name="Signal Line", line=dict(color="#624990", width=1)), row=5, col=1)

    tech_chart.add_trace(bullish_crossover, row=5, col=1)
    tech_chart.add_trace(bearish_crossover, row=5, col=1)

    # Subplot tech_chart Update
    tech_chart.update_xaxes(title_text="Time (Date)", showgrid=False, row=5, col=1)
    tech_chart.update_yaxes(title_text="Numerical difference", showgrid=False, row=5, col=1)


    # Define button options
    button_labels = ['None', 'Moving Averages', 'Buy/Sell Indicators', 'Bollinger Bands', 'All']
    button_visibility = [[True, False, False, False, False], [True, True, True, False, False], [False, False, True, True, False], [True, True, True, True, True]]


    # Add buttons
    buttons = [dict(label=label, method="update", args=[{"visible": update_traces(idx)}, {"title": f"Chart Analysis: {stock_symbol} - {label}"}]) for idx, label in enumerate(button_labels)]
    tech_chart.update_layout(updatemenus=[dict(type="buttons", direction="right", active=0, x=0.57, y=1.2, buttons=buttons)])

    # Set title and show plot
    tech_chart.update_layout(title_text=f"Chart Analysis: {stock_symbol} - None", xaxis_domain=[0.05, 1.0], yaxis_title=f"{company_information_df.loc[0, 'symbol']} Stock Price ({company_information_df.loc[0, 'currency']}", xaxis_rangeslider_visible=False,
    xaxis_title="Time (Date)")
    
    print("SUCCESS: Charts successfully created.")
except Exception as e:
    print("\n","ERROR: Failure to create charts.","\n", e)

# =============================================================================
# CREATION OF NUMBERS AND SCORE FOR DASHBOARD
# =============================================================================

try:
    number_score = pn.indicators.Number(
        name='Total Investment Score', value=round(finance_df.loc[0,'score'],2), format='{value}',
        font_size='28px', title_size='16px'
    )
    
    number_debtratio = pn.indicators.Number(
        name='Debt to Equity ratio', value=round(finance_df.loc[0,'debtToEquity'],1), format='{value}%',
        font_size='28px', title_size='16px'
    )
    
    linear_gauge_roic = pn.indicators.LinearGauge(
        name='Return on Invested Capital (ROIC)', value=round(finance_df.loc[0,'roic'],2), bounds=(0, 60), format='{value}%',
        colors=[(0.083333333, '#ff6666'), (0.66666666, '#ffff66'), (1, '#85e085')], horizontal = True, show_boundaries=True, title_size='16px'
    )
    
    linear_gauge_evoverebit = pn.indicators.LinearGauge(
        name='Enterprise Value over EBIT', value=round(finance_df.loc[0,'enterprisevalue_over_ebit'],2), bounds=(0, 40), format='{value}',
        colors=[(0.25, '#85e085'), (0.395, '#ffff66'), (1, '#ff6666')], horizontal = True, show_boundaries=True, title_size='16px'
    )
    
    linear_gauge_growth = pn.indicators.LinearGauge(
        name='Minimum Growth (CAGR)', value=round(finance_df.loc[0,'lowest_growth'],2), bounds=(0, 5), format='{value}%',
        colors=[(0.25, '#ff6666'), (0.35, '#ffff66'), (1, '#85e085')], horizontal = True, show_boundaries=True, title_size='16px'
    )
    
    print("SUCCESS: Numbers successfully created.")
except Exception as e:
    print("\n","ERROR: Failure to create numbers.","\n", e)

# =============================================================================
# CREATION OF BUTTONS FOR DASHBOARD
# =============================================================================

try:
    button_1 = pn.widgets.CheckButtonGroup(name='Unstructured', button_type='success', button_style='outline', options=['Regulations', 'Competition', 'Competence'])
    button_2 = pn.widgets.CheckButtonGroup(name='Unstructured', button_type='warning', button_style='outline', options=['Regulations', 'Competition', 'Competence'])
    button_3 = pn.widgets.CheckButtonGroup(name='Unstructured', button_type='danger', button_style='outline', options=['Regulations', 'Competition', 'Competence'])
    layout_1 = pn.Column(button_1, button_2, button_3)
    
    
    button_21 = pn.widgets.CheckButtonGroup(name='Structured', button_type='success', button_style='outline', options=['Stable Growth', 'Low Debt', 'Reliability Earnings'])
    button_22 = pn.widgets.CheckButtonGroup(name='Structured', button_type='warning', button_style='outline', options=['Stable Growth',  'Low Debt', 'Reliability Earnings'])
    button_23 = pn.widgets.CheckButtonGroup(name='Structured', button_type='danger', button_style='outline', options=['Stable Growth', 'Low Debt', 'Reliability Earnings'])
    layout_2 = pn.Column(button_21, button_22, button_23)
    
    tabs = pn.Tabs((('Unstructured Filters', layout_1)),(('Structured Filters'), layout_2))
    
    print("SUCCESS: Buttons successfully created.")
except Exception as e:
    print("\n","ERROR: Failure to create buttons.","\n", e)
    
# =============================================================================
# CREATION OF SIDEBAR FOR DASHBOARD
# =============================================================================

text = f"""
#  {company_information_df.loc[0, 'companyName']}

### Finance
Ticker symbol: **{company_information_df.loc[0, 'symbol']}**
Exchange: **{company_information_df.loc[0, 'exchangeShortName']}**
Currency: **{company_information_df.loc[0, 'currency']}**
Current price: **{company_information_df.loc[0, 'price']} {company_information_df.loc[0, 'currency']}**
Range (TTM): **{company_information_df.loc[0, 'range']} {company_information_df.loc[0, 'currency']}**
Changes: **{company_information_df.loc[0, 'changes']}%**

### Insides
Industry: **{company_information_df.loc[0, 'industry']}**
Sector: **{company_information_df.loc[0, 'sector']}**
Head of the company: **{company_information_df.loc[0, 'ceo']}**
Full time employees: **{company_information_df.loc[0, 'fullTimeEmployees']}**

### Description
*{company_information_df.loc[0, 'description']}*

### Location
**{company_information_df.loc[0, 'companyName']}**
{company_information_df.loc[0, 'address']}
{company_information_df.loc[0, 'zip']} {company_information_df.loc[0, 'city']}, ({company_information_df.loc[0, 'state']}), {company_information_df.loc[0, 'country']}

### Contact
{company_information_df.loc[0, 'phone']} or {company_information_df.loc[0, 'website']}
"""

explanation = """
### References
*DISCLAIMER:* No guarantee can be given with regard to the correctness, accuracy, up-to-dateness, reliability and completeness of this information.
The visualized data is sourced from [this Finance API](https://site.financialmodelingprep.com). The curated and visualized financial data is inspired by the [magic formula](https://en.wikipedia.org/wiki/Magic_formula_investing) from [Mr. Joel Greenblatt](https://en.wikipedia.org/wiki/Joel_Greenblatt) and the [value investing principles](https://en.wikipedia.org/wiki/Value_investing) from [Benjamin Graham](https://en.wikipedia.org/wiki/Benjamin_Graham).
"""

sidebar = pn.layout.WidgetBox(
    pn.pane.Markdown(text, margin=(0, 10)),
    explanation,
    max_width=350,
    sizing_mode='stretch_width'
).servable(area='sidebar')

sidebar

# =============================================================================
# IMPORT OF LOGO FOR HEADER
# =============================================================================

logo_path = '/Users/fabiothon/Desktop/Code/API_Finance/logo.png'
logo = Image.open(logo_path)

# Convertion from Pillow to BytesIO
logo_bytes = BytesIO()
logo.save(logo_bytes, format='PNG')
logo_data = logo_bytes.getvalue()

# Creation of Panel Logo
logo_component = pn.pane.PNG(logo_data, width=90, height=90)

header = pn.Row(
    pn.layout.Spacer(width=400),
    logo_component,
    pn.layout.Spacer(),
)

# =============================================================================
# CREATION OF DASHBOARD
# =============================================================================

template = pn.template.BootstrapTemplate(
    title="Investment Portfolio",
    header_background=accent,
    header=header,
    sidebar=sidebar,
    main=[
        pn.Row(
            pn.Column(
                pn.Card(linear_gauge_roic, linear_gauge_evoverebit, linear_gauge_growth, number_score, number_debtratio, tabs),
                align='start',
                sizing_mode='stretch_width',
                max_width = 360
            ),
            pn.Column(
                pn.Card(revenue_chart, earning_chart, operating_margin_chart),
                align='start',
                sizing_mode='stretch_width'
            ),
            align='start',
            sizing_mode='stretch_width'
        ),
        pn.Row(
            pn.Card(daily_chart_eod),
            align='start',
            sizing_mode='stretch_width'
        ),
        pn.Row(
            pn.Card(tech_chart),
            align='start',
            sizing_mode='stretch_width',
            height=1200
        ),
        pn.Row(
            pn.Card(montecarlo_chart),
            align='start',
            sizing_mode='stretch_width'
        ),
        pn.Row(
            pn.Card(pn.Row(montecarlo_hist_chart, montecarlo_box_chart)),
            align='start',
            sizing_mode='stretch_width'
        )
    ]
)

template.show()

print('*********** END OF SCRIPT ***********')
# =============================================================================
# END OF SCRIPT
# =============================================================================


# =============================================================================
# NOTES:
# =============================================================================
# =============================================================================
# CHANGELOG
#
# 1. Add logging so that errors and nominal runs can be tracked better (loguru)
# 2. Find a better solution to save a temp file -> File or Server (SQLite)
# 3. Clustering Marketdata (YahooFinance?)
#
# =============================================================================

# =============================================================================
# # UNSTRUCTURED FILTERS
# Regulations
# Competition
# Circle of competence
# =============================================================================

# =============================================================================
# STRUCTURED FILTERS
# Too low growth -> Can be seen in the lowest growth parameter
# Too shorted stock -> Percentage or No data available
# Too much debt -> Net debt over the last 5 years
# Unreliable Earnings -> Histogram last 5 years (including: Operating margins)
# =============================================================================

# =============================================================================
# KEY PARAMETER NOTES
# lowest_growth (20%)             US Median = 1%          Good >2% Bad <0%
# roic (20%)                      US Median = 24%         Good >40%   Bad <5%
# EnterprisevalueoverEBIT (40%)   US Median = 15.8        Good <10    Bad >10
# Insider purchases (20%)         US Average = 0.084%     Good >0.1%  Bad <0.08%
# =============================================================================












