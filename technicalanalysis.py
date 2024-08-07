# =============================================================================
# TECHNICAL ANALYSIS
# =============================================================================

# STATUS: OBSOLETE

# This script is a seperate document that was used to develop a technical analysis (chart analysis)
# that was implemented into the main.py (Smart Investment Screener). The code is documented
# and commented in the main.py file and additional information is found in the README File!


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import yfinance as yf
from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.trend import MACD


# Just for testing
import plotly.io as pio
pio.renderers.default='browser'

# =============================================================================
# TECHNICAL ANALYSIS CHART
# =============================================================================

upwards_pyramid_symbol = "triangle-up"
downwards_pyramid_symbol = "triangle-down"

# BOLLINGER BANDS
def calculate_bb_indicators(technical_analysis_df):
    indicator_bb = BollingerBands(close=technical_analysis_df["Close"], window=20, window_dev=2)
    technical_analysis_df['bb_bbm'] = indicator_bb.bollinger_mavg()
    technical_analysis_df['bb_bbh'] = indicator_bb.bollinger_hband()
    technical_analysis_df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    return technical_analysis_df

# MACD INDICATORS
def calculate_MACD_indicators(technical_analysis_df):
    macd_object = MACD(technical_analysis_df['Close'])
    technical_analysis_df['MACD'] = macd_object.macd()
    technical_analysis_df['MACD_Signal'] = macd_object.macd_signal()
    technical_analysis_df['MACD_Diff'] = macd_object.macd_diff()
    technical_analysis_df['SMA50'] = technical_analysis_df['Close'].rolling(window=50).mean()
    technical_analysis_df['SMA200'] = technical_analysis_df['Close'].rolling(window=200).mean()
    
    return technical_analysis_df
    
# UPDATE FUNCTION
def update_traces(button_idx):
    visibility = [
        [True, True, False, False, False, False, False, False, False, True, True, True, True],    # None
        [True, True, True, True, True, False, False, False, False, True, True, True, True],       # MA
        [True, True, True, True, True, True, True, False, False, True, True, True, True],         # Buy/Sell
        [True, True, False, False, False, False, False, True, True, True, True, True, True],      # BB
        [True, True, True, True, True, True, True, True, True, True, True, True, True]]           # All
    
    return visibility[button_idx]


# Get stock symbol from user input
stock_symbol = input("Put in stock symbol: ")
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
tech_chart.update_layout(title_text=f"Chart Analysis: {stock_symbol} - None", xaxis_domain=[0.05, 1.0], yaxis_title=" Stock Price (X)", xaxis_rangeslider_visible=False,
xaxis_title="Time (Date)")

tech_chart.show()




