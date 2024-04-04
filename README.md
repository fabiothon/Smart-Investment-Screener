# Smart Investment Screener

This Python code creates an interactive dashboard for smart investment screening, leveraging data fetched from a financial data API. The goal is to provide a comprehensive analysis of a single company based on its stock symbol. The dashboard includes various charts such as revenue history, earnings history, operating margin trends, historical stock prices, Monte Carlo simulation of stock prices, and MACD chart for technical analysis. Additionally, key financial indicators and scores like total investment score, debt-to-equity ratio, return on invested capital (ROIC), enterprise value over EBIT, and minimum growth rate are displayed. Users can also apply structured and unstructured filters using check buttons for further analysis. Company information including ticker symbol, exchange, currency, current price, industry, CEO, description, location, and contact details are displayed in a sidebar. The dashboard layout is organized using the BootstrapTemplate from Panel library, providing an intuitive and interactive interface for investment decision-making.

<img width="1464" alt="Screenshot 2024-04-03 at 13 47 22" src="https://github.com/fabiothon/Smart-Investment-Screener/blob/0cfab892b688413e425464c0a862374ffa8d98c8/picture_1.png">
<img width="1463" alt="https://github.com/fabiothon/Smart-Investment-Screener/blob/e53856470329947e877dea88b904b7447498d335/picture_2.png">
<img width="1466" alt="Screenshot 2024-04-03 at 13 48 28" src="https://github.com/fabiothon/Smart-Investment-Screener/blob/0cfab892b688413e425464c0a862374ffa8d98c8/picture_3.png">

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) and the provided requirments.txt file to install this script.

```bash
pip install -r /path/to/requirements.txt
`````

### API's
The API's used in the script are [Yahoo Finance](https://finance.yahoo.com), which is accessed via the yfinance library, and the [Financial Modeling Prep](https://site.financialmodelingprep.com), which is accessed directly. This script is using the free version of the Financial Modeling Prep API, leading to a limited amount of API calls per day, only end-of-day data, a 5 year limit of histrocial data and limited amount of geographical regions. You may notice some commented-out code within the script, which remains unused at present. However, it could become relevant of a paid plan is opted for in the future. Find more information on: [Financial Modeling Prep Pricing](https://site.financialmodelingprep.com/developer/docs/pricing).

## Usage

1. Download of all necessary files (main.py, logo.png, requirements.txt)
2. Install necessary libraries on your local environment or virtual environment via the requirement.txt
3. Get your own API key for FMP-API and either hardcode it into the script or use keyring (only for MacOS users)
4. Run application

## Contributing

Pull requests are welcome! For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
This script uses the [MIT](https://choosealicense.com/licenses/mit/) License.