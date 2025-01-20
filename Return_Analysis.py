import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import os
from stargazer.stargazer import Stargazer

# Define tickers and their ESG risk ratings
green_tickers = {
    "NEE": 25.05,
    "BEP": 15.82,
    "IBDRY": 16.92,
    "ENPH": 19.90,
    "SEDG": 15.31
}

brown_tickers = {
    "XOM": 43.66,
    "CVX": 38.36,
    "COP": 33.14,
    "PSX": 35.43,
    "MPC": 30.34
}

all_tickers = list(green_tickers.keys()) + list(brown_tickers.keys())

# Define event dates
event_dates = {"Brown Event": "2022-07-14", "Green Event": "2022-07-27"}
event_dates = {k: pd.to_datetime(v) for k, v in event_dates.items()}

# Download stock data (Close prices only)
start_date = "2022-01-01"
end_date = "2022-08-31"
stock_data = yf.download(all_tickers, start=start_date, end=end_date)["Close"]

# Function to calculate event window returns (1-day and 3-day windows)
def calculate_event_returns(data, event_dates, window):
    event_returns = []
    for event_name, event_date in event_dates.items():
        if event_date in data.index:
            try:
                start_date = event_date
                end_date = data.index[data.index.get_loc(event_date) + window]

                event_return = (data.loc[end_date] - data.loc[start_date]) / data.loc[start_date]
                for ticker in data.columns:
                    event_returns.append([start_date, event_name, ticker, event_return[ticker]])
            except IndexError:
                print(f"Warning: Not enough data for {event_name} with window {window}")

    return pd.DataFrame(event_returns, columns=['Date', 'Event', 'Ticker', 'Return'])

# Calculate 1-day and 3-day event returns separately for green and brown firms
green_returns_1d = calculate_event_returns(stock_data[list(green_tickers.keys())], event_dates, window=1)
green_returns_3d = calculate_event_returns(stock_data[list(green_tickers.keys())], event_dates, window=3)
brown_returns_1d = calculate_event_returns(stock_data[list(brown_tickers.keys())], event_dates, window=1)
brown_returns_3d = calculate_event_returns(stock_data[list(brown_tickers.keys())], event_dates, window=3)

# Prepare ESG data with Green and Brown dummy variables
esg_data = pd.DataFrame({
    "Ticker": all_tickers,
    "ESG Risk Rating": [green_tickers.get(t, brown_tickers.get(t, None)) for t in all_tickers],
    "Green Dummy": [1 if t in green_tickers else 0 for t in all_tickers],
    "Brown Dummy": [1 if t in brown_tickers else 0 for t in all_tickers]
})

# Merge event returns with ESG data separately for green and brown firms
green_returns_1d = green_returns_1d.merge(esg_data, on="Ticker", how="left")
green_returns_3d = green_returns_3d.merge(esg_data, on="Ticker", how="left")
brown_returns_1d = brown_returns_1d.merge(esg_data, on="Ticker", how="left")
brown_returns_3d = brown_returns_3d.merge(esg_data, on="Ticker", how="left")

# Fetch financial metrics using Yahoo Finance
def fetch_financial_proxies(tickers):
    data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        data.append({
            'Ticker': ticker,
            'Beta': info.get('beta', None),
            'Debt To Equity': info.get('debtToEquity', None),
            'Revenue Growth': info.get('revenueGrowth', None),
            'Market Cap': info.get('marketCap', None),
            'Return On Assets': info.get('returnOnAssets', None),
        })
    return pd.DataFrame(data)

financial_data = fetch_financial_proxies(all_tickers)

# Merge financial data with returns separately for green and brown firms
green_returns_1d = green_returns_1d.merge(financial_data, on="Ticker", how="left")
green_returns_3d = green_returns_3d.merge(financial_data, on="Ticker", how="left")
brown_returns_1d = brown_returns_1d.merge(financial_data, on="Ticker", how="left")
brown_returns_3d = brown_returns_3d.merge(financial_data, on="Ticker", how="left")

# Function to run regression analysis
def run_regression(df, dummy_col):
    X = df[['Beta', 'Debt To Equity', 'Revenue Growth', 'Market Cap', 'Return On Assets', dummy_col]]
    X = sm.add_constant(X)
    y = df['Return']
    X = X.apply(pd.to_numeric, errors='coerce').dropna()
    y = pd.to_numeric(y, errors='coerce').dropna()
    model = sm.OLS(y, X).fit()
    return model

# Run regression for 1-day and 3-day returns with respective dummy variables
green_model_1d = run_regression(green_returns_1d, 'Green Dummy')
green_model_3d = run_regression(green_returns_3d, 'Green Dummy')
brown_model_1d = run_regression(brown_returns_1d, 'Brown Dummy')
brown_model_3d = run_regression(brown_returns_3d, 'Brown Dummy')

# Print regression summaries
print("Green Firms - 1 Day Return:\n", green_model_1d.summary())
print("Green Firms - 3 Day Return:\n", green_model_3d.summary())
print("Brown Firms - 1 Day Return:\n", brown_model_1d.summary())
print("Brown Firms - 3 Day Return:\n", brown_model_3d.summary())

# Create results folder and save LaTeX output using Stargazer
output_folder = "Results"
os.makedirs(output_folder, exist_ok=True)

stargazer = Stargazer([green_model_1d, green_model_3d, brown_model_1d, brown_model_3d])
stargazer.title("Determinants of Green and Brown Firms' Event Returns")
stargazer.custom_columns(["Green 1-Day", "Green 3-Day", "Brown 1-Day", "Brown 3-Day"], [1, 1, 1, 1])
stargazer.significant_digits(3)


# Automatically get correct covariate names
covariate_names = list(green_model_1d.params.index) + ["Brown Dummy"]
stargazer.covariate_order(covariate_names)

# Save the LaTeX output
latex_output = stargazer.render_latex().replace("_", " ")
output_file_path = os.path.join(output_folder, "Return_Analysis.tex")

with open(output_file_path, "w") as file:
    file.write(latex_output)

print(f"LaTeX table saved in '{output_file_path}'.")
