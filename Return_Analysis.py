import pandas as pd
import statsmodels.api as sm
import yfinance as yf
import os
import numpy as np
import matplotlib.pyplot as plt
from stargazer.stargazer import Stargazer

# Define tickers and event dates
green_tickers = ["NEE", "BEP", "IBDRY", "ENPH", "SEDG"]
brown_tickers = ["XOM", "CVX", "COP", "PSX", "MPC"]

# Download stock data (Close prices only)
green_data = yf.download(green_tickers, start="2022-01-01", end="2022-08-31")["Close"]
brown_data = yf.download(brown_tickers, start="2022-01-01", end="2022-08-31")["Close"]
market_data = yf.download("^GSPC", start="2022-01-01", end="2022-08-31")["Close"]

# Calculate daily returns
green_returns = green_data.pct_change().dropna()
brown_returns = brown_data.pct_change().dropna()
market_returns = market_data.pct_change().dropna()

# Calculate rolling returns for 1-day and 3-day windows
green_returns_1d = green_returns
green_returns_3d = green_returns.rolling(window=3).sum().dropna()
brown_returns_1d = brown_returns
brown_returns_3d = brown_returns.rolling(window=3).sum().dropna()

# Merge control variables
control_vars = pd.DataFrame(index=green_returns.index)

# Add market returns as a control
control_vars['Market_Return'] = market_returns

# Add volatility (standard deviation of returns) as a proxy for risk
control_vars['Green_Volatility'] = green_returns.std(axis=1)
control_vars['Brown_Volatility'] = brown_returns.std(axis=1)

# Add trading volume data for additional control if available
green_volume = yf.download(green_tickers, start="2022-01-01", end="2022-08-31")["Volume"].mean(axis=1)
brown_volume = yf.download(brown_tickers, start="2022-01-01", end="2022-08-31")["Volume"].mean(axis=1)

control_vars['Green_Volume'] = green_volume.pct_change().dropna()
control_vars['Brown_Volume'] = brown_volume.pct_change().dropna()

# Prepare regression data
def prepare_regression_data(returns, control_vars):
    df = returns.mean(axis=1).reset_index()
    df.columns = ['Date', 'Return']
    df = df.merge(control_vars, on='Date', how='left').dropna()
    return df

# Prepare data for regression analysis
green_1d_data = prepare_regression_data(green_returns_1d, control_vars)
green_3d_data = prepare_regression_data(green_returns_3d, control_vars)
brown_1d_data = prepare_regression_data(brown_returns_1d, control_vars)
brown_3d_data = prepare_regression_data(brown_returns_3d, control_vars)

# Run regression analysis
def run_regression(df):
    X = df[['Market_Return', 'Green_Volatility', 'Green_Volume']]
    X = sm.add_constant(X)
    y = df['Return']
    model = sm.OLS(y, X).fit()
    return model

# Run regressions for 1-day and 3-day returns
green_1d_model = run_regression(green_1d_data)
green_3d_model = run_regression(green_3d_data)
brown_1d_model = run_regression(brown_1d_data)
brown_3d_model = run_regression(brown_3d_data)

# Print regression summaries
print("Green Firms - 1 Day Return:\n", green_1d_model.summary())
print("Green Firms - 3 Day Return:\n", green_3d_model.summary())
print("Brown Firms - 1 Day Return:\n", brown_1d_model.summary())
print("Brown Firms - 3 Day Return:\n", brown_3d_model.summary())

# Create results folder and save LaTeX table
output_folder = "Results"
os.makedirs(output_folder, exist_ok=True)

stargazer = Stargazer([green_1d_model, green_3d_model, brown_1d_model, brown_3d_model])
stargazer.title("Regression of 1-Day and 3-Day Returns with Control Variables")
stargazer.custom_columns(["Green 1-Day", "Green 3-Day", "Brown 1-Day", "Brown 3-Day"], [1, 1, 1, 1])
stargazer.significant_digits(3)
stargazer.show_degrees_of_freedom(False)
stargazer.show_r2 = True
stargazer.show_adj_r2 = True
stargazer.show_f_statistic = True

# Save LaTeX table
latex_output = stargazer.render_latex()
output_file_path = os.path.join(output_folder, "control_variable_event_analysis.tex")

with open(output_file_path, "w") as file:
    file.write(latex_output)

print(f"LaTeX table saved in '{output_file_path}'.")

