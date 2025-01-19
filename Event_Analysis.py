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
event_dates = {"Brown Event": "2022-07-14", "Green Event": "2022-07-27"}
event_dates = {k: pd.to_datetime(v) for k, v in event_dates.items()}

# Download stock data (Close prices only)
green_data = yf.download(green_tickers, start="2022-01-01", end="2022-08-31")["Close"]
brown_data = yf.download(brown_tickers, start="2022-01-01", end="2022-08-31")["Close"]

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

# Calculate 1-day and 3-day event returns for green and brown firms
green_returns_1d = calculate_event_returns(green_data, event_dates, window=1)
green_returns_3d = calculate_event_returns(green_data, event_dates, window=3)
brown_returns_1d = calculate_event_returns(brown_data, event_dates, window=1)
brown_returns_3d = calculate_event_returns(brown_data, event_dates, window=3)

# Function to prepare regression data
def prepare_regression_data(green_returns, brown_returns):
    df_green = green_returns.copy()
    df_brown = brown_returns.copy()

    df_green['Green_Event'] = df_green['Event'].apply(lambda x: 1 if "Green" in x else 0)
    df_green['Brown_Event'] = df_green['Event'].apply(lambda x: 1 if "Brown" in x else 0)

    df_brown['Green_Event'] = df_brown['Event'].apply(lambda x: 1 if "Green" in x else 0)
    df_brown['Brown_Event'] = df_brown['Event'].apply(lambda x: 1 if "Brown" in x else 0)

    return df_green[['Return', 'Green_Event', 'Brown_Event']], df_brown[['Return', 'Green_Event', 'Brown_Event']]

# Prepare regression data for each case separately
green_1d_data, brown_1d_data = prepare_regression_data(green_returns_1d, brown_returns_1d)
green_3d_data, brown_3d_data = prepare_regression_data(green_returns_3d, brown_returns_3d)

# Run regression analysis for Green and Brown events as independent variables
def run_regression(df):
    X = df[['Green_Event', 'Brown_Event']]
    X = sm.add_constant(X)
    y = df['Return']
    X = X.apply(pd.to_numeric, errors='coerce').dropna()
    y = pd.to_numeric(y, errors='coerce').dropna()
    model = sm.OLS(y, X).fit()
    return model

# Run regressions for green and brown firms separately
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
stargazer.title("Impact of Green and Brown Events on Energy Firm Returns")
stargazer.custom_columns(["Green 1-Day", "Green 3-Day", "Brown 1-Day", "Brown 3-Day"], [1, 1, 1, 1])
stargazer.significant_digits(3)
stargazer.covariate_order(['const', 'Green_Event', 'Brown_Event'])
stargazer.add_line("Event Dates", ["2022-07-27", "2022-07-27", "2022-07-14", "2022-07-14"])
stargazer.show_degrees_of_freedom(False)
stargazer.show_r2 = True
stargazer.show_adj_r2 = True
stargazer.show_f_statistic = True

# Save LaTeX table
latex_output = stargazer.render_latex()
output_file_path = os.path.join(output_folder, "split_event_impact_results.tex")

with open(output_file_path, "w") as file:
    file.write(latex_output)

print(f"LaTeX table saved in '{output_file_path}'.")

# Function to plot combined cumulative percentage returns for green and brown firms
def plot_combined_closing_returns(green_data, brown_data, title, event_dates, filename):
    plt.figure(figsize=(12, 8))

    green_filtered = green_data.loc["2022-07-01":"2022-08-01"]
    brown_filtered = brown_data.loc["2022-07-01":"2022-08-01"]

    green_pct_returns = green_filtered.pct_change().dropna().cumsum() * 100
    brown_pct_returns = brown_filtered.pct_change().dropna().cumsum() * 100

    green_pct_returns.mean(axis=1).plot(label='Green Firms', color='green')
    brown_pct_returns.mean(axis=1).plot(label='Brown Firms', color='brown')

    for event_name, event_date in event_dates.items():
        if event_date in green_pct_returns.index:
            plt.axvline(event_date, linestyle='--', label=f'{event_name}',
                        color='brown' if 'Brown' in event_name else 'green')

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative % Return")
    plt.legend()
    plt.grid()

    output_folder = "Graphs"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, filename)
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.show()

plot_combined_closing_returns(
    green_data, 
    brown_data, 
    "Comparison of Green and Brown Firms Cumulative % Return (July 1 - Aug 1)", 
    event_dates, 
    "Combined_Green_Brown_Closing_Returns.png"
)
