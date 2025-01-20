import pandas as pd
import statsmodels.api as sm
import yfinance as yf
import os
import matplotlib.pyplot as plt
from stargazer.stargazer import Stargazer

# Define tickers and event dates
green_tickers = ["NEE", "BEP", "IBDRY", "ENPH", "SEDG"]
brown_tickers = ["XOM", "CVX", "COP", "PSX", "MPC"]
event_dates = {
    "Brown Event": pd.to_datetime("2022-07-14"),
    "Green Event": pd.to_datetime("2022-07-27")
}

# Download stock data (Close prices only)
green_data = yf.download(green_tickers, start="2022-01-01", end="2022-10-31")["Close"]
brown_data = yf.download(brown_tickers, start="2022-01-01", end="2022-10-31")["Close"]

# Function to calculate event window returns
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

# Calculate 30-day and 60-day event returns for both Green and Brown firms
green_returns_30d = calculate_event_returns(green_data, event_dates, window=30)
green_returns_60d = calculate_event_returns(green_data, event_dates, window=60)
brown_returns_30d = calculate_event_returns(brown_data, event_dates, window=30)
brown_returns_60d = calculate_event_returns(brown_data, event_dates, window=60)

# Function to prepare regression data with Green and Brown event indicators
def prepare_regression_data(green_returns, brown_returns):
    df_green = green_returns.copy()
    df_brown = brown_returns.copy()

    df_green['Green_Event'] = df_green['Event'].apply(lambda x: 1 if "Green" in x else 0)
    df_green['Brown_Event'] = df_green['Event'].apply(lambda x: 1 if "Brown" in x else 0)

    df_brown['Green_Event'] = df_brown['Event'].apply(lambda x: 1 if "Green" in x else 0)
    df_brown['Brown_Event'] = df_brown['Event'].apply(lambda x: 1 if "Brown" in x else 0)

    return df_green[['Return', 'Green_Event', 'Brown_Event']], df_brown[['Return', 'Green_Event', 'Brown_Event']]

# Prepare regression data for all scenarios
green_30d_data, brown_30d_data = prepare_regression_data(green_returns_30d, brown_returns_30d)
green_60d_data, brown_60d_data = prepare_regression_data(green_returns_60d, brown_returns_60d)

# Run regression analysis with both events as independent variables
def run_regression(df):
    X = df[['Green_Event', 'Brown_Event']]
    X = sm.add_constant(X)
    y = df['Return']
    X = X.apply(pd.to_numeric, errors='coerce').dropna()
    y = pd.to_numeric(y, errors='coerce').dropna()
    model = sm.OLS(y, X).fit()
    return model

# Run regressions for green and brown firms separately for 30-day and 60-day returns
green_30d_model = run_regression(green_30d_data)
green_60d_model = run_regression(green_60d_data)
brown_30d_model = run_regression(brown_30d_data)
brown_60d_model = run_regression(brown_60d_data)

# Print regression summaries
print("Green Firms - 30 Day Return:\n", green_30d_model.summary())
print("Green Firms - 60 Day Return:\n", green_60d_model.summary())
print("Brown Firms - 30 Day Return:\n", brown_30d_model.summary())
print("Brown Firms - 60 Day Return:\n", brown_60d_model.summary())

# Create results folder and save LaTeX table
output_folder = "Results"
os.makedirs(output_folder, exist_ok=True)

# Stargazer table for regression results
stargazer = Stargazer([green_30d_model, green_60d_model, brown_30d_model, brown_60d_model])
stargazer.title("Long-Term Impact of the IRA")
stargazer.custom_columns(["Green 30-Day", "Green 60-Day", "Brown 30-Day", "Brown 60-Day"], [1, 1, 1, 1])
stargazer.significant_digits(3)
stargazer.covariate_order(['const', 'Green_Event', 'Brown_Event'])
stargazer.add_line("Event Dates", ["2022-07-27", "2022-07-27", "2022-07-14", "2022-07-14"])
stargazer.show_degrees_of_freedom(False)
stargazer.show_r2 = True
stargazer.show_adj_r2 = True
stargazer.show_f_statistic = True

# Save LaTeX table
latex_output = stargazer.render_latex()
output_file_path = os.path.join(output_folder, "Long_Term_Event_Analysis.tex")

with open(output_file_path, "w") as file:
    file.write(latex_output)

print(f"LaTeX table saved in '{output_file_path}'.")

# Function to plot cumulative percentage returns with both events included
def plot_combined_closing_returns(green_data, brown_data, title, event_dates, filename):
    plt.figure(figsize=(14, 8))

    # Define analysis period (July 1st to 60 days after the Green Event)
    analysis_start = "2022-07-01"
    analysis_end = "2022-09-25"

    green_filtered = green_data.loc[analysis_start:analysis_end]
    brown_filtered = brown_data.loc[analysis_start:analysis_end]

    green_pct_returns = green_filtered.pct_change().dropna().cumsum() * 100
    brown_pct_returns = brown_filtered.pct_change().dropna().cumsum() * 100

    # Plot cumulative average returns for Green and Brown firms
    green_pct_returns.mean(axis=1).plot(label='Green Firms', color='green')
    brown_pct_returns.mean(axis=1).plot(label='Brown Firms', color='brown')

    # Add vertical lines for event dates
    for event_name, event_date in event_dates.items():
        plt.axvline(event_date, linestyle='--', label=event_name, 
                    color='green' if 'Green' in event_name else 'brown')

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative % Return")
    plt.legend()
    plt.grid()

    output_file = os.path.join(output_folder, filename)
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.show()

plot_combined_closing_returns(
    green_data, 
    brown_data, 
    "Long-Term Cumulative % Return of Green and Brown Firms", 
    event_dates, 
    "Long Term Returns.png"
)
