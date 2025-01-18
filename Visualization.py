import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

# Define tickers
green_tickers = ["NEE", "BEP", "IBDRY", "ENPH", "SEDG"]
brown_tickers = ["XOM", "CVX", "COP", "PSX", "MPC"]

# Define the date range for estimation and event windows
estimation_start = "2022-01-01"  # Historical period for OLS estimation
estimation_end = "2022-06-30"    # End of estimation window
event_window_start = "2022-07-01"
event_window_end = "2022-08-31"
event_dates = {"Brown Event": "2022-07-14", "Green Event": "2022-07-27"}
event_dates = {k: pd.to_datetime(v) for k, v in event_dates.items()}

# Download stock and market data
green_data = yf.download(green_tickers, start=estimation_start, end=event_window_end)['Close']
brown_data = yf.download(brown_tickers, start=estimation_start, end=event_window_end)['Close']
market_data = yf.download("^GSPC", start=estimation_start, end=event_window_end)['Close']  # S&P 500 as market proxy

# Calculate returns
green_returns = green_data.pct_change().dropna()
brown_returns = brown_data.pct_change().dropna()
market_returns = market_data.pct_change().dropna()

# Function to estimate alpha and beta using OLS
def estimate_market_model(returns, market_returns):
    X = sm.add_constant(market_returns)  # Add intercept
    betas = {}
    for ticker in returns.columns:
        y = returns[ticker]
        model = sm.OLS(y, X).fit()
        betas[ticker] = model.params  # Store alpha and beta
    return betas

# Estimate market model for green and brown indices
green_betas = estimate_market_model(green_returns.loc[estimation_start:estimation_end],
                                    market_returns.loc[estimation_start:estimation_end])
brown_betas = estimate_market_model(brown_returns.loc[estimation_start:estimation_end],
                                    market_returns.loc[estimation_start:estimation_end])

# Predict expected returns
def predict_returns(betas, market_returns):
    # Ensure market_returns is treated as a Series (1D)
    if isinstance(market_returns, pd.DataFrame):
        market_returns = market_returns.iloc[:, 0]
    
    expected_returns = {}
    for ticker, params in betas.items():
        # Use the column name dynamically from market_returns
        market_key = market_returns.name
        expected_returns[ticker] = params['const'] + params[market_key] * market_returns
    return pd.DataFrame(expected_returns, index=market_returns.index)

green_expected = predict_returns(green_betas, market_returns)
brown_expected = predict_returns(brown_betas, market_returns)

# Calculate abnormal returns
green_abnormal = green_returns.sub(green_expected, axis=0)
brown_abnormal = brown_returns.sub(brown_expected, axis=0)

# Create a folder for saving plots
output_folder = "Graphs"
os.makedirs(output_folder, exist_ok=True)

# Plot CARs
def plot_cars(abnormal_returns, title, event_dates, filename):
    car = abnormal_returns.cumsum()
    plt.figure(figsize=(10, 6))
    car = car[(car.index >= "2022-07-09") & (car.index <= "2022-08-01")]  # Zoom into the event window
    car.mean(axis=1).plot(label='Average CAR', color='blue')
    for event_name, event_date in event_dates.items():
        plt.axvline(event_date, linestyle='--', label=f'{event_name}',
                    color='brown' if 'Brown' in event_name else 'green')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Abnormal Return")
    plt.legend()
    plt.grid()
    output_file = os.path.join(output_folder, filename)
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.show()

# Plot and save results
plot_cars(green_abnormal, "Green Index CARs", event_dates, "Green_Index_CARs.png")
plot_cars(brown_abnormal, "Brown Index CARs", event_dates, "Brown_Index_CARs.png")

# Plot combined CARs for Green and Brown Indexes
def plot_combined_cars(green_abnormal, brown_abnormal, title, event_dates, filename):
    # Calculate CARs
    green_car = green_abnormal.cumsum().mean(axis=1)
    brown_car = brown_abnormal.cumsum().mean(axis=1)

    # Filter CARs for event window
    green_car = green_car[(green_car.index >= "2022-07-09") & (green_car.index <= "2022-08-01")]
    brown_car = brown_car[(brown_car.index >= "2022-07-09") & (brown_car.index <= "2022-08-01")]

    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(green_car, label='Green Index Average CAR', color='green')
    plt.plot(brown_car, label='Brown Index Average CAR', color='brown')

    # Mark event dates
    for event_name, event_date in event_dates.items():
        plt.axvline(event_date, linestyle='--', label=f'{event_name}',
                    color='brown' if 'Brown' in event_name else 'green')

    # Add title and labels
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Abnormal Return")
    plt.legend()
    plt.grid()

    # Save the plot
    output_file = os.path.join(output_folder, filename)
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.show()

# Call the combined plot function
plot_combined_cars(
    green_abnormal, brown_abnormal,
    "Comparison of Green and Brown Index CARs",
    event_dates,
    "Combined_CARs.png"
)
