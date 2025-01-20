import yfinance as yf
import pandas as pd
import os

# Define ticker
single_ticker = ["NEE"]

def fetch_financial_proxies(tickers):
    data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        data.append({
            'Ticker': ticker,
            'Beta': info.get('beta', None),
            'DebtToEquity': info.get('debtToEquity', None),
            'RevenueGrowth': info.get('revenueGrowth', None),
            'MarketCap': info.get('marketCap', None),
            'ReturnOnAssets': info.get('returnOnAssets', None),
            'ESG_Score': info.get('esgScore', None),
            'Sustainability_Score': info.get('sustainability', None)
        })
    return pd.DataFrame(data)

# Fetch data for the specified ticker
nee_proxies = fetch_financial_proxies(single_ticker)

'''# Save results to CSV
output_folder = "Results"
os.makedirs(output_folder, exist_ok=True)
output_file_path = os.path.join(output_folder, "nee_financial_proxies.csv")
nee_proxies.to_csv(output_file_path, index=False)

print(f"Financial proxy data saved to {output_file_path}")
'''
# Display the collected data
print(nee_proxies)
