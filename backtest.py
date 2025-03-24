import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
import os
import streamlit as st

def fetch_crypto_data(symbol, exchange, start_date, end_date, timeframe='4h', retries=3, delay=2):
    """Fetch historical crypto price data from an exchange using CCXT with error handling."""
    limit = 1000      # Max candles per API call (Binance limit)
    
    # Convert datetime to milliseconds timestamp
    since = int(start_date.timestamp() * 1000)
    
    all_data = []
    while True:
        attempt = 0
        while attempt < retries:
            try:
                # Fetch candles
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                
                # If no data returned, exit
                if not ohlcv:
                    st.warning(f"No data returned for {symbol}")
                    return pd.DataFrame()
                
                # Add fetched data to our collection
                all_data.extend(ohlcv)
                
                # If we got fewer candles than the limit, we've reached the end
                if len(ohlcv) < limit:
                    break
                
                # Update since to get the next batch
                since = ohlcv[-1][0] + 1
                
                # Check if we've reached or exceeded end_date
                last_candle_date = datetime.fromtimestamp(ohlcv[-1][0]/1000)
                if last_candle_date >= end_date:
                    break
                
                # Success, exit the retry loop
                break
                
            except Exception as e:
                attempt += 1
                st.warning(f"Error fetching {symbol}: {str(e)}. Retrying ({attempt}/{retries})...")
                time.sleep(delay)
                
        # If we failed all retries
        if attempt == retries:
            st.error(f"Failed to fetch data for {symbol} after {retries} attempts.")
            # Return whatever data we got so far, or empty DataFrame if none
            if not all_data:
                return pd.DataFrame()
            break
            
        # Rate limiting to avoid API restrictions
        time.sleep(delay)
        
        # Check if we've reached the end date
        if ohlcv and datetime.fromtimestamp(ohlcv[-1][0]/1000) >= end_date:
            break
    
    # If we collected data, convert to DataFrame
    if all_data:
        df = pd.DataFrame(all_data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
        df.set_index("Date", inplace=True)
        
        # Filter by date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Keep only the Close price and rename column to the symbol
        df = df[["Close"]].rename(columns={"Close": symbol})
        
        # Clean up data
        df.dropna(inplace=True)
        return df
    else:
        st.error(f"No data collected for {symbol}")
        return pd.DataFrame()

def simulate_portfolio(prices_df, target_weights, initial_capital, threshold):
    """Simulate portfolio value over time with threshold-based rebalancing."""
    assets = prices_df.columns.tolist()
    target_w = np.array(target_weights, dtype=float)
    
    # Normalize weights if they don't sum to 1
    if not np.isclose(target_w.sum(), 1.0):
        target_w = target_w / target_w.sum()
    
    # Calculate initial quantities based on initial prices and weights
    first_prices = prices_df.iloc[0].values.astype(float)
    quantities = (target_w * initial_capital) / first_prices

    portfolio_values = []
    events = []

    # Record initial allocation
    initial_event = {
        "time": prices_df.index[0],
        "type": "initial",
        "quantities": {asset: qty for asset, qty in zip(assets, quantities)},
        "portfolio_value": initial_capital
    }
    events.append(initial_event)

    # Simulate through time
    for t, (time, prices) in enumerate(prices_df.iterrows()):
        prices = prices.values.astype(float)
        
        # Calculate portfolio value
        total_value = float(np.dot(quantities, prices))
        portfolio_values.append(total_value)
        
        # Calculate current weights
        current_w = (quantities * prices) / total_value
        
        # Skip first day (no rebalancing needed)
        if t == 0:
            continue
            
        # Check if rebalancing is needed
        if np.any(np.abs(current_w - target_w) > threshold):
            # Record rebalancing event
            event = {
                "time": time,
                "type": "rebalance",
                "weights_before": current_w.copy(),
                "portfolio_value": total_value
            }
            
            # Rebalance
            quantities = (target_w * total_value) / prices
            
            # Record new quantities
            event["quantities_after"] = {asset: qty for asset, qty in zip(assets, quantities)}
            events.append(event)
    
    return np.array(portfolio_values), events, quantities

def run_backtest(portfolio, initial_capital, threshold, start_date, end_date, timeframe):
    # Initialize exchange with rate limit handling
    exchange = ccxt.binance({
        'enableRateLimit': True,  # This enables the built-in rate limiter
        'options': {
            'defaultType': 'spot',  # Use spot markets by default
        }
    })

    # Status updater
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Fetch Data for Each Asset
    dataframes = []
    for i, asset in enumerate(portfolio):
        if asset["type"] == "crypto":
            status_text.text(f"Fetching data for {asset['symbol']}...")
            progress_bar.progress((i + 1) / len(portfolio))
            try:
                df = fetch_crypto_data(
                    asset["symbol"], 
                    exchange, 
                    start_date, 
                    end_date,
                    timeframe
                )
                
                if not df.empty:
                    dataframes.append(df)
                else:
                    st.warning(f"No data available for {asset['symbol']} in the specified range.")
            except Exception as e:
                st.error(f"Error processing {asset['symbol']}: {str(e)}")

    if not dataframes:
        st.error("No data fetched for any assets. Check symbols and time range.")
        return None, None, None, None

    # Combine all asset data and handle missing values
    combined_df = pd.concat(dataframes, axis=1, join="outer").sort_index()
    combined_df.fillna(method="ffill", inplace=True)
    combined_df.dropna(inplace=True)

    if combined_df.empty:
        st.error("Combined dataframe is empty after handling missing values.")
        return None, None, None, None

    st.info(f"Data shape: {combined_df.shape}")
    st.info(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")

    # Run the Simulation
    weights = [asset["weight"] for asset in portfolio if asset["symbol"] in combined_df.columns]
    assets_included = [asset["symbol"] for asset in portfolio if asset["symbol"] in combined_df.columns]
    
    if len(weights) != len(combined_df.columns):
        st.warning("Warning: Some assets were not included in the simulation.")
        st.warning(f"Assets in simulation: {combined_df.columns.tolist()}")
    
    status_text.text("Running portfolio simulation...")
    portfolio_values, events, final_quantities = simulate_portfolio(combined_df, weights, initial_capital, threshold)
    
    # Clear progress indicators
    status_text.empty()
    progress_bar.empty()
    
    return combined_df, portfolio_values, events, final_quantities

def display_results(combined_df, portfolio_values, events, final_quantities):
    # Calculate Performance Metrics
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    cumulative_return = final_value / initial_value - 1
    
    # Calculate returns and metrics only if we have enough data points
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        avg_ret = returns.mean()
        volatility = returns.std()
        sharpe_ratio = (avg_ret / volatility) * np.sqrt(6 * 365) if volatility and not np.isnan(volatility) else float('nan')
    else:
        avg_ret = volatility = sharpe_ratio = float('nan')
    
    # Calculate drawdowns
    cum_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (cum_max - portfolio_values) / cum_max
    max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0
    
    # Display Results in Streamlit
    st.subheader("Backtest Performance Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Initial Capital", f"${initial_value:,.2f}")
        st.metric("Final Portfolio Value", f"${final_value:,.2f}")
        st.metric("Cumulative Return", f"{cumulative_return*100:.2f}%")
    
    with col2:
        st.metric("Annualized Sharpe Ratio", f"{sharpe_ratio:.2f}")
        st.metric("Maximum Drawdown", f"{max_drawdown*100:.2f}%")
        st.metric("Total Rebalances", f"{len(events)-1}")

    # Print initial allocation
    initial_event = events[0]
    init_time = initial_event["time"]
    init_quantities = initial_event["quantities"]
    
    st.subheader("Initial Portfolio Allocation")
    st.write(f"Initial positions at {init_time}:")
    
    init_data = []
    for asset, qty in init_quantities.items():
        init_data.append({"Asset": asset, "Quantity": f"{qty:.4f}"})
    
    st.table(pd.DataFrame(init_data))
    st.write(f"Initial Portfolio Value = ${initial_event['portfolio_value']:,.2f}")

    # Print rebalancing events
    if len(events) > 1:
        st.subheader("Rebalancing Events")
        
        for i, event in enumerate(events[1:], start=1):
            with st.expander(f"Rebalance #{i} at {event['time']}"):
                st.write(f"Portfolio Value = ${event['portfolio_value']:,.2f}")
                
                # Weights before rebalance
                weights_df = pd.DataFrame({
                    "Asset": combined_df.columns,
                    "Weight Before (%)": [f"{w*100:.2f}%" for w in event["weights_before"]]
                })
                st.write("Weights before rebalance:")
                st.table(weights_df)
                
                # Quantities after rebalance
                quantities_data = []
                for asset, qty in event["quantities_after"].items():
                    quantities_data.append({"Asset": asset, "Quantity After": f"{qty:.4f}"})
                
                st.write("Token quantities after rebalance:")
                st.table(pd.DataFrame(quantities_data))

    # Print token amount changes
    st.subheader("Token Amounts Comparison")
    
    token_changes = []
    for i, asset in enumerate(combined_df.columns):
        init_qty = init_quantities[asset]
        final_qty = final_quantities[i]
        pct_change = ((final_qty - init_qty) / init_qty) * 100 if init_qty != 0 else float('nan')
        
        token_changes.append({
            "Asset": asset,
            "Initial": f"{init_qty:.4f}",
            "Final": f"{final_qty:.4f}",
            "Change (%)": f"{pct_change:.2f}%"
        })
    
    st.table(pd.DataFrame(token_changes))

    # Plot portfolio value
    st.subheader("Portfolio Equity Curve")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot portfolio value
    ax1.plot(combined_df.index, portfolio_values, label="Portfolio Value", color='blue')
    ax1.set_title("Portfolio Equity Curve")
    ax1.set_ylabel("Portfolio Value (USD)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot drawdowns
    ax2.fill_between(combined_df.index, -drawdowns * 100, 0, color='red', alpha=0.3)
    ax2.plot(combined_df.index, -drawdowns * 100, color='red', label='Drawdown')
    ax2.set_title("Portfolio Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Crypto Portfolio Backtester", layout="wide")
    
    st.title("Crypto Portfolio Backtester")
    st.write("Backtest your crypto portfolio with threshold rebalancing")
    
    # Sidebar inputs
    st.sidebar.header("Portfolio Settings")
    
    # Date inputs
    start_date = st.sidebar.date_input(
        "Start Date",
        datetime(2023, 1, 1)
    )
    end_date = st.sidebar.date_input(
        "End Date",
        datetime(2023, 12, 31)
    )
    
    # Convert to datetime
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.min.time())
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1h", "4h", "1d", "1w"],
        index=1
    )
    
    # Initial capital
    initial_capital = st.sidebar.number_input(
        "Initial Capital (USD)",
        min_value=100,
        value=10000,
        step=100
    )
    
    # Rebalancing threshold
    threshold = st.sidebar.slider(
        "Rebalancing Threshold",
        min_value=0.01,
        max_value=0.20,
        value=0.04,
        step=0.01,
        format="%.2f"
    )
    
    # Portfolio management
    st.sidebar.header("Portfolio Assets")
    
    # Dynamic asset management
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = [
            {"symbol": "BTC/USDT", "type": "crypto", "weight": 0.20},
            {"symbol": "SOL/USDT", "type": "crypto", "weight": 0.35},
            {"symbol": "ETH/USDT", "type": "crypto", "weight": 0.05},
            {"symbol": "LINK/USDT", "type": "crypto", "weight": 0.05},
            {"symbol": "USDC/USDT", "type": "crypto", "weight": 0.35}
        ]
    
    # Add new asset button
    if st.sidebar.button("Add Asset"):
        st.session_state.portfolio.append({"symbol": "BTC/USDT", "type": "crypto", "weight": 0.10})
    
    # Asset management interface
    updated_portfolio = []
    total_weight = 0
    
    for i, asset in enumerate(st.session_state.portfolio):
        st.sidebar.markdown(f"**Asset #{i+1}**")
        col1, col2 = st.sidebar.columns([3, 1])
        
        with col1:
            symbol = st.text_input(f"Symbol {i}", value=asset["symbol"], key=f"symbol_{i}")
        
        with col2:
            weight = st.number_input(
                f"Weight {i}",
                min_value=0.0,
                max_value=1.0,
                value=float(asset["weight"]),
                step=0.05,
                format="%.2f",
                key=f"weight_{i}"
            )
        
        total_weight += weight
        
        # Remove asset button
        if st.sidebar.button(f"Remove Asset #{i+1}", key=f"remove_{i}"):
            continue
        
        updated_portfolio.append({
            "symbol": symbol,
            "type": "crypto",
            "weight": weight
        })
    
    # Update portfolio with changes
    st.session_state.portfolio = updated_portfolio
    
    # Weight warning
    if len(updated_portfolio) > 0 and not np.isclose(total_weight, 1.0, atol=0.01):
        st.sidebar.warning(f"Total weight is {total_weight:.2f}. It should sum to 1.0. Weights will be normalized.")
    
    # Run backtest button
    if st.sidebar.button("Run Backtest"):
        if len(updated_portfolio) == 0:
            st.error("Please add at least one asset to your portfolio.")
        else:
            with st.spinner("Running backtest..."):
                combined_df, portfolio_values, events, final_quantities = run_backtest(
                    updated_portfolio, 
                    initial_capital, 
                    threshold, 
                    start_date, 
                    end_date,
                    timeframe
                )
                
                if combined_df is not None:
                    display_results(combined_df, portfolio_values, events, final_quantities)
    
    # Initial instructions
    if 'portfolio' in st.session_state and len(st.session_state.portfolio) > 0:
        st.info("Configure your portfolio in the sidebar and click 'Run Backtest' to start.")
    else:
        st.warning("Please add at least one asset to your portfolio in the sidebar.")

if __name__ == "__main__":
    main()