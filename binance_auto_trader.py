import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import time

# Binance Testnet API credentials
api_key = 'your_testnet_api_key'
secret_key = 'your_testnet_api_secret'

# Initialize the Binance Testnet client
client = Client(api_key=api_key, api_secret=secret_key, tld="com", testnet=True)

# Trading parameters
symbol = "BTCUSDT"
leverage = 15
size = 0.01  # Adjust position size
lookback = 60  # Lookback period for training the model
epochs = 10  # Number of epochs for LSTM model training
batch_size = 32

# Set leverage for futures positions
def set_leverage(symbol, leverage):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        print(f"Leverage set to {leverage}x for {symbol}")
    except Exception as e:
        print(f"Error setting leverage: {e}")

# Fetch historical price data
def get_historical_data(symbol, interval="1m", limit=100):
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            "time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
            "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["close"] = df["close"].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

# Preprocess data for LSTM
def preprocess_data(data, lookback=60):
    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['close']])

    # Create dataset with lookback
    X, y = [], []
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i-lookback:i, 0])  # use previous `lookback` prices
        y.append(1 if data_scaled[i, 0] > data_scaled[i-1, 0] else 0)  # 1 for buy (up), 0 for sell (down)

    X, y = np.array(X), np.array(y)

    # Reshape for LSTM [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Create and compile the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))  # Binary classification (1 for long, 0 for short)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Place a futures market order
def place_order(symbol, side, quantity, reduce_only=False):
    try:
        print(f"Placing {'reduce-only ' if reduce_only else ''}{side} market order for {quantity} {symbol}...")
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=quantity,
            reduceOnly=reduce_only
        )
        print("Order placed successfully:", order)
    except Exception as e:
        print(f"Order failed: {e}")

# Check if there is an open position
def check_open_position(symbol):
    try:
        positions = client.futures_position_information(symbol=symbol)
        for position in positions:
            if float(position['positionAmt']) != 0:
                return True  # There is an open position
        return False  # No open position
    except Exception as e:
        print(f"Error checking open position: {e}")
        return False

# Main trading function
def main():
    # Set leverage
    set_leverage(symbol, leverage)

    # Fetch initial historical data
    print("Fetching historical data...")
    data = get_historical_data(symbol, limit=100)
    if data is None:
        print("Failed to fetch data. Exiting.")
        return

    # Preprocess data
    X, y, scaler = preprocess_data(data, lookback=lookback)

    # Create LSTM model
    model = create_lstm_model((X.shape[1], 1))

    # Train the model
    print("Training model...")
    model.fit(X, y, epochs=epochs, batch_size=batch_size)

    # Auto-trading loop
    print("\n--- Starting Auto-Trading ---")
    while True:
        try:
            # Check if there is an open position
            if check_open_position(symbol):
                print("There is already an open position. Skipping trade.")
                time.sleep(60)  # Wait before checking again
                continue

            # Fetch the latest data
            data = get_historical_data(symbol, limit=lookback)
            if data is None:
                time.sleep(60)
                continue

            # Preprocess the latest data
            X_new, _, _ = preprocess_data(data, lookback=lookback)

            # Predict the signal
            prediction = model.predict(X_new[-1].reshape(1, X_new.shape[1], 1))
            signal = "BUY" if prediction[0][0] > 0.5 else "SELL"
            print(f"Predicted signal: {signal}")

            # Open position logic
            if signal == "BUY":
                place_order(symbol, "BUY", size)
            elif signal == "SELL":
                place_order(symbol, "SELL", size)

            # Wait for some time before checking if the position should be closed
            time.sleep(60)

            # Close the position after some conditions (this can be based on a different strategy)
            print("Checking for position closure...")
            # Here you could implement more conditions for closing the position.
            # For simplicity, we will close it after every trade.
            place_order(symbol, "SELL" if signal == "BUY" else "BUY", size, reduce_only=True)

            # Wait before the next prediction
            time.sleep(60)
        except KeyboardInterrupt:
            print("Exiting trading bot.")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
