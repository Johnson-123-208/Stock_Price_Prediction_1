import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

def predict_stock_price(ticker):
    df = yf.download(ticker, period="5y")
    if df.empty or 'Close' not in df.columns:
        raise ValueError(f"Could not download data for ticker: {ticker}")

    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    predicted_price = model.predict(np.array([X[-1]]))
    price = scaler.inverse_transform(predicted_price)[0][0]

    # Save Plot 1: Line Chart with Predicted Line
    plot1_path = f'static/{ticker}_line.png'
    plt.figure(figsize=(8, 4))
    plt.plot(data, label='Real Price')
    plt.axhline(price, color='red', linestyle='--', label=f'Predicted: ${price:.2f}')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot1_path)
    plt.close()

    # Save Plot 2: Moving Averages
    plot2_path = f'static/{ticker}_ma.png'
    ma20 = df['Close'].rolling(window=20).mean()
    ma50 = df['Close'].rolling(window=50).mean()
    plt.figure(figsize=(8, 4))
    plt.plot(df['Close'], label='Close')
    plt.plot(ma20, label='MA 20')
    plt.plot(ma50, label='MA 50')
    plt.title(f'{ticker} Moving Averages')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot2_path)
    plt.close()

    # Save Plot 3: RSI
    plot3_path = f'static/{ticker}_rsi.png'
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    plt.figure(figsize=(8, 4))
    plt.plot(rsi, label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.title(f'{ticker} RSI')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot3_path)
    plt.close()

    return round(price, 2), plot1_path, plot2_path, plot3_path
