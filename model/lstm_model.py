import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

def predict_stock_price(ticker):
    df = yf.download(ticker, period="5y")
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

    # Plotting
    plot_path = f'static/{ticker}_plot.png'
    plt.figure(figsize=(8, 4))
    plt.plot(data, label='Real Price')
    plt.axhline(price, color='red', linestyle='--', label=f'Predicted: ${price:.2f}')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return round(price, 2), plot_path
