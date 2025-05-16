import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 1. Load CSV data
df = pd.read_csv("data.csv")  # Make sure your CSV has a 'value' column
data = df['value'].values.reshape(-1, 1)  # Reshape to 2D for scaler

# 2. Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 3. Create sequences (sliding window)
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_length = 10  # Last 10 values to predict next one
X, y = create_sequences(scaled_data, seq_length)

# 4. Build the LSTM model
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(seq_length, 1)),
    Dense(1)  # Predict single value
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 5. Train the model
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

# 6. Predict and inverse transform
predictions = model.predict(X)
predictions_actual = scaler.inverse_transform(predictions)
y_actual = scaler.inverse_transform(y)

# 7. Print a few predictions
print("\nPredicted vs Actual:")
for pred, actual in zip(predictions_actual[:5], y_actual[:5]):
    print(f"Predicted: {pred[0]:.2f}, Actual: {actual[0]:.2f}")
what does this do and help explain the code completely
