import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load dataset
df = pd.read_csv("mars_ephemeris_20251120.csv")

# Use RA and DEC as features
data = df[["ra_hours", "dec_degrees"]].values

# Normalize data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# Prepare sequences
X, y = [], []
window = 7

for i in range(len(scaled) - window):
    X.append(scaled[i:i + window])
    y.append(scaled[i + window])

X = np.array(X)
y = np.array(y)

# Build LSTM model
model = Sequential([
LSTM(64, input_shape=(window, 2), return_sequences=False, implementation=2),
    Dense(2)
])

model.compile(optimizer="adam", loss="mse")

# Train model
model.fit(X, y, epochs=30, batch_size=8)

# Save model and scaler
model.save("mars_lstm.h5")
joblib.dump(scaler, "mars_scaler.pkl")

print("✅ LSTM model trained and saved successfully")
