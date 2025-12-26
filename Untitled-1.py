cat > ~/SmartSky/backend/train_lstm.py << 'PY'
# train_lstm.py
"""
Train a simple LSTM to predict next-day RA (Right Ascension) for a planet.
Assumes CSV produced by fetch_data.py exists in the same folder and has columns:
date,mode,ra_hours,dec_degrees,distance_au,altitude_deg,azimuth_deg,distance_km
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
import joblib

# CONFIG
WINDOW = 7          # days of history used to predict next-day
EPOCHS = 80
BATCH_SIZE = 8
MODEL_FN = "mars_lstm.h5"
SCALER_FN = "mars_scaler.save"

def find_csv():
    files = sorted([f for f in os.listdir('.') if f.startswith('mars_ephemeris_') and f.endswith('.csv')])
    if not files:
        raise FileNotFoundError("No mars_ephemeris_*.csv file found in current directory.")
    return files[-1]

def load_series(csvfile):
    df = pd.read_csv(csvfile, parse_dates=["date"])
    df = df[df['mode'] == 'radec'].reset_index(drop=True)
    if df.empty:
        raise ValueError("No RA/Dec rows (mode=='radec') found in CSV. Use fetch_data.py without lat/lon.")
    df = df.sort_values('date')
    ra_hours = df['ra_hours'].astype(float).values  # 0-24 range
    return df, ra_hours

def ra_hours_to_unwrapped_radians(ra_hours):
    # convert hours (0..24) to radians (0..2pi)
    ra_rad = (ra_hours / 24.0) * 2 * np.pi
    # unwrap to make time-series continuous
    ra_unwrapped = np.unwrap(ra_rad)
    return ra_unwrapped

def create_windows(series, window):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    X = np.array(X)
    y = np.array(y)
    return X, y

def build_model(window):
    model = Sequential([
        LSTM(64, input_shape=(window, 1), return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    csvfile = find_csv()
    print("Using CSV:", csvfile)
    df, ra_hours = load_series(csvfile)
    ra_unwrapped = ra_hours_to_unwrapped_radians(ra_hours).reshape(-1, 1)

    # scale
    scaler = StandardScaler()
    scaled = scaler.fit_transform(ra_unwrapped)

    # windows
    X_raw, y_raw = create_windows(scaled.flatten(), WINDOW)
    X = X_raw.reshape((X_raw.shape[0], X_raw.shape[1], 1))
    y = y_raw.reshape(-1, 1)

    # split
    split = int(0.9 * len(X)) if len(X) > 10 else int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Dataset shape: X={X.shape}, y={y.shape}; Train={X_train.shape[0]}, Val={X_val.shape[0]}")

    model = build_model(WINDOW)
    checkpoint = ModelCheckpoint(MODEL_FN, save_best_only=True, monitor='val_loss', mode='min')
    early = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val)>0 else None,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early],
        verbose=2
    )

    # Save scaler
    joblib.dump(scaler, SCALER_FN)
    print("Saved scaler ->", SCALER_FN)
    print("Best model saved as ->", MODEL_FN)

    # sample prediction using last window
    last_window = scaled[-WINDOW:].reshape(1, WINDOW, 1)
    pred_scaled = model.predict(last_window)
    pred_unscaled = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
    pred_hours = (pred_unscaled / (2 * np.pi)) * 24.0
    print(f"Sample next-day prediction (RA hours, unwrapped): {pred_hours:.6f}")

    # save small CSV comparing last actual vs predicted
    try:
        recent_dates = df['date'].dt.strftime('%Y-%m-%d').values
        last_actual_hours = df['ra_hours'].values[-1]
        with open("train_prediction_sample.csv", "w") as f:
            f.write("last_date,last_actual_ra_hours,predicted_ra_hours\n")
            f.write(f"{recent_dates[-1]},{last_actual_hours:.6f},{pred_hours:.6f}\n")
        print("Saved train_prediction_sample.csv")
    except Exception:
        pass

if __name__ == "__main__":
    main()
PY