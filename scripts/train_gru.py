import os
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

warnings.filterwarnings("ignore")

# 1. Asset and Data Configuration
file_name = 'JPM_data.csv'
model_name = 'model_jpm_gru.h5'

# Define strict calendar boundaries for the training phase
train_start_date = '2014-01-01'
train_end_date = '2022-12-31'
window_size = 60

# Resolve absolute path setup dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if "scripts" in script_dir else script_dir

# Load dataset
data_path = os.path.join(project_root, 'data', file_name)
df = pd.read_csv(data_path, index_col=0)
df.index = pd.to_datetime(df.index)
print(f"Dataset successfully loaded from: {data_path}", flush=True)

# Feature extraction and normalization
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Average']
matrix_data = df[feature_cols].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(matrix_data)

# Reconstruct data structure using sliding lookback windows
X, y = [], []
for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size:i])
    y.append(scaled_data[i, 5])  # Index 5 maps directly to the 'Average' target column

X, y = np.array(X), np.array(y)

# Isolate the precise slice of time matching the sliding window index matrix
dates_for_X = df.index[window_size:]

# Separate training and testing sets using target date boundaries
train_mask = (dates_for_X >= train_start_date) & (dates_for_X <= train_end_date)
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]

print(f"Training structural span: {train_start_date} -> {train_end_date} ({len(X_train)} samples)")
print(f"Testing structural span: Remaining validation horizon ({len(X_test)} samples)", flush=True)

# 2. Build and Train the Recurrent Neural Network
model = Sequential([
    GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    GRU(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
print(f"\nInitializing optimized GRU training process for {file_name}...", flush=True)
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Export weights directly to the models directory
models_dir = os.path.join(project_root, 'models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, model_name)

model.save(model_path)
print(f"Model successfully saved to absolute path: {model_path}", flush=True)