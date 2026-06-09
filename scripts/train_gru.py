import os
import sys
import warnings

# Eliminare avertismente OpenSSL/urllib3 de pe macOS pentru o consolă curată
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", message=".*OpenSSL.*")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

# Configurare Active (Modifică aici numele fișierului pentru celelalte companii)
file_name = 'NVDA_data.csv'
model_name = 'model_nvda_gru.h5'

# REPARARE FILE_NOT_FOUND: Determinăm folderul rădăcină al proiectului în mod dinamic
script_dir = os.path.dirname(os.path.abspath(__file__)) # Calea către folderul 'scripts'
project_root = os.path.dirname(script_dir)             # Urcăm un nivel în rădăcina proiectului

data_path = os.path.join(project_root, 'data', file_name)
df = pd.read_csv(data_path, index_col=0)
print(f"--- [OK] Date încărcate cu succes din: {data_path} ---", flush=True)

# Selectăm exact cele 6 coloane obligatorii pentru antrenare
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Average']
data = df[feature_cols].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

X, y = [], []
window_size = 60

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size:i])
    y.append(scaled_data[i, 5])  # Indexul 5 este coloana 'Average'

X, y = np.array(X), np.array(y)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential([
    GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    GRU(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
print(f"\n--- Pornire Antrenare GRU pentru {file_name} ---", flush=True)
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Salvare model folosind calea absolută către folderul 'models'
models_dir = os.path.join(project_root, 'models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, model_name)
model.save(model_path)
print(f"[SUCCES] Model salvat în mod absolut: {model_path}", flush=True)