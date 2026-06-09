import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", message=".*OpenSSL.*")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

# 1. Configurare
file_name = 'AAPL_data.csv'
model_name = 'model_aapl_gru.h5'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

df = pd.read_csv(os.path.join(project_root, 'data', file_name), index_col=0)
model = load_model(os.path.join(project_root, 'models', model_name))

dates = pd.to_datetime(df.index)
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Average']
data = df[feature_cols].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

window_size = 60
split_idx = int(len(scaled_data) * 0.8)

# Extragem datele și axa temporală pentru TOATĂ perioada post-antrenare (Setul de Testare)
test_data = scaled_data[split_idx - window_size:]
test_dates = dates[split_idx:]

X_test = []
y_actual = data[split_idx:, 5]

for i in range(window_size, len(test_data)):
    X_test.append(test_data[i - window_size:i])

X_test = np.array(X_test)
predictions = model.predict(X_test)

dummy_matrix = np.zeros((len(predictions), 6))
dummy_matrix[:, 5] = predictions.flatten()
inv_predictions = scaler.inverse_transform(dummy_matrix)[:, 5]

# Calcul metrici
rmse = np.sqrt(mean_squared_error(y_actual, inv_predictions))
mae = mean_absolute_error(y_actual, inv_predictions)
residuals_usd = y_actual - inv_predictions
ape_percentage = (np.abs(residuals_usd) / y_actual) * 100
mape = np.mean(ape_percentage)

# ==============================================================================
# GRAFIC MASTER: MARCAREA EXPLICITĂ A ORIZONTULUI DE PREDICȚIE
# ==============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [2.5, 1, 1]})

# --- PANELUL 1: PREȚURI NOMINALE ȘI DELIMITARE REZULTATE ---
ax1.plot(test_dates, y_actual, label="Preț Real Bursier (VWAP)", color='#1f77b4', linewidth=1.8)
ax1.plot(test_dates, inv_predictions, label=f"Predicție Model ({model_name.split('_')[-1].split('.')[0].upper()})",
         color='#d62728', linestyle='--', linewidth=1.6)

# LINIA DE DEMARCAȚIE FONDAMENTALĂ PENTRU REZULTATE
prediction_start_date = test_dates[0]
ax1.axvline(prediction_start_date, color='darkred', linestyle='-', linewidth=2.5,
            label=f"Trecere în Regim de Predicție Pură ({prediction_start_date.strftime('%Y-%m-%d')})")

# Evidențierea regimului de stress test (Bula AI 2024-2025)
ax1.axvspan(pd.to_datetime('2024-01-01'), pd.to_datetime('2025-12-31'),
            color='gold', alpha=0.12, label='Orizont Temporal Evaluat (Bula AI 2024-2025)')

# Adăugare text explicativ pe grafic pentru validarea în fața comisiei
ax1.text(0.3, 0.85, "ZONA DE PREDICȚIE DIRECTĂ\n(Date complet noi pentru rețea)",
         transform=ax1.transAxes, color='black', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='square,pad=0.3', facecolor='#fff2f2', edgecolor='red', alpha=0.7))

metric_text = f"Performanță Generală:\nRMSE: {rmse:.4f} USD\nMAE: {mae:.4f} USD\nMAPE: {mape:.2f}%"
ax1.text(0.02, 0.95, metric_text, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

ax1.set_title(f"Evaluarea Capabilității Predictive pe Orizontul de Testare ({file_name.split('_')[0]})", fontsize=14, fontweight='bold')
ax1.set_ylabel("Valoare Activ (USD)", fontsize=11)
ax1.legend(loc="lower right", fontsize=9)
ax1.grid(True, linestyle=':', alpha=0.5)

# --- PANELUL 2: ERORI REZIDUALE (USD) ---
ax2.plot(test_dates, residuals_usd, label="Eroare Nominală (USD)", color='#2ca02c', linewidth=1.2, alpha=0.8)
ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax2.axvline(prediction_start_date, color='darkred', linestyle='-', linewidth=2)
ax2.set_ylabel("Eroare (USD)", fontsize=11)
ax2.legend(loc="upper right")
ax2.grid(True, linestyle=':', alpha=0.5)

# --- PANELUL 3: ERORI PROCENTUALE LIVE (%) ---
ax3.plot(test_dates, ape_percentage, label="Eroare Procentuală Absolută (APE)", color='#9467bd', linewidth=1.2)
ax3.axhline(mape, color='darkviolet', linestyle=':', linewidth=1.2, label=f"Media MAPE ({mape:.2f}%)")
ax3.axvline(prediction_start_date, color='darkred', linestyle='-', linewidth=2)
ax3.set_xlabel("Orizont Temporal (An Calendaristic)", fontsize=11)
ax3.set_ylabel("Deviație (%)", fontsize=11)
ax3.legend(loc="upper right")
ax3.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()