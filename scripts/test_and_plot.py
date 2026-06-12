import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Suppress redundant system and third-party library warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", message=".*OpenSSL.*")

# Prevent macOS thread locks and memory allocation conflicts within TensorFlow
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

# 1. Asset and Model Configuration
file_name = 'JPM_data.csv'
model_name = 'model_jpm_lstm.h5'

# Resolve absolute paths dynamically based on the directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if "scripts" in script_dir else script_dir

# Load the historical dataset and the pre-trained recurrent model
df = pd.read_csv(os.path.join(project_root, 'data', file_name), index_col=0)
model = load_model(os.path.join(project_root, 'models', model_name), compile=False)

dates = pd.to_datetime(df.index)
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Average']
data = df[feature_cols].values

# Normalize the input matrix across all 6 technical features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

window_size = 60

# DYNAMIC SPLIT: Locate the exact index where the year 2023 begins
target_split_date = pd.to_datetime('2023-01-01')
split_idx = np.where(dates >= target_split_date)[0][0]

# Extract the complete historical target series (VWAP Average) from 2014 to 2026
y_all_actual = data[:, 5]

# Isolate and format the testing sub-dataset using sliding lookback windows
test_data = scaled_data[split_idx - window_size:]
X_test = []
for i in range(window_size, len(test_data)):
    X_test.append(test_data[i - window_size:i])
X_test = np.array(X_test)

# Execute model inference exclusively over the unseen test horizon (from 2023 to 2026)
predictions = model.predict(X_test, verbose=0)

# Invert matrix scaling using a dummy framework to reconstruct true USD values
dummy_matrix = np.zeros((len(predictions), 6))
dummy_matrix[:, 5] = predictions.flatten()
inv_predictions = scaler.inverse_transform(dummy_matrix)[:, 5]

# Align predictions onto a global timeline matrix padded with NaN for the training phase
y_all_pred = np.empty_like(y_all_actual)
y_all_pred[:] = np.nan
y_all_pred[split_idx:] = inv_predictions

# ==============================================================================
# X_train_full = []
# for i in range(window_size, split_idx):
#     X_train_full.append(scaled_data[i - window_size:i])
# X_train_full = np.array(X_train_full)
# train_predictions = model.predict(X_train_full, verbose=0)
# dummy_train = np.zeros((len(train_predictions), 6))
# dummy_train[:, 5] = train_predictions.flatten()
# inv_train_preds = scaler.inverse_transform(dummy_train)[:, 5]
# y_all_pred[window_size:split_idx] = inv_train_preds
# ==============================================================================

# Calculate standard statistical evaluation metrics on the test segment
y_test_actual = y_all_actual[split_idx:]
rmse = np.sqrt(mean_squared_error(y_test_actual, inv_predictions))
mae = mean_absolute_error(y_test_actual, inv_predictions)
residuals_usd = y_test_actual - inv_predictions
ape_percentage = (np.abs(residuals_usd) / y_test_actual) * 100
mape = np.mean(ape_percentage)

# ==============================================================================
# MASTER PLOT GENERATION: COMPREHENSIVE TIMELINE VISUALIZATION (2014 - 2026)
# ==============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [2.5, 1, 1]})

prediction_start_date = dates[split_idx]

# --- PANEL 1: FULL HORIZON AND PREDICTIVE FORKING ---
ax1.plot(dates, y_all_actual, label="True Market Price (Yahoo Finance - VWAP)", color='#1f77b4', linewidth=1.5)
ax1.plot(dates, y_all_pred, label=f"Model Prediction ({model_name.split('_')[-1].split('.')[0].upper()})",
         color='#d62728', linestyle='--', linewidth=1.8)

ax1.axvline(prediction_start_date, color='darkred', linestyle='-', linewidth=2.5,
            label=f"Training / Testing Boundary ({prediction_start_date.strftime('%Y-%m-%d')})")
ax1.axvspan(pd.to_datetime('2023-01-01'), pd.to_datetime('2026-05-31'),
            color='gold', alpha=0.10, label='Evaluated Stress Horizon (AI Market Wave)')


# Summary performance text box
metric_text = f"Out-of-Sample Performance:\nRMSE: {rmse:.4f} USD\nMAE: {mae:.4f} USD\nMAPE: {mape:.2f}%"
ax1.text(0.02, 0.95, metric_text, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))

ax1.set_title(f"Predictive Capability Evaluation over Full Historical Horizon - Comparative Analysis ({file_name.split('_')[0]})", fontsize=14, fontweight='bold')
ax1.set_ylabel("Asset Value (USD)", fontsize=11)

ax1.legend(loc='lower left', bbox_to_anchor=(0.02, 0.15), fontsize=9, framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.5)

# --- PANEL 2: NOMINAL TRACKING RESIDUALS (USD) ---
y_all_residuals = np.empty_like(y_all_actual)
y_all_residuals[:] = np.nan
y_all_residuals[split_idx:] = residuals_usd

ax2.plot(dates, y_all_residuals, label="Nominal Tracking Error (USD)", color='#2ca02c', linewidth=1.2)
ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax2.axvline(prediction_start_date, color='darkred', linestyle='-', linewidth=2)
ax2.set_ylabel("Error (USD)", fontsize=11)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, linestyle=':', alpha=0.5)

# --- PANEL 3: LIVE ABSOLUTE PERCENTAGE ERRORS (%) ---
y_all_ape = np.empty_like(y_all_actual)
y_all_ape[:] = np.nan
y_all_ape[split_idx:] = ape_percentage

ax3.plot(dates, y_all_ape, label="Absolute Percentage Error (APE)", color='#9467bd', linewidth=1.2)
ax3.axhline(mape, color='darkviolet', linestyle=':', linewidth=1.2, label=f"Mean MAPE ({mape:.2f}%)")
ax3.axvline(prediction_start_date, color='darkred', linestyle='-', linewidth=2)
ax3.set_xlabel("Historical Timeline (Calendar Scale 2014 - 2026)", fontsize=11)
ax3.set_ylabel("Deviation (%)", fontsize=11)
ax3.legend(loc="upper left", fontsize=9)
ax3.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()

# Save a clean 300 DPI image directly to the project root
plt.savefig(os.path.join(project_root, f"{file_name.split('_')[0].lower()}_orizont_total.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"Execution complete. Plot rendered and saved successfully.")
