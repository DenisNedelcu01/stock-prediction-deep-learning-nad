# Analysis of stock market prediction algorithms based on deep learning models

This repository contains the complete implementation framework for a deep learning pipeline designed to evaluate and forecast stock market trends within modern e-business environments. The project strictly challenges the Efficient Market Hypothesis (EMH) by testing non-linear sequential topologies under real-world market stress and structural regime shifts.

## Project Overview
The core objective of this study is to perform a rigorous comparative analysis between two primary Recurrent Neural Network (RNN) architectures: Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU). 

To evaluate their performance across varying levels of volatility, the pipeline tests three strategically selected corporate assets spanning a 12-year historical timeline (2014 – 2026):
* Apple Inc. (AAPL): Serving as a stable, high-liquidity technology benchmark.
* NVIDIA Corporation (NVDA): Capturing hyper-volatile momentum during the recent AI hardware expansion.
* JPMorgan Chase & Co. (JPM): Serving as a cyclical financial sector counterweight.

## Technical Contributions & Pipeline Design
* Feature Engineering (VWAP): Instead of standard closing prices, the ingestion layer calculates a daily Volume Weighted Average Price (VWAP) approximation to represent a truer financial equilibrium.
* Anti-Leakage Data Split: To prevent data leakage and guarantee a genuine out-of-sample stress test, a strict chronological calendar split is hardcoded at December 31, 2022. Models train on 3,288 days (2014–2022) and evaluate blindly over the subsequent 2023–2026 window.
* Systems Performance Optimization: Explicit intra-op and inter-op thread parallelisms are enforced directly inside the TensorFlow backend to optimize system footprint and eliminate thread contention on modern compute hardware.
* Regularization Stack: Overfitting challenges are controlled using 60-day sliding lookback windows and embedding Dropout(0.2) layers into the stacked hidden recurrent paths.

## Repository Structure
```text
├── data/           # Stored historical asset datasets (.csv files)
├── models/         # Serialized, pre-trained neural network weights (.h5 files)
├── scripts/        # Core execution engines
├── prediction_photos/ # Contains the photos for each stock graph
│   ├── download_data.py   # Ingestion layer via Yahoo Finance API
│   ├── train_gru.py       # Stacked GRU model training pipeline
│   ├── train_lstm.py      # Stacked LSTM model training pipeline
│   └── test_and_plot.py   # Out-of-sample inference, error evaluation & plotting
├── documentation.docx    # Documentation of the project
├── requirements.txt       # Environment software dependencies log
└── README.md              # Project documentation repository index
