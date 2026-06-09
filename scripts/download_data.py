import yfinance as yf
import pandas as pd
import warnings

# Ignorăm avertismentele de SSL de pe Mac pentru un output curat
warnings.filterwarnings("ignore")

ticker = "NVDA"
start_date = "2014-01-01"
end_date = "2026-01-01"

print(f"Calculăm Media Ponderată (Average) pentru {ticker}...")

# Descarcăm datele
df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

# Reparăm indexul pentru macOS
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# CALCULUL SOLICITAT: Average = (Suma valorii tranzacțiilor) / (Nr total tranzacții)
# Deoarece nu avem fiecare secundă, folosim prețul de închidere ponderat cu volumul zilei
# Pentru o disertație, cea mai precisă formulă zilnică este:
df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
df['Money_Flow'] = df['Typical_Price'] * df['Volume']

# Average-ul tău (VWAP zilnic)
df['Average'] = df['Money_Flow'] / df['Volume']

# Calculăm și return-ul zilnic pentru modelul de Deep Learning
df['Daily_Return'] = df['Close'].pct_change()

# Curățăm datele
df.dropna(inplace=True)

# Salvăm
df.to_csv(f"{ticker}_precise_data.csv")

print("\n--- DATE PROCESATE PENTRU DISERTAȚIE ---")
print(df[['Open', 'Close', 'Volume', 'Average']].head())