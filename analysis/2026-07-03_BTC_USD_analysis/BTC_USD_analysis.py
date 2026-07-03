import numpy as np 
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import os

# --------- Define the Parameters ---------

SYMBOL = 'BTC-USD'
START = '2016-01-01'
END = '2026-06-30'
SMA1 = 14
SMA2 = 180
RATIO = 0.7

# --------- Import the Data ---------------

raw = yf.download(SYMBOL, start=START, end=END, auto_adjust=True)
data = pd.DataFrame(raw['Close'])
data.columns = [SYMBOL]
data['return'] = np.log(data[SYMBOL]/data[SYMBOL].shift(1))
data.dropna(inplace=True)

# ---------- Download and Store the Data

save_dir = r'C:\Users\flyin\OneDrive\桌面\新代碼\Simple Moving Average\data'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f'{SYMBOL}_{START}_{END}.csv')
data.to_csv(save_path)
print(f'Data has been stored to {save_path}')

# --------- Split the Training/Testing Data ------------------

split = int(len(data)* RATIO)
trained_data = data[: split].copy()
tested_data = data[split:].copy()

# --------- Define the SMA Strategy Function -----------------

def SMA_strategy(data_train, sma1, sma2):
    
    local_data = data_train.copy()

    local_data['sma1'] = local_data[SYMBOL].rolling(sma1).mean()
    local_data['sma2'] = local_data[SYMBOL].rolling(sma2).mean()
    local_data.dropna(inplace=True)
    local_data['position'] = np.where(local_data['sma1'].shift(1) >= local_data['sma2'].shift(1), 1, -1)
    local_data['strategy'] = local_data['position'] * local_data['return']

    perf_train = np.exp(local_data['strategy'].sum())

    return perf_train

# -------- Cylicallly Find the Best (sma1, sma2) from the SMA strategy -----------------

long_range = range(50, 180, 10)
short_range = range(7, 50, 3)

perfs = []

for s1, s2 in product(short_range, long_range):
    n = SMA_strategy(trained_data, s1, s2)

    perf = {'perf': n,
            'sma1': s1,
            'sma2': s2}
    
    perfs.append(perf)

perf_values = [d['perf'] for d in perfs]
best_perf = max(perf_values)

best_sma = [[d['sma1'], d['sma2']] for d in perfs if d['perf'] == best_perf]
print(best_sma)

# ----------- Plot and Print the result with the Best Parameters ----------------

best_sma1, best_sma2 = best_sma[0] # Choose just one set (sma1, sma2)
perf = SMA_strategy(tested_data, best_sma1, best_sma2)
price = np.exp(tested_data['return'].sum())
print(f'BTC Performance is: {round(price, 4)}')
print(f'The Performance using simple moving average strategy is: {round(perf, 4)}')

tested_data['sma1'] = tested_data[SYMBOL].rolling(best_sma1).mean()
tested_data['sma2'] = tested_data[SYMBOL].rolling(best_sma2).mean()
tested_data.dropna(inplace=True)
tested_data['position'] = np.where(tested_data['sma1'].shift(1) >= tested_data['sma2'].shift(1), 1, -1)
tested_data['strategy'] = tested_data['position'] * tested_data['return']

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(tested_data['return'].cumsum().apply(np.exp), label='return')
ax.plot(tested_data['strategy'].cumsum().apply(np.exp), label='strategy')
ax.legend()
ax.set_title('Comparison between real prices and strategy result')
plt.tight_layout()
plt.savefig('BTC_USD_with_sma_strategy.png', dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(tested_data['sma1'], label='sma1')
ax.plot(tested_data['sma2'], label='sma2')
ax.plot(tested_data[SYMBOL], label='BTC-USD')
ax.legend()
ax.set_title('BTC-USD time series')
plt.tight_layout()
plt.savefig('BTC_USD_time_series.png', dpi=150)
plt.show()

