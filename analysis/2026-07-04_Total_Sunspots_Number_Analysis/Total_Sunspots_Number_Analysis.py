# ------------------------ Import Modules -----------------------------------
import numpy as np
import requests
import matplotlib.pyplot as plt
import pandas as pd
import certifi
import io
from itertools import product
from scipy.signal import find_peaks

# ------------------------ Parameters Setup ---------------------------------

RATIO = 0.7
SYMBOL = 'SUNSPOT'
URL = 'https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv'
SMA = 50
SEPERATOR = 550

# ----------------------- Grab Data -----------------------------------------

response = requests.get(URL, verify=certifi.where())
response.raise_for_status()

columns = ['Year', 'Month', 'Day', 'DecimalYear', 'Value', 'Std', 'Obs', 'Definitive']
raw = pd.read_csv(io.StringIO(response.text), sep=';', header=None, names=columns)

raw['Date'] = pd.to_datetime(raw[['Year', 'Month', 'Day']])
raw.set_index('Date', inplace=True)

data = pd.DataFrame(raw['Value'])
data.columns = [SYMBOL]
data[SYMBOL] = data[SYMBOL].where(data[SYMBOL] >= 0, np.nan)
data['sma'] = data[SYMBOL].rolling(SMA).mean()
data['monthly_average'] = data[SYMBOL].rolling(30).mean()
data.dropna(inplace=True)

data['return'] = np.log((data['sma'] + 1) / (data['sma'].shift(1) + 1))
data.dropna(inplace=True)
data['direction'] = np.sign(data['return']).astype(int)

split_index = int(len(data) * RATIO)
trained_data = data[:split_index].copy()
tested_data = data[split_index:].copy()

# --------------------- Define the SMA Strategy Function -----------------------

def SMA_strategy(data_train, sma1, sma2):

    local_data = data_train.copy()

    local_data['sma1'] = local_data['sma'].rolling(sma1).mean()
    local_data['sma2'] = local_data['sma'].rolling(sma2).mean()
    local_data.dropna(inplace=True)

    local_data['position'] = np.where(local_data['sma1'].shift(1) >= local_data['sma2'].shift(1), 1, -1)
    local_data['strategy'] = local_data['position'] * local_data['return']
    local_data.dropna(inplace=True)

    valid = local_data['direction'] != 0
    accu_train = (local_data['position'][valid] == local_data['direction'][valid]).mean()

    return accu_train

# --------------------- Find the Best (sma1, sma2) ------------------------------

long_range  = range(SEPERATOR, 1000, 10)
short_range = range(30, SEPERATOR, 5)

accu = []

for s1, s2 in product(short_range, long_range):
    a = SMA_strategy(trained_data, s1, s2)
    accu.append({'accu': a, 'sma1': s1, 'sma2': s2})

accu_values = [d['accu'] for d in accu]
best_accu = max(accu_values)

best_sma = [[d['sma1'], d['sma2']] for d in accu if d['accu'] == best_accu]

print('===== Best (sma1, sma2)\'s =====')
print(best_sma)

best_sma1, best_sma2 = best_sma[0]

# -------------------- Find the Best Accuracy --------------------------------------

accuracy = SMA_strategy(tested_data, best_sma1, best_sma2)
print('===== Accuracy based on the best (sma1, sma2) =====')
print(f'{accuracy:.2%}')

# -------------------- Plot the Position Changes on Tested Data based on the SMA Strategy ----------

tested_data['sma1'] = tested_data['sma'].rolling(best_sma1).mean()
tested_data['sma2'] = tested_data['sma'].rolling(best_sma2).mean()
tested_data.dropna(inplace=True)

# 產生部位訊號，短天期均線大於等於長天期均線時做多，否則做空
tested_data['position'] = np.where(tested_data['sma1'].shift(1) >= tested_data['sma2'].shift(1), 1, -1)


fig, ax = plt.subplots(figsize=(12, 4))
ax1 = ax.twinx()
ax.plot(tested_data[SYMBOL], label='Total Number of Sunspots',
        color='#B0B0B0', linewidth=0.5, zorder=0)
ax.plot(tested_data['sma1'], label='sma1',
        color="#F80303", linewidth=1, zorder=1)
ax.plot(tested_data['sma2'], label='sma2',
        color="#0320f8", linewidth=1, zorder=2)
ax1.plot(tested_data['position'], label='Position', 
         color="#F98857", linewidth=1.2, zorder=3)

ax.set_xlim(tested_data.index.min(), tested_data.index.max())
ax.set_title('Position Changes by SMA Strategy on the Tested Data')
ax.set_axisbelow(True)
ax.grid()
ax.legend()
ax1.legend()
plt.tight_layout()
plt.savefig('Position_changes_by_SMA_Strategy_on_the_Tested_Data.png', dpi=150)
plt.show()

# -------------------- Plot the Accuracy Distribution on (sma1, sma2) Plane ------------------------

fig, ax = plt.subplots(figsize=(9, 8))

acc_grid = np.zeros((len(long_range), len(short_range)))

for i, s2 in enumerate(long_range):
    for j, s1 in enumerate(short_range):
        acc_grid[i, j] = SMA_strategy(trained_data, s1, s2)

s1_edges = np.append(short_range, short_range.stop)
s2_edges = np.append(long_range, long_range.stop)

mesh = ax.pcolormesh(s1_edges, s2_edges, acc_grid,
                      cmap='coolwarm', vmin=acc_grid.min(), vmax=acc_grid.max())
plt.colorbar(mesh, ax=ax, label='Accuracy')
ax.scatter(best_sma1, best_sma2, s=80, facecolors='white', edgecolors='black', zorder=5)
ax.set_title('Accuracy by Different (sma1, sma2) on the Trained Data')
ax.set_xlabel('sma1')
ax.set_ylabel('sma2')
plt.tight_layout()
plt.savefig('Accuracy_by_Different_(sma1_sma2)_on_the_Trained_Data.png', dpi=150)
plt.show()

# -------- Find the Years where Total Number of Sunspots is at Maximum -------------

distance = 365 * 7.5

peak_idx, _ = find_peaks(tested_data['monthly_average'], distance=distance)
peak_dates = tested_data['monthly_average'].index[peak_idx]

trough_idx, _ = find_peaks(-tested_data['monthly_average'], distance=distance)
trough_dates = tested_data['monthly_average'].index[trough_idx]

fig, ax = plt.subplots(figsize=(12, 4))

trans = ax.get_xaxis_transform()

ax.plot(tested_data[SYMBOL], label='Total Number of Sunspots on the Tested Data',
        color='#B0B0B0', linewidth=0.5, zorder=0)

for i, date in enumerate(peak_dates):
    ax.axvline(date, color="#F80303", alpha=0.5, linestyle='--',
               label='Peaks' if i == 0 else None)
    ax.text(date, 1.02, date.year, color="#F80303", ha='center', transform=trans)

for i, date in enumerate(trough_dates):
    ax.axvline(date, color="#0320f8", alpha=0.5, linestyle='--',
               label='Troughs' if i == 0 else None)
    ax.text(date, 1.02, date.year, color="#0320f8", ha='center', transform=trans)

ax.set_xlim(tested_data.index.min(), tested_data.index.max())
ax.set_ylim(bottom=0)
ax.set_xlabel('years')
ax.set_ylabel('sunspots number')
ax.legend()
plt.tight_layout()
plt.savefig('local_Extremums_of_Sunspots_Number_on_the_Tested_Data.png', dpi=150)
plt.show()

# --------- Predict Next Time when Maximum Occurs -------------------------
