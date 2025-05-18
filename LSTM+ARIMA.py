import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. 载入数据（先按常规读）
df = pd.read_csv('financial_fraud_detection_dataset.csv', parse_dates=['timestamp'])

# 2. 强制将 timestamp 列转成 datetime 类型
df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True, errors='coerce')

# （可选）检查有没有无法解析的行：  
# print(df[df['timestamp'].isna()])

# 3. 把它设为索引
df.set_index('timestamp', inplace=True)

# 4. 再次确认索引类型必为 DatetimeIndex
df.index = pd.DatetimeIndex(df.index)  # ← 这一步保证索引就是 DatetimeIndex
# print(type(df.index))  # <class 'pandas.core.indexes.datetimes.DatetimeIndex'>

# 5. 构造按日统计的欺诈次数序列
ts = (
    df[df['is_fraud'] == 1]  # 先筛出 fraud=1
    .resample('D')  # 此时索引是真正的 DatetimeIndex，resample 安全
    .size()
    .asfreq('D')
    .fillna(0)
)

# 6. 划分训练 / 测试
train_size = int(len(ts) * 0.7)
train, test = ts[:train_size], ts[train_size:]

# 7. SARIMAX（ARIMA）预测
arima_order = (5, 1, 2)
arima_model = SARIMAX(train, order=arima_order).fit(disp=0)
arima_mean = arima_model.get_forecast(steps=len(test)).predicted_mean
y_arima = arima_mean.to_numpy()
x_arima = test.index.to_numpy()

# 8. LSTM 预测
def make_seqs(vals, w):
    X, y = [], []
    for i in range(len(vals) - w):
        X.append(vals[i:i+w])
        y.append(vals[i+w])
    return np.array(X), np.array(y)

window = 7
train_vals = train.values.astype('float32')
test_vals = test.values.astype('float32')

X_tr, y_tr = make_seqs(train_vals, window)
X_te, y_te = make_seqs(test_vals, window)

X_tr = X_tr.reshape(-1, window, 1)
X_te = X_te.reshape(-1, window, 1)

lstm = Sequential([LSTM(50, activation='relu', input_shape=(window,1)), Dense(1)])
lstm.compile('adam', 'mse')
lstm.fit(X_tr, y_tr, epochs=50, batch_size=16,
        validation_split=0.2, callbacks=[EarlyStopping(patience=5)],
        verbose=0)


y_lstm = lstm.predict(X_te).flatten()
x_lstm = test.index.to_numpy()[window:]
x_true = x_lstm
y_true = y_te

# 9. 评估
def mape(a,b): return np.mean(np.abs((a-b)/(a+1e-9))) * 100

metrics = {
    'Model': ['ARIMA','LSTM'],
    'RMSE': [
        np.sqrt(mean_squared_error(test.values, y_arima)),
        np.sqrt(mean_squared_error(y_true, y_lstm))
    ],
    'MAE': [
        mean_absolute_error(test.values, y_arima),
        mean_absolute_error(y_true, y_lstm)
    ],
    'MAPE': [
        mape(test.values, y_arima),
        mape(y_true, y_lstm)
    ]
}
print(pd.DataFrame(metrics))

plt.plot(x_true,  y_true, label='True')
plt.plot(x_lstm,  y_lstm, label='LSTM Pred')
plt.plot(x_true, y_true, label='True')
plt.plot(x_lstm, y_lstm, label='LSTM Pred')
plt.plot(x_arima, y_arima, label='ARIMA Pred') 
plt.xlabel('Date')
plt.ylabel('Daily Fraud Count')
plt.title('Fraud Count Forecast: ARIMA vs LSTM')
plt.legend()
plt.tight_layout()
plt.show()