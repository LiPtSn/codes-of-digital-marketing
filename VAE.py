import os
os.environ["OMP_NUM_THREADS"] = "4"  # 防止 MKL 内存泄漏
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 优化 PyTorch 内存管理

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc, confusion_matrix
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import seaborn as sns

# 设置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# -------------------------------
# 1. 数据加载与特征工程
# -------------------------------
print("加载数据...")
df = pd.read_csv('financial_fraud_detection_dataset.csv', low_memory=False)
# 确保 timestamp 为 datetime 类型
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# 时间特征
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# 增强特征
df['amount_deviation'] = df['amount'] - df['amount'].mean()
df['velocity_score'] = df['amount'] / (df['time_since_last_transaction'] + 1e-6)

# 频率特征（全表一次 groupby）
txn_counts = df['sender_account'].value_counts()
df['transaction_freq'] = df['sender_account'].map(txn_counts) / txn_counts.max()
ip_counts = df['ip_address'].value_counts()
df['ip_freq'] = df['ip_address'].map(ip_counts) / ip_counts.max()

# 滑动统计只对活跃账号（交易≥5次）
active_accounts = txn_counts[txn_counts >= 5].index
active_idx = df['sender_account'].isin(active_accounts)
df.loc[active_idx, 'rolling_amount_mean'] = df[active_idx].groupby('sender_account')['amount'] \
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
df['rolling_amount_mean'].fillna(df['amount'].mean(), inplace=True)
df['amount_to_mean_ratio'] = df['amount'] / (df['rolling_amount_mean'] + 1e-6)

# 标签
y = df['is_fraud'].astype(int).values

# 特征列
num_cols = ['amount', 'time_since_last_transaction', 'spending_deviation_score',
            'velocity_score', 'geo_anomaly_score', 'amount_deviation',
            'transaction_freq', 'ip_freq', 'hour', 'is_weekend', 'amount_to_mean_ratio']
cat_cols = ['transaction_type', 'merchant_category', 'location', 'device_used', 'payment_channel']

# -------------------------------
# 2. 预处理管道
# -------------------------------
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(sparse_output=True, handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
], remainder='drop')

X = preprocessor.fit_transform(df)
print(f"预处理后矩阵稀疏维度: {X.shape}")

# -------------------------------
# 3. VAE 训练（下采样正常样本）
# -------------------------------
normal_idx = np.where(y == 0)[0]
sub_idx = np.random.choice(normal_idx, size=200_000, replace=False)
X_norm = X[sub_idx].toarray() if hasattr(X, 'toarray') else X[sub_idx]

tensor_ds = TensorDataset(torch.tensor(X_norm, dtype=torch.float32), torch.tensor(X_norm, dtype=torch.float32))
train_loader = DataLoader(tensor_ds, batch_size=1024, shuffle=True)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=20):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim*2)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + std * torch.randn_like(std)
    def forward(self, x):
        h = self.enc(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparam(mu, logvar)
        return self.dec(z), mu, logvar

vae = VAE(X_norm.shape[1]).to(device)
opt = optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(30):
    total_loss = 0
    for xb, _ in train_loader:
        xb = xb.to(device)
        opt.zero_grad()
        recon, mu, logvar = vae(xb)
        mse = nn.MSELoss()(recon, xb)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = mse + 0.01*kl
        loss.backward()
        opt.step()
        total_loss += loss.item()*xb.size(0)
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader.dataset):.6f}")

# -------------------------------
# 4. 重构误差 + IsolationForest
# -------------------------------
# 4.1 计算重构误差
X_arr = X.toarray() if hasattr(X, 'toarray') else X
recon_err = []
vae.eval()
with torch.no_grad():
    for i in range(0, X_arr.shape[0], 10000):
        batch = torch.tensor(X_arr[i:i+10000], dtype=torch.float32).to(device)
        rec, mu, _ = vae(batch)
        err = ((rec - batch)**2).mean(dim=1).cpu().numpy()
        recon_err.append(err)
recon_err = np.concatenate(recon_err)

# 4.2 IsolationForest 下采样训练
cont = y.mean()
iso = IsolationForest(contamination=cont, max_samples=200_000, n_jobs=-1, random_state=42)
iso.fit(X_arr[sub_idx])
iso_pred = (iso.predict(X_arr) == -1).astype(int)

# 评估对比
def eval_model(y_true, y_pred, name):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc = roc_auc_score(y_true, y_pred)
    prc, rec, _ = precision_recall_curve(y_true, y_pred)
    prauc = auc(rec, prc)
    print(f"{name} -> Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}, ROC-AUC: {roc:.4f}, PR-AUC: {prauc:.4f}")

# 4.3 重构误差阈值检测
thr = np.percentile(recon_err, 95)
err_pred = (recon_err > thr).astype(int)

eval_model(y, iso_pred, 'IsolationForest')
eval_model(y, err_pred, 'Reconstruction Error')

# 4.4 混淆矩阵可视化
for name, pred in [('IsolationForest', iso_pred), ('Reconstruction Error', err_pred)]:
    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.savefig(f'{name.replace(" ", "_")}_conf.png')
    plt.close()
