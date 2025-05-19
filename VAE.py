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
from sklearn.model_selection import train_test_split
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
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

df['amount_deviation'] = df['amount'] - df['amount'].mean()
df['velocity_score'] = df['amount'] / (df['time_since_last_transaction'] + 1e-6)

txn_counts = df['sender_account'].value_counts()
df['transaction_freq'] = df['sender_account'].map(txn_counts) / txn_counts.max()
ip_counts = df['ip_address'].value_counts()
df['ip_freq'] = df['ip_address'].map(ip_counts) / ip_counts.max()

active_accounts = txn_counts[txn_counts >= 5].index
active_idx = df['sender_account'].isin(active_accounts)
df.loc[active_idx, 'rolling_amount_mean'] = df[active_idx].groupby('sender_account')['amount'] \
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
df['rolling_amount_mean'].fillna(df['amount'].mean(), inplace=True)
df['amount_to_mean_ratio'] = df['amount'] / (df['rolling_amount_mean'] + 1e-6)

y = df['is_fraud'].astype(int).values

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
print(f"预处理后矩阵维度: {X.shape}")
X_arr = X.toarray() if hasattr(X, 'toarray') else X

# -------------------------------
# 3. VAE 训练（使用早停）
# -------------------------------
X_train, X_val = train_test_split(X_arr, test_size=0.2, random_state=42)
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_train, dtype=torch.float32))
val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32))
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

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
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def reparam(self, mu, logvar):
        logvar = torch.clamp(logvar, -10, 10)
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = h.chunk(2, dim=-1)
        return self.dec(self.reparam(mu, logvar)), mu, logvar

vae = VAE(X_arr.shape[1]).to(device)
opt = optim.Adam(vae.parameters(), lr=5e-4)
patience, best_val_loss, counter = 5, float('inf'), 0
best_path = 'best_vae.pth'

for epoch in range(30):
    vae.train()
    train_loss = 0
    for xb, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        xb = xb.to(device)
        opt.zero_grad()
        recon, mu, logvar = vae(xb)
        mse = nn.MSELoss(reduction='sum')(recon, xb)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = mse + 0.01 * kl
        if torch.isnan(loss): break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()
        train_loss += loss.item()
    # 验证
    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            recon, mu, logvar = vae(xb)
            mse = nn.MSELoss(reduction='sum')(recon, xb)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            val_loss += (mse + 0.01*kl).item()
    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}: Train {train_loss:.6f}, Val {val_loss:.6f}")
    if val_loss < best_val_loss:
        best_val_loss, counter = val_loss, 0
        torch.save(vae.state_dict(), best_path)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            vae.load_state_dict(torch.load(best_path))
            break

# -------------------------------
# 4. 利用 VAE 重构误差进行异常检测
# -------------------------------
recon_err = []
vae.eval()
with torch.no_grad():
    for i in range(0, X_arr.shape[0], 10000):
        batch = torch.tensor(X_arr[i:i+10000], dtype=torch.float32).to(device)
        rec, _, _ = vae(batch)
        recon_err.append(((rec - batch) ** 2).mean(dim=1).cpu().numpy())
recon_err = np.concatenate(recon_err)

thr = np.percentile(recon_err, 95)
err_pred = (recon_err > thr).astype(int)

# -------------------------------
# 5. 评估指标
# -------------------------------
def eval_model(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc = roc_auc_score(y_true, y_pred)
    prc, rec, _ = precision_recall_curve(y_true, y_pred)
    prauc = auc(rec, prc)
    return p, r, f, roc, prauc

p, r, f, roc, prauc = eval_model(y, err_pred)
print(f"VAE 重构误差 -> Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}, ROC-AUC: {roc:.4f}, PR-AUC: {prauc:.4f}")

cm = confusion_matrix(y, err_pred)
print(f"Confusion Matrix:\n{cm}")
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('VAE Confusion Matrix')
plt.xlabel('Pred')
plt.ylabel('True')
plt.savefig('VAE_conf.png')
plt.close()
print("检测完成，混淆矩阵已保存。")
