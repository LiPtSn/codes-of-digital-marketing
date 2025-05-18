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
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc, confusion_matrix
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import seaborn as sns

# 设置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检测可用设备（GPU 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# -------------------------------
# 1. 读取数据集并进行特征工程
# -------------------------------
print("加载数据...")
df = pd.read_csv('financial_fraud_detection_dataset.csv', parse_dates=['timestamp'])

# 添加衍生特征
df['amount_deviation'] = df['amount'] - df['amount'].mean()
df['velocity_score'] = df['amount'] / (df['time_since_last_transaction'] + 1e-6)
df['transaction_freq'] = df.groupby('sender_account')['timestamp'].transform('count') / (df['time_since_last_transaction'] + 1e-6)
df['ip_freq'] = df.groupby('ip_address')['timestamp'].transform('count') / len(df)
df['device_freq'] = df.groupby('device_hash')['timestamp'].transform('count') / len(df)

# 提取特征和标签
y = df['is_fraud'].astype(int).values

# 定义特征列表
num_cols = [
    'amount', 'time_since_last_transaction', 'spending_deviation_score',
    'velocity_score', 'geo_anomaly_score', 'amount_deviation',
    'transaction_freq', 'ip_freq', 'device_freq'
]
cat_cols = [
    'transaction_type', 'merchant_category', 'location',
    'device_used', 'payment_channel'
]

# 构建预处理流水线
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols),
], remainder='drop')

# 对全数据进行预处理
X_full = preprocessor.fit_transform(df).astype(np.float32)
print(f"预处理后全数据形状: {X_full.shape}")

# -------------------------------
# 2. 自编码器训练（正常交易样本，全数据集）
# -------------------------------
# 使用全数据集，正常样本训练自编码器
normal_idx = y == 0
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_full[normal_idx]), torch.tensor(X_full[normal_idx])),
    batch_size=1024, shuffle=True
)

# 自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Dropout(0.2),  
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

input_dim = X_full.shape[1]
model = Autoencoder(input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 训练（固定轮次，添加正则化）
max_epochs = 50  # 减少轮次以防止过拟合
train_losses = []
for epoch in tqdm(range(max_epochs), desc="Training Autoencoder"):
    model.train()
    total_train = 0
    for xb, _ in train_loader:
        xb = xb.to(device)
        optimizer.zero_grad()
        recon, _ = model(xb)
        loss = criterion(recon, xb)
        loss.backward()
        optimizer.step()
        total_train += loss.item() * xb.size(0)
    train_loss = total_train / len(train_loader.dataset)
    train_losses.append(train_loss)
    print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.6f}")
print("自编码器训练完成")
model.eval()

# 可视化训练损失
plt.figure(figsize=(8, 6))
plt.plot(range(1, max_epochs + 1), train_losses, label='训练损失')
plt.xlabel('轮次')
plt.ylabel('均方误差损失')
plt.title('自编码器训练损失曲线')
plt.legend()
plt.savefig('autoencoder_train_loss.png')
plt.close()

# -------------------------------
# 3. K-Means 聚类两个簇，全数据集
# -------------------------------
# 编码全数据特征
def batch_encode(data, model, device, batch_size=10000):
    encs = []
    for i in range(0, len(data), batch_size):
        batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32).to(device)
        with torch.no_grad():
            encs.append(model.encoder(batch).cpu().numpy())
    return np.vstack(encs)

Z_full = batch_encode(X_full, model, device)

# 2 类
km = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = km.fit_predict(Z_full)

# 确定风险簇（fraud比例更高）
fraud_rate_by_cluster = [y[clusters == i].mean() for i in range(2)]
risk_cluster = np.argmax(fraud_rate_by_cluster)

# 风险簇预测标签
y_pred_risk = (clusters == risk_cluster).astype(int)

# 评估风险簇相关指标
precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred_risk, average='binary')
roc_auc = roc_auc_score(y, y_pred_risk)
prec_curve, rec_curve, _ = precision_recall_curve(y, y_pred_risk)
pr_auc = auc(rec_curve, prec_curve)

print("\n【K-Means 聚类(2 簇）风险簇评估】")
print(f'簇 {risk_cluster} 视为风险簇')
print(f'簇 0 欺诈率: {fraud_rate_by_cluster[0]:.5f}')
print(f'簇 1 欺诈率: {fraud_rate_by_cluster[1]:.5f}')
print(f'精确率: {precision:.4f} | 召回率: {recall:.4f} | F1 分数: {f1:.4f}')
print(f'ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}')

# 混淆矩阵
cm = confusion_matrix(y, y_pred_risk)
print("\n混淆矩阵:")
print(cm)

# 可视化混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.savefig('confusion_matrix.png')
plt.close()

# 可视化 PR 曲线
plt.figure(figsize=(8, 6))
plt.plot(rec_curve, prec_curve, label=f'PR-AUC = {pr_auc:.4f}')
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('精确率-召回率曲线')
plt.legend()
plt.savefig('pr_curve.png')
plt.close()

# 可视化风险簇与非风险簇分布（潜在特征前两维）
plt.figure(figsize=(8, 6))
plt.scatter(Z_full[clusters == 0][:, 0], Z_full[clusters == 0][:, 1], alpha=0.5, label='簇 0')
plt.scatter(Z_full[clusters == 1][:, 0], Z_full[clusters == 1][:, 1], alpha=0.5, label='簇 1')
plt.xlabel('潜在特征 1')
plt.ylabel('潜在特征 2')
plt.title('K-Means 簇分布（潜在空间前两维）')
plt.legend()
plt.savefig('k-means_cluster_scatter.png')
plt.close()