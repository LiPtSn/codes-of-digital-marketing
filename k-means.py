import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 读取数据（全量作为训练集与评估集）
df = pd.read_csv('financial_fraud_detection_dataset.csv', parse_dates=['timestamp'])

# 2. 定义预处理 Pipeline
num_cols = [
    'amount',
    'time_since_last_transaction',
    'spending_deviation_score',
    'velocity_score',
    'geo_anomaly_score'
]
cat_cols = [
    'transaction_type',
    'merchant_category',
    'location',
    'device_used',
    'payment_channel'
]

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols),
], remainder='drop')

# 3. 对全量数据做预处理
X = preprocessor.fit_transform(df)
y = df['is_fraud'].astype(int).values

# 4. 用 KMeans 划分 2 个簇（风险 / 无风险）
k = 2
km = KMeans(n_clusters=k, random_state=42, n_init=30, max_iter=500, algorithm='elkan')
clusters = km.fit_predict(X)

# 5. 统计每个簇的欺诈比例，选取高欺诈率簇为“风险簇”
fraud_ratios = []
for cluster_id in range(k):
    mask = (clusters == cluster_id)
    if mask.sum() > 0:
        fraud_ratio = y[mask].mean()
    else:
        fraud_ratio = 0.0
    fraud_ratios.append(fraud_ratio)

risk_cluster = int(np.argmax(fraud_ratios))
print(f'簇 0 欺诈率 = {fraud_ratios[0]:.5f}')
print(f'簇 1 欺诈率 = {fraud_ratios[1]:.5f}')
print(f'高风险簇 = {risk_cluster}（欺诈率 {fraud_ratios[risk_cluster]:.5f}）\n')

# 6. 将该簇内样本标记为预测“欺诈”（风险）类
y_pred = (clusters == risk_cluster).astype(int)

# 7. 计算风险类簇的相关评估指标
prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
roc_auc = roc_auc_score(y, y_pred)

print('=== 风险簇分类指标 ===')
print(f'Precision: {prec:.4f}')
print(f'Recall:    {rec:.4f}')
print(f'F1-score:  {f1:.4f}')
print(f'ROC-AUC:   {roc_auc:.4f}')

# 8. 混淆矩阵（表格）
cm = confusion_matrix(y, y_pred, labels=[1, 0])
cm_df = pd.DataFrame(
    cm,
    index=['实际_欺诈(1)', '实际_非欺诈(0)'],
    columns=['预测_欺诈(1)', '预测_非欺诈(0)']
)

print('\n=== 混淆矩阵 ===')
print(cm_df)

# 9. 绘制并保存混淆矩阵热力图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Risk Cluster Method)', fontsize=14)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('k-means混淆矩阵.png', dpi=300)
plt.close()
