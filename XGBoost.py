import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
data = pd.read_csv("financial_fraud_detection_dataset.csv")


# 数据预处理函数
def preprocess_data(data):
    # 转换时间戳
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data = data.dropna(subset=['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['month'] = data['timestamp'].dt.month

    # 处理缺失值
    categorical_columns = ['transaction_type', 'merchant_category', 'location', 'device_used', 'payment_channel']
    for col in categorical_columns:
        data[col] = data[col].fillna('unknown')

    # 创建目标变量
    print("欺诈类型唯一值：", data['fraud_type'].unique())  # 验证欺诈类型
    data['target'] = data.apply(lambda row: 'normal' if not row['is_fraud'] else row['device_used'] + '_fraud', axis=1)
    print("目标变量分布：", data['target'].value_counts())  # 验证目标变量分布

    # 编码目标变量
    le = LabelEncoder()
    data['fraud_label'] = le.fit_transform(data['target'])

    # 转换为哑变量
    data = pd.get_dummies(data, columns=categorical_columns)

    # 删除无关列
    columns_to_drop = ['transaction_id', 'timestamp', 'sender_account', 'receiver_account', 'ip_address', 'device_hash',
                    'is_fraud', 'fraud_type', 'target']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    return data, le


# 预处理数据
data, label_encoder = preprocess_data(data)

# 分离特征和目标
X = data.drop('fraud_label', axis=1)
y = data['fraud_label']

# 划分训练集和测试集
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, stratify=y, random_state=42)

# 标准化特征
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)


# 划分训练集和测试集（先用 20% 数据训练）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_scaled = scaler.transform(X)  # 对整个数据集标准化

# 用整个数据集作为测试集
X_test = X.copy()
y_test = y.copy()
X_test_scaled = X_scaled



# 计算类权重
class_weights = dict(zip(np.unique(y_train), len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))))
sample_weights = [class_weights[yi] for yi in y_train]

# 训练 XGBoost
xgb = XGBClassifier(objective='multi:softmax', max_depth=6, learning_rate=0.1, n_estimators=300, random_state=42,
                    tree_method='hist', eval_metric='mlogloss')
xgb.fit(X_train_scaled, y_train, sample_weight=sample_weights, verbose=True)

# 预测
#y_pred = xgb.predict(X_test_scaled)

# 获取预测概率
#y_pred_proba = xgb.predict_proba(X_test_scaled)

# 设置高阈值 T
T = 0.95  # 可以根据需要调整

# 调整预测：只有当欺诈类别的最大概率大于 T 时，才预测为欺诈
# 预测
y_pred = xgb.predict(X_test_scaled)
y_pred_proba = xgb.predict_proba(X_test_scaled)
n_samples = y_pred_proba.shape[0]
y_pred_new = np.zeros(n_samples, dtype=int)
for i in range(n_samples):
    probs = y_pred_proba[i]
    # 获取欺诈类别的概率（假设类别 0 是 'normal'，1-4 是欺诈类别）
    fraud_probs = probs[1:]
    max_fraud_prob = np.max(fraud_probs)
    if max_fraud_prob > T:
        # 找到概率最高的欺诈类别
        max_idx = np.argmax(fraud_probs)
        y_pred_new[i] = max_idx + 1  # 欺诈类别从 1 开始
    else:
        y_pred_new[i] = 0  # 'normal'

# 评估调整后的模型
print("XGBoost 分类报告（调整阈值）：")
print(classification_report(y_test, y_pred_new, target_names=label_encoder.classes_))

# 评估
#print("XGBoost 分类报告：")
#print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 可视化混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('XGBoost 混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.show()

# 可视化特征重要性
feature_importance = xgb.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('XGBoost 特征重要性')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.show()


from sklearn.metrics import roc_curve, auc
from itertools import cycle

y_pred_proba = xgb.predict_proba(X_test_scaled)
n_classes = y_test.nunique()
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    y_true_i = (y_test == i)
    y_score_i = y_pred_proba[:, i]
    fpr[i], tpr[i], _ = roc_curve(y_true_i, y_score_i)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=next(colors), lw=2,
            label=f'{label_encoder.classes_[i]} (AUC={roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC 曲线')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import precision_recall_curve

precision = dict()
recall = dict()
for i in range(n_classes):
    y_true_i = (y_test == i)
    y_score_i = y_pred_proba[:, i]
    precision[i], recall[i], _ = precision_recall_curve(y_true_i, y_score_i)

plt.figure(figsize=(10, 8))
colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
for i in range(n_classes):
    plt.plot(recall[i], precision[i], color=next(colors), lw=2,
            label=f'{label_encoder.classes_[i]}')

plt.xlabel('召回率')
plt.ylabel('精确率')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('精确率-召回率曲线')
plt.legend(loc="lower left")
plt.show()


from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, test_scores = learning_curve(
    xgb, X_train_scaled, y_train, cv=3,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1_macro'
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 8))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.1,
                color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.1,
                color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
        label="训练集得分")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
        label="交叉验证得分")
plt.xlabel("训练样本数")
plt.ylabel("F1 分数")
plt.title("学习曲线")
plt.legend(loc="best")
plt.show()


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(xgb, X_train_scaled, y_train, cv=5, scoring='f1_macro')
print("交叉验证 F1 分数：", cv_scores)
print("平均 F1 分数：", cv_scores.mean())