import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *
from model import SleepStageModel


# 在开始之前设置随机种子
torch.manual_seed(2023)
cudnn.deterministic = True
cudnn.benchmark = False  # 设置为 False 可以确保一些特定操作的确定性
# plt画图的中文字体
plt.rcParams['font.sans-serif']=['SimHei']


# 读入数据，其中len(ECG_data) = 4065000，len(sleep_labels) = 1084
ECG_data, sleep_labels = get_data()

# 数据预处理
ECG_data = np.array(ECG_data)
# ECG_data = rawdata_remove_noise(ECG_data)  # 变差，可能是因为信息丢失了一部分
# ECG_data = rawdata_normalize(ECG_data)  # 不变，神经网络很强
# ECG_data = rawdata_z_score_normalize(ECG_data)  # 变差，改变了特征特性

# 提取特征
features = get_features(ECG_data)
features = np.array(features)
# features = feature_minmax_normalize(features)
# features = feature_z_score_normalize(features)

# 转成tensor，为深度学习做准备
ECG_data = torch.tensor(ECG_data.copy())
sleep_labels = torch.tensor(sleep_labels)
features = torch.tensor(features)

# print(ECG_data.shape, sleep_labels.shape)  # ECG_data.shape = ([4065000])，sleep_labels.shape = ([1084])
# print(features.shape)  # features.shape = ([1084, 31])

# 初始化模型和损失函数、优化器
input_size = features.shape[1]  # 特征的数量
NUM_CLASSES = 5  # 分类的类别数
model = SleepStageModel(input_size, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 假设 features 和 sleep_labels 是张量
train_size = int(0.8 * len(features))
test_size = len(features) - train_size
print(f"数据集划分比例为 训练集{8} : 测试集{2}")

# 使用切片进行划分
train_features = features[:train_size]
train_labels = sleep_labels[:train_size]
test_features = features[train_size:]
test_labels = sleep_labels[train_size:]

# 创建 DataLoader
BATCH_SIZE = 32
train_dataset = TensorDataset(train_features, train_labels)
test_dataset = TensorDataset(test_features, test_labels)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 训练模型
NUM_EPOCHS = 80

for epoch in range(NUM_EPOCHS):
    model.train()
    for features, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(features.unsqueeze(1))  # 添加1维作为通道维度
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 在测试集上评估模型
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for features, labels in test_dataloader:
        outputs = model(features.unsqueeze(1))
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# 计算准确率
accuracy = accuracy_score(all_labels, all_predictions)
print(f'accuracy: {accuracy:.3f}')

# 计算查准率、查全率和 F1 分数
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', zero_division=1)
print(f'precision: {precision:.3f}, recall: {recall:.3f}, F1 score: {f1:.3f}')

# 绘制 ROC 曲线
fpr, tpr, thresholds = roc_curve(all_labels, all_predictions, pos_label=1)
roc_auc = auc(fpr, tpr)
print(f"ROC 面积: {roc_auc:.3f}")


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲线 (面积 = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive')
plt.ylabel('Ture Positive')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.show()

# 绘制混淆矩阵
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['N1', 'N2', 'N3', 'R', 'W'], yticklabels=['N1', 'N2', 'N3', 'R', 'W'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()