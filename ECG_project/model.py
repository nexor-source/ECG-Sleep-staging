import torch.nn as nn

# CNN模型
class SleepStageModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SleepStageModel, self).__init__()
        self.cnn1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(128 * (input_size // 4), num_classes)

    def forward(self, x):
        x = x.float()  # 将输入数据类型转换为 torch.float32, x.shape = (32,1,31)
        x = self.cnn1d(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x