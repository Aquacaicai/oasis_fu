import torch.nn as nn
import torch.nn.functional as F

class FLNet(nn.Module):
    def __init__(self, num_classes=10, img_size=128):
        super(FLNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        fc_input_size = (img_size // 4) * (img_size // 4) * 64
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
