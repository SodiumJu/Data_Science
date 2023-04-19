import torch.nn as nn
import torch.nn.functional as F

from .prune import PruningModule


class AlexNet(PruningModule):
    def __init__(self, n_classes=10):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, n_classes)

    def forward(self, x):
        x = F._max_pool2d(F.relu(self.conv1(x), inplace=True), kernel_size=2)
        x = F._max_pool2d(F.relu(self.conv2(x), inplace=True), kernel_size=2)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = F._max_pool2d(F.relu(self.conv5(x), inplace=True), kernel_size=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x



