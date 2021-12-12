import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        
        super(RNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 16, 9, stride=4)
        self.batch1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU() #output now: (2248,32) 2248 is seq_len

        self.conv2 = nn.Conv1d(16, 32, 9, stride=4)
        self.batch2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU() #output now: (560,64)

        self.conv3 = nn.Conv1d(32, 64, 9, stride=4)
        self.batch3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU() #output now: torch.Size([32, 128, 138])

        self.gru1 = nn.GRU(input_size = 138, hidden_size=64, batch_first=True)
        self.dense1 = nn.Linear(64, 32)
        self.dense2 = nn.Linear(32, output_size)
        self.sof = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu3(x)

        x,_ = self.gru1(x)
        x = x[:, -1, :]
        x = self.dense1(x.squeeze())
        x = self.dense2(x)
        x = self.sof(x)

        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        crit = torch.nn.CrossEntropyLoss()
        bce_loss = crit(inputs.squeeze(),  targets)
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss