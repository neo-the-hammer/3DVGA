import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        assert inputs.shape == targets.shape

        probs = F.softmax(inputs + 1e-8, dim=1)
        loss = -torch.sum(torch.log(probs + 1e-8) * targets, dim=1).mean()

        return loss


class SoftmaxRankingLoss2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        assert inputs.shape == targets.shape

        probs = F.softmax(inputs + 1e-8, dim=1)
        loss = -torch.sum(torch.log(1 - probs + 1e-8) * (1 - targets), dim=1).mean()

        return loss

class SoftmaxRankingLoss3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        assert inputs.shape == targets.shape

        sigmoid = nn.Sigmoid()
        probs = sigmoid(inputs + 1e-8)
        loss = -torch.sum(torch.log(probs + 1e-8) * targets, dim=1).mean()

        return loss