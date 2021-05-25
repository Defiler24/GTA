import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.utils.model_zoo as model_zoo
import math

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np

class Classifier(nn.Module):
    def __init__(self, num_classes=101):
        super(Classifier, self).__init__()

        self.fc = nn.Linear(101, num_classes)

    def forward(self, x):
        out = self.fc(x)
        out = F.softmax(out, dim=1)
        return out

class CosineMarginProduct(nn.Module):
    def __init__(self, in_feature=512, out_feature=85742, s=30.0, m=0.35):
        super(CosineMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)


    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = self.s * (cosine - one_hot * self.m)
        return output

def ClassifierF(num_classes=85742):
    return Classifier(num_classes=num_classes)