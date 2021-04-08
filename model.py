import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, num_classes=101):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.fc(x)
        out = F.softmax(out, dim=1)
        return out

def ClassifierA(num_classes=101):
    return Classifier(num_classes=num_classes)



