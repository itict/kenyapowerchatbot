import torch
import torch.nn as nn

class KPLCChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(KPLCChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  # Update num_classes to match your pretrained model

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
