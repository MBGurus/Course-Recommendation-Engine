import torch
import torch.nn as nn


class Chat(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Chat, self).__init__()
        self.line1 =nn.Linear(input_size, hidden_size)
        self.line2 =nn.Linear(hidden_size, hidden_size)
        self.line3 =nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self,x):
        out= self.line1(x)
        out= self.relu(out)
        out= self.line2(out)
        out= self.relu(out)
        out= self.line3(out)
# Final output layer (no activation here)

        return out
        