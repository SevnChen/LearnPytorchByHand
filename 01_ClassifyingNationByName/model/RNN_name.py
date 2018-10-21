import torch
import torch.nn as nn
from torch.autograd import Variable

class rnn_name(nn.Module):
    def __init__(self, size_input, size_hidden, size_output):
        super(rnn_name, self).__init__()

        self.size_input = size_input
        self.size_hidden = size_hidden
        self.size_output = size_output

        self.i2h = nn.Linear(size_input+size_hidden, size_hidden)
        self.i2o = nn.Linear(size_input+size_hidden, size_output)
        self.softmax = nn.LogSoftmax()
    def forward(self, input, hidden):
        combined = torch.cat(input, hidden)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    def init_hidden(self):
        return Variable(torch.zeros(1, self.size_hidden))
