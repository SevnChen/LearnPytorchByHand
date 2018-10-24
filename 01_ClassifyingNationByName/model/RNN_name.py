import torch
import torch.nn as nn
from torch.autograd import Variable

class rnn_name(nn.Module):
    def __init__(self, size_input, size_hidden, size_output, criterion=nn.NLLLoss, learning_rate=0.1
                 , optimizer=torch.optim.SGD):
        super(rnn_name, self).__init__()

        self.size_input = size_input
        self.size_hidden = size_hidden
        self.size_output = size_output

        self.i2h = nn.Linear(size_input+size_hidden, size_hidden)
        self.i2o = nn.Linear(size_input+size_hidden, size_output)
        self.softmax = nn.LogSoftmax()
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)

    def forward(self, input, hidden):
        combined = torch.cat(input, hidden)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.size_hidden))
    
    def train(self, category_tensor, name_tensor):
        self.zero_grad()
        hidden = self.init_hidden()

        for i in range(name_tensor.size()[0]):
            output, hidden = self(name_tensor[i], hidden)

        loss = self.criterion(output, category_tensor)
        loss.backward()

        self.optimizer.step()

        return output, loss
