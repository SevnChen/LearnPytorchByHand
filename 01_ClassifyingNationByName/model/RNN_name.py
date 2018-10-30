import torch
import torch.nn as nn
from torch.autograd import Variable

class rnn_name(nn.Module):
    def __init__(self, size_input, size_hidden, size_output, size_batch
                 , max_length, feature_dim
                 , criterion=nn.NLLLoss(), learning_rate=0.1
                 , optimizer=torch.optim.SGD):
        super(rnn_name, self).__init__()

        self.size_input = size_input
        self.size_hidden = size_hidden
        self.size_output = size_output
        self.size_batch = size_batch
        self.max_length = max_length
        self.feature_dim = feature_dim

        self.i2h = nn.Linear(size_input+size_hidden, size_hidden)
        self.i2o = nn.Linear(size_input+size_hidden, size_output)
        self.softmax = nn.LogSoftmax()
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.size_hidden))

    def train(self, category_tensor, name_tensor):
        loss_all = []
        for name_ix in range(name_tensor.size()[0]):
            self.zero_grad()
            hidden = self.init_hidden()
            for i in range(name_tensor[name_ix].size()[0]):
                if name_tensor[name_ix][i].sum().item()>0:
                    output, hidden = self.forward(name_tensor[name_ix][i].view(1,-1), hidden)
                else:
                    break
            loss = self.criterion(output, category_tensor[name_ix])
            loss.backward()
            loss_all.append(loss.item())
            self.optimizer.step()
        return output, loss_all

    def evaluate(self, name_tensor):
        hidden = self.init_hidden()
        for i in range(name_tensor.size()[0]):
            if name_tensor[i].sum().item()>0:
                output, hidden = self.forward(name_tensor[i].view(1,-1), hidden)
            else:
                break
        return output

    def predict(self, input_name, transform, all_category, n_predictions=3):
        print('\n>{}'.format(input_name))
        output = self.evaluate(Variable(transform(input_name, self.max_length)))
        # get top
        topv, topi = output.data.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i]
            category_index = topi[0][i]
            print('{:.2f} {}'.format(value, all_category[category_index]))
            predictions.append([value, all_category[category_index]])
        return predictions
