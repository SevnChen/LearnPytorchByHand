import string
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import RNN_name
from utils import data_process, data_loaders
from utils import data_loaders
from importlib import reload

ALL_LETTERS = string.ascii_letters + '.,;'
N_LETTERS = len(ALL_LETTERS)

reload(data_loaders)
reload(data_process)
reload(RNN_name)

def category_from_output(output):
    top_n, top_i = output.data.topl(1)
    category_i = top_i[0][0]
    return ALL_LETTERS[category_i], category_i


def train(rnn, category_tensor, name_tensor):
    rnn.zero_grad()
    hidden = rnn.init_hidden()

    for i in range(name_tensor.size()[0]):
        output, hidden = rnn(name_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data[0]


class Config():
    size_input = 55
    size_hidden = 128
    size_output = 18
    criterion = nn.NLLLoss
    optimizer = torch.optim.SGD
    learning_rate = 0.005

def main():
    # read data
    category_names, all_category, n_category = data_process.read_files(
        './data', 'txt')
    # create dataste
    name_dataset = data_loaders.NameDataset(
        data=category_names, allcategory=all_category, transform=data_process.name_to_tensor)
    # create dataloader
    name_dataloader = DataLoader(name_dataset, batch_size=1, shuffle=True)
    # set config
    config = Config()
    dir(Config)
    rnnname = RNN_name(**config)
    for i_batch, sample_batch in enumerate(name_dataloader):
        output, loss = rnnname.train(sample_batch[0], sample_batch[1])
        current_loss += loss

if __name__ == '__main__':
    main()
    len(name_dataset)
    name_dataset.len()
    i = list(category_names.keys())[0]
    tmp = category_names[i]
    list(zip([i]*len(tmp), tmp))
