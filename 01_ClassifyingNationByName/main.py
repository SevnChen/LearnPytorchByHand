import random
import torch
import torch.nn as nn
from model import RNN_name
from utils import data_process


ALL_LETTERS = string.ascii_letters + '.,;'
N_LETTERS = len(ALL_LETTERS)


def category_from_output(output):
    top_n, top_i = output.data.topl(1)
    category_i = top_i[0][0]
    return ALL_LETTERS[category_i], category_i


def main():
    # read data
    category_names, all_category, n_category = data_process.read_files(
        './data', 'txt')
    # process data
    n_hidden = 128
    rnnname = RNN_name(N_LETTERS, n_hidden, n_category)
    criterion = nn.NLLLoss()
    learning_rate = 0.005
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

if __name__ == '__main__':
    main()
