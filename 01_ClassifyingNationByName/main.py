import os
import string
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import RNN_name
from utils import data_process, data_loaders, looger
from importlib import reload

ALL_LETTERS = string.ascii_letters + '.,;'
N_LETTERS = len(ALL_LETTERS)

# reload(data_loaders)
# reload(data_process)
# reload(RNN_name)


def category_from_output(output):
    top_n, top_i = output.data.topl(1)
    category_i = top_i[0][0]
    return ALL_LETTERS[category_i], category_i


def read_config(addr='config.json'):
    con = open(addr, 'r').read()
    config = json.loads(con)
    return config


def save_model(model, save_dir, savr_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    savr_prefix = os.path.join(save_dir, savr_prefix)
    save_path = '{}_steps_{}.pt'.format(savr_prefix, steps)
    torch.save(model.state_dict(), save_path)

def main():
    # read config
    config = read_config()
    config['criterion'] = nn.NLLLoss()
    config['optimizer'] = torch.optim.SGD
    config['size_batch'] = 1
    config['feature_dim'] = N_LETTERS
    config['size_input'] = N_LETTERS
    # read data
    category_names, all_category, n_category, max_length = data_process.read_files(
        './data', 'txt')
    config['max_length'] = max_length
    # create dataste
    name_dataset = data_loaders.NameDataset(
        data=category_names, allcategory=all_category, max_length=max_length
        , transform=data_process.name_to_tensor)
    # create dataloader
    name_dataloader = DataLoader(name_dataset, batch_size=config['size_batch'], shuffle=True)
    # set config
    rnnname = RNN_name.rnn_name(**config)

    loss_all = []
    loss_all_batch = []
    min_loss = 1000
    for i_batch, sample_batch in enumerate(name_dataloader):
        output, loss_batch = rnnname.train(sample_batch[0], sample_batch[1])
        loss_all += loss_batch
        tmp = sum(loss_batch)/len(loss_batch)
        if tmp<min_loss: min_loss=tmp
        if i_batch%100==0: loss_all_batch.append(min_loss)
        if i_batch%1000==0:
            print('i_batch:{} | min_loss:{:0.7f}'.format(i_batch, min_loss))
            save_model(rnnname, './snapshot', 'RnnName', i_batch)
    return rnnname

if __name__ == '__main__':
    main()
    rnnname.predict(
        input_name='Hui'
        , transform=data_process.name_to_tensor
        , all_category=all_category
    )
