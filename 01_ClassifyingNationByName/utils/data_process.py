import glob
import string
import unicodedata
import torch


ALL_LETTERS = string.ascii_letters + '.,;'
N_LETTERS = len(ALL_LETTERS)
 

def unicode_to_ascii(s):
    """
    将unicode转化成ascii
        :param s: 
        :return s_ascii: 
    """
    s_ascii = ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS
    )
    return s_ascii


def read_files(dir_root_data='./data', file_type_data='txt'):
    """
    读取全部文件
    """
    dir_data_all=glob.glob(dir_root_data + '/*.' + file_type_data)
    category_names = {}
    all_category = []
    for dir_file in dir_data_all:
        category = dir_file.split('/')[-1].split('.')[0]
        all_category.append(category)
        names = [unicode_to_ascii(name) for name in open(
            dir_file).read().strip().split('\n')]
        category_names[category] = names
    n_category = len(all_category)
    print('n_category = {}'.format(n_category))
    return category_names, all_category, n_category
    

def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    letter_index = ALL_LETTERS.find(letter)
    tensor[0][letter_index] = 1
    return tensor


def name_to_tensor(name):
    tensor = torch.zeros(len(name), N_LETTERS)
    for index, letter in enumerate(name):
        letter_index = ALL_LETTERS.find(letter)
        tensor[index][letter_index] = 1
    return tensor


def main():
    pass

if __name__ == '__main__':
    main()

    unicode_to_ascii('Ślusàrski')
