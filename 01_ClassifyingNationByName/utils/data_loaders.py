import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from utils import data_process

class NameDataset(Dataset):
    """
        Name dataset.
        use:DataLoader(train_dataset2, batch_size=8, shuffle=True)
    """

    def __init__(self, data, allcategory, transform=None):
        self.data = data
        self.all_category = {
            category: Variable(torch.LongTensor([allcategory.index(category)]))
            for category in allcategory}
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        lable, name = self.data[index]
        if self.transform:
            name = self.transform(name)
            lable = self.all_category[lable]
        return lable, name
