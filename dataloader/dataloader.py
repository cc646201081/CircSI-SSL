import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from data_preprocessing.CRBP.getDataView import get_data


class Load_Dataset(Dataset):
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()

        X_data1 = dataset["samples1"]
        X_data2 = dataset["samples2"]
        X_data3 = dataset["samples3"]

        y_data = dataset["labels"]

        self.x_data1 = X_data1
        self.x_data2 = X_data2
        self.x_data3 = X_data3
        self.y_data = y_data.long()
        self.len = X_data1.shape[0]


    def __getitem__(self, index):

        return self.x_data1[index], self.x_data2[index], self.x_data3[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator(protein, configs, training_mode, ratio):

    all_dataset = get_data(protein)

    train_dataset = dict()
    test_dataset = dict()

    if training_mode == 'self_supervised':
        train_dataset = all_dataset
        test_dataset = all_dataset
    else:
        m = all_dataset["samples1"].shape[0]
        train_dataset["samples1"] = all_dataset["samples1"][:round((m / 10) * ratio)]
        test_dataset["samples1"] = all_dataset["samples1"][round((m / 10) * ratio):]

        train_dataset["samples2"] = all_dataset["samples2"][:round((m / 10) * ratio)]
        test_dataset["samples2"] = all_dataset["samples2"][round((m / 10) * ratio):]

        train_dataset["samples3"] = all_dataset["samples3"][:round((m / 10) * ratio)]
        test_dataset["samples3"] = all_dataset["samples3"][round((m / 10) * ratio):]

        train_dataset["labels"] = all_dataset["labels"][:round((m / 10) * ratio)]
        test_dataset["labels"] = all_dataset["labels"][round((m / 10) * ratio):]


    train_dataset = Load_Dataset(train_dataset)
    test_dataset = Load_Dataset(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=True,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=True, drop_last=True,
                                              num_workers=0)

    return train_loader, test_loader
