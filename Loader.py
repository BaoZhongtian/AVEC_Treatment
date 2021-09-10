import os
import json
import tqdm
import numpy
import torch
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, treat_data, treat_label):
        self.data = treat_data
        self.label = treat_label
        assert len(self.data) == len(self.label)

    def __getitem__(self, index):
        raw_data = self.data[index]
        padding_data = []
        data_length = [len(_) for _ in raw_data]
        for sample in raw_data:
            padding_data.append(numpy.concatenate(
                [sample, numpy.zeros([max(data_length) - numpy.shape(sample)[0], numpy.shape(sample)[1]])], axis=0))

        padding_data = torch.FloatTensor(padding_data)
        data_length = torch.LongTensor(data_length)
        label = torch.FloatTensor([self.label[index]])
        return padding_data, data_length, label

    def __len__(self):
        return len(self.data)


def loader_audio(appoint_part):
    def loader_treatment(part_name, shuffle_flag):
        treat_data, treat_label = [], []

        treat_label_raw = numpy.genfromtxt(
            fname=os.path.join(load_path, '%s_split_Depression_AVEC2017.csv' % part_name),
            dtype=str, delimiter=',')
        treat_label_dictionary = {}
        for index in range(1, len(treat_label_raw)):
            treat_label_dictionary[int(treat_label_raw[index][0])] = int(treat_label_raw[index][2])

        for key in tqdm.tqdm(treat_label_dictionary.keys()):
            current_data = json.load(
                open(os.path.join(load_path, 'AudioData_' + appoint_part, part_name, '%d_P.json' % key), 'r'))
            treat_data.append(current_data)
            treat_label.append(treat_label_dictionary[key])

            # if len(treat_data) > 1: break

        train_dataset = AudioDataset(treat_data, treat_label)
        treat_loader = DataLoader(train_dataset, batch_size=1, shuffle=shuffle_flag)
        return treat_loader

    load_path = 'C:/ProjectData/AVEC2017_Treatment/TreatedData/'
    train_loader = loader_treatment(part_name='train', shuffle_flag=True)
    val_loader = loader_treatment(part_name='dev', shuffle_flag=False)
    return train_loader, val_loader


def loader_text(appoint_part):
    def loader_treatment(part_name, shuffle_flag):
        treat_data, treat_label = [], []

        treat_label_raw = numpy.genfromtxt(
            fname=os.path.join(load_path, '%s_split_Depression_AVEC2017.csv' % part_name),
            dtype=str, delimiter=',')
        treat_label_dictionary = {}
        for index in range(1, len(treat_label_raw)):
            treat_label_dictionary[int(treat_label_raw[index][0])] = int(treat_label_raw[index][2])

        for key in tqdm.tqdm(treat_label_dictionary.keys()):
            current_data = json.load(
                open(os.path.join(load_path, 'Text_Change2Vec_%s' % appoint_part, part_name, '%d_P.json' % key), 'r'))
            treat_data.append(current_data)
            treat_label.append(treat_label_dictionary[key])

            # if len(treat_data) > 1: break

        train_dataset = AudioDataset(treat_data, treat_label)
        treat_loader = DataLoader(train_dataset, batch_size=1, shuffle=shuffle_flag)
        return treat_loader

    load_path = 'C:/ProjectData/AVEC2017_Treatment/TreatedData/'
    train_loader = loader_treatment(part_name='train', shuffle_flag=True)
    val_loader = loader_treatment(part_name='dev', shuffle_flag=False)
    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = loader_audio('ALL')
    for treat_data, treat_seq, treat_label in train_loader:
        print(numpy.shape(treat_data), numpy.shape(treat_seq), treat_label)
