from data_reader.dataReader import DataReader
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import numpy as np
import torch

def compute_adjustment(train_loader, tro=1.0):
    """
    compute the base probabilities
    :param train_loader:
    :param tro: 温度常数默认为1
    :return:
    """
    label_freq = {}
    for i, batch in enumerate(train_loader):
        label = batch[-1].tolist()
        for j in label:
            if j not in label_freq:
                label_freq[j] = 1
            else:
                label_freq[j] += 1

    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    return adjustments











if __name__ == '__main__':
    path = '../data/train_clean.xlsx'
    model_path = '../pretrained_models/chinese-bert-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    train_dataset = DataReader(tokenizer=tokenizer, filepath=path, max_len=102)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    adjustments = compute_adjustment(train_dataloader)
    print(adjustments)

    output = torch.randn((2,3))
    print(output)
    output += adjustments
    print(output)