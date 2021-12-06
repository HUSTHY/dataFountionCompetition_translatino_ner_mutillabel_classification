
from tqdm import tqdm
import torch
import pandas as pd

class DataReader(object):
    def __init__(self,tokenizer,filepath,max_len):
        self.tokenizer = tokenizer
        self.filepath = filepath
        self.max_len = max_len
        self.dataList = self.datas_to_torachTensor()
        self.allLength = len(self.dataList)

    def convert_text2ids(self,text):
        text = text[0:self.max_len-2]
        inputs = self.tokenizer(text)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        input_length = len(input_ids)
        paddings = [0] * (self.max_len - len(input_ids))
        input_ids += paddings
        attention_mask += paddings

        token_type_id = [0] * self.max_len

        return input_ids, attention_mask, token_type_id, input_length


    def datas_to_torachTensor(self):
        # with open(self.filepath,'r',encoding='utf-8') as f:
        #     lines = f.readlines()
        if type(self.filepath) is str:
            df = pd.read_csv(self.filepath)
        else:
            df = self.filepath
        lines_a = df['text'].values.tolist()
        labels = df['label'].values.tolist()


        res = []
        for line_a, label in tqdm(zip(lines_a, labels), desc='tokenization', ncols=50):
            label =str(label)
            temp = []
            input_ids_a, attention_mask_a, token_type_id_a, input_len = self.convert_text2ids(text=line_a)
            input_ids_a = torch.as_tensor(input_ids_a, dtype=torch.long)
            attention_mask_a = torch.as_tensor(attention_mask_a, dtype=torch.long)
            token_type_id_a = torch.as_tensor(token_type_id_a, dtype=torch.long)
            input_len = torch.as_tensor(input_len, dtype=torch.long)
            label = [ int(ele) for ele in label.split('\t')]
            label = torch.as_tensor(label, dtype=torch.long)
            temp.append(input_ids_a)
            temp.append(attention_mask_a)
            temp.append(token_type_id_a)
            temp.append(label)
            temp.append(input_len)

            res.append(temp)
        return res

    def __getitem__(self, item):
        input_ids_a = self.dataList[item][0]
        attention_mask_a = self.dataList[item][1]
        token_type_id_a = self.dataList[item][2]
        label = self.dataList[item][3]
        input_len = self.dataList[item][4]
        return (input_ids_a, attention_mask_a, token_type_id_a, label, input_len)


    def __len__(self):
        return self.allLength