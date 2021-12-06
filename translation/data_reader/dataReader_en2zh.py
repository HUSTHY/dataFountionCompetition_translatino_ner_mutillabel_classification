
from tqdm import tqdm
import torch
import pandas as pd

class DataReader(object):
    def __init__(self,tokenizer,filepath,max_len = 512):
        self.tokenizer = tokenizer
        self.filepath = filepath
        self.max_len = max_len
        self.dataList = self.datas_to_torachTensor()
        self.allLength = len(self.dataList)

    def convert_text2ids_source(self,text):
        # text = text[0:self.max_len-2]

        inputs = self.tokenizer(text, max_length=512, padding='max_length', truncation=True)
        input_ids = inputs['input_ids']

        attention_mask = inputs['attention_mask']
        # input_length = len(input_ids)

        return input_ids, attention_mask

    def convert_text2ids_target(self, text):
        # text = text[0:self.max_len-2]

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            inputs = self.tokenizer(text, max_length=512, padding='max_length', truncation=True)

        input_ids = inputs['input_ids']

        input_ids = [  l if l != self.tokenizer.pad_token_id else -100  for l in input_ids]


        # attention_mask = inputs['attention_mask']
        # input_length = len(input_ids)

        return input_ids



    def datas_to_torachTensor(self):
        with open(self.filepath,'r',encoding='utf-8') as f:
            lines = f.readlines()
        res = []
        for line in tqdm(lines[0:], desc='tokenization', ncols=50):
            line = line.strip('\n').split('\t')
            temp = []
            if len(line) > 1:

                input_ids_srg, attention_mask_srg = self.convert_text2ids_source(text=line[1])
                # 目标语言的Ids就是训练的labels
                input_ids_tag = self.convert_text2ids_target(text=line[0])
                labels = input_ids_tag


            else:
                input_ids_srg, attention_mask_srg = self.convert_text2ids_source(text=line[0])
                input_ids_srg = torch.as_tensor(input_ids_srg, dtype=torch.long)
                attention_mask_srg = torch.as_tensor(attention_mask_srg, dtype=torch.long)

                labels = torch.zeros(1,1)


                # input_ids_srg, attention_mask_srg = self.convert_text2ids_source(text=line[0])
                # labels = [1]

                
            temp.append(input_ids_srg)
            temp.append(attention_mask_srg)
            temp.append(labels)

            res.append(temp)

        return res

    def __getitem__(self, item):
        input_ids_srg = self.dataList[item][0]
        attention_mask_srg = self.dataList[item][1]

        labels = self.dataList[item][2]

        return {'input_ids':input_ids_srg,'attention_mask':attention_mask_srg,'labels':labels}




    def __len__(self):
        return self.allLength