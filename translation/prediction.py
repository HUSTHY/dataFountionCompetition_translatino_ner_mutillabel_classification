import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/MarianMTModel_zh2en")

    model = AutoModelForSeq2SeqLM.from_pretrained("./pretrained_models/MarianMTModel_zho_eng")


    #
    # text = ['我爱你中国','我爱<unk>中国']
    # a = tokenizer.tokenize('我爱你中国')
    # print(a)
    # a = tokenizer.tokenize('我爱<unk>中国')
    # print(a)
    # inputs = tokenizer(text,max_length=10, padding='max_length', truncation=True)
    # print(inputs)



    dataset = DataReader(tokenizer,filepath='data/test_dataset.csv')

    test_dataloader = DataLoader(dataset=dataset,batch_size=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    finanl_result = []

    for batch in tqdm(test_dataloader,desc='translation prediction'):
        batch = [ t.to(device) for t in batch]
        batch = {'input_ids':batch[0],'attention_mask':batch[1]}
        # Perform the translation and decode the output
        translation = model.generate(**batch, top_k=5, num_return_sequences=1,num_beams=1)
        batch_result = tokenizer.batch_decode(translation, skip_special_tokens=True)
        finanl_result.extend(batch_result)


    print(len(finanl_result))

    with open('submit/submit_example.txt','w',encoding='utf-8') as f:
        for line in finanl_result:
            f.write(line+'\n')
