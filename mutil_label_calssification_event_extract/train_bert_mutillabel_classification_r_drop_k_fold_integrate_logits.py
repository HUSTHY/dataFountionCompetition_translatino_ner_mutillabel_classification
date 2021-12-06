import torch
import argparse
from data_reader.dataReader import DataReader
from model.mutilLabel_classification import MutilLabelClassification
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import BertTokenizer,BertConfig
import os
from tools.log import Logger
from tools.progressbar import ProgressBar
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

log_file = 'log_files/'+'k_fold_MLCE_L512_bert_rdrop0_logit_integrate.log'
logger = Logger('mutil_label_logger',log_level=10,log_file=log_file)
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import pandas as pd
import json
import numpy as np
import random
from glob import glob

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--max_len",type=int,default=512)
    # parser.add_argument("--train_file", type=str,default='./data/train.csv', help="train text file")
    # parser.add_argument("--val_file", type=str, default='./data/dev.csv',help="val text file")
    parser.add_argument("--test_file", type=str, default='./data/test.csv', help="val text file")
    parser.add_argument("--pretrained", type=str, default="./pretrained_models/chinese-bert-wwm-ext", help="huggingface pretrained model")
    parser.add_argument("--model_out", type=str, default="./output", help="model output path")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=20, help="epochs")
    parser.add_argument("--lr", type=int, default=1e-5, help="epochs")
    parser.add_argument("--loss_function_type",type=str,default='MLCE')
    parser.add_argument('--is_rdrop',type=bool,default=False)
    parser.add_argument("--alpha", type=float, default=0, help="alpha")
    args = parser.parse_args()
    return args


def multilabel_crossentropy(output,label):
    """
    多标签分类的交叉熵
    说明：label和output的shape一致，label的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证output的值域是全体实数，换言之一般情况下output
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出output大于0的类。如有疑问，请仔细阅读并理解
         本文。
    :param output: [B,C]
    :param label:  [B,C]
    :return:
    """
    output = (1-2*label)*output

    #得分变为负1e12
    output_neg = output - label* 1e12
    output_pos = output-(1-label)* 1e12

    zeros = torch.zeros_like(output[:,:1])

    # [B, C + 1]
    output_neg = torch.cat([output_neg,zeros],dim=1)
    # [B, C + 1]
    output_pos = torch.cat([output_pos,zeros],dim=1)

    loss_pos = torch.logsumexp(output_pos,dim=1)
    loss_neg = torch.logsumexp(output_neg,dim=1)
    loss = (loss_neg + loss_pos).sum()
    return loss



def train(args):
    logger.info(args)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open('data/event_types.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
    event_types = [ele.strip('\n') for ele in lines]

    test_dataset = DataReader(tokenizer=tokenizer, filepath=args.test_file, max_len=args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    save_paths = ['./output/k_fold/MLCE/1/bert','./output/k_fold/MLCE/3/bert','./output/k_fold/MLCE/4/bert']

    models = []

    for save_path in save_paths:
        config = BertConfig.from_pretrained(save_path)
        config.num_labels = len(lines)
        model = MutilLabelClassification.from_pretrained(config=config, pretrained_model_name_or_path=save_path,max_len=args.max_len)
        model.to(device)
        model.eval()
        models.append(model)


    predicts_labels = prediction(models, test_dataloader, device, args)

    submit_path = 'submit/k_fold_20211112/kfold_MLCE_L512_bert_logit_integrate_submition_1_3_4.json'

    test_df = pd.read_csv(args.test_file)
    ids = test_df['id']
    texts = test_df['text']

    with open(submit_path, 'w', encoding='utf-8') as f:
        for id, text, predicts_label in zip(ids, texts, predicts_labels):
            dict = {}
            dict['id'] = id
            dict['text'] = text
            event_chain = []
            for index, val in enumerate(predicts_label):
                if val == 1:
                    event_chain.append(event_types[index])
            dict['event_chain'] = event_chain
            s = json.dumps(dict, ensure_ascii=False)
            f.write(s + '\n')








def compute_kl_loss(p, q,pad_mask = None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)
    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()
    loss = (p_loss + q_loss) / 2
    return loss


def prediction(models,test_dataloader,device,args):
    predicts_labels = []

    with torch.no_grad():
        pbar = ProgressBar(n_total=len(test_dataloader), desc='prediction')
        for step, batch in enumerate(test_dataloader):
            batch = [t.to(device) for t in batch]
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
            outputs = []
            for model in models:
                output = model(inputs)
                if args.loss_function_type == "BCE":
                    output = F.log_softmax(output)
                outputs.append(output)
            t = torch.zeros_like(outputs[0])
            for output in outputs:
                t += output

            # 注意这里统计模型指标正确率的代码逻辑，torch.where()和torch.equal()
            if args.loss_function_type == "BCE":
                output = torch.sigmoid(t)
                pred = torch.where(output > 0.5, 1, 0)
            else:
                pred = torch.where(t > 0, 1, 0)
            predicts_labels.extend(pred.detach().cpu().tolist())
            pbar(step, {'step':step})

    return predicts_labels








def mertics_compute(label,predict):
    micro_f1 = f1_score(label, predict, average='micro')
    macro_f1 = f1_score(label, predict, average='macro')
    acc = accuracy_score(label, predict)
    return acc, micro_f1, macro_f1



def valdation(model,val_dataloader,device,args):
    model.eval()
    predict_labels = []
    labels_all = []
    with torch.no_grad():
        pbar = ProgressBar(n_total=len(val_dataloader), desc='evaldation')
        for step, batch in enumerate(val_dataloader):
            batch = [t.to(device) for t in batch]
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
            labels = batch[3]
            output = model(inputs)

            #注意这里统计模型指标正确率的代码逻辑，torch.where()和torch.equal()
            if args.loss_function_type == "BCE":
                output = torch.sigmoid(output)
                pred = torch.where(output>0.5,1,0)
            else:
                pred = torch.where(output>0,1,0)

            # correct = 0
            # for i in range(labels.size()[0]):
            #     if torch.equal(pred[i],labels[i]):
            #         correct +=1
            labels_all.extend(labels.detach().cpu().tolist())
            predict_labels.extend(pred.detach().cpu().tolist())

            pbar(step,{'step':step})

    acc, micro_f1, macro_f1 = mertics_compute(label=labels_all,predict=predict_labels)

    return acc, micro_f1, macro_f1


def main():
    seed = 1
    setup_seed(seed)
    args =parse_args()
    train(args)


if __name__ == '__main__':
    main()