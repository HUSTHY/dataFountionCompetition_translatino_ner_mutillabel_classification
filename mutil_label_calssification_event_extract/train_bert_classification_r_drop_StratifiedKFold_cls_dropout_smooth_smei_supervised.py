import torch
import argparse
from data_reader.dataReader import DataReader
from model.mutilLabel_classification import MutilLabelClassification

from torch.utils.data import DataLoader
import random
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import BertTokenizer,BertConfig, get_linear_schedule_with_warmup
import os
from tools.progressbar import ProgressBar
from datetime import datetime
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tools.log import Logger
log_file = 'log_files/'+'v0_bert_integrate_MLCE_rdrop0_RRL_scheduler.log'
logger = Logger('mutil_label_logger',log_level=10,log_file=log_file)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from loss_function.labelsmoothloss import LabelSmoothingLoss
from sklearn.model_selection import KFold
from utils.adversarial_model import FGM,PGD
from sklearn.metrics import accuracy_score, f1_score
import json
from collections import defaultdict
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--max_len",type=int,default=512)
    parser.add_argument("--train_file", type=str, default='./data/train_dev.csv', help="val text file")
    parser.add_argument("--test_file", type=str, default='./data/test.csv', help="val text file")
    parser.add_argument("--pretrained", type=str, default="./pretrained_models/chinese-bert-wwm-ext", help="huggingface pretrained model")
    parser.add_argument("--model_out", type=str, default="./output", help="model output path")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=20, help="epochs")
    parser.add_argument("--random_seed", type=int, default=1, help="random_seed")
    parser.add_argument("--lr", type=float, default=1e-5, help="epochs")
    parser.add_argument('--warmup_steps', default=290, type=int, required=False, help='warm up步数')
    parser.add_argument("--integrate_type", type=str, default="logits", help="model output path")

    # adversarial training
    parser.add_argument("--adversarial", default=False, type=bool,
                        help="Whether to adversarial training.")
    parser.add_argument('--adv_epsilon', default=0.5, type=float,
                        help="Epsilon for adversarial.")
    parser.add_argument('--adv_name', default='word_embeddings', type=str,
                        help="name for adversarial layer.")

    parser.add_argument('--adversarial_type', default='PGD', type=str, help="")

    #论文中1-5之间
    parser.add_argument("--alpha", type=float, default=0, help="alpha")
    args = parser.parse_args()
    return args


def compute_kl_loss(p, q,pad_mask = None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)
    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()
    loss = (p_loss + q_loss) / 2
    return loss


def collate_fn(batch):
    # 动态处理batch中seq的长度,保持等长，节约显存
    input_ids_a, attention_mask_a, token_type_id_a, label, input_len = map(torch.stack, zip(*batch))
    max_len = max(input_len).item()
    input_ids_a = input_ids_a[:, :max_len]
    attention_mask_a = attention_mask_a[:, :max_len]
    token_type_id_a = token_type_id_a[:, :max_len]
    return input_ids_a, attention_mask_a, token_type_id_a, label




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




def train_and_prediction(args):
    logger.info("args: %s"%args)


    with open('data/event_types.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
    event_types = [ele.strip('\n') for ele in lines]

    tokenizer = BertTokenizer.from_pretrained(args.pretrained)
    config = BertConfig.from_pretrained(args.pretrained)
    config.num_labels = len(lines)
    args.num_labels = len(lines)
    # config.attention_probs_dropout_prob = 0.3
    # config.hidden_dropout_prob = 0.3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = DataReader(tokenizer=tokenizer, filepath=args.test_file, max_len=args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    train_df = pd.read_csv('data/train.csv')
    dev_df = pd.read_csv('data/dev.csv')
    df = pd.concat([train_df, dev_df])
    df['id'] = list(range(len(df)))
    print(len(df))
    skf = KFold(n_splits=5)

    model_save_paths = []
    model_type = args.pretrained.split('/')[-1]
    best_mico_f1_save_path = defaultdict(float)

    best_mico_f1s_json_path = 'submit/v0_submition_MLCE_L512_rdrop_' + str(args.alpha) + '_RRL_scheduler_mico_f1s_save_path.json'

    for fold,(train_index,dev_index) in enumerate(skf.split(df)):
        logger.info('================fold {}==============='.format(fold))
        time_srt = datetime.now().strftime('%Y-%m-%d')
        save_path = os.path.join(args.model_out, "MutilLabelClassification","v0" + time_srt + '_rdrop_' + str(args.alpha) + '_' + model_type, '5_kfold_RRL_scheduler',str(fold))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_save_paths.append(save_path)

        writer = SummaryWriter('./runs/fold'+str(fold))
        train_df = df[df['id'].isin(train_index)]
        dev_df = df[df['id'].isin(dev_index)]

        train_dataset = DataReader(tokenizer=tokenizer, filepath=train_df, max_len=args.max_len)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        dev_dataset = DataReader(tokenizer=tokenizer, filepath=dev_df, max_len=args.max_len)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)


        #model init
        model = MutilLabelClassification.from_pretrained(config=config, pretrained_model_name_or_path=args.pretrained)

        model.to(device)

        # # 优化器定义
        # param_optimizer = list(model.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        #
        # optimizer = AdamW(optimizer_grouped_parameters,lr=args.lr)



        # total_steps = int(train_dataset.__len__() * args.epochs / args.batch_size)
        # print('total_steps', total_steps)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,num_training_steps=total_steps)




        optimizer = AdamW(model.parameters(), lr=args.lr)
        RRL_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=5)

        best_val_acc = 0.0
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d"% (len(train_dataloader)))
        logger.info("  Num Epochs = %d"%(args.epochs) )
        global_step = 0

        if args.adversarial:
            if args.adversarial_type == 'GFM':
                fgm = FGM(model, epsilon=0.5, emb_name='word_embeddings.')
            else:
                pgd = PGD(model,emb_name='word_embeddings.',epsilon=0.5,alpha=0.3)

        for epoch in range(args.epochs):
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
            model.train()
            for step,batch in enumerate(train_dataloader):
                batch = [t.to(device) for t in batch]
                inputs = {'input_ids':batch[0],'attention_mask':batch[1],'token_type_ids':batch[2]}
                labels = batch[3]

                #r-drop
                logits1 = model(inputs)

                if args.alpha>0:
                    logits2 = model(inputs)
                    # cross entropy loss for classifier
                    ce_loss = 0.5 * (multilabel_crossentropy(logits1, labels) + multilabel_crossentropy(logits2, labels))
                    kl_loss = compute_kl_loss(logits1, logits2)
                    # carefully choose hyper-parameters
                    loss = ce_loss +  args.alpha * kl_loss
                else:
                    loss = multilabel_crossentropy(logits1, labels)

                loss.backward()

                if args.adversarial:
                    if args.adversarial_type == "FGM":
                        fgm.attack()  # 在embedding上添加对抗扰动
                        output_adv, _ = model(inputs)
                        loss_adv = F.cross_entropy(output_adv, labels)
                        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                        fgm.restore()  # 恢复embedding参数
                    else:
                        pgd.backup_grad()
                        K= 3
                        for t in range(K):
                            pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                            if t != K - 1:
                                model.zero_grad()
                            else:
                                pgd.restore_grad()
                            output_adv, _ = model(inputs)
                            loss_adv = F.cross_entropy(output_adv, labels)
                            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                        pgd.restore()  # 恢复embedding参数

                    writer.add_scalar('Train/loss_adv', loss_adv.item(), global_step)


                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()
                pbar(step, {'loss':loss.item()})
                global_step += 1
                # scheduler.step()
                writer.add_scalar('Train/Loss', loss.item(), global_step)

            train_acc, _, _ = valdation(model, dev_dataloader, device)

            val_acc, micro_f1, macro_f1 = valdation(model, dev_dataloader, device)

            RRL_scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                logger.info("save model")
                model.save_pretrained(save_path)
                tokenizer.save_vocabulary(save_path)
            logger.info("Epoch: %d  train_acc:%.6f  val_acc:%.6f     best_val_acc:%.6f ------macro_f1:%.6f   micro_f1:%.6f  " % (epoch,train_acc,val_acc, best_val_acc,macro_f1, micro_f1))
            writer.add_scalar('Dev/val_acc', val_acc, global_step)
            writer.add_scalar('Dev/best_val_acc', best_val_acc, global_step)
            writer.add_scalar('Dev/micro_f1', micro_f1, global_step)



        best_mico_f1_save_path[save_path] = best_val_acc


        writer.close()


    with open(best_mico_f1s_json_path,'w',encoding='utf-8') as f:
        json.dump(best_mico_f1_save_path,f,ensure_ascii=False,indent='   ')

    with open(best_mico_f1s_json_path,'r',encoding='utf-8') as f:
        best_mico_f1_save_path = json.load(f)

    best_mico_f1_save_path = sorted(best_mico_f1_save_path.items(),key=lambda x:x[1],reverse=True)

    model_save_paths = [ ele[0]  for ele in best_mico_f1_save_path[0:3]]


    args.integrate_type = 'vote'
    final_predict_result = integrate_predicting(model_save_paths,device,test_dataloader,args,vote_alpha=0.6)

    file_name = 'submit/v0_submition_MLCE_L512_rdrop_'+ str(args.alpha) + '_RRL_scheduler_vote_integrate_max3_20211113.json'
    submit_json_files(file_name, final_predict_result, args, event_types)

    args.integrate_type = 'logits'
    final_predict_result = integrate_predicting(model_save_paths, device, test_dataloader, args)

    file_name = 'submit/v0_submition_MLCE_L512_rdrop_'+ str(args.alpha) + '_RRL_scheduler_logits_integrate_max3_20211113.json'
    submit_json_files(file_name, final_predict_result, args, event_types)




def submit_json_files(file_name,predicts_labels,args,event_types):
    test_df = pd.read_csv(args.test_file)
    ids = test_df['id']
    texts = test_df['text']

    with open(file_name, 'w', encoding='utf-8') as f:
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
    print('model inference finished!')









def integrate_predicting(model_save_paths,device,test_dataloader,args,vote_alpha=None):
    final_predict_result = []

    models = []
    for save_path in model_save_paths:
        logger.info('------------{}------------'.format(save_path))
        config = BertConfig.from_pretrained(save_path)
        config.num_labels = args.num_labels
        model = MutilLabelClassification.from_pretrained(config=config, pretrained_model_name_or_path=save_path)
        model.to(device)
        model.eval()
        models.append(model)


    if args.integrate_type == 'vote':
        vote_num = len(models) * vote_alpha
        with torch.no_grad():
            pbar = ProgressBar(n_total=len(test_dataloader), desc='prediction')
            for step, batch in enumerate(test_dataloader):
                batch = [t.to(device) for t in batch]
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
                for index,model in enumerate(models):
                    output = model(inputs)
                    pred = torch.where(output > 0, 1, 0)
                    if index == 0:
                        batch_allmodel_pred = pred
                    else:
                        batch_allmodel_pred += pred

                pred = torch.where(batch_allmodel_pred>vote_num,1,0)
                pred = pred.detach().cpu().tolist()
                final_predict_result.extend(pred)
                pbar(step, {'step': step})

        return final_predict_result
    else:
        with torch.no_grad():
            pbar = ProgressBar(n_total=len(test_dataloader), desc='prediction')
            for step, batch in enumerate(test_dataloader):
                batch = [t.to(device) for t in batch]
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
                for index,model in enumerate(models):
                    output = model(inputs)
                    if index == 0:
                        logits = output
                    else:
                        logits += output
                pred = torch.where(output > 0, 1, 0)
                pbar(step, {'step': step})
                pred = pred.detach().cpu().tolist()
                final_predict_result.extend(pred)
        return final_predict_result




def prediction(model,val_dataloader,device):
    model.eval()
    predictlabels = []
    with torch.no_grad():
        pbar = ProgressBar(n_total=len(val_dataloader), desc='prediction')
        for step, batch in enumerate(val_dataloader):
            batch = [t.to(device) for t in batch]
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
            output,_ = model(inputs)
            pred = torch.argmax(output, dim=1)
            pbar(step, {'step': step})
            pred = pred.detach().cpu().tolist()
            predictlabels.extend(pred)

    return predictlabels




def valdation(model,val_dataloader,device):
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

            # 注意这里统计模型指标正确率的代码逻辑，torch.where()和torch.equal()
            pred = torch.where(output > 0, 1, 0)

            # correct = 0
            # for i in range(labels.size()[0]):
            #     if torch.equal(pred[i],labels[i]):
            #         correct +=1
            labels_all.extend(labels.detach().cpu().tolist())
            predict_labels.extend(pred.detach().cpu().tolist())

            pbar(step, {'step': step})

    acc, micro_f1, macro_f1 = mertics_compute(label=labels_all, predict=predict_labels)

    return acc, micro_f1, macro_f1

def mertics_compute(label,predict):
    micro_f1 = f1_score(label, predict, average='micro')
    macro_f1 = f1_score(label, predict, average='macro')
    acc = accuracy_score(label, predict)
    return acc, micro_f1, macro_f1


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True





def main():
    args =parse_args()
    setup_seed(args.random_seed)
    train_and_prediction(args)




if __name__ == '__main__':
    main()











