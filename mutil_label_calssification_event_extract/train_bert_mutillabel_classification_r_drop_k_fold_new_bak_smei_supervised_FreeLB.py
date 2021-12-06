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
from sklearn.model_selection import KFold
from utils.adversarial_model import FGM,PGD
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from model.ema import EMA

log_file = 'log_files/'+'v20_k_5fold_MLCE_L512_bert_rdrop_0_FGM_ema_semisupervised_1_double_copy.log'
logger = Logger('mutil_label_logger',log_level=10,log_file=log_file)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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
    parser.add_argument("--model_type", type=str, default="bert", help="model output path")

    parser.add_argument("--model_out", type=str, default="./output", help="model output path")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=20, help="epochs")
    parser.add_argument("--lr", type=int, default=1e-5, help="epochs")
    parser.add_argument("--loss_function_type",type=str,default='MLCE')
    parser.add_argument("--version", type=str, default='v20', help="epochs")

    # adversarial training
    parser.add_argument("--adversarial", default=True, type=bool,
                        help="Whether to adversarial training.")
    parser.add_argument('--adv_epsilon', default=0.5, type=float,
                        help="Epsilon for adversarial.")
    parser.add_argument('--adv_name', default='word_embeddings', type=str,
                        help="name for adversarial layer.")

    parser.add_argument('--adversarial_type', default='FGM', type=str, help="")

    # 论文中1-5之间
    parser.add_argument("--alpha", type=float, default=0, help="alpha")

    #ema
    parser.add_argument("--ema", default=True, type=bool,
                        help="Whether to ema training.")

    parser.add_argument("--double_copy", default=True, type=bool,
                        help="Whether to ema training.")

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





def train(args):
    logger.info(args)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained)
    config = BertConfig.from_pretrained(args.pretrained)

    with open('data/event_types.txt','r',encoding='utf-8') as f:
        lines = f.readlines()

    event_types = [ele.strip('\n') for ele in lines]

    config.num_labels = len(lines)

    args.num_labels = config.num_labels

    device = "cuda" if torch.cuda.is_available() else "cpu"




    # paths = glob('data/k_fold/*/')
    # paths.sort()
    # print(paths)
    test_dataset = DataReader(tokenizer=tokenizer, filepath=args.test_file, max_len=args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    time_srt = datetime.now().strftime('%Y-%m-%d')


    if args.adversarial:
        best_mico_f1s_json_path = 'submit/'+ args.version + '_' + args.model_type + '_' + time_srt + '_'+ args.loss_function_type+'_L' + str(args.max_len) + '_rdrop_' + str(args.alpha) + '_semi_superviseed_double_copy_RRL_scheduler_best_val_acc_'+ args.adversarial_type+'_save_path.json'
    else:
        best_mico_f1s_json_path = 'submit/'+ args.version + '_' + args.model_type + '_' + time_srt + '_' + args.loss_function_type + '_L' + str(args.max_len) + '_rdrop_' + str(args.alpha) + '_semi_superviseed_double_copy_RRL_scheduler_best_val_acc_save_path.json'


    train_df = pd.read_csv('data/train.csv')[['id','text','label','text_length']]
    dev_df = pd.read_csv('data/dev.csv')[['id','text','label','text_length']]
    # df = pd.concat([train_df, dev_df])


    # # print('len(df)', len(df))
    # # # 长于600的数据直接丢弃
    # # df = df[df['text_length'] < 600]
    # # print('len(df)',len(df))
    # df['id'] = list(range(len(df)))
    # print('len(df)', len(df))
    # skf = KFold(n_splits=5)




    # test_df = pd.read_csv('submit/20211124_prediact_result_MLCE_vote_0.csv')
    # print('len(test_df)',len(test_df))
    # test_df = test_df[test_df['smallest_prob']>0.90][['id','text','label','text_length']]
    # print('len(test_df)',len(test_df))
    # df = pd.concat([train_df, dev_df, test_df])
    # df['id'] = list(range(len(df)))
    # print('len(df)', len(df))
    # skf = KFold(n_splits=5)



    test_df = pd.read_csv('submit/20211124_prediact_result_MLCE_vote_1.csv')
    print('len(test_df)',len(test_df))
    test_df = test_df[test_df['smallest_prob']>0.80][['id','text','label','text_length']]
    print('len(test_df)',len(test_df))
    df = pd.concat([train_df, dev_df,test_df])
    df['id'] = list(range(len(df)))
    print('len(df)',len(df))
    skf = KFold(n_splits=5)



    model_save_paths = []
    best_mico_f1_save_path = defaultdict(float)

    for fold, (train_index, dev_index) in enumerate(skf.split(df)):
        logger.info('================fold {}==============='.format(fold))
        time_srt = datetime.now().strftime('%Y-%m-%d')
        if args.adversarial:
            save_path = os.path.join(args.model_out, "MutilLabelClassification",
                                     args.version +"_" + time_srt + '_rdrop_' + str(args.alpha) + '_' + args.model_type + '_5_kfold_double_copy_RRL_scheduler_semi_supervised_1_' + args.loss_function_type + '_' + args.adversarial_type, str(fold))
        else:
            save_path = os.path.join(args.model_out, "MutilLabelClassification",
                                     args.version + "_" + time_srt + '_rdrop_' + str(args.alpha) + '_' + + args.model_type + '_5_kfold_double_copy_RRL_scheduler_semi_supervised_1_'+args.loss_function_type, str(fold))

        writer = SummaryWriter('./runs/fold' + str(fold))

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_save_paths.append(save_path)

        train_df = df[df['id'].isin(train_index)]
        dev_df = df[df['id'].isin(dev_index)]


        if args.double_copy:
            args.batch_size = int(args.batch_size/2)

        train_dataset = DataReader(tokenizer=tokenizer, filepath=train_df, max_len=args.max_len)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        val_dataset = DataReader(tokenizer=tokenizer, filepath=dev_df, max_len=args.max_len)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        # model init
        model = MutilLabelClassification.from_pretrained(config=config, pretrained_model_name_or_path=args.pretrained)

        model.to(device)


        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=5)

        if args.adversarial:
            if args.adversarial_type == 'FGM':
                fgm = FGM(model, epsilon=0.5, emb_name='word_embeddings.')
            else:
                pgd = PGD(model,emb_name='word_embeddings.',epsilon=0.1,alpha=0.3)


        if args.ema:
            ema = EMA(model=model, decay=0.999)
            ema.register()


        model.train()
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d" % len(train_dataloader))
        logger.info("  Num Epochs = %d" % args.epochs)
        logger.info("  args.loss_function_type : %s" % args.loss_function_type)
        if args.adversarial:
            logger.info(" adversarial_type: %s" % args.adversarial_type)
        best_acc = 0.0
        global_step = 0
        for epoch in range(args.epochs):
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
            for step, batch in enumerate(train_dataloader):
                batch = [t.to(device) for t in batch]

                if args.double_copy:
                    batch = [ t.repeat(2,1) for t in batch]

                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
                labels = batch[3]
                logits1 = model(inputs)
                if args.alpha>0:
                    logits2 = model(inputs)
                    # cross entropy loss for classifier
                    if args.loss_function_type == "BCE":
                        # 此处BCELoss的输入labels类型是必须和output一样的
                        ce_loss = 0.5 * (F.binary_cross_entropy_with_logits(logits1, labels.float()) + F.binary_cross_entropy_with_logits(logits2,labels.float()))
                    else:
                        # 多标签分类交叉熵
                        ce_loss = 0.5 * (multilabel_crossentropy(logits1, labels) + multilabel_crossentropy(logits2, labels))
                    kl_loss = compute_kl_loss(logits1, logits2)
                    # carefully choose hyper-parameters
                    loss = ce_loss +  args.alpha * kl_loss
                else:
                    if args.loss_function_type == "BCE":
                        # 此处BCELoss的输入labels类型是必须和output一样的
                        loss = F.binary_cross_entropy_with_logits(logits1, labels.float())
                    else:
                        # 多标签分类交叉熵
                        loss = multilabel_crossentropy(logits1, labels)

                loss.backward()


                if args.adversarial:
                    if args.adversarial_type == "FGM":
                        fgm.attack()  # 在embedding上添加对抗扰动
                        output_adv  = model(inputs)
                        if args.loss_function_type == "BCE":
                            loss_adv =  F.binary_cross_entropy_with_logits(output_adv, labels.float())
                        else:
                            loss_adv = multilabel_crossentropy(output_adv, labels)
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
                            output_adv = model(inputs)
                            if args.loss_function_type == "BCE":
                                loss_adv = F.binary_cross_entropy_with_logits(output_adv, labels.float())
                            else:
                                loss_adv = multilabel_crossentropy(output_adv, labels)
                            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                        pgd.restore()  # 恢复embedding参数

                    writer.add_scalar('Train/loss_adv', loss_adv.item(), global_step)

                writer.add_scalar('Train/Loss', loss.item(), global_step)

                optimizer.step()
                optimizer.zero_grad()
                pbar(step, {'loss': loss.item()})
                global_step += 1

                if args.ema:
                    ema.update()

            if args.ema:
                ema.apply_shadow()
            val_acc, micro_f1, macro_f1 = valdation(model, val_dataloader, device, args)
            scheduler.step(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                logger.info("save model")
                model.save_pretrained(save_path)#保存的是最佳影子权重
                tokenizer.save_vocabulary(save_path)
            logger.info(args.loss_function_type + "  Epoch %d    val_acc:%.6f------best_acc:%.6f   micro_f1:%.6f macro_f1:%.6f" % (epoch,val_acc, best_acc, micro_f1, macro_f1))

            writer.add_scalar('Dev/val_acc', val_acc, global_step)
            writer.add_scalar('Dev/best_val_acc', best_acc, global_step)
            writer.add_scalar('Dev/micro_f1', micro_f1, global_step)

            if args.ema:
                ema.restore()#恢复权重系数

        best_mico_f1_save_path[save_path] = best_acc


        config = BertConfig.from_pretrained(save_path)
        config.num_labels = len(lines)
        model = MutilLabelClassification.from_pretrained(config=config, pretrained_model_name_or_path=save_path,max_len=args.max_len)
        model.to(device)
        predicts_labels = prediction(model, test_dataloader, device, args)
        test_df = pd.read_csv(args.test_file)
        ids = test_df['id']
        texts = test_df['text']
        submit_dir = 'submit/5_fold_20211124/'
        if not os.path.exists(submit_dir):
            os.makedirs(submit_dir)
        submit_path = submit_dir + 'submition_' + args.loss_function_type + '_L' + str(args.max_len) + '_bert_double_copy_' + str(fold) + '.json'
        print(save_path)

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
        print(fold,'model training finished!')
        print('\n')
        print('\n')
        print('\n')


    best_mico_f1_save_path = sorted(best_mico_f1_save_path.items(), key=lambda x: x[1], reverse=True)


    with open(best_mico_f1s_json_path,'w',encoding='utf-8') as f:
        json.dump(best_mico_f1_save_path,f,ensure_ascii=False,indent='   ')

    with open(best_mico_f1s_json_path,'r',encoding='utf-8') as f:
        best_mico_f1_save_path = json.load(f)
    count = 5
    model_save_paths = [ele[0] for ele in best_mico_f1_save_path[0:count]]



    args.integrate_type = 'vote'
    vote_alpha = 0.6
    final_predict_result,final_prob_result = integrate_predicting(model_save_paths, device, test_dataloader, args, vote_alpha=vote_alpha)

    file_name = 'submit/' + args.version + '_' + args.model_type + '_submition_'+ args.loss_function_type + '_L' + str(args.max_len) + '_rdrop_' + str(
        args.alpha) + '_RRL_scheduler_semi_supervised_1_double_copy_' + args.adversarial_type + '_vote_integrate_vote_alpha_' + str(vote_alpha) + '_max_' + str(count) + '_20211124.json'
    submit_json_files(file_name, final_predict_result, args, event_types,final_prob_result)

    # args.integrate_type = 'logits'
    # final_predict_result,final_prob_result = integrate_predicting(model_save_paths, device, test_dataloader, args)
    #
    # file_name = 'submit/'+ args.version + '_submition_'+ args.loss_function_type + '_L512_rdrop_' + str(
    #     args.alpha) + '_RRL_scheduler_' + args.adversarial_type + '_logits_integrate_max_' + str(count) + '_20211123.json'
    # submit_json_files(file_name, final_predict_result, args, event_types,final_prob_result)



def integrate_predicting(model_save_paths,device,test_dataloader,args,vote_alpha=None):
    final_predict_result = []
    final_prob_result = []
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
            pbar = ProgressBar(n_total=len(test_dataloader), desc='vote prediction')
            for step, batch in enumerate(test_dataloader):
                batch = [t.to(device) for t in batch]
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
                for index,model in enumerate(models):
                    output = model(inputs)

                    if args.loss_function_type == "BCE":
                        output = torch.sigmoid(output)
                        pred = torch.where(output > 0.5, 1, 0)
                    else:
                        pred = torch.where(output > 0, 1, 0)
                        output = torch.sigmoid(output)


                    if index == 0:
                        prob = output
                        batch_allmodel_pred = pred
                    else:
                        prob += output
                        batch_allmodel_pred += pred

                pred = torch.where(batch_allmodel_pred>vote_num,1,0)
                pred = pred.detach().cpu().tolist()

                prob = prob/len(models)
                prob = prob.detach().cpu().tolist()

                final_prob_result.extend(prob)
                final_predict_result.extend(pred)
                pbar(step, {'step': step})

        return final_predict_result,final_prob_result
    else:
        with torch.no_grad():
            pbar = ProgressBar(n_total=len(test_dataloader), desc='logits prediction')
            for step, batch in enumerate(test_dataloader):
                batch = [t.to(device) for t in batch]
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}

                for index, model in enumerate(models):
                    output = model(inputs)
                    if index == 0:
                        logits = output
                    else:
                        logits += output

                if args.loss_function_type == "BCE":
                    logits = logits/len(models)
                    logits = torch.sigmoid(logits)
                    pred = torch.where(logits > 0.5, 1, 0)
                    prob = logits
                else:
                    pred = torch.where(logits > 0, 1, 0)
                    logits = torch.sigmoid(logits)
                    prob = logits

                pbar(step, {'step': step})
                pred = pred.detach().cpu().tolist()
                prob = prob.detach().cpu().tolist()
                final_predict_result.extend(pred)
                final_prob_result.extend(prob)
        return final_predict_result,final_prob_result




def submit_json_files(file_name,predicts_labels,args,event_types,final_prob_result=None):
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

    # if final_prob_result:
    #     df = pd.DataFrame()
    #     all_event_probs = []
    #     smalles_threshold = []
    #     all_event_chain = []
    #     for t, (predicts_label,probs) in enumerate(zip(predicts_labels,final_prob_result)):
    #         event_probs = []
    #         event_chain = []
    #         for index, val in enumerate(predicts_label):
    #             if val == 1:
    #                 event_probs.append(probs[index])
    #                 event_chain.append(event_types[index])
    #
    #         temp = event_probs
    #         event_probs = [str(ele) for ele in event_probs]
    #         event_probs = '\t'.join(event_probs)
    #
    #         event_chain = '\t'.join(event_chain)
    #
    #         predicts_label = [str(ele) for ele in predicts_label]
    #         predicts_label = '\t'.join(predicts_label)
    #         predicts_labels[t] = predicts_label
    #
    #
    #         all_event_probs.append(event_probs)
    #         all_event_chain.append(event_chain)
    #         temp.sort(reverse=False)
    #         if len(temp) == 0:
    #             smalles_threshold.append(0)
    #         else:
    #             smalles_threshold.append(temp[0])
    #
    #
    #     text_lengths = [len(text) for text in texts]
    #
    #     df['id'] = ids
    #     df['text'] = texts
    #     df['label'] = predicts_labels
    #     df['probs'] = all_event_probs
    #     df['smallest_prob'] = smalles_threshold
    #     df['event_type'] = all_event_chain
    #     df['text_length'] = text_lengths
    #     df.to_csv('submit/20211124_prediact_result_' + args.loss_function_type + '_'+ args.integrate_type + '_2.csv',index=False)


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


def prediction(model,test_dataloader,device,args):
    predicts_labels = []
    model.eval()
    with torch.no_grad():
        pbar = ProgressBar(n_total=len(test_dataloader), desc='prediction')
        for step, batch in enumerate(test_dataloader):
            batch = [t.to(device) for t in batch]
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
            output = model(inputs)

            # 注意这里统计模型指标正确率的代码逻辑，torch.where()和torch.equal()
            if args.loss_function_type == "BCE":
                output = torch.sigmoid(output)
                pred = torch.where(output > 0.5, 1, 0)
            else:
                pred = torch.where(output > 0, 1, 0)
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