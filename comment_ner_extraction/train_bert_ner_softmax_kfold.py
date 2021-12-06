import argparse
import glob
import logging
import os
import json
import time
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from callback.adversarial import FGM,PGD
from tools.common import seed_everything, json_to_text
from tools.common import init_logger, logger

from models.transformers import WEIGHTS_NAME, BertConfig, AlbertConfig
from models.bert_for_ner import BertSoftmaxForNer
from models.albert_for_ner import AlbertSoftmaxForNer
from processors.utils_ner import CNerTokenizer,get_entities
from processors.ner_seq import convert_examples_to_features
from processors.ner_seq import ner_processors as processors
from processors.ner_seq import collate_fn
from metrics.ner_metrics import SeqEntityScore
from processors.utils_ner import bert_extract_item,bert_extract_item_batch
from tools.finetuning_argparse import get_argparse
from tqdm import tqdm
from sklearn.model_selection import KFold
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertSoftmaxForNer, CNerTokenizer),
    'roberta': (BertConfig, BertSoftmaxForNer, CNerTokenizer),
    'albert': (AlbertConfig, AlbertSoftmaxForNer, CNerTokenizer),
}


def evaluate(args, model, dev_dataloader, prefix=""):
    metric = SeqEntityScore(args.id2label)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Batch size = %d", args.per_gpu_dev_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(dev_dataloader), desc="Evaluating")
    model.eval()
    count = 0
    for step, batch in enumerate(dev_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if args.model_type != "distilbert":
            # XLM and RoBERTa don"t use segment_ids
            inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)

        input_lens = batch[4].cpu().numpy().tolist()

        with torch.no_grad():
            outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        preds = torch.argmax(logits,dim=2).detach().cpu().tolist()
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i]-1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(args.id2label[out_label_ids[i][j]])
                    temp_2.append(preds[i][j])
        pbar(step)
    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)

    avg_f1 = results["f1"]

    return results,avg_f1

def load_and_cache_examples(args,path,task, tokenizer, data_type='train'):
    processor = processors[task]()
    logger = args.logger
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
    if data_type == 'train':
        examples = processor.get_train_examples(path)
    elif data_type == 'dev':
        examples = processor.get_dev_examples(path)
    else:
        examples = processor.get_test_examples(path)

    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            label_list=label_list,
                                            max_seq_length=args.train_max_seq_length if data_type == 'train' \
                                                else args.eval_max_seq_length,
                                            cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                            pad_on_left=bool(args.model_type in ['xlnet']),
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                            sep_token=tokenizer.sep_token,
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                            )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset





def data_save_to_txt(df,path):
    texts = df['text'].values.tolist()
    labels = df['BIO_anno'].values.tolist()
    with open(path, 'w', encoding='utf-8') as f:
        for text, label in tqdm(zip(texts, labels)):
            label = label.split(' ')
            for word, symbol in zip(text, label):
                s = word + '\t' + symbol
                f.write(s + '\n')
            f.write('\n')


def main():
    args = get_argparse().parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_dir = args.output_dir + '_{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    time_ = time.strftime("%Y-%m-%d", time.localtime())
    logger = init_logger(log_file='log/' + f'v30_{args.model_type}_span_5kfold_lsr_FGM_supervised{time_}.log')
    args.logger = logger
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    args.device = "cuda" if torch.cuda.is_available() else "cpu"


    seed_everything(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    logger.info('id2label----{}'.format(args.id2label))
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    args.num_labels = num_labels

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None, )
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=args.num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None, )
    config.soft_label = False
    config.loss_type = args.loss_type
    print('config.loss_type',config.loss_type)

    test_path = args.test_path
    test_dataset = load_and_cache_examples(args, test_path, args.task_name, tokenizer, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32, collate_fn=collate_fn)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    model_save_paths = []

    import pandas as pd
    df = pd.read_csv(args.train_data)
    # df['BIO_anno'] = [ ele.replace('B-','').replace('I-','')  for ele in df['BIO_anno'].values.tolist()]
    labels = df['class'].values.tolist()
    print('len(df)', len(df))
    for fold, (train_index, dev_index) in enumerate(skf.split(df, labels)):
        logger.info("================================fold{}================================".format(fold))
        args.run_time = datetime.now().strftime('%Y-%m-%d')

        args.model_save = os.path.join(args.output_dir, 'v6_' + args.run_time, '5fold_softmax_lsr', str(fold))
        logger.info("args.model_save:{}".format(args.model_save))
        model_save_paths.append(args.model_save)

        train_df = df[df['id'].isin(train_index)]
        dev_df = df[df['id'].isin(dev_index)]

        save_dir = os.path.join(args.data_dir, 'kflod_span', str(fold))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        train_save_path = os.path.join(save_dir, 'comment_train.txt')
        dev_save_path = os.path.join(save_dir, 'comment_dev.txt')

        data_save_to_txt(train_df,train_save_path)
        data_save_to_txt(dev_df,dev_save_path)

        train_dataset = load_and_cache_examples(args, train_save_path, args.task_name, tokenizer, data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size,
                                      collate_fn=collate_fn)

        dev_dataset = load_and_cache_examples(args, dev_save_path, args.task_name, tokenizer, data_type='dev')
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.per_gpu_dev_batch_size,
                                    collate_fn=collate_fn)

        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config, cache_dir=args.cache_dir if args.cache_dir else None)

        model.to(args.device)

        t_total = len(train_dataloader) * args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay, },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        args.warmup_steps = int(t_total * args.warmup_proportion)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total optimization steps = %d", t_total)

        if args.do_adv:
            if args.adv_type=='FGM':
                fgm = FGM(model, emb_name=args.adv_name, epsilon=args.adv_epsilon)
            else:
                pgd = PGD(model,emb_name=args.adv_name,epsilon=args.adv_epsilon,alpha=0.3)

        seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
        train_max_f1 = 0.0
        for epoch in range(int(args.num_train_epochs)):
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    # XLM and RoBERTa don"t use segment_ids
                    inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
                loss.backward()

                # 对抗训练
                if args.do_adv:
                    if args.adv_type == 'FGM':
                        fgm.attack()
                        loss_adv = model(**inputs)[0]
                        loss_adv.backward()
                        fgm.restore()
                    else:
                        pgd.backup_grad()
                        K = 3
                        for t in range(K):
                            pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                            if t != K - 1:
                                model.zero_grad()
                            else:
                                pgd.restore_grad()
                            loss_adv = model(**inputs)[0]
                            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                        pgd.restore()  # 恢复embedding参数


                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()
                pbar(step, {'loss': loss.item()})
                scheduler.step()
            _, avg_f1 = evaluate(args,model, dev_dataloader)

            if train_max_f1 < avg_f1:
                train_max_f1 = avg_f1
                if not os.path.exists(args.model_save):
                    os.makedirs(args.model_save)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(args.model_save)
                logger.info("Epoch: %d Saving model checkpoint to %s"%(epoch,args.model_save))
                tokenizer.save_vocabulary(args.model_save)
            logger.info('Epoch: %d avg_f1: %.6f max_f1:%.6f'%(epoch, avg_f1,train_max_f1))

        del outputs
        del model
        del inputs
        torch.cuda.empty_cache()



    # inference below
    model_save_paths.sort()


    args.integrate_type = 'vote'
    final_predict_result = integrate_predicting_batch(model_save_paths, test_dataloader, args)

    re_df = pd.read_csv('result/v18_Submission_rdrop_4.0_bert_classficaition_bert_5logits_crf_ner_cls_droprout_FGM.csv')
    print(len(re_df))
    re_df['BIO_anno'] = final_predict_result
    path = 'result/v31_Submission_rdrop_4.0_bert_cls_dropout_classficaition_' + args.model_type + '_' + args.integrate_type +'_5_softmax_ner_lsr_FGM_semi_supervised.csv'
    re_df.to_csv(path, index=False)




    args.integrate_type = 'logits'
    final_predict_result = integrate_predicting_batch(model_save_paths, test_dataloader, args)
    re_df = pd.read_csv('result/v18_Submission_rdrop_4.0_bert_classficaition_bert_5logits_crf_ner_cls_droprout_FGM.csv')
    re_df['BIO_anno'] = final_predict_result
    print(len(re_df))
    path = 'result/v31_Submission_rdrop_4.0_bert_cls_dropout_classficaition_' + args.model_type + '_' + args.integrate_type + '_5_softmax_ner_lsr_FGM_semi_supervised.csv'
    re_df.to_csv(path, index=False)




def ner_tags_entities_correct(length,label_entities):
    tags = ['O']*length
    for label_entity in label_entities:
        type_ = label_entity[0]
        start = label_entity[1]
        end = label_entity[2]
        for i in range(start,end+1):
            if i == start:
                tags[i] = 'B-'+type_
            else:
                tags[i] = 'I-' + type_
    tags = ' '.join(tags)
    return tags



def integrate_predicting_batch(model_save_paths, test_dataloader, args):
    logger = args.logger
    final_predict_result = []
    if args.integrate_type == 'vote':
        all_model_predicts = []
        for save_path in model_save_paths:
            logger.info('------------{}------------'.format(save_path))
            config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(save_path, num_labels=args.num_labels,cache_dir=args.cache_dir if args.cache_dir else None)
            model = model_class.from_pretrained(save_path,from_tf=bool(".ckpt" in args.model_name_or_path),config=config, cache_dir=args.cache_dir if args.cache_dir else None)
            model.to(args.device)
            model.eval()
            each_model_predictlabels = []
            with torch.no_grad():
                pbar = ProgressBar(n_total=len(test_dataloader), desc='vote prediction')
                for step, batch in enumerate(test_dataloader):
                    batch = tuple(t.to(args.device) for t in batch)
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}
                    if args.model_type != "distilbert":
                        # XLM and RoBERTa don"t use segment_ids
                        inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
                    outputs = model(**inputs)
                    logits = outputs[0]
                    tags = torch.argmax(logits,dim=2)
                    batch_tags = tags.squeeze(0).cpu().numpy().tolist()
                    input_lens = batch[4].cpu().numpy().tolist()
                    preds = []
                    for tags, length in zip(batch_tags, input_lens):
                        tags = tags[1:length - 1]
                        label_entities = get_entities(tags, args.id2label, args.markup)
                        preds.append((label_entities, length - 2))
                    each_model_predictlabels.extend(preds)
                    pbar(step)
            all_model_predicts.append(each_model_predictlabels)

        def crf_vote(all_model_predicts,threshold=0.8):
            """
            :param all_model_predicts: 所有模型的预测结果
            :param threshold: 阈值 大于80%模型预测出来的实体才能被选中
            :return:
            """
            threshold_nums = len(all_model_predicts)*threshold
            all_vote_predict_entities = []
            all_vote_predict_lentgh = []
            for i in range(len(all_model_predicts[0])):#每条测试集
                each_entities_dict = defaultdict(int)
                for each_model_predicts in all_model_predicts:
                    entities,length = each_model_predicts[i]
                    for entity in entities:
                        each_entities_dict[(entity[0],entity[1],entity[2])] += 1
                all_vote_predict_lentgh.append(length)
                each_vote_entities = []
                for entity,count in each_entities_dict.items():
                    if count >= threshold_nums:
                        each_vote_entities.append(entity)
                all_vote_predict_entities.append(each_vote_entities)

            return all_vote_predict_lentgh,all_vote_predict_entities

        all_vote_predict_lentgh, all_vote_predict_entities = crf_vote(all_model_predicts)

        final_predict_result = [ ner_tags_entities_correct(length,label_entities) for length,label_entities in zip(all_vote_predict_lentgh,all_vote_predict_entities)]
        return final_predict_result
    else:

        def weight_init(t):
            import math
            lamb = 1/3
            return math.exp(-lamb * t)

        models = []
        for save_path in model_save_paths:
            logger.info('------------{}------------'.format(save_path))
            config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(save_path, num_labels=args.num_labels,
                                                  cache_dir=args.cache_dir if args.cache_dir else None)
            model = model_class.from_pretrained(save_path,
                                                from_tf=bool(".ckpt" in args.model_name_or_path), config=config,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
            model.to(args.device)
            model.eval()
            models.append(model)

        with torch.no_grad():
            pbar = ProgressBar(n_total=len(test_dataloader), desc='logits prediction')
            for step, batch in enumerate(test_dataloader):
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}
                if args.model_type != "distilbert":
                    # XLM and RoBERTa don"t use segment_ids
                    inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)

                weight_sum = 0.0
                for i,model in enumerate(models):
                    # # 牛顿冷却
                    # weight = weight_init(i)

                    # 平均概率融合
                    weight = 1/len(models)
                    output = model(**inputs)[0]

                    #起始位置概率融合
                    if i == 0:
                        logits = output
                    else:
                        logits += output
                    weight_sum += weight

                logits = logits / weight_sum  # 注意要平均
                tags = torch.argmax(logits, dim=2)
                batch_tags = tags.squeeze(0).cpu().numpy().tolist()
                input_lens = batch[4].cpu().numpy().tolist()
                preds = []
                for tags, length in zip(batch_tags, input_lens):
                    tags = tags[1:length - 1]
                    label_entities = get_entities(tags, args.id2label, args.markup)
                    tag2seq = ner_tags_entities_correct(length-2, label_entities)
                    preds.append(tag2seq)
                final_predict_result.extend(preds)
                pbar(step, {'step': step})
        return final_predict_result


if __name__ == "__main__":
    main()

