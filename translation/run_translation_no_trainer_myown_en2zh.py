#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on text translation.
"""
# You can also adapt this script on your own text translation task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)



from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from data_reader.dataReader_en2zh import DataReader

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    MBartTokenizer,
    MBartTokenizerFast,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./runs')


from sacrebleu.metrics import BLEU

import datetime

from tools.progressbar import ProgressBar
from tools.log import Logger

log_file = 'log_files/'+'v3_eng_to_zh_seed_1.log'
logger = Logger('mutil_label_logger',log_level=10,log_file=log_file)


# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# Parsing input arguments
def parse_args():

    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--validation_file", type=str, default='data/dev_dataset.csv', help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--train_file", type=str, default='data/train_dataset.csv', help="A csv or a json file containing the training data."
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=512,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=False,
        help="Whether to pad all samples to model maximum sentence "
        "length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More"
        "efficient on GPU but very bad for TPU.",
    )

    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument("--source_lang", type=str, default=None, help="Source language id for translation.")
    parser.add_argument("--target_lang", type=str, default=None, help="Target language id for translation.")
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        default='./pretrained_models/MarianMTModel_en2zh',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=8, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=10,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default='./output', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    # Sanity checks

    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    if args.validation_file is not None:
        extension = args.validation_file.split(".")[-1]
        assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args


def main():
    # Parse the arguments
    args = parse_args()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)




    # strtime = datetime.datetime.now().strftime('%Y-%m-%d')
    strtime = '2021-12-02'
    args.output_dir = os.path.join(args.output_dir,strtime + '_v3_seed_' + str(args.seed) + '_en2zh')






    #
    # # download model & vocab.
    # if args.config_name:
    #     config = AutoConfig.from_pretrained(args.model_name_or_path)
    # elif args.model_name_or_path:
    #     config = AutoConfig.from_pretrained(args.model_name_or_path)
    # else:
    #     config = CONFIG_MAPPING[args.model_type]()
    #     logger.warning("You are instantiating a new config instance from scratch.")
    #
    # if args.tokenizer_name:
    #     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    # elif args.model_name_or_path:
    #     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    # else:
    #     raise ValueError(
    #         "You are instantiating a new tokenizer from scratch. This is not supported by this script."
    #         "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    #     )
    #
    # if args.model_name_or_path:
    #     model = AutoModelForSeq2SeqLM.from_pretrained(
    #         args.model_name_or_path,
    #         from_tf=bool(".ckpt" in args.model_name_or_path),
    #         config=config,
    #     )
    # else:
    #     logger.info("Training new model from scratch")
    #     model = AutoModelForSeq2SeqLM.from_config(config)
    #
    # # logger.info(model)
    #
    # model.resize_token_embeddings(len(tokenizer))
    #
    # # Set decoder_start_token_id
    # if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
    #     assert (
    #         args.target_lang is not None and args.source_lang is not None
    #     ), "mBart requires --target_lang and --source_lang"
    #     if isinstance(tokenizer, MBartTokenizer):
    #         model.config.decoder_start_token_id = tokenizer.lang_code_to_id[args.target_lang]
    #     else:
    #         model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(args.target_lang)
    #
    # if model.config.decoder_start_token_id is None:
    #     raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    #
    #
    # # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # # ignore those attributes).
    # if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
    #     if args.source_lang is not None:
    #         tokenizer.src_lang = args.source_lang
    #     if args.target_lang is not None:
    #         tokenizer.tgt_lang = args.target_lang
    #
    #
    #
    #
    # # # Log a few random samples from the training set:
    # # for index in random.sample(range(len(train_dataset)), 1):
    # #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    #
    # # DataLoaders creation:
    # label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    # if args.pad_to_max_length:
    #     data_collator = default_data_collator
    # else:
    #     data_collator = DataCollatorForSeq2Seq(
    #         tokenizer,
    #         model=model,
    #         label_pad_token_id= label_pad_token_id,
    #         pad_to_multiple_of= None,
    #     )
    #
    # train_dataset = DataReader(tokenizer,filepath=args.train_file)
    # eval_dataset = DataReader(tokenizer,filepath=args.validation_file)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    # eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    #
    # # Optimizer
    # # Split weights in two groups, one with weight decay and the other not.
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    #
    # # Scheduler and math around the number of training steps.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if args.max_train_steps is None:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # else:
    #     args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    #
    # lr_scheduler = get_scheduler(
    #     name=args.lr_scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.num_warmup_steps,
    #     num_training_steps=args.max_train_steps,
    # )
    #
    #
    #
    #
    # # Train!
    # total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    #
    # logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num Epochs = {args.num_train_epochs}")
    # logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    # logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    # logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # # Only show the progress bar once on each machine.
    #
    # # progress_bar = tqdm(range(args.max_train_steps))
    #
    # progress_bar = ProgressBar(n_total=args.max_train_steps, desc='Training')
    # completed_steps = 0
    #
    #
    # model.to(device)
    #
    # best_score= 0
    #
    #
    # bleu = BLEU()
    #
    # for epoch in range(args.num_train_epochs):
    #     model.train()
    #     for step, batch in enumerate(train_dataloader):
    #         for k,v in batch.items():
    #             batch[k] = v.to(device)
    #
    #         outputs = model(**batch)
    #         loss = outputs.loss
    #         loss = loss / args.gradient_accumulation_steps
    #         loss.backward()
    #         if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
    #             optimizer.step()
    #             lr_scheduler.step()
    #             optimizer.zero_grad()
    #             completed_steps += 1
    #             progress_bar(step=completed_steps, info={'step': completed_steps})
    #         if completed_steps >= args.max_train_steps:
    #             break
    #
    #         writer.add_scalar('train/Loss', loss.item(), completed_steps)
    #
    #     score = evaluation_step(model, eval_dataloader, tokenizer, device, bleu, args)
    #
    #
    #     if best_score < score:
    #         best_score = score
    #         logger.info("save model")
    #         model.save_pretrained(args.output_dir)
    #         tokenizer.save_pretrained(args.output_dir)
    #
    #     logger.info("Epoch:%d  score:%.6f   best_score:%.6f"%(epoch, score, best_score))
    #
    #     writer.add_scalar('Val/bleu_score', score, completed_steps)
    #
    #     writer.add_scalar('Val/bleu_best_score', best_score, completed_steps)
    #
    # logger.info('training finished!')

    logger.info('prediction begin!')
    logger.info(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir)

    dataset = DataReader(tokenizer, filepath='data/part_single_en_3W.csv')

    test_dataloader = DataLoader(dataset=dataset, batch_size=16)



    model.to(device)

    finanl_result = []

    for batch in tqdm(test_dataloader, desc='translation prediction',ncols = 100 ):
        for k, v in batch.items():
            batch[k] = v.to(device)
        batch = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        # Perform the translation and decode the output
        translation = model.generate(**batch, top_k=5, num_return_sequences=1, num_beams=1)
        batch_result = tokenizer.batch_decode(translation, skip_special_tokens=True)
        finanl_result.extend(batch_result)

    logger.info(len(finanl_result))


    with open('data/part_single_en_3W.csv','r',encoding='utf-8') as f:
        engs = f.readlines()


    with open('data/seed_0_en2zh_3W_' + strtime + '.txt', 'w', encoding='utf-8') as f:
        for zh, en in zip(finanl_result,engs[0:len(finanl_result)]):
            f.write(zh.replace('\n','')+'\t'+en.replace('\n','')+'\n')

    logger.info('prediction finished!')


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def evaluation_step(model, eval_dataloader, tokenizer, device, bleu, args):
    model.eval()

    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": 512,
        "num_beams": args.num_beams,
    }
    refs = [] #label 需要二维list
    sys = [] #预测值 一维list

    # progress_bar = tqdm(len(eval_dataloader),desc='evaluation')
    progress_bar = ProgressBar(n_total=len(eval_dataloader), desc='evaldation')
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            for k, v in batch.items():
                batch[k] = v.to(device)
            generated_tokens = model.generate(batch["input_ids"],attention_mask=batch["attention_mask"],**gen_kwargs,)
            labels = batch["labels"]

            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # print('decoded_preds', decoded_preds)
            # print('decoded_labels',decoded_labels)
            # exit()

            progress_bar(step = step,info = {'step':step})
            for decoded_pred, decoded_label in zip(decoded_preds,decoded_labels):
                if len(decoded_label) >0 and decoded_label != '' and decoded_label != ' ' and len(decoded_label.split(' ')) > 0:
                    refs.append(decoded_label.strip())
                    sys.append(decoded_pred.strip())

    print('len(refs)',len(refs))
    print('len(sys)', len(sys))
    assert len(sys) == len(refs)
    refs = [refs]
    print('len(refs)',len(refs))
    try:
        result = bleu.corpus_score(hypotheses=sys,references=refs)
        logger.info('\n')
        logger.info({"bleu": result.score})
    except Exception as e:
        logger.info(e)
        return 0.3
    return result.score


if __name__ == "__main__":
    main()
