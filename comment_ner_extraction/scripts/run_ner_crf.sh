CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/pretrained_models/chinese-bert-wwm-ext
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/model_output
TASK_NAME="cner"

python run_ner_crf.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_predict \
  --do_lower_case \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=1e-5 \
  --crf_learning_rate=5e-4 \
  --num_train_epochs=10 \
  --logging_steps=212 \
  --save_steps=212 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42