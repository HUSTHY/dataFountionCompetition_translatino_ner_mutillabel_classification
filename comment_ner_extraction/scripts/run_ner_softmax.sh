CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/ernie_softmax_output/tianchi_output/bert/checkpoint-1800
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/ernie_softmax_output
export CUDA_VISIBLE_DEVICES=0
TASK_NAME="tianchi"

python run_ner_softmax.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_predict \
  --do_lower_case \
  --loss_type=ce \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=48 \
  --per_gpu_eval_batch_size=48 \
  --learning_rate=3e-5 \
  --num_train_epochs=10.0 \
  --logging_steps=200 \
  --save_steps=200 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42
