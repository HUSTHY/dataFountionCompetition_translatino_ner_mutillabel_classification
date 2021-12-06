CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/ernie_1.0_tianchi_pretrained
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/ernie_span_outputs
export CUDA_VISIBLE_DEVICES=0
TASK_NAME="tianchi"

python run_ner_span.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --do_lower_case \
  --loss_type=ce \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=48 \
  --per_gpu_eval_batch_size=48 \
  --learning_rate=2e-5 \
  --num_train_epochs=10.0 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42

