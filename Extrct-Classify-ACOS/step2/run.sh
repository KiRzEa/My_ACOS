export BERT_BASE_DIR=/home/hjcai/BERT/uncased_L-12_H-768_A-12
export DATA_DIR=/home/hjcai/8RTX/BERT/pytorch_pretrained_BERT/Extract-Classify4Quad-2nd
export TASK_NAME=categorysenti
export MODEL=categorysenti
export DOMAIN=rest16

python run_classifier_step2.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --domain_type $DOMAIN \
  --model_type $MODEL\
  --do_lower_case \
  --data_dir $DATA_DIR \
  --bert_model $BERT_BASE_DIR\
  --max_seq_length 128 \
  --train_batch_size 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 30.0 \
  --output_dir /home/hjcai/8RTX/BERT/pytorch_pretrained_BERT/output/Extract-Classify4Quad-2nd/rest16/test-all_subtasks
