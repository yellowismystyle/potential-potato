DOMAIN_NAME=$1
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
DATA_PATH=data/amazon_review/inst/test.parquet
SAVE_DIR=results/amazon_review
MODEL_NAME=Qwen-inst-amazon-review


CUDA_VISIBLE_DEVICES=6 python src/eval/amazon_review/model_generate.py \
    --domain_name $DOMAIN_NAME \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME