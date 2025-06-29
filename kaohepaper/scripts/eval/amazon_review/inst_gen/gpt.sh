DOMAIN_NAME=$1
MODEL_NAME=gpt-4o
SAVE_DIR=results/amazon_review
DATA_PATH=data/amazon_review/inst/test.parquet

python src/eval/amazon_review/gpt.py \
    --domain_name $DOMAIN_NAME \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR