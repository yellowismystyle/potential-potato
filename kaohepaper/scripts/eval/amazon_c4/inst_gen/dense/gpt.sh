DOMAIN_NAME=$1
MODEL_NAME=gpt-4o
SAVE_DIR=results_dense/amazon_c4
DATA_PATH=data/amazon_c4/inst/dense/subset_other/test.parquet

python src/eval/amazon_c4/gpt.py \
    --domain_name $DOMAIN_NAME \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR