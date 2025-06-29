DOMAIN_NAME=$1
MODEL_NAME=gpt-4o
SAVE_DIR=results_dense/esci
DATA_PATH=data/esci/inst/dense/subset/test.parquet

python src/eval/esci/gpt.py \
    --domain_name $DOMAIN_NAME \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR