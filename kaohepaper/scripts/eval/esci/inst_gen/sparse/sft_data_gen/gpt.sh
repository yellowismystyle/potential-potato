DOMAIN_NAME=esci
SPLIT=$1
MODEL_NAME=gpt-4o
SAVE_DIR=results/esci/$SPLIT
DATA_PATH=data/esci/inst/sparse/subset/$SPLIT.parquet

python src/eval/esci/gpt.py \
    --domain_name $DOMAIN_NAME \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR