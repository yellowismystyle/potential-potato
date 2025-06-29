DOMAIN_NAME=esci
SPLIT=$1
MODEL_NAME=claude-3.5
SAVE_DIR=results/esci/$SPLIT
DATA_PATH=data/esci/inst/sparse/subset/$SPLIT.parquet

python src/eval/esci/claude.py \
    --domain_name $DOMAIN_NAME \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR