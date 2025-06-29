DOMAIN_NAME=$1
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
DATA_PATH=data/amazon_c4/inst/sparse/subset_other/test.parquet
SAVE_DIR=results/amazon_c4
MODEL_NAME=Qwen-inst-amazon-c4


CUDA_VISIBLE_DEVICES=0 python src/eval/amazon_c4/model_generate.py \
    --domain_name $DOMAIN_NAME \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME