DOMAIN_NAME=$1
MODEL_PATH=checkpoints/qwen-sft-full-with_reason_no_json/checkpoint-562
DATA_PATH=data/esci/inst/sparse/sft/with_reason_no_json/test.parquet
SAVE_DIR=results/esci
MODEL_NAME=Qwen-sft-with-reason-no-json-esci


CUDA_VISIBLE_DEVICES=4 python src/eval/esci/model_generate.py \
    --domain_name $DOMAIN_NAME \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME