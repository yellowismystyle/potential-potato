DOMAIN_NAME=$1
MODEL_PATH=checkpoints/Rec-R1-AC4-Mix-Dense/AC4-dense-new-qwen2.5-3b-inst-grpo-2gpus/actor/global_step_800
DATA_PATH=data/amazon_c4/inst/dense/subset_other/test.parquet
SAVE_DIR=results_dense/amazon_c4
MODEL_NAME=rec-r1-amazon-c4


CUDA_VISIBLE_DEVICES=0 python src/eval/amazon_c4/model_generate.py \
    --domain_name $DOMAIN_NAME \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME