DOMAIN_NAME=$1
DATA_PATH=data/amazon_c4/inst/subset/$DOMAIN_NAME/test.parquet
MODEL_PATH=checkpoints/Rec-R1-C4/AC4-qwen2.5-3b-inst-grpo-1gpus/actor/global_step_200
MODEL_NAME=rec-r1
SAVE_DIR=results/amazon_c4/$DOMAIN_NAME

CUDA_VISIBLE_DEVICES=0 python src/eval/amazon_c4/eval_inst.py \
    --data_path $DATA_PATH \
    --model_path $MODEL_PATH \
    --model_name $MODEL_NAME \
    --save_dir $SAVE_DIR