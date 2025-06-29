DATA_PATH=data/amazon_c4/inst/qwen-instruct/test.parquet
MODEL_PATH=checkpoints/Rec-R1-C4/AC4-qwen2.5-3b-inst-grpo-1gpus/actor/global_step_250
MODEL_NAME=rec-r1
SAVE_DIR=results/amazon_c4
PROCESS_ID=$1
GPU_ID=$2

CUDA_VISIBLE_DEVICES=$GPU_ID python src/eval/amazon_c4/eval_inst.py \
    --data_path $DATA_PATH \
    --model_path $MODEL_PATH \
    --model_name $MODEL_NAME \
    --save_dir $SAVE_DIR \
    --process_id $PROCESS_ID