DATASET=esci
CACHE_DIR=data/esci/raw
PLM_NAME=FacebookAI/roberta-large
FEAT_NAME=roberta-large
GPU_ID=6

python src/baselines/generate_emb.py \
    --dataset $DATASET \
    --cache_path $CACHE_DIR \
    --plm_name $PLM_NAME \
    --feat_name $FEAT_NAME \
    --gpu_id $GPU_ID