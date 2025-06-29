DATASET=esci
CACHE_DIR=data/esci/raw
PLM_NAME=FacebookAI/roberta-base
FEAT_NAME=roberta-base
GPU_ID=0

python src/blair/generate_emb.py \
    --dataset $DATASET \
    --cache_path $CACHE_DIR \
    --plm_name $PLM_NAME \
    --feat_name $FEAT_NAME \
    --gpu_id $GPU_ID