DATASET=esci
CACHE_DIR=data/esci/raw
PLM_NAME=hyp1231/blair-roberta-large
FEAT_NAME=blair-large
GPU_ID=1

python src/baselines/generate_emb.py \
    --dataset $DATASET \
    --cache_path $CACHE_DIR \
    --plm_name $PLM_NAME \
    --feat_name $FEAT_NAME \
    --gpu_id $GPU_ID