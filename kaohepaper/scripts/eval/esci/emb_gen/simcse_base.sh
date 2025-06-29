DATASET=esci
CACHE_DIR=data/esci/raw
PLM_NAME=princeton-nlp/sup-simcse-roberta-base
FEAT_NAME=simcse-base
GPU_ID=5

python src/baselines/generate_emb.py \
    --dataset $DATASET \
    --cache_path $CACHE_DIR \
    --plm_name $PLM_NAME \
    --feat_name $FEAT_NAME \
    --gpu_id $GPU_ID