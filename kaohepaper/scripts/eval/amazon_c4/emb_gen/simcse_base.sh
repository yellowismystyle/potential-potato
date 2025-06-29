DATASET=McAuley-Lab/Amazon-C4
CACHE_DIR=data/amazon_c4/raw/cache
PLM_NAME=princeton-nlp/sup-simcse-roberta-base
FEAT_NAME=simcse-base
GPU_ID=5

python src/blair/generate_emb.py \
    --dataset $DATASET \
    --cache_path $CACHE_DIR \
    --plm_name $PLM_NAME \
    --feat_name $FEAT_NAME \
    --gpu_id $GPU_ID