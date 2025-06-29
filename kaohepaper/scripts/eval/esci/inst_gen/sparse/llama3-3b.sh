
for DOMAIN_NAME in 'Video_Games' 'Baby_Products' 'Office_Products' 'Sports_and_Outdoors'; do

    MODEL_PATH=/shared/eng/jl254/server-05/code/TinyZero/models/llama3
    DATA_PATH=data/esci/inst/sparse/subset/llama3/test.parquet
    SAVE_DIR=results/esci
    MODEL_NAME=llama3-3b-inst-esci
    

    CUDA_VISIBLE_DEVICES=0 python src/eval/esci/model_generate.py \
        --domain_name $DOMAIN_NAME \
        --model_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --save_dir $SAVE_DIR \
        --model_name $MODEL_NAME

done