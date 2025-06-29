
for DOMAIN_NAME in 'Video_Games' 'Baby_Products' 'Office_Products' 'Sports_and_Outdoors'; do

    MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
    DATA_PATH=data/esci/inst/sparse/subset/test.parquet
    SAVE_DIR=results/esci
    MODEL_NAME=Qwen-7b-inst-esci
    

    CUDA_VISIBLE_DEVICES=0 python src/eval/esci/model_generate.py \
        --domain_name $DOMAIN_NAME \
        --model_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --save_dir $SAVE_DIR \
        --model_name $MODEL_NAME \
        --batch_size 8

done