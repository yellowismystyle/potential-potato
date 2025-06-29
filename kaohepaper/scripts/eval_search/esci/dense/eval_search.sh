DOMAIN_NAME=$1
# MODEL_NAME=blair-base
MODEL_NAME=blair-large
# QUERY_GEN_MODEL_NAME=Qwen-inst
QUERY_GEN_MODEL_NAME=gpt-4o
TEST_FILE_PATH=results_dense/esci/$QUERY_GEN_MODEL_NAME-esci_$DOMAIN_NAME.json


CUDA_VISIBLE_DEVICES=0 python src/eval_search/Dense/esci.py \
    --model_name $MODEL_NAME \
    --test_file_path $TEST_FILE_PATH