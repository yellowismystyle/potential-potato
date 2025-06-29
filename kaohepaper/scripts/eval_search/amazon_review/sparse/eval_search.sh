SETTING_NAME=$1
DOMAIN_NAME=All_Beauty
# QUERY_GEN_MODEL_NAME=Qwen-inst
QUERY_GEN_MODEL_NAME=gpt-4o
# QUERY_GEN_MODEL_NAME=Rec-r1
TEST_FILE_PATH=results/amazon_review/$QUERY_GEN_MODEL_NAME-amazon-review_$SETTING_NAME.json
INDEX_DIR=database/amazon_review/$DOMAIN_NAME/pyserini_index
METRIC_RES_SAVE_DIR=results/amazon_review/metric_res/query_metric_results-$QUERY_GEN_MODEL_NAME-$SETTING_NAME.json

python src/eval_search/BM25/amazon_review.py \
    --res_path $TEST_FILE_PATH \
    --index_dir $INDEX_DIR \
    --save_path $METRIC_RES_SAVE_DIR