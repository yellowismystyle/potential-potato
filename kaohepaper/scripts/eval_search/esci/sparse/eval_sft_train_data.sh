
# QUERY_GEN_MODEL_NAME=gpt-4o
# QUERY_GEN_MODEL_NAME=claude-3.5
QUERY_GEN_MODEL_NAME=claude-haiku
TEST_FILE_PATH=results/esci/train/$QUERY_GEN_MODEL_NAME-esci_esci.json
METRIC_RES_SAVE_DIR=results/esci/metric_res/train/query_metric_results-$QUERY_GEN_MODEL_NAME.json


python src/eval_search/BM25/esci.py \
    --res_path $TEST_FILE_PATH \
    --save_path $METRIC_RES_SAVE_DIR \
