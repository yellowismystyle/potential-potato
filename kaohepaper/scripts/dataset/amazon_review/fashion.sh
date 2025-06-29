AREA_NAME=Fashion
RAW_REVIEW_PATH=data/amazon_review/raw/Fashion/Amazon_Fashion.jsonl
RAW_ITEM_PATH=data/amazon_review/raw/Fashion/meta_Amazon_Fashion.jsonl
OUTPUT_PATH=data/amazon_review/split/$AREA_NAME

python src/dataset/amazon_review/data_split.py \
    --data_area_name $AREA_NAME \
    --raw_review_path $RAW_REVIEW_PATH \
    --raw_item_path $RAW_ITEM_PATH \
    --output_dir $OUTPUT_PATH