import pandas as pd
import json
import os
import pdb

PROMPT = """You are an expert in query generation. Given a query, your task is to create query terms to retrieve retrieve the most relevant products, ensuring they best meet customer needs.
Below is the query:
```{user_query}```"""

def make_prefix(dp):
    input_str = PROMPT.format(user_query=dp['query'])
    input_str = """<|im_start|>system\nYou are a helpful AI assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\n""" + input_str
    input_str += """\nShow your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. The generated query should use Boolean operators (AND, OR) to structure your query logically. For example,
<answer>
{
    "query": xxx
}
</answer>.<|im_end|>
<|im_start|>assistant\nLet me solve this step by step.\n"""
    input_str += f'{dp["output"]}<|im_end|>'
    return input_str

def construct_merged_rejection_data(split, model_names, ndcg_threshold=0.5):
    data_path = f'data/esci/inst/sparse/subset/{split}.parquet'
    data = pd.read_parquet(data_path)

    with open(f'results/esci/metric_res/{split}/query_metric_results-gpt-4o.json', 'r') as f:
        retrieval_eval_dict = json.load(f)

    merged_rows = []

    for model_name in model_names:
        print(f"Processing model: {model_name}")
        model_output_path = f'results/esci/{split}/{model_name}-esci_esci.json'
        if not os.path.exists(model_output_path):
            print(f"Warning: {model_output_path} not found, skipping.")
            continue

        with open(model_output_path, 'r') as f:
            model_res_dict = json.load(f)

        for i in range(len(data)):
            query = data.loc[i, 'query']
            output = model_res_dict[str(i)]['generated_text']
            entry = list(retrieval_eval_dict.values())[i]
            ndcg = entry['ndcg@100']

            if ndcg >= ndcg_threshold:
                dp = {
                    'query': query,
                    'output': output
                }
                text = make_prefix(dp)
                merged_rows.append({
                    'text': text,
                    'ndcg': ndcg,
                    'model': model_name,
                    'example_id': i
                })

    df = pd.DataFrame(merged_rows)
    os.makedirs(f'data/esci/inst/sparse/rsft/merged', exist_ok=True)
    df.to_parquet(f'data/esci/inst/sparse/rsft/merged/{split}.parquet')
    print(f"Saved merged RSFT dataset with {len(df)} rows to merged/{split}.parquet")

if __name__ == '__main__':
    model_list = ['gpt-4o', 'claude-haiku', 'claude-3.5']
    construct_merged_rejection_data('train', model_list)