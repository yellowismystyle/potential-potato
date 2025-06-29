import pandas as pd
import json
import os
import pdb

PROMPT = """You are an expert in query generation. Given a query, your task is to create query terms to retrieve retrieve the most relevant products, ensuring they best meet customer needs.
Below is the query:
```{user_query}```"""

def make_prompt(query):
    input_str = PROMPT.format(user_query=query)
    input_str = """<|im_start|>system\nYou are a helpful AI assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\n""" + input_str
    input_str += """\nShow your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. The generated query should use Boolean operators (AND, OR) to structure your query logically. For example,
<answer>
{
    "query": xxx
}
</answer>.<|im_end|>
<|im_start|>assistant\nLet me solve this step by step.\n"""
    return input_str

def construct_dpo_dataset(split, model_names):
    # Load original query data
    data_path = f'data/esci/inst/sparse/subset/{split}.parquet'
    data = pd.read_parquet(data_path)

    model_outputs = {}  # model_outputs[model][i] = generated_text
    model_ndcg = {}     # model_ndcg[model][i] = ndcg_score

    for model in model_names:
        # Load model-generated outputs
        with open(f'results/esci/{split}/{model}-esci_esci.json', 'r') as f:
            model_outputs[model] = json.load(f)

        # Load per-model evaluation scores
        with open(f'results/esci/metric_res/{split}/query_metric_results-{model}.json', 'r') as f:
            ndcg_json = json.load(f)
            model_ndcg[model] = {}
            for k, v in ndcg_json.items():
                id_str = v['id']
                try:
                    id_int = int(id_str)
                    model_ndcg[model][id_int] = v['ndcg@100']
                except ValueError:
                    print(f"⚠️ Warning: invalid id in model {model}, key: {k}, id: {id_str}")

    dpo_rows = []

    for i in range(len(data)):
        query = data.loc[i, 'query']
        prompt = make_prompt(query)

        model_infos = []
        for model in model_names:
            if str(i) not in model_outputs[model] or i not in model_ndcg[model]:
                continue
            generated = model_outputs[model][str(i)]['generated_text']
            ndcg = model_ndcg[model][i]
            model_infos.append((model, generated, ndcg))

        if len(model_infos) < 2:
            continue  # Need at least two model outputs to compare

        # Sort by NDCG descending: best first, worst last
        model_infos.sort(key=lambda x: x[2], reverse=True)
        best_model, best_output, best_ndcg = model_infos[0]
        worst_model, worst_output, worst_ndcg = model_infos[-1]

        if best_ndcg == worst_ndcg:
            continue  # Skip if no preference can be established

        if best_output.strip() == worst_output.strip():
            continue  # Skip if identical output

        dpo_rows.append({
            "prompt": prompt,
            "chosen": f"{best_output}<|im_end|>",
            "rejected": f"{worst_output}<|im_end|>",
            "chosen_model": best_model,
            "rejected_model": worst_model,
            "chosen_ndcg": best_ndcg,
            "rejected_ndcg": worst_ndcg,
            "query_id": i
        })

    # Save the dataset
    os.makedirs(f'data/esci/inst/sparse/dpo', exist_ok=True)
    df = pd.DataFrame(dpo_rows)
    df.to_parquet(f'data/esci/inst/sparse/dpo/{split}.parquet', index=False)
    print(f"✅ Saved {len(df)} DPO pairs to data/esci/inst/sparse/dpo/{split}.parquet")

if __name__ == '__main__':
    model_list = ['gpt-4o', 'claude-haiku', 'claude-3.5']
    construct_dpo_dataset('train', model_list)
