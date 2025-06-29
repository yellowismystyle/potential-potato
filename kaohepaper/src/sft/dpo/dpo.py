from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import pandas as pd

# Load DPO training data from a Parquet file
def load_dpo_dataset(parquet_path):
    df = pd.read_parquet(parquet_path)
    return Dataset.from_pandas(df)

# Model and tokenizer setup
model_name = "Qwen/Qwen2.5-3B-Instruct"
cache_dir = "/srv/local/data/linjc/hub"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token  # Required for DPOTrainer

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=cache_dir,
    attn_implementation="flash_attention_2"
)

# Load training data
train_dataset = load_dpo_dataset("data/esci/inst/sparse/dpo/train.parquet")

# Training configuration
training_args = DPOConfig(
    per_device_train_batch_size=2,
    learning_rate=5e-6,
    num_train_epochs=1,
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    output_dir="checkpoints/qwen-dpo",
    bf16=True,
    remove_unused_columns=False,
    report_to="none",
    max_prompt_length=512,
    max_length=1024,
)

# DPO trainer instantiation (no eval dataset)
trainer = DPOTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=None,
)

# Train
trainer.train()
