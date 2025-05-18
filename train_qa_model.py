import numpy as np
import pandas as pd
from tqdm import tqdm
import mlflow
from mlflow.models.signature import infer_signature
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from datasets import load_dataset, Dataset
import gradio as gr
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)



dataset = load_dataset("squad_v2")
print(f"Dataset structure: {dataset}\n")
print("Sample training example:")
print(dataset["train"][0])  # Show first example

# Convert to pandas for easier processing
train_df = pd.DataFrame(dataset["train"])
valid_df = pd.DataFrame(dataset["validation"])

def preprocess_data(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answers"]

    # Tokenize and prepare inputs
    inputs = tokenizer(
        questions,
        contexts,
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Process answers
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0] if answer["answer_start"] else 0
        end_char = start_char + len(answer["text"][0]) if answer["text"] else 0

        # Find token positions
        sequence_ids = inputs.sequence_ids(i)
        idx = 0
        while sequence_ids[idx] != 1:  # Find context start
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:  # Find context end
            idx += 1
        context_end = idx - 1

        # If answer is out of span, label is (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise find start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

mlflow.set_experiment("QA_System")
mlflow.start_run()

# Log parameters
mlflow.log_params({
    "model": "deepset/roberta-base-squad2",
    "batch_size": 8,
    "learning_rate": 2e-5,
    "epochs": 3
})

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Preprocess datasets
tokenized_train = dataset["train"].map(
    preprocess_data,
    batched=True,
    remove_columns=dataset["train"].column_names
)
tokenized_valid = dataset["validation"].map(
    preprocess_data,
    batched=True,
    remove_columns=dataset["validation"].column_names
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    max_steps=500,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    gradient_accumulation_steps=2,
    optim="adamw_torch_fused",
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    report_to="none",



)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# Start training
trainer.train()
mlflow.end_run()
