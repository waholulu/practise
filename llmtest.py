import numpy as np
import pandas as pd
from scipy.special import softmax

from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

from datasets import Dataset

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# PEFT / LoRA imports
from peft import LoraConfig, TaskType, get_peft_model

# ----------------------------------------------------------------------------
#                               CONFIGURATION
# ----------------------------------------------------------------------------
model_name = "Qwen/Qwen2.5-0.5B"  # Example model name

# Example DataFrame `dataset` must have columns: ["input_text", "label"]
# dataset = pd.DataFrame({
#     "input_text": ["sample1", "sample2", "sample3", ...],
#     "label": [0, 1, 0, ...]
# })

# ----------------------------------------------------------------------------
#                               TOKENIZER
# ----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------------------------------------------------------
#                               BASE MODEL
# ----------------------------------------------------------------------------
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
base_model.config.pad_token_id = tokenizer.pad_token_id

# ----------------------------------------------------------------------------
#                               LoRA CONFIG
# ----------------------------------------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Adjust based on model architecture
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# ----------------------------------------------------------------------------
#                               DATA SPLIT
# ----------------------------------------------------------------------------
train_df, temp_df = train_test_split(dataset, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# ----------------------------------------------------------------------------
#                               TOKENIZE
# ----------------------------------------------------------------------------
def tokenize_function(example):
    encoding = tokenizer(
        example["input_text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    encoding["labels"] = int(example["label"])
    return encoding

train_dataset = train_dataset.map(tokenize_function, remove_columns=["input_text", "label"])
val_dataset = val_dataset.map(tokenize_function, remove_columns=["input_text", "label"])
test_dataset = test_dataset.map(tokenize_function, remove_columns=["input_text", "label"])

# ----------------------------------------------------------------------------
#                               METRICS
# ----------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    prec = precision_score(labels, predictions, average="binary", zero_division=0)
    return {"precision": prec}

# ----------------------------------------------------------------------------
#                               TRAINING ARGS
# ----------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./medical_lora_finetuned2",
    overwrite_output_dir=True,
    num_train_epochs=1,              # Adjust as needed
    per_device_train_batch_size=8,   # Adjust for GPU memory
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    warmup_steps=10,
    logging_steps=10,
    save_total_limit=2,
    fp16=True,  # Use if GPU supports FP16
)

# ----------------------------------------------------------------------------
#                               TRAINER
# ----------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# ----------------------------------------------------------------------------
#                               EVALUATION
# ----------------------------------------------------------------------------
predictions = trainer.predict(test_dataset)
logits = predictions.predictions
labels = predictions.label_ids

# Convert logits to probabilities
probs = softmax(logits, axis=1)
pred_prob_class1 = probs[:, 1]
pred_class = np.argmax(logits, axis=1)

precision = precision_score(labels, pred_class, average="binary", zero_division=0)
print(f"Test Precision: {precision:.4f}")
print("Sample probabilities for class=1:", pred_prob_class1[:10])
