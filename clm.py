import os
import torch
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

# PEFT/LoRA imports
from peft import LoraConfig, get_peft_model, TaskType

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

def merge_intervals(intervals):
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or interval[0] > merged[-1][1]:
            merged.append(interval)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
    return merged

def generate_fake_medical_claims_data_faster(
    n_individuals=500,
    min_claims_per_individual=3,
    max_claims_per_individual=15,
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)

    # ICD codes categorized by risk for future inpatient event
    high_risk_icd_codes = ["C50.9", "I10", "E78.5"]
    medium_risk_icd_codes = ["E11.9", "J45.909"]
    low_risk_icd_codes = ["Z79.4", "M54.5", "F32.9"]
    inpatient_code = "INPAT"

    # Vectorized random dates
    total_days = (end_date - start_date).days

    def random_dates(num_dates):
        day_offsets = np.random.randint(0, total_days + 1, size=num_dates)
        return [start_date + timedelta(days=int(offset)) for offset in day_offsets]

    all_claims = []

    code_map = {
        'low': low_risk_icd_codes,
        'medium': medium_risk_icd_codes,
        'high': high_risk_icd_codes,
        'none': low_risk_icd_codes
    }

    for individual_id in range(1, n_individuals + 1):
        n_claims = np.random.randint(min_claims_per_individual, max_claims_per_individual + 1)
        claim_dates = sorted(random_dates(n_claims))
        base_inpatient_risk = random.random() * 0.1

        categories = np.random.choice(
            ['low','medium','high','none'],
            size=n_claims,
            p=[0.4, 0.3, 0.2, 0.1]
        )

        codes = np.empty_like(categories, dtype=object)
        for cat_value in np.unique(categories):
            mask = (categories == cat_value)
            codes_subset = np.random.choice(code_map[cat_value], size=mask.sum())
            codes[mask] = codes_subset

        found_high_risk = np.isin(codes, high_risk_icd_codes).any()
        found_medium_risk = np.isin(codes, medium_risk_icd_codes).any()
        inpatient_prob = base_inpatient_risk
        if found_high_risk:
            inpatient_prob += 0.4
        elif found_medium_risk:
            inpatient_prob += 0.2

        random_vals = np.random.rand(n_claims)
        final_codes = np.where(random_vals < inpatient_prob, inpatient_code, codes)

        for dt_val, code_val in zip(claim_dates, final_codes):
            all_claims.append({
                "individual_id": individual_id,
                "icd_cd": code_val,
                "dt": dt_val
            })

    df = pd.DataFrame(all_claims).sort_values(["individual_id", "dt"], ignore_index=True)

    # Mark inpatient_flag if within 6 months of any INPAT date
    df["inpatient_flag"] = 0
    six_months = pd.Timedelta(days=180)

    df_list = []
    for ind_id, sub_df in df.groupby("individual_id", group_keys=False):
        sub_df = sub_df.copy()
        inpat_dates = sub_df.loc[sub_df['icd_cd'] == inpatient_code, 'dt'].sort_values()
        claim_dates = sub_df['dt'].sort_values()

        if len(inpat_dates) > 0:
            intervals = [(d, d + six_months) for d in inpat_dates]
            merged_intervals = merge_intervals(intervals)

            for (start_date_int, end_date_int) in merged_intervals:
                start_idx = np.searchsorted(claim_dates, start_date_int, side='left')
                end_idx = np.searchsorted(claim_dates, end_date_int, side='right')
                claim_indexes = claim_dates.index[start_idx:end_idx]
                sub_df.loc[claim_indexes, 'inpatient_flag'] = 1

        df_list.append(sub_df)

    df = pd.concat(df_list, ignore_index=True)
    return df

# --- Create some fake data ---
fake_data = generate_fake_medical_claims_data_faster(
    n_individuals=1600,
    min_claims_per_individual=3,
    max_claims_per_individual=50,
    seed=42
)

###############################################################################
#       Summarize repeated ICD codes (icd_code, first_date, recent_date, total_count)
###############################################################################
data_list = []
grouped = fake_data.groupby('individual_id')

for individual_id, df_individual in grouped:
    # Sort by date so we can incrementally build history up to each claim
    df_individual = df_individual.sort_values('dt').reset_index(drop=True)
    
    # For each row => the "current" date is df_individual.iloc[idx]['dt']
    # We'll gather all claims up to that date and build the summary
    for idx in range(len(df_individual)):
        current_row = df_individual.iloc[idx]

        # All claims up to and including this row (i.e. up to this date index)
        history_df = df_individual.loc[:idx, ['dt', 'icd_cd']].copy()

        # Group by ICD code and compute first_date, recent_date, total_count
        grouped_icd = history_df.groupby('icd_cd').agg(
            first_date=('dt', 'min'),
            recent_date=('dt', 'max'),
            total_count=('dt', 'count')
        ).reset_index()

        # Convert timestamps to strings
        grouped_icd['first_date'] = grouped_icd['first_date'].dt.strftime('%Y-%m-%d')
        grouped_icd['recent_date'] = grouped_icd['recent_date'].dt.strftime('%Y-%m-%d')

        # Convert to list of dicts => each dict is {icd_cd, first_date, recent_date, total_count}
        diagnosis_history = grouped_icd.to_dict(orient='records')

        # Build the input dict
        input_text_dict = {
            'diagnosis_history': diagnosis_history
        }
        
        # Convert to JSON
        input_text = json.dumps(input_text_dict, separators=(',', ':'))

        # Label is the current_row's inpatient_flag => "0" or "1"
        label = str(current_row['inpatient_flag'])

        # Append to overall data list
        data_list.append({
            'input_text': input_text,
            'label': label
        })

# Convert into DataFrame for further processing
dataset = pd.DataFrame(data_list)
print(dataset.head(5))

###############################################################################
#       Tokenization / Train-Test split / LoRA fine-tuning
###############################################################################
model_name = "Qwen/Qwen2.5-0.5B"  # Example model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# If no pad token, define one
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

base_model.config.pad_token_id = tokenizer.pad_token_id

# --- LoRA Configuration ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # depends on your model's internals
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Data split
train_df, temp_df = train_test_split(dataset, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenize
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

import evaluate
precision_metric = evaluate.load("precision")
import numpy as np
from sklearn.metrics import precision_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    prec = precision_score(labels, predictions, average="binary", zero_division=0)
    return {"precision": prec}

training_args = TrainingArguments(
    output_dir="./medical_lora_finetuned2",
    overwrite_output_dir=True,
    num_train_epochs=1,             # Adjust as needed
    per_device_train_batch_size=16, # Adjust for GPU memory
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    warmup_steps=10,
    logging_steps=10,
    save_total_limit=2,
    fp16=True,  # Use if GPU supports FP16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()

predictions = trainer.predict(test_dataset)
logits = predictions.predictions
labels = predictions.label_ids

# Convert logits to probabilities
from scipy.special import softmax
probs = softmax(logits, axis=1)   
pred_prob_class1 = probs[:, 1]    
pred_class = np.argmax(logits, axis=1)

precision = precision_score(labels, pred_class, average="binary")
print(f"Test Precision: {precision:.4f}")
print("Sample probabilities for class=1:", pred_prob_class1[:10])
