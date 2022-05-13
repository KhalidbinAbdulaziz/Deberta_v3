import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import os

import torch
from datasets import load_dataset, load_metric, Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification,AutoModelWithLMHead, AutoTokenizer, AutoConfig, DataCollatorWithPadding, DataCollator
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer


## load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"

# label2id mapping
label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=512)  
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, label2id=label2id, id2label=id2label).to(device)  

## MNLI
dataset_train_mnli = load_dataset('multi_nli', split="train") 
dataset_test_mnli_m = load_dataset('multi_nli', split="validation_matched") 
#dataset_test_mnli_mm = load_dataset('multi_nli', split="validation_mismatched") 

print(dataset_train_mnli['premise'][:4])
print(dataset_train_mnli['hypothesis'][:4])
print(dataset_train_mnli['label'][:4])

#### tokenization

dynamic_padding = True

if dynamic_padding == False:
    def tokenize_func(examples):
        return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=256)  # max_length=512,  padding=True

## dynamic padding
elif  dynamic_padding == True:
    def tokenize_func(examples):
        return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)  # max_length=512,  padding=True

encoded_dataset_train = dataset_train_mnli.map(tokenize_func, batched=True)
encoded_dataset_test = dataset_test_mnli_m.map(tokenize_func, batched=True)

if  dynamic_padding == True:
    data_collator = DataCollatorWithPadding(tokenizer)


from datasets import list_metrics

metric = load_metric('accuracy') 


training_directory = "nli-few-shot/mnli-3c/"

train_args = TrainingArguments(
    output_dir=f'./results/{training_directory}',
    overwrite_output_dir=True,
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    #warmup_steps=0,  # 1000,
    warmup_ratio=0.06,  #0.1, 0.06
    weight_decay=0.1,  #0.1,
    fp16=True,
    fp16_full_eval=True,
    seed=42,
    prediction_loss_only=True,
    
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(references=labels,predictions=predictions)


trainer = Trainer( 
    model=model,
    tokenizer=tokenizer,
    args=train_args,
    data_collator=data_collator,
    train_dataset=encoded_dataset_train,  
    eval_dataset=encoded_dataset_test,  
    compute_metrics=compute_metrics
)

# train the model
trainer.train()
model.save_pretrained('./results/')

# Test the model

premise = "The Movie have been criticized for the story. However, I think it was a great movie."
hypothesis = "I liked the movie"

input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")

output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"

prediction = torch.softmax(output["logits"][0], -1)

label_names = ["entailment", "neutral", "contradiction"]

print(label_names[prediction.argmax(0).tolist()])


