import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import nltk
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from filelock import FileLock
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    AutoConfig
)

# params setting
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", required=True, type=str, help=".csv format")
parser.add_argument("--valid_file", required=True, type=str, help=".csv format")
parser.add_argument("--pre_trained_model", default='google/mt5-small', type=str, help="from hugging face")
parser.add_argument("--epoch", default=40, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--strategy", default='beam', type=str, choices=['greedy', 'beam', 'temp', 'topk', 'topp'])
args = parser.parse_args()

train_file = args.train_file
valid_file = args.valid_file
model_name = args.pre_trained_model
epoch = args.epoch
batch_size = args.batch_size
strategy = args.strategy # greedy, beam, temp, topk, topp
prefix = "summarize: "
os.makedirs('./tmp', exist_ok=True)

# load data
train_data = load_dataset('csv', data_files=train_file)
valid_data = load_dataset('csv', data_files=valid_file)

# preprocess
tokenizer = AutoTokenizer.from_pretrained(model_name)
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True)

    labels = tokenizer(examples["summary"], max_length=64, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_valid = valid_data.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

# model config
if strategy == 'greedy':
    mt5_config = AutoConfig.from_pretrained(
        model_name,
        max_length = 64,
    )

if strategy == 'beam':
    mt5_config = AutoConfig.from_pretrained(
        model_name,
        max_length = 64,
        num_beams=8, 
        early_stopping=True
    )

if strategy == 'temp': 
    mt5_config = AutoConfig.from_pretrained(
        model_name,
        max_length = 64,
        do_sample=True, 
        top_k=0,
        temperature=0.5
    )

if strategy == 'topk': 
    mt5_config = AutoConfig.from_pretrained(
        model_name,
        max_length = 64,
        do_sample=True, 
        top_k=8,
    )

if strategy == 'topp': 
    mt5_config = AutoConfig.from_pretrained(
        model_name,
        max_length = 64,
        do_sample=True, 
        top_p=0.90, 
        top_k=0
    )

# load model 
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=mt5_config)

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./tmp/epoch_{epoch}_batch_{batch_size}_strategy_{strategy}",
    evaluation_strategy="no",
    save_steps = 5000,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    num_train_epochs=epoch,
    warmup_steps = 100,
    predict_with_generate=False,
    fp16=True,
    push_to_hub=False,
    adafactor=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train['train'],
    eval_dataset=tokenized_valid['train'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# main
print('='*20)
print(f'model_name:{model_name}', f'epoch:{epoch}', f'batch_size:{batch_size}', f'strategy:{strategy}')
print('='*20)
trainer.train()