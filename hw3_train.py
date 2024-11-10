import os
import sys
import json
import argparse
import logging
import math
import numpy as np
from functools import partial
from time import strftime, localtime
import datasets
from datasets import load_dataset, load_metric
from tqdm.auto import tqdm
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils import clip_grad_norm_
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    default_data_collator,
    AutoTokenizer,
    get_scheduler,
    set_seed,
    MT5ForConditionalGeneration, MT5Tokenizer,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
    GenerationConfig,
)
from utils import get_prompt
from utils import get_bnb_config
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    AdaLoraConfig,
    PeftType,
    PeftConfig,
    PrefixTuningConfig,
    PromptEncoderConfig, LoraConfig, PromptTuningConfig, PeftModel,
)
#from trl import SFTTrainer
#from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description="Language Generation")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="The path of the model.",
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        default=None,
        help="The path of the adapter.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochs.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--valid_file", type=str, default=None, help="A csv or a json file containing the validation data or testing data."
    )
    parser.add_argument(
        "--output_file", type=str, default="output.jsonl", help="A jsonl file intended to be output."
    )
    args = parser.parse_args()
    return args


args = parse_args()
num_epochs = args.epochs
accelerator = Accelerator()

device_map = {"":0}
tokenizer = None
model = None
if args.model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

else:
    tokenizer = AutoTokenizer.from_pretrained("yentinglin/Taiwan-LLM-7B-v2.0-chat", revision="5073b2bbc1aa5519acdc865e99832857ef47f7c9")
bnb_config = get_bnb_config()


if args.model_name_or_path:
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.bfloat16,quantization_config=bnb_config)

else:
    model = AutoModelForCausalLM.from_pretrained("yentinglin/Taiwan-LLM-7B-v2.0-chat", revision="5073b2bbc1aa5519acdc865e99832857ef47f7c9",torch_dtype=torch.bfloat16,quantization_config=bnb_config)


# Old
raw_datasets = load_dataset("json", data_files={"train": args.train_file, "valid": args.valid_file} )

column_names = raw_datasets["train"].column_names
def prepare_train_features(examples, indices):
    inputs = get_prompt(examples['instruction'])+examples['output']
    only_inputs = get_prompt(examples['instruction'])
    answers = examples['output']
    """
    for question, answer in zip(examples['instruction'], examples['output']):
        q = question
        a = answer
        input = get_prompt(q)+a
        inputs.append(input)
        answers.append(a)
    """
    inputs = tokenizer(inputs,return_tensors=None, padding="max_length", truncation=True, max_length=256)
    only_inputs = tokenizer(only_inputs,return_tensors=None, padding="max_length", truncation=True, max_length=256)
    labels = tokenizer(answers,return_tensors=None, padding="max_length", truncation=True, max_length=64)
    #print(labels['input_ids'])
    #print(torch.tensor([-100]*len(inputs['input_ids'])))
    inputs['labels'] = [-100]*(len(only_inputs['input_ids'])-1)+labels['input_ids']
    #print(inputs['input_ids'])
    #print(inputs['labels'])
    #inputs['labels'] = torch.cat(torch.tensor([-100]*len(inputs['input_ids'])),labels['input_ids'])
    #inputs['labels'] = labels['input_ids']
    return inputs

train_examples = raw_datasets["train"]

train_dataset = train_examples.map(
    prepare_train_features,
        with_indices=True,
    remove_columns=column_names,

  )
print(train_dataset)

def prepare_valid_features(examples, indices):
    inputs = get_prompt(examples['instruction'])
    """
    for question, answer in zip(examples['instruction'], examples['output']):
        q = question
        input = get_prompt(q)
        inputs.append(input)
    """
    inputs = tokenizer(inputs, return_tensors="pt")
    #inputs = tokenizer(inputs, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
    return inputs

valid_examples = raw_datasets["valid"]

valid_dataset = valid_examples.map(
    prepare_valid_features,
    with_indices=True,
    batched=False,
    remove_columns=column_names,
  )
print(valid_dataset)

train_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=train_collator,
                                batch_size=2, num_workers=4)
valid_collator = default_data_collator
valid_loader = DataLoader(valid_dataset, shuffle=False, collate_fn=valid_collator,
                                batch_size=1, num_workers=4)

#model = prepare_model_for_kbit_training(model)
#model.to('cuda')
#peft_config.init_lora_weights = False
#print(peft_config)
"""
peft_config = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.1,
    r=4,
    bias="none",
    task_type="CAUSAL_LM",
)
"""
peft_config = PeftConfig.from_pretrained(args.peft_path)
model.add_adapter(peft_config)
#model = get_peft_model(model, peft_config)
model = PeftModel.from_pretrained(model, args.peft_path)
model.to('cuda')
model.enable_adapters()

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=0,
        num_train_epochs=num_epochs,
        learning_rate=3e-5,
        fp16=True,
        output_dir="outputs",
        optim="paged_adamw_32bit",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
if(num_epochs>0):
    trainer.train()


#Predict
model.save_pretrained("./gdrive/MyDrive/adapter_checkpoint2", safe_serialization=True)


