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
from utils import get_prompt, get_few_shot_prompt
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
        "--peft_path",
        type=str,
        default=None,
        help="The path of the adapter.",
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
num_epochs = 0
accelerator = Accelerator()

device_map = {"":0}
tokenizer = AutoTokenizer.from_pretrained("yentinglin/Taiwan-LLM-7B-v2.0-chat", revision="5073b2bbc1aa5519acdc865e99832857ef47f7c9")
bnb_config = get_bnb_config()
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
    inputs['labels'] = [-100]*len(only_inputs['input_ids'])+labels['input_ids']
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


first_shot = "翻譯成文言文：\n五年春正月丙午，齊獻武王在晉陽逝世，秘密不公布喪事。"
first_shot_answer = "五年春正月丙午，齊獻武王薨於晉陽，秘不發喪。"
second_shot = "翻譯成現代文：\n祿山構逆，承嗣與張忠誌等為前鋒，陷河洛。\n答案："
second_shot_answer = "安祿山叛亂，田承嗣和張忠誌等擔任先鋒，攻陷河洛。"
third_shot = "翻譯成文言文：\n因此忠貞的臣子，並非不想竭盡忠誠，竭盡忠誠實在太難瞭。"
third_shot_answer = "故忠貞之臣，非不欲竭誠。竭誠者，乃是極難。"
#total_shots = get_prompt(first_shot)+first_shot_answer+get_prompt(second_shot)+second_shot_answer+get_prompt(third_shot)+third_shot_answer

def prepare_valid_features(examples, indices):
    inputs = get_few_shot_prompt(first_shot, first_shot_answer, second_shot, second_shot_answer, third_shot, third_shot_answer, examples['instruction'])
    print(inputs)
    inputs = tokenizer(inputs, return_tensors="pt")
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

model = prepare_model_for_kbit_training(model)
#model.to('cuda')



#Predict
#model.save_pretrained("./gdrive/MyDrive/ADL_Homework3", safe_serialization=True)


predictions = []
generation_config = GenerationConfig(
        temperature=1.2,
        top_p=0.9,
        top_k=10,
        num_beams=1,
        do_sample=True,
        )
"""
gen_kwargs = {
            "max_length": 64,
            "num_beams":  5,
            "do_sample": True,
            "top_k": 10,
            "top_p": 0.9,
            "temperature": 1.2,
            }
            """
model.eval()
for step, batch in enumerate(tqdm(valid_loader)):
  #batch.to('cuda')
  with torch.no_grad():
    #print(batch["input_ids"][0])
    tokens = model.generate(input_ids=batch["input_ids"][0].to('cuda'), generation_config=generation_config, return_dict_in_generate=True, max_new_tokens=256)
    #print(tokens)
    tokens = tokens.sequences[0]
    pred = tokenizer.decode(tokens, skip_special_tokens=True)
#    tokens = accelerator.gather(tokens).cpu().numpy()
#    pred = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    print("Length", len(pred.split("ASSISTANT:")))
    if len(pred.split("ASSISTANT:"))>4:
        pred = pred.split("ASSISTANT:")[4].strip()
    #pred.replace("你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: ", "")
    print(pred)
    predictions += [pred]
#print("Prediction: ", predictions)

with open("gdrive/MyDrive/data/public_test.json", 'r') as f:
  data = json.load(f)
for i, entry in enumerate(predictions):
  if len(entry.split("ASSISTANT: "))>1:
    entry = entry.split("ASSISTANT: ")[1]
  data[i]['output'] = entry
print(data)
with open(args.output_file, 'w', encoding='utf-8', errors='ignore') as file:
    json.dump(data, file, ensure_ascii=False)
