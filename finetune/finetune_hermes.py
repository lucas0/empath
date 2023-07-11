import torch
torch.cuda.is_available()
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

filename = sys.argv[0]
cwd = os.path.abspath(filename+"/..")

model_name_or_path = "TheBloke/Chronos-Hermes-13B-SuperHOT-8K-fp16"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
config.max_position_embeddings = 4096
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
    config=config,
    trust_remote_code=True,
    load_in_8bit=True,
    device_map='auto')

for param in model.parameters():
    param.requires_grad = False  # freeze the model - train adapters later
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)

model.lm_head = CastOutputToFloat(model.lm_head)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

from peft import LoraConfig, get_peft_model 

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
    )

model = get_peft_model(model, config)
print_trainable_parameters(model)

import pandas as pd
from datasets import Dataset

df = pd.read_csv(cwd+"/sentences.csv")
df = df.rename(columns={"sentences": "train"})
ds = Dataset.from_pandas(df)
tokenized_data = ds.map(lambda samples: tokenizer(samples["train"]))

import transformers

trainer = transformers.Trainer(
    model=model, 
    train_dataset=tokenized_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=100,
        learning_rate=1e-3, 
        fp16=False,
        logging_steps=1, 
        output_dir='empath-hermes-13b',
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
trainer.push_to_hub("empath-hermes-13b")


