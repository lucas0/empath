from datasets import Dataset
import pandas as pd 
import os
os.environ["WANDB_DISABLED"] = "true"

df = pd.read_csv("verb_sentences.csv").head()
df.rename(columns={"sentences": "text"}, inplace=True)
df.drop(columns=["pagenames"], inplace=True)
dataset = Dataset.from_pandas(df)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

model_name = "tiiuae/falcon-40b-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto"
    )

print("Loaded model")

model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

from peft import LoraConfig
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


peft_config = LoraConfig(
    inference_mode = False,
    lora_alpha=32,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
    "query_key_value",
#    "dense",
#    "dense_h_to_4h",
#    "dense_4h_to_h",
    ]
)

import transformers
from transformers import TrainingArguments

training_arguments = TrainingArguments(
    output_dir="empath-falcon-40b",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    logging_steps=10,
    learning_rate=1e-3,
    fp16=True,
    max_grad_norm=0.3,
    report_to=None,
    max_steps=10,
    warmup_ratio= 0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    )


max_seq_length = 512
tokenized_data = dataset.map(lambda samples: tokenizer(samples['text']))

trainer = transformers.Trainer(
    model=model, 
    train_dataset=tokenized_data,
    args=training_arguments,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

#for name, module in trainer.model.named_modules():
#    if "norm" in name:
#        module = module.to(torch.float32)

trainer.train()
trainer.save_model("empath-falcon-40b/model")
trainer.mode.save_config("empath-falcon-40b/config.json")
#trainer.push_to_hub("empath-falcon-40b")



