from datasets import Dataset
import pandas as pd 
import os

df = pd.read_csv("verb_sentences.csv")
df.rename(columns={"sentences": "text"}, inplace=True)
df.drop(columns=["pagenames"], inplace=True)
dataset = Dataset.from_pandas(df)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import transformers
import torch
#from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer

#model_id = "vilsonrodrigues/falcon-7b-instruct-sharded" # sharded model by vilsonrodrigues
model_id = "tiiuae/falcon-40b-instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)

print("Loaded model")

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
    )

model = get_peft_model(model, config)

tokenized_data = dataset.map(lambda samples: tokenizer(samples['text']))
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data,
    # eval_dataset=val_dataset,
    args=transformers.TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        #max_steps=100,
        learning_rate=2e-4,
        fp16=False,
        logging_steps=50,
        output_dir="empath-falcon-40b",
        optim="paged_adamw_8bit",
        lr_scheduler_type='cosine',
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer.train()
#trainer.save_model("empath-falcon-40b/model")
trainer.push_to_hub()



