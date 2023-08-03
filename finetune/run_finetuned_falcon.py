import os,sys
from transformers import AutoTokenizer, AutoModelForCausalLM

filename = sys.argv[0]
cwd = os.path.abspath(filename+"/..")

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

peft_model_id = "lucas0/empath-falcon-40b"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_4bit=True, device_map='auto', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

pipe = pipeline("text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
    )

while True:
    print("===================")
    question = input("Input your question:\n>>> ")
    output = pipe(question)[0]
    answer = output['generated_text']
    print(answer)
