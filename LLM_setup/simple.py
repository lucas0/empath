from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, pipeline

model_name = "TheBloke/Chronos-Hermes-13B-SuperHOT-8k-fp16" 

config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=2048,
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.15)

output = pipe("What is Cisco Webex?")
print(output)
