import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use forward slashes for compatibility
tokenizer = AutoTokenizer.from_pretrained(r"D:\Imran Nur\model test finetune\Model Call\Qwen2.5-0.5B-Instruct-GPTQ-Int8")
model = AutoModelForCausalLM.from_pretrained(r"D:\Imran Nur\model test finetune\Model Call\Qwen2.5-0.5B-Instruct-GPTQ-Int8", torch_dtype="auto", device_map="auto")

inputs = tokenizer("hi", return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

