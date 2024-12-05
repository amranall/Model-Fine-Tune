import torch
from transformers import pipeline
import os
# os.environ['TRANSFORMERS_CACHE'] = 'C:\\path\\to\\custom\\cache'
os.environ["HF_API_TOKEN"] = "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"

# Specify the model ID
model_id = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8"

# Initialize the text generation pipeline
pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device=-1,
    #device=0 if torch.cuda.is_available() else -1
)

# Generate text based on a prompt
# prompt = "to do crud in fastapi"
# output = pipe(prompt, max_new_tokens=500)

# # Print the generated output
# print(output)

import gc

# Delete the pipeline and collect garbage
del pipe
gc.collect()










# from huggingface_hub import login
# # Replace 'YOUR_HUGGINGFACE_TOKEN' with your actual token
# login(token='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')