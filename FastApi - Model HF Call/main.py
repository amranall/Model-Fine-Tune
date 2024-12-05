import os
import torch
import multiprocessing
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from accelerate import Accelerator
from typing import List, Tuple
# Load environment variables from a .env file (useful for local development)
load_dotenv()

# HTML for the Buy Me a Coffee badge
html_content = """
<!DOCTYPE html>
<html>
    <head>
        <title>Llama-3.2-1B-Instruct-API</title>
    </head>
    <body>
        <div style="text-align: center;">
            <a href="https://buymeacoffee.com/xxparthparekhxx" target="_blank">
                <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" 
                     alt="Buy Me A Coffee" 
                     height="40px">
            </a>
            <h2>Please Chill Out! 😎</h2>
            <p>This API takes around <strong>5.62 minutes</strong> to process a single request due to current hardware limitations.</p>
            <h3>Want Faster Responses? Help Me Out! 🚀</h3>
            <p>If you'd like to see this API running faster on high-performance <strong>A100</strong> hardware, please consider buying me a coffee. ☕ Your support will go towards upgrading to <strong>Hugging Face Pro</strong>, which will allow me to run A100-powered spaces for everyone! 🙌</p>
            <h4>Instructions to Clone and Run Locally:</h4>
            <ol>
                <li><strong>Clone the Repository:</strong>   
                <div>
                    <code>git clone https://huggingface.co/spaces/xxparthparekhxx/llama-3.2-1B-FastApi</code>
                 </div> 
                 <div>
                    <code>cd llama-3.2-1B-FastApi</code>
                 </div>     
                </li>
                <li><strong>Run the Docker container:</strong>
                   <div>  <code>
                    docker build -t llama-api .  </code> </div>
                 <div>  <code>   docker run -p 7860:7860 llama-api </code> </div>
                </li>
                <li><strong>Access the API locally:</strong>
                    <p>Open <a href="http://localhost:7860">http://localhost:7860</a> to access the API docs locally.</p>
                </li>
            </ol>
        </div>
    </body>
</html>
"""

# FastAPI app with embedded Buy Me a Coffee badge and instructions
app = FastAPI(
    title="Llama-3.2-1B-Instruct-API",
    description= html_content,
    docs_url="/",  # URL for Swagger docs
    redoc_url="/doc"  # URL for ReDoc docs
)

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.set_num_threads(multiprocessing.cpu_count())
accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    token=HF_TOKEN,
    torch_dtype=torch.float16,
    device_map=device
)

model, tokenizer = accelerator.prepare(model, tokenizer)
# Pydantic models for request validation
class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7

class ChatRequest(BaseModel):
    message: str
    history: List[Tuple[str, str]] = []
    max_new_tokens: int = 100
    temperature: float = 0.7
    system_prompt: str = "You are a helpful assistant."


# Endpoints
@app.post("/generate/")
async def generate_text(request: PromptRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
   
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
   
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

@app.post("/chat/")
async def chat(request: ChatRequest):
    conversation = [
        {"role": "system", "content": request.system_prompt}
    ]
    for human, assistant in request.history:
        conversation.extend([
            {"role": "user", "content": human},
            {"role": "assistant", "content": assistant}
        ])
    conversation.append({"role": "user", "content": request.message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    assistant_response = response.split("Assistant:")[-1].strip()
    
    return {"response": assistant_response}