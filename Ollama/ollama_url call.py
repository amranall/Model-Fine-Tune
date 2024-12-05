
import asyncio
from asyncio import subprocess
from fastapi import FastAPI, HTTPException
import requests
from enum import Enum
from typing import Generator
from fastapi.responses import StreamingResponse
from enum import Enum
import httpx
from typing import AsyncGenerator

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama API endpoint
#model= llama3.2:1b | qwen2.5:3b

class Model(str, Enum):
    llama3_2_1b = "llama3.2:1b"
    qwen2_5_3b = "qwen2.5:3b"


async def stream_ollama_response(model: str, prompt: str):
    # Call ollama using asyncio subprocess, asynchronously
    process = await asyncio.create_subprocess_exec(
        "ollama", "generate", "-m", model, "-p", prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Stream the output from ollama
    while True:
        output = await process.stdout.readline()
        if output == b"" and process.poll() is not None:
            break
        if output:
            yield output.decode('utf-8')

    await process.wait()

@app.post("/generate-stream")
async def generate_stream(model: Model, prompt: str):
    return StreamingResponse(stream_ollama_response(model, prompt), media_type="text/plain")



async def generate_response(model: Model, prompt: str):
    """Generate a response from the specified model and stream it back to the client."""
    return StreamingResponse(stream_ollama_response(model, prompt), media_type="text/plain")



@app.post("/generate")
async def generate_response(model: Model, prompt: str):
    """Generate a response from the specified model."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()  # Return the JSON response from Ollama
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


# Streaming generator for progressive responses
def stream_response(response: requests.Response) -> Generator[str, None, None]:
    try:
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                yield chunk.decode("utf-8")  # Stream each chunk as UTF-8 text
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")

@app.post("/generate_formatted")
async def generate_formatted_response(model: Model, prompt: str):
    """Generate a formatted response from the specified model."""
    return await generate_response(model=model, prompt=prompt, stream=False)


# To run the app, use: uvicorn app:app --reload