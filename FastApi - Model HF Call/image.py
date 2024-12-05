from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
import requests
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

app = FastAPI()


hf_token = os.getenv("HF_TOKEN")


IMAGE_API_URL = "https://api-inference.huggingface.co/models/nvidia/segformer-b0-finetuned-ade-512-512"
TEXT_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"

headers = {"Authorization": f"Bearer {hf_token}"}

def query_image(image_bytes: bytes) -> bytes:
    response = requests.post(IMAGE_API_URL, headers=headers, data=image_bytes)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error contacting Hugging Face API")
    return response.content

@app.post("/image_segment/")
async def remove_background(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result_image_bytes = query_image(image_bytes)
        return StreamingResponse(BytesIO(result_image_bytes), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TextGenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

@app.post("/generate-text/")
async def generate_text(request: TextGenerationRequest):
    """Generate text based on the provided prompt."""
    try:
        response = requests.post(
            TEXT_API_URL,
            headers=headers,
            json={
                "inputs": request.prompt,
                "parameters": {
                    "max_new_tokens": request.max_tokens
                }
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error contacting Hugging Face API")
        
        
        response_json = response.json()
        print(response_json)  

        
        if isinstance(response_json, list):
            generated_text = ''.join([message.get('generated_text', '') for message in response_json])
        else:
            generated_text = response_json.get('generated_text', '')

        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
