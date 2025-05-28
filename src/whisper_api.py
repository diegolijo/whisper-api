from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Form
from fastapi.middleware.cors import CORSMiddleware
import whisper
import os
import uuid
import time
import magic
import base64
#import numpy as np
#from io import BytesIO

app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    allow_credentials=True,
)

# modelo Whisper ("tiny" 75MB, "base" 140MB, "small" 480MB, "medium" 1.5GB, "large" 3.1GB)
model = whisper.load_model("base")
print(f"Model device: {model.device}")

def is_audio_file(file_bytes):
    """Verifica si los bytes corresponden a un archivo de audio válido"""
    mime = magic.from_buffer(file_bytes, mime=True)
    return mime.startswith('audio/') or mime.startswith('video/') or mime in ['application/octet-stream']

def save_temp_file(file_bytes, extension=".wav"):
    """Guarda bytes en un archivo temporal"""
    temp_file = f"temp_{uuid.uuid4()}{extension}"
    with open(temp_file, "wb") as f:
        f.write(file_bytes)
    return temp_file

@app.post("/transcribe_base64")
async def transcribe_audio(
    base64_audio: str = Body(None),
    language: str = Body(None),
):
    try:
        if base64_audio:
            file_bytes = base64.b64decode(base64_audio.split(",")[-1])
        else:
            raise HTTPException(status_code=400, detail="Debe proporcionar un archivo de audio o datos base64")
        
        if not is_audio_file(file_bytes):
            raise HTTPException(status_code=400, detail="El archivo proporcionado no es un audio válido")

        temp_file = save_temp_file(file_bytes)
        start_time = time.time()

        result = model.transcribe(
            temp_file,
            language=language if language else None,
            fp16=False,         # Para evitar warnings en CPU
            verbose=False       # Para menos output en consola
        )

        processing_time = time.time() - start_time

        os.remove(temp_file)

        return {
            "processing_time":  processing_time,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe_file")
async def transcribe_audio(
    file: UploadFile = File(None),
    language: str = Form(None)  
):
    try:
        if file:
            file_bytes = await file.read()
        else:
            raise HTTPException(status_code=400, detail="Debe proporcionar un archivo de audio o datos base64")

        if not is_audio_file(file_bytes):
            raise HTTPException(status_code=400, detail="El archivo proporcionado no es un audio válido")

        temp_file = save_temp_file(file_bytes)
        start_time = time.time()

        result = model.transcribe(
            temp_file,
            language=language if language else None,
            fp16=False,         # Para evitar warnings en CPU
            verbose=False       # Para menos output en consola
        )

        processing_time = time.time() - start_time

        os.remove(temp_file)

        return {
            "processing_time": processing_time,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False, log_level="error" )