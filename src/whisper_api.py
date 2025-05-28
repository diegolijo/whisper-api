from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import os
import uuid
import time
import numpy as np
from io import BytesIO
import magic
import base64

app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga el modelo Whisper ("tiny" 75MB, "base" 140MB, "small" 480MB, "medium" 1.5GB, "large" 3.1GB)
model = whisper.load_model("base")
print(f"Model device: {model.device}")

def is_audio_file(file_bytes):
    """Verifica si los bytes corresponden a un archivo de audio válido"""
    mime = magic.from_buffer(file_bytes, mime=True)
    return mime.startswith('audio/') or mime in ['application/octet-stream']

def save_temp_file(file_bytes, extension=".wav"):
    """Guarda bytes en un archivo temporal"""
    temp_file = f"temp_{uuid.uuid4()}{extension}"
    with open(temp_file, "wb") as f:
        f.write(file_bytes)
    return temp_file

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(None),
    base64_audio: str = None,
    language: str = None 
):
    try:
        # Manejo de ambos tipos de entrada (BLOB o Base64)
        if file:
            file_bytes = await file.read()
        elif base64_audio:
            file_bytes = base64.b64decode(base64_audio.split(",")[-1])
        else:
            raise HTTPException(status_code=400, detail="Debe proporcionar un archivo de audio o datos base64")

        # Validación del archivo de audio
        if not is_audio_file(file_bytes):
            raise HTTPException(status_code=400, detail="El archivo proporcionado no es un audio válido")

        # Guardado temporal y transcripción
        temp_file = save_temp_file(file_bytes)
        start_time = time.time()

        result = model.transcribe(
            temp_file,
            language=language,  # <-- Usar el idioma proporcionado
            fp16=False,         # Para evitar warnings en CPU
            verbose=False       # Para menos output en consola
        )

        processing_time = time.time() - start_time

        # Limpieza
        os.remove(temp_file)

        return {
            "processing_time": f"{processing_time:.2f} segundos",
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False, log_level="error" )