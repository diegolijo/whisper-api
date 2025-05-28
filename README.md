# Whisper API con FastAPI

## Crear entorno virtual

```bash
python -m venv venv
```
### activar 

```bash
venv/scripts/activate
```

## 📦 Instalación de dependencias

```bash
pip install fastapi uvicorn openai-whisper numpy python-multipart python-magic-bin librosa soundfile
```

### Dependencias adicionales requeridas:

- **FFmpeg** (para procesamiento de audio):

  ```bash
  # Ubuntu/Debian
  sudo apt update && sudo apt install ffmpeg

  # macOS (con Homebrew)
  brew install ffmpeg

  # Windows (via Chocolatey)
  choco install ffmpeg
  ```

## 🚀 Ejecución del servidor

```bash
python whisper_api.py
```

El servidor estará disponible en:

- API: `http://localhost:8000`
- Documentación interactiva: `http://localhost:8000/docs`

## 🔧 Funcionamiento del código

### Estructura principal:

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper  # Modelo de reconocimiento de voz
import magic    # Validación de tipos MIME
import base64   # Decodificación de audio en base64
```

### Flujo de procesamiento:

1. **Recepción de audio**:

   - Acepta dos formatos de entrada:
     - Archivo binario (BLOB) vía `UploadFile`
     - String en base64 vía campo `base64_audio`

2. **Validación**:

   - Usa `python-magic` para verificar que el archivo sea un audio válido
   - Soporta formatos comunes (MP3, WAV, OGG, etc.)

3. **Procesamiento**:

   - Guarda temporalmente el archivo en disco
   - Usa Whisper para transcribir el audio
   - Elimina el archivo temporal después de procesar

4. **Respuesta**:
   - Devuelve el objeto completo de resultado de Whisper
   - Incluye métricas de tiempo de procesamiento

### Ejemplo de solicitud:

```bash
curl -X POST -F "file=@audio.mp3" http://localhost:8000/transcribe
```

O con base64:

```bash
curl -X POST -d "{\"base64_audio\":\"$(base64 audio.mp3)\"}" http://localhost:8000/transcribe
```

## 📊 Estructura de la respuesta

La API devuelve un JSON con:

```json
{
  "result": {
    "text": "Texto transcrito completo",
    "language": "es",
    "segments": [
      {
        "id": 0,
        "start": 0.0,
        "end": 4.0,
        "text": "Fragmento de texto",
        "no_speech_prob": 0.02
      }
    ],
    "language_probability": 0.95
  },
  "processing_time": "3.25 segundos"
}
```

## ⚠️ Consideraciones importantes

#### **1. Modelos disponibles**

Whisper ofrece varios modelos pre-entrenados con diferentes equilibrios entre precisión y rendimiento:

| Modelo   | Tamaño  | RAM Requerida | Calidad Relativa | Uso Recomendado                         |
| -------- | ------- | ------------- | ---------------- | --------------------------------------- |
| `tiny`   | ~75 MB  | ~1 GB         | Baja             | Pruebas rápidas, dispositivos limitados |
| `base`   | ~140 MB | ~1 GB         | Básica           | Uso general en CPU                      |
| `small`  | ~480 MB | ~2 GB         | Media            | Buen equilibrio velocidad/precisión     |
| `medium` | ~1.5 GB | ~5 GB         | Alta             | Servidores con buena RAM                |
| `large`  | ~3.1 GB | ~10 GB+       | Máxima           | Máxima precisión (requiere GPU)         |

**Para cambiar el modelo**, modifica esta línea:

```python
model = whisper.load_model("base")  # Reemplaza "base" por el modelo deseado
```

2. **Requisitos hardware**:

   - Modelo `base` requiere ~1GB de RAM
   - Modelo `large` requiere ~10GB de RAM (recomendado GPU)

3. **Seguridad**:
   - En producción, restringir CORS (`allow_origins`) a dominios específicos
   - Considerar autenticación para endpoints públicos

# EJECUTAR:

```bash
python whisper_api.py
```
