# Imagem de produção — CPU only (Railway não tem GPU por defeito)
FROM python:3.11-slim

WORKDIR /app

# Torch CPU separado para manter a imagem menor (~200MB vs ~2GB com CUDA)
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY artifacts/ ./artifacts/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
