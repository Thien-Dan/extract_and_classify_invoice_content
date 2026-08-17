# ============================================================
# Stage 1: Build React Frontend
# ============================================================
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ============================================================
# Stage 2: Python Backend + Serve Frontend
# ============================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python & system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy backend code & modules
COPY backend/ ./backend/
COPY module/ ./module/
COPY config.json ./

# Copy built frontend to be served by FastAPI
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Download DBNet model if not present
RUN mkdir -p ch_PP-OCRv4_det_infer && \
    wget -q https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar -O /tmp/det.tar && \
    tar -xf /tmp/det.tar -C . && \
    pip3 install paddle2onnx && \
    paddle2onnx --model_dir ch_PP-OCRv4_det_infer \
        --model_filename inference.pdmodel \
        --params_filename inference.pdiparams \
        --save_file ch_PP-OCRv4_det_infer/model.onnx && \
    rm /tmp/det.tar

# NOTE: saved_models/ must be mounted as a volume or copied manually
# They are too large for Docker image (500MB+ ONNX + 150MB VietOCR)
VOLUME ["/app/saved_models"]

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
