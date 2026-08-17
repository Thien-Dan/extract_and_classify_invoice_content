# Invoice AI - Hệ thống Trích xuất Hóa đơn Tự động

Hệ thống AI end-to-end nhận diện và trích xuất thông tin từ hóa đơn/biên nhận tiếng Việt, sử dụng kiến trúc Client-Server hiện đại.

## Pipeline AI

```
Input Image -> Preprocessing (Deskew) -> DBNet (Text Detection) -> VietOCR (Text Recognition) -> LayoutLMv3 (Token Classification) -> Structured JSON
```

| Module | Model | Vai trò |
|--------|-------|---------|
| Text Detection | DBNet (PaddleOCR v4, ONNX) | Phát hiện vùng chữ trên ảnh |
| Text Recognition | VietOCR (Fine-tuned) | Nhận diện ký tự tiếng Việt |
| Information Extraction | LayoutLMv3 (Fine-tuned, ONNX) | Gán nhãn NER: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST |

## Tech Stack

- **Backend:** FastAPI + Uvicorn (Python)
- **Frontend:** React + TypeScript (Vite)
- **AI/ML:** PyTorch, ONNX Runtime (GPU), PaddleOCR, VietOCR, LayoutLMv3
- **GPU:** CUDA 12.x (NVIDIA)
- **Deployment:** Docker + docker-compose (nvidia-docker)

## Cấu trúc dự án

```
invoice/
├── backend/
│   └── main.py              # FastAPI Server
├── frontend/                 # React + TypeScript (Vite)
│   ├── src/
│   │   ├── App.tsx
│   │   └── App.css
│   └── package.json
├── module/
│   ├── preprocessing/        # Deskew, crop hóa đơn
│   ├── DBnet/                # Text Detection (PaddleOCR)
│   ├── recognition/          # VietOCR Text Recognition
│   ├── extraction/           # Clustering, LayoutLM formatting
│   └── finetune/             # LayoutLMv3 ONNX Inferencer
├── saved_models/             # Pre-trained & fine-tuned weights
│   ├── layoutlmv3_onnx/      # LayoutLMv3 ONNX model (~500MB)
│   └── vietocr_mcocr/        # VietOCR fine-tuned weights (~150MB)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Cài đặt & Chạy

### Cách 1: Chạy trực tiếp (Development)

**Yêu cầu:** Python 3.10+, Node.js 18+, NVIDIA GPU + CUDA 12.x

```bash
# 1. Clone repo
git clone <repo-url> && cd invoice

# 2. Cài Python dependencies
pip install -r requirements.txt

# 3. Tải model DBNet (PaddleOCR)
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar
tar -xf ch_PP-OCRv4_det_infer.tar
paddle2onnx --model_dir ch_PP-OCRv4_det_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ch_PP-OCRv4_det_infer/model.onnx

# 4. Cài Frontend dependencies
cd frontend && npm install && cd ..

# 5. Khởi chạy Backend (Terminal 1)
uvicorn backend.main:app --reload

# 6. Khởi chạy Frontend (Terminal 2)
cd frontend && npm run dev
```

Truy cập:
- **Frontend:** http://localhost:5173
- **API Docs (Swagger):** http://localhost:8000/docs

### Cách 2: Docker (Production)

**Yêu cầu:** Docker, nvidia-docker2

```bash
# Build & chạy (GPU)
docker compose up --build

# Truy cập tại http://localhost:8000
```

> **Lưu ý:** Thư mục `saved_models/` chứa model fine-tuned (~650MB) cần được tải riêng.
> 
> 📥 **Tải models tại đây:** [Google Drive - saved_models](https://drive.google.com/drive/folders/121WlFDBltmqu1lJrwINh3xev309ns3P1?hl=vi)
> 
> Sau khi tải, đặt vào thư mục gốc dự án sao cho cấu trúc là `saved_models/layoutlmv3_onnx/` và `saved_models/vietocr_mcocr/`.

## API Reference

### `POST /api/v1/extract`

Upload ảnh hóa đơn, nhận về kết quả trích xuất.

**Request:** `multipart/form-data` với field `file` (JPG/PNG)

**Response:**
```json
{
  "success": true,
  "data": [
    {"label": "SELLER", "text": "CỬA HÀNG ABC"},
    {"label": "ADDRESS", "text": "123 Nguyễn Trãi, Q.5, TP.HCM"},
    {"label": "TIMESTAMP", "text": "15/08/2026"},
    {"label": "TOTAL_COST", "text": "1.250.000"}
  ],
  "annotated_image_base64": "data:image/jpeg;base64,...",
  "message": "Extraction successful."
}
```

### `GET /health`

Kiểm tra trạng thái Server.

## License

MIT

