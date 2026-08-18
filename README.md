# Invoice AI — Hệ thống Trích xuất Hóa đơn Tự động

Hệ thống AI **end-to-end** nhận diện và trích xuất thông tin có cấu trúc từ hóa đơn/biên nhận tiếng Việt, sử dụng kiến trúc **Client-Server** hiện đại (FastAPI + React).

---

## 🏗 Kiến trúc Pipeline

```
Input Image
    │
    ▼
┌───────────────────┐
│  1. Preprocessing │  ← Deskew, Crop hóa đơn (OpenCV + Homography)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  2. DBNet         │  ← Phát hiện vùng chữ (PaddleOCR v4, ONNX Runtime)
└───────┬───────────┘
        │  danh sách bbox + crops
        ▼
┌───────────────────┐
│  3. VietOCR       │  ← Nhận diện ký tự (Fine-tuned Seq2Seq Transformer)
└───────┬───────────┘
        │  danh sách (word, bbox)
        ▼
┌───────────────────┐
│  4. LayoutLMv3    │  ← Gán nhãn NER (Fine-tuned, ONNX Runtime)
└───────┬───────────┘
        │
        ▼
  Structured JSON Output
  {SELLER, ADDRESS, TIMESTAMP, TOTAL_COST}
```

---

## 📦 Mô tả các Module

### Module 1 — Preprocessing
| Thông số | Chi tiết |
|----------|----------|
| Kỹ thuật | Saliency Map + Contour Detection + Homography Transform |
| Đầu vào | Ảnh hóa đơn gốc (bất kỳ góc chụp) |
| Đầu ra | Ảnh được căn thẳng, crop sát viền hóa đơn |
| Framework | OpenCV |

---

### Module 2 — Text Detection (DBNet)
| Thông số | Chi tiết |
|----------|----------|
| Model | DBNet — Differentiable Binarization Network |
| Pretrained | PaddleOCR v4 (`ch_PP-OCRv4_det`) |
| Backbone | ResNet + FPN (Feature Pyramid Network) |
| Runtime | ONNX Runtime (CUDAExecutionProvider) |
| Fine-tune | ❌ Không (pretrained weights đủ tốt cho hóa đơn tiếng Việt) |
| Đầu ra | Danh sách Bounding Boxes (đa giác 4 điểm) |

> **Lý do chọn:** DBNet sử dụng "Differentiable Binarization" thay vì ngưỡng cứng, cho phép huấn luyện End-to-End và phát hiện chữ sát nhau, chữ cong cực hiệu quả.

---

### Module 3 — Text Recognition (VietOCR) — Fine-tuned ✅

| Thông số | Chi tiết |
|----------|----------|
| Model | VietOCR (Seq2Seq: VGG19-CNN Encoder + Transformer Decoder) |
| Base pretrained | `vgg_transformer` trên bộ dữ liệu tiếng Việt tổng quát |
| Fine-tune dataset | ~4.500 crops hóa đơn thực tế từ MCOCR dataset + tự tổng hợp |
| Optimizer | AdamW |
| Scheduler | Cosine Annealing |
| Max iterations | 10.000 |
| Batch size | 32 |

#### 📊 Kết quả Fine-tune (Validation set)

| Checkpoint | Train Loss | Valid Loss | Acc Full Seq | Acc Per Char |
|-----------|-----------|-----------|-------------|-------------|
| Iter 2.000 | 0.659 | 0.653 | 62.15% | 79.50% |
| Iter 4.000 | 0.614 | 0.634 | 63.19% | 78.86% |
| Iter 6.000 | 0.588 | 0.617 | **65.00%** | **81.68%** |
| Iter 7.000 | 0.577 | 0.618 | 66.46% | 82.68% |
| Iter 9.000 | 0.557 | 0.613 | 66.67% | **83.14%** |
| **Iter 10.000** | **0.563** | **0.611** | **66.04%** | **83.16%** |

> **Best checkpoint:** Iter 9.000 (Acc per char: **83.14%**, Valid Loss: 0.613)
>
> **Cải thiện nổi bật:** Mô hình sau fine-tune xử lý chính xác các số tiền có dấu chấm phân cách hàng nghìn (vd: `3.500.000`), ký hiệu VAT (`GTGT`, `%`), tên sản phẩm đặc thù của hóa đơn Việt Nam mà mô hình gốc hay nhầm.

---

### Module 4 — Information Extraction (LayoutLMv3) — Fine-tuned ✅

| Thông số | Chi tiết |
|----------|----------|
| Model | `microsoft/layoutlmv3-base` |
| Task | Token Classification (NER — Named Entity Recognition) |
| Labels | `B/I-SELLER`, `B/I-ADDRESS`, `B/I-TIMESTAMP`, `B/I-TOTAL_COST`, `O` |
| Fine-tune dataset | ~3.000 ảnh hóa đơn thực tế (MCOCR public dataset) |
| Max seq length | 512 tokens |
| Batch size | 4 |
| Learning rate | 2e-5 |
| Epochs | 5 |
| Runtime | ONNX Runtime (CUDAExecutionProvider) — xuất từ PyTorch checkpoint |
| Mixed Precision | FP16 (giảm VRAM) |

> **Đặc điểm:** LayoutLMv3 là mô hình đa phương thức (Multimodal) kết hợp đồng thời:
> - **Text** (nội dung từ VietOCR)
> - **Layout** (tọa độ bbox chuẩn hóa 0–1000)
> - **Vision** (patch ảnh 16×16 từ ViT)
>
> Điều này cho phép mô hình phân biệt `50.000` ở dòng "Tiền hàng" khác với `50.000` ở dòng "Thuế VAT" dựa trên vị trí không gian, không chỉ dựa vào nội dung text đơn thuần.

---

## 🧰 Tech Stack

| Layer | Công nghệ |
|-------|-----------|
| **Backend API** | FastAPI + Uvicorn |
| **Frontend UI** | React 19 + TypeScript + Vite |
| **AI Runtime** | PyTorch, ONNX Runtime (GPU) |
| **GPU** | CUDA 12.x — NVIDIA RTX 40-series tested |
| **Detection** | PaddleOCR v4 |
| **Recognition** | VietOCR |
| **Extraction** | HuggingFace Transformers (`layoutlmv3-base`) |
| **Deployment** | Docker + docker-compose (nvidia-docker) |

---

## 📁 Cấu trúc dự án

```
invoice/
├── backend/
│   └── main.py                  # FastAPI Server (API + static file serving)
├── frontend/                    # React + TypeScript (Vite)
│   ├── src/
│   │   ├── App.tsx              # Main component (Drag & Drop UI)
│   │   └── App.css              # Glassmorphism styling
│   └── package.json
├── module/
│   ├── preprocessing/           # Deskew, crop hóa đơn
│   ├── DBnet/                   # Text Detection (PaddleOCR + ONNX)
│   ├── recognition/             # VietOCR Text Recognition
│   ├── extraction/              # Line clustering, LayoutLM data preparation
│   ├── finetune/                # LayoutLMv3 Training + ONNX Exporter
│   ├── synthetic_data/          # Data augmentation / generation utilities
│   ├── config.py                # Centralized config
│   └── pipeline.py              # End-to-end pipeline orchestrator
├── saved_models/                # Fine-tuned model weights (tải từ Drive)
│   ├── layoutlmv3_onnx/         # LayoutLMv3 ONNX model (~480 MB)
│   └── vietocr_mcocr/           # VietOCR fine-tuned weights (~145 MB)
├── Dockerfile                   # Multi-stage: Node build → CUDA runtime
├── docker-compose.yml           # GPU passthrough + volume mounts
├── requirements.txt
└── README.md
```

---

## 🚀 Cài đặt & Chạy

### Cách 1: Development (Local)

**Yêu cầu:** Python 3.10+, Node.js 18+, NVIDIA GPU + CUDA 12.x

```bash
# 1. Clone repo
git clone https://github.com/Thien-Dan/extract_and_classify_invoice_content.git
cd extract_and_classify_invoice_content

# 2. Cài Python dependencies
pip install -r requirements.txt

# 3. Tải & chuyển đổi model DBNet sang ONNX
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar
tar -xf ch_PP-OCRv4_det_infer.tar
paddle2onnx \
  --model_dir ch_PP-OCRv4_det_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ch_PP-OCRv4_det_infer/model.onnx

# 4. Tải fine-tuned models (~625 MB) về thư mục saved_models/
# 📥 Link: https://drive.google.com/drive/folders/121WlFDBltmqu1lJrwINh3xev309ns3P1?hl=vi

# 5. Cài Frontend
cd frontend && npm install && cd ..

# 6. Khởi động Backend (Terminal 1)
uvicorn backend.main:app --reload

# 7. Khởi động Frontend (Terminal 2)
cd frontend && npm run dev
```

| Service | URL |
|---------|-----|
| Frontend (React UI) | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| Swagger API Docs | http://localhost:8000/docs |

---

### Cách 2: Docker (Production)

**Yêu cầu:** Docker Desktop + `nvidia-docker2`

```bash
# Đặt saved_models/ vào thư mục gốc trước
# 📥 Link: https://drive.google.com/drive/folders/121WlFDBltmqu1lJrwINh3xev309ns3P1?hl=vi

docker compose up --build
# Truy cập: http://localhost:8000
```

---

## 📡 API Reference

### `POST /api/v1/extract`

Upload ảnh hóa đơn, nhận về dữ liệu đã trích xuất.

**Request:** `multipart/form-data` — field `file` (JPG / PNG)

**Response:**
```json
{
  "success": true,
  "data": [
    {"label": "SELLER",     "text": "CỬA HÀNG TIỆN LỢI ABC"},
    {"label": "ADDRESS",    "text": "123 Nguyễn Trãi, Q.5, TP.HCM"},
    {"label": "TIMESTAMP",  "text": "15/08/2026"},
    {"label": "TOTAL_COST", "text": "1.250.000"}
  ],
  "annotated_image_base64": "data:image/jpeg;base64,...",
  "message": "Extraction successful."
}
```

### `GET /health`
```json
{"status": "ok", "message": "AI Pipeline is running."}
```

---

## 📋 Dataset

| Dataset | Số lượng | Dùng cho |
|---------|----------|----------|
| [MCOCR Public](https://drive.google.com/drive/folders/1yuuRkBXzEQMHMLs7f5qCOAL_HqL_CHPK) | ~3.000 ảnh hóa đơn thực tế có nhãn | Fine-tune LayoutLMv3 |
| MCOCR crops | ~4.500 text crops | Fine-tune VietOCR |

---

## License

MIT
