# ⚙️ Module Tinh chỉnh (Finetuning)

Module này đóng vai trò như "phòng thí nghiệm" của dự án, chứa toàn bộ các kịch bản (scripts) và siêu tham số (hyperparameters) dùng để đào tạo (Train) lại các mô hình lõi khi có dữ liệu mới.

## 🛠 Kiến trúc & Quy trình
1. **VietOCR Fine-tuner:** 
   - Quản lý quá trình đào tạo mô hình Seq2Seq. Sử dụng Optimizer AdamW, hàm mất mát Cross-Entropy Loss. Script hỗ trợ Resume Checkpoint để huấn luyện nhiều đợt.
2. **LayoutLMv3 Fine-tuner:**
   - Dựa trên thư viện HuggingFace 	ransformers và datasets. 
   - Xử lý việc căn chỉnh token (Token Alignment) - một bài toán rất phức tạp khi một từ trong bounding box bị chia nhỏ thành nhiều sub-tokens (BPE).
   - Huấn luyện trên GPU với mixed precision (FP16) để giảm mức tiêu thụ VRAM.
3. **ONNX Exporter:** 
   - Đoạn mã dùng để đóng băng (freeze) mạng nơ-ron PyTorch động và chuyển đổi nó sang biểu diễn đồ thị tĩnh của ONNX, hỗ trợ cắt giảm kích thước mô hình (Quantization) để deploy thực tế.
