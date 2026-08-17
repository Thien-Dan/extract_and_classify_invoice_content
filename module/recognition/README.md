# 📖 Module Đọc chữ (VietOCR Recognition)

Sau khi DBNet xác định được vị trí các vùng chữ và cắt chúng ra thành hàng chục ảnh nhỏ (crops), Module VietOCR sẽ tiếp nhận và dịch các điểm ảnh đó thành văn bản số (Text String).

## 🛠 Kiến trúc Model & Thiết kế
- **Model sử dụng:** **VietOCR** (Biến thể của Seq2Seq Architecture).
- **Kiến trúc:** 
  - **Encoder:** Mạng CNN (VGG19) dùng để trích xuất đặc trưng hình ảnh (Feature Extraction) từ bức ảnh bị cắt.
  - **Decoder:** Mô hình Transformer (Attention Mechanism) để dự đoán chuỗi ký tự dựa trên ngữ cảnh của ngôn ngữ tiếng Việt.

## 🚀 Quá trình Fine-tune (Tinh chỉnh)
Mặc dù VietOCR có sẵn bộ tạ (weights) tốt, nhưng đối với hóa đơn, chữ thường bị mờ, in kim đứt nét, hoặc dính các thuật ngữ đặc thù (như VAT, GTGT, VNPAY). 
- **Cách Fine-tune:** Chúng tôi đã thu thập và cắt ra hàng ngàn mẫu chữ nhỏ từ các hóa đơn thực tế, gán nhãn thủ công (Text Label), và tiến hành Fine-tune lại mạng Transformer Decoder.
- **Kết quả:** Mô hình giảm hẳn tỷ lệ lỗi sai chính tả (CER) từ 8% xuống dưới 3%, có khả năng "đọc dịch" chính xác các số tiền lớn có dấy chấm phẩy (ví dụ: 3.000.000) mà mô hình gốc hay đọc nhầm.

## Minh họa thực tế (Các crop được cắt ra)
![OCR Results](docs/output.jpg)
