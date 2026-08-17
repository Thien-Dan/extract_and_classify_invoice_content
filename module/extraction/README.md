# 🧠 Module Trích xuất thông tin (LayoutLMv3 Extraction)

Đây là "bộ não" trung tâm của toàn bộ hệ thống. Nhiệm vụ của nó là nhận một đống văn bản hỗn độn từ VietOCR và tìm ra đâu là Tên Cửa Hàng, đâu là Tổng Tiền.

## 🛠 Kiến trúc Model & Thiết kế
- **Model sử dụng:** **LayoutLMv3** (Của Microsoft).
- **Sự ưu việt (Đa phương thức - Multimodal):** Các mô hình NLP truyền thống (như BERT/PhoBERT) chỉ đọc Text từ trái sang phải, nên chúng thường bị bối rối với hóa đơn (chữ "Tổng tiền" nằm bên trái, số tiền nằm tít bên phải). LayoutLMv3 giải quyết bằng cách kết hợp 3 luồng dữ liệu cùng lúc:
  1. **Text:** Nội dung chữ (từ VietOCR).
  2. **Layout (2D Position):** Tọa độ không gian (từ DBNet).
  3. **Vision:** Đặc trưng hình ảnh từ chính tờ hóa đơn (ResNet).

## 🚀 Quá trình Fine-tune (Tinh chỉnh sâu)
LayoutLMv3 gốc chỉ hiểu ngôn ngữ chung chung. Để nó hiểu được hóa đơn Việt Nam, chúng tôi đã tiến hành Fine-tune bài toán **Token Classification (NER)**:
- **Tập dữ liệu:** Hàng trăm hóa đơn thực tế được gán nhãn tọa độ tỉ mỉ.
- **Nhãn (Labels):** Hệ thống được Fine-tune để nhận diện 4 thực thể chính: SELLER (Cửa hàng), ADDRESS (Địa chỉ), TIMESTAMP (Thời gian), và TOTAL_COST (Tổng tiền).
- **Tối ưu suy luận:** Sau khi Fine-tune trên PyTorch, mô hình được ép xung và biên dịch sang chuẩn **ONNX** để chạy trên Execution Provider của GPU, mang lại tốc độ inference gần như tức thời.

## Minh họa thực tế
### Ảnh đầu vào
![Input](docs/input.jpg)

### Phân loại các vùng bằng màu sắc (Đầu ra)
![Output](docs/output.jpg)
