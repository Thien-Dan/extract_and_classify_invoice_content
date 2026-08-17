# 🔍 Module Phát hiện vùng chữ (DBNet Detection)

Module này sử dụng mô hình AI chuyên dụng để tìm kiếm và xác định chính xác tọa độ các hộp giới hạn (Bounding Boxes) bao quanh các đoạn văn bản (Text Lines) trên hóa đơn.

## 🛠 Kiến trúc Model & Thiết kế
- **Model sử dụng:** **DBNet** (Differentiable Binarization Network).
- **Backbone:** ResNet18 hoặc ResNet50 kết hợp với FPN (Feature Pyramid Network).
- **Nguyên lý hoạt động:** Thay vì dùng các ngưỡng binarization cứng ngắc như các mô hình cũ (EAST, CRAFT), DBNet học cách dự đoán một "ngưỡng binarization linh hoạt" cho từng pixel thông qua một hàm có thể vi phân (Differentiable). Nhờ đó, mạng có thể huấn luyện End-to-End.
- **Ưu điểm:** Khả năng bắt các dòng chữ siêu sát nhau, cong vênh và tốc độ cực nhanh trên GPU.

*Lưu ý: Module này hiện tại sử dụng pretrained weights (đã được huấn luyện sẵn trên bộ dữ liệu đa ngôn ngữ) do hiệu năng mặc định đã rất tốt, không yêu cầu Fine-tune thêm.*

## Minh họa thực tế
### Hình ảnh đầu vào (Từ bước Preprocessing)
![Input](docs/input.jpg)

### Hộp xanh lá bao quanh vùng chữ (Đầu ra)
![Output](docs/output.jpg)
