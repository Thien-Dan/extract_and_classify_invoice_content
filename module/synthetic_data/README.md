# 🧪 Module Sinh dữ liệu giả (Synthetic Data Generation)

Các mô hình Deep Learning (như VietOCR và LayoutLMv3) nổi tiếng là cực kỳ "đói" dữ liệu (Data Hungry). Thu thập hóa đơn thực tế vừa tốn kém, vừa vi phạm quyền riêng tư của khách hàng. Module này sinh ra để giải quyết bài toán đó.

## 🛠 Kỹ thuật & Thiết kế
- **Engine sinh ảnh:** Sử dụng thư viện Pillow (PIL) kết hợp OpenCV để render text lên các nền giấy trắng/hóa đơn trống.
- **Quy trình sinh dữ liệu:**
  1. **Tạo nội dung ngẫu nhiên (Faker):** Lấy dữ liệu tên siêu thị, địa chỉ, số tiền và thời gian ngẫu nhiên từ thư viện Faker.
  2. **Bố cục động:** Sắp xếp text theo hàng cột hoặc lệch dòng ngẫu nhiên để LayoutLMv3 học cách tổng quát hóa (generalize).
  3. **Nhiễu loạn (Data Augmentation):** Áp dụng hàng loạt phép biến đổi thị giác:
     - Làm mờ (Gaussian Blur) mô phỏng camera điện thoại rung.
     - Giảm chất lượng (JPEG Artifacts).
     - Thêm nhiễu hột (Salt & Pepper Noise).
     - Ám màu vàng ố để mô phỏng giấy hóa đơn để lâu ngày.

Việc sinh ra 10,000 hóa đơn nhân tạo chỉ mất vài phút, cung cấp nguồn dữ liệu vô tận để Fine-tune các mô hình AI phía sau!
