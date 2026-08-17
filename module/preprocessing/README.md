# 📐 Module Tiền xử lý (Preprocessing)

Module này chịu trách nhiệm chuẩn hóa hình ảnh đầu vào trước khi đưa vào các mô hình AI. Hình ảnh hóa đơn trong thực tế thường bị cong vênh, bóng mờ, chụp ở các góc độ nghiêng và có lẫn nhiều hậu cảnh thừa.

## 🛠 Thiết kế Kỹ thuật & Thuật toán
Module này **không sử dụng Deep Learning** mà dựa hoàn toàn vào các thuật toán Xử lý Ảnh truyền thống (Computer Vision) với thư viện **OpenCV**, nhằm đảm bảo tốc độ siêu nhanh và ổn định:
1. **Document Cropping (Cắt viền):** 
   - Sử dụng **Canny Edge Detection** để tìm các cạnh.
   - Dùng **Morphological Operations** (Dilate/Erode) để nối liền các đường viền đứt nét.
   - Tìm **Contours** lớn nhất có 4 góc (tương đương với tờ giấy) và dùng phép biến đổi phối cảnh (**Perspective Transform**) để kéo phẳng hình ảnh.
2. **Angle Deskew (Chống xoay):** 
   - Ứng dụng **Hough Line Transform** để tính toán góc nghiêng trung bình của các dòng chữ.
   - Tự động xoay ảnh ngược lại (Rotate) với góc tương ứng để văn bản nằm ngang hoàn toàn.

## Minh họa thực tế
### Hình ảnh đầu vào (Gốc)
![Input](docs/input.jpg)

### Hình ảnh sau khi xoay & cắt phẳng (Đầu ra)
![Output](docs/output_crop.jpg)
