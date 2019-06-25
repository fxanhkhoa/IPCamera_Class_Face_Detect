***********************************************************************
Kiểm tra phát hiện khuôn mặt:
- Bước 1: Bỏ các class của hình ảnh muốn kiểm tra vào thư mục images/original (Xem trong thư mục có hình ảnh ví dụ, mục đích để thuận tiện khi kiểm tra nhận diện khuôn mặt nên để theo class luôn cho tiện).
- Bước 2: Chạy python face_detection_acc.py, code tự động tính tỉ lệ phát hiện được khuôn mặt bao gồm:
+ Số hình phát hiện được: Detected
+ Số hình không phát hiện được: Undetected
+ Tổng số hình ảnh: Total = Detected+Undetected
+ Tỉ lệ số hình phát hiện được khuôn mặt: Detected/Total
+ Các hình phát hiện được khuôn mặt sẽ được chứa trong đường dẫn images/detected
+ Các hình phát hiện được khuôn mặt sẽ được chứa trong đường dẫn images/undetected
***********************************************************************
Kiểm tra nhận diện khuôn mặt:
- Các hình trong đường dẫn images/detected là những hình phát hiện được khuôn mặt khi chạy script phát hiện khuôn mặt bên trên, vì vậy sử dụng các hình ấy để nhận diện khuôn mặt.
- Chạy python face_recognition_acc.py, code tự động kiểm tra độ chính xác của các khuôn mặt phát hiện được ở bước trên bao gồm:
+ Số hình nhận diện đúng: Recognized
+ Số hình nhận diện sai: Unrecognized
+ Tổng số hình ảnh: Total = Recognized+Unrecognized
+ Độ chính xác của thuật toán: Recognized/Total
***********************************************************************

