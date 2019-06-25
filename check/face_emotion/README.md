***********************************************************************
Kiểm tra nhận diện cảm xúc khuôn mặt:
- Bước 1: Download The Japanese Female Facial Expression (JAFFE) Database tại đây [http://www.kasrl.org/jaffedbase.zip] và thông tin của bộ dataset tại đây [http://www.kasrl.org/jaffe.html].
- Bước 2: Đặt dataset vào thư mục images/original (mình có để sẵn trong folder, bạn xem thử).
- Bước 3: Chạy python face_tion_acc.py, cemoode tự động kiểm tra độ chính xác cảm xúc của các khuôn mặt bao gồm 7 cảm xúc với các thông tin:
+ Số hình nhận diện đúng của mỗi cảm xúc.
+ Số hình nhận diện sai của mỗi cảm xúc.
+ Tổng số hình ảnh của mỗi cảm xúc.
+ Độ chính xác của thuật toán trên mỗi cảm xúc.
+ Các hình ảnh sau khi nhận diện cảm xúc được lưu trong đường dẫn images/recognized và images/unrecognized tương ứng với nhận diện đúng và sai.
***********************************************************************

