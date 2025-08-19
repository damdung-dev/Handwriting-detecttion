# Handwritten Digit Recognition (Django + PyTorch)

Dự án web nhận diện chữ số viết tay sử dụng **Django** và **PyTorch**.  
Người dùng có thể upload ảnh chữ số viết tay và hệ thống sẽ dự đoán kết quả bằng mô hình deep learning (MNIST).

---

## 🚀 Yêu cầu hệ thống
- Python >= 3.8
- pip (Python package manager)
- virtualenv (khuyến nghị)

---

## ⚙️ Cài đặt môi trường

1. Clone repository:
gh repo clone damdung-dev/Handwriting-detecttion
cd digit-recognition-django

2. Cài đặt thư viện:

pip install -r requirements.txt

3. Sau khi cài xong, kiểm tra phiên bản thư viện:
Đảm bảo các phiên bản chính xác (cần cho project chạy ổn định):
python -m pip show django
python -m pip show torch
python -m pip show pillow
Ví dụ output mong đợi:
Name: Django
Version: 5.0.x

Name: torch
Version: 2.2.x

Name: Pillow
Version: 10.2.x
## ▶️ Chạy project
1. Di chuyển tới thư mục project (chứa manage.py) và migrate database:
python manage.py migrate
2. Chạy server:
python manage.py runserver
Sau khi chạy xong, sẽ hiện đường link đến với trang web, Ctrl + Click vào đường link đó là bạn đã mở được trang web rồi.

## 📝 Demo sử dụng
1. Vào giao diện web.
2. Nhấn nút Chọn ảnh và upload một chữ số viết tay (ví dụ 7.png).
3. Nhấn Dự đoán.
Hệ thống sẽ trả về:
Số dự đoán: Ví dụ: 7
Độ tin cậy: Ví dụ: 95.3%

## 🎥 Demo Video
[![Demo][(https://drive.google.com/file/d/1RwvZA4xBxMpc9j59yhVHGxpNKsCJtmk2/view?usp=sharing)


📚 Thư viện chính sử dụng
Django – Web framework
PyTorch – Deep Learning framework
Pillow – Xử lý ảnh

💡 Ghi chú
Mô hình sử dụng MNIST (28x28 grayscale).
Người dùng có thể train lại model hoặc sử dụng pretrained model kèm theo.

