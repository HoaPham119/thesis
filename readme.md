# Hướng dẫn cấu trúc thư mục

## Cấu trúc thư mục

Để đảm bảo chương trình hoạt động đúng, hãy lưu thư mục `data` ngang hàng với thư mục `code`. Cấu trúc thư mục nên như sau:

```
project_root/
│-- data/       # Thư mục chứa dữ liệu
│-- code/       # Thư mục chứa mã nguồn
│   │-- main.py
│   │-- module.py
│-- README.md   # Hướng dẫn sử dụng
```
## Bắt đầu từ file `code/data_analysis.ipynb`

Bạn có thể bắt đầu quá trình phân tích dữ liệu bằng cách mở và chạy file `code/data_analysis.ipynb`. File này sẽ sử dụng các hàm cần thiết từ các file `.py` để xử lý và phân tích dữ liệu từ thư mục `data`.


## Lưu ý

- Thư mục `data` chứa các tệp dữ liệu cần thiết để chương trình chạy.
- Thư mục `code` chứa mã nguồn của dự án.
- Khi tham chiếu đến tệp trong thư mục `data`, sử dụng đường dẫn tương đối, ví dụ:
  ```python
  import os
  data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/my_file.txt")
  ```


.\env\Scripts\Activate
Set-ExecutionPolicy Unrestricted -Scope Process
.\env\Scripts\Activate