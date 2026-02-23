vector_db_project/
│
├── docker-compose.yml       # File để khởi chạy Qdrant Database nhanh chóng
├── requirements.txt         # Chứa danh sách các thư viện Python cần cài đặt
├── test_quality.py          # Script để test chất lượng embedding và search
│
└── src/                     # Thư mục chứa mã nguồn chính
    ├── __init__.py
    ├── config.py            # Cấu hình các tham số (tên model, kết nối DB)
    ├── models.py            # Tích hợp Embedding models (all-MiniLM & bge-m3)
    ├── database.py          # Script kết nối Qdrant và cấu hình HNSW / Indexing
    ├── schemas.py           # Định nghĩa cấu trúc dữ liệu đầu vào/ra (Pydantic)
    └── main.py              # Xây dựng API Endpoints (FastAPI)


                    <!-- Cách chạy hệ thống -->
Mở terminal tại thư mục vector_db_project

Chạy Qdrant: docker-compose up -d

Cài thư viện: pip install -r requirements.txt

Khởi chạy API (sẽ mất một chút thời gian ở lần đầu để tải các file model): uvicorn src.main:app --reload

Mở một terminal khác và chạy file test: python test_quality.py