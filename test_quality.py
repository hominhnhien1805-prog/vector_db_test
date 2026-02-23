import requests

BASE_URL = "http://localhost:8000"

print("--- 1. Tiến hành Indexing Dữ liệu ---")
docs = [
    {"id": 1, "text": "Hệ thống cơ sở dữ liệu phân tán giúp mở rộng quy mô dễ dàng.", "language": "vi"},
    {"id": 2, "text": "Apache Airflow là một công cụ lập lịch luồng công việc tuyệt vời.", "language": "vi"},
    {"id": 3, "text": "Mô hình ngôn ngữ lớn (LLM) tốn rất nhiều tài nguyên để huấn luyện.", "language": "vi"},
]

for doc in docs:
    res = requests.post(f"{BASE_URL}/index", json=doc)
    print(res.json())

print("\n--- 2. Tiến hành Hybrid Search ---")
# Cố tình dùng từ khóa không giống 100% để test Semantic Search
queries = [
    "Công cụ nào dùng để orchestrate data pipeline?", 
    "Database nào có thể scale tốt?"
]

for q in queries:
    print(f"\nCâu hỏi: '{q}'")
    res = requests.post(f"{BASE_URL}/search", json={"text": q, "top_k": 1})
    for item in res.json().get("results", []):
        print(f" -> Kết quả: {item['text']} (Độ tự tin: {item['score']:.4f})")