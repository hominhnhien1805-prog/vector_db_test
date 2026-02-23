from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def init_db():
    """Khởi tạo Collection với cấu hình HNSW và Hybrid Search"""
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            # Cấu hình Dense Vector (dùng bge-m3 là 1024 chiều)
            vectors_config={
                "dense_vector": models.VectorParams(
                    size=1024, 
                    distance=models.Distance.COSINE
                )
            },
            # Cấu hình Sparse Vector cho từ khóa
            sparse_vectors_config={
                "sparse_vector": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
            # Tối ưu hóa thuật toán HNSW (Hierarchical Navigable Small World)
            hnsw_config=models.HnswConfigDiff(
                m=16, # Số lượng liên kết tối đa cho mỗi node
                ef_construct=100 # Càng lớn index càng lâu nhưng tìm càng chính xác
            )
        )
        print(f"Đã tạo collection: {COLLECTION_NAME}")
    else:
        print(f"Collection {COLLECTION_NAME} đã tồn tại.")

def get_client():
    return client