from fastapi import FastAPI
from qdrant_client.http import models
from src.database import init_db, get_client
from src.models import embedding_service
from src.schemas import Document, SearchQuery
from src.config import COLLECTION_NAME

app = FastAPI(title="VectorDB & RAG API")
db_client = get_client()

@app.on_event("startup")
def startup_event():
    init_db()

@app.post("/index")
async def index_document(doc: Document):
    """Implement Qdrant indexing pipeline"""
    if doc.language == "vi":
        dense_vec, sparse_indices, sparse_values = embedding_service.embed_vietnamese_hybrid(doc.text)
        
        db_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=doc.id,
                    payload={"text": doc.text, "language": doc.language},
                    vector={
                        "dense_vector": dense_vec,
                        "sparse_vector": models.SparseVector(
                            indices=sparse_indices, values=sparse_values
                        )
                    }
                )
            ]
        )
        return {"status": "success", "message": f"Đã index tài liệu ID {doc.id} (Hybrid)"}
    else:
        # Nếu là tiếng Anh, bạn có thể thiết kế một collection riêng hoặc lưu dạng vector khác. 
        # Trong ví dụ này ta tập trung hoàn thiện pipeline Tiếng Việt Hybrid.
        return {"status": "info", "message": "Pipeline tiếng Anh đang được phát triển."}

@app.post("/search")
async def search_documents(query: SearchQuery):
    """Create hybrid search (dense + sparse) using latest Qdrant API"""
    dense_vec, sparse_indices, sparse_values = embedding_service.embed_vietnamese_hybrid(query.text)
    
    # Sử dụng query_points thay cho search (API mới của Qdrant)
    search_result = db_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            # 1. Tìm kiếm theo Dense Vector (Ngữ nghĩa)
            models.Prefetch(
                query=dense_vec,
                using="dense_vector",
                limit=query.top_k,
            ),
            # 2. Tìm kiếm theo Sparse Vector (Từ khóa)
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                using="sparse_vector",
                limit=query.top_k,
            ),
        ],
        # 3. Trộn và xếp hạng lại kết quả bằng thuật toán RRF (Reciprocal Rank Fusion)
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=query.top_k,
        with_payload=True
    )
    
    return {
        "query": query.text,
        # Lưu ý: Kết quả trả về nằm trong thuộc tính .points
        "results": [{"score": res.score, "text": res.payload["text"]} for res in search_result.points]
    }