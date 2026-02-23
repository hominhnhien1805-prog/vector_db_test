from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
from src.config import ENG_MODEL_NAME, VIE_MODEL_NAME

class EmbeddingService:
    def __init__(self):
        print("Đang tải model tiếng Anh...")
        self.eng_model = SentenceTransformer(ENG_MODEL_NAME)
        
        print("Đang tải model tiếng Việt (BGE-M3)...")
        # BGE-M3 hỗ trợ tạo cả dense và sparse vector
        self.vie_model = BGEM3FlagModel(VIE_MODEL_NAME, use_fp16=True)

    def embed_english(self, text: str):
        # Chỉ trả về dense vector (384 chiều)
        return self.eng_model.encode(text).tolist()

    def embed_vietnamese_hybrid(self, text: str):
        # Trả về cả dense (1024 chiều) và sparse (cho hybrid search)
        embeddings = self.vie_model.encode([text], return_dense=True, return_sparse=True)
        
        dense_vec = embeddings['dense_vecs'][0].tolist()
        sparse_dict = embeddings['lexical_weights'][0]
        
        # Xử lý format sparse cho Qdrant
        sparse_indices = [int(k) for k in sparse_dict.keys()]
        sparse_values = list(sparse_dict.values())
        
        return dense_vec, sparse_indices, sparse_values

# Khởi tạo instance để dùng chung
embedding_service = EmbeddingService()