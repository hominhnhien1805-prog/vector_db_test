from pydantic import BaseModel

class Document(BaseModel):
    id: int
    text: str
    language: str = "vi" 

class SearchQuery(BaseModel):
    text: str
    top_k: int = 3