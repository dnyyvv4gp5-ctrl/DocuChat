from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int = Field(ge=1)
    content: str = Field(min_length=1)

class RetrievedSource(BaseModel):
    chunk_id: str
    document_name: str
    page_number: int = Field(ge=1)
    content: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[RetrievedSource] = Field(default_factory=list)
    grounded: bool = False
