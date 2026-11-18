from modules.schemas import DocumentChunk

def chunk_document_pages(
    pages: list[dict],
    document_id: str,
    document_name: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[DocumentChunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap.")

    chunks = []
    for page in pages:
        text = page["content"].strip()
        page_number = page["page_number"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            content = text[start:end].strip()

            if content:
                chunks.append(DocumentChunk(
                    chunk_id=f"{document_id}-p{page_number}-c{len(chunks) + 1}",
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page_number,
                    content=content,
                ))

            if end == len(text):
                break
            start = end - overlap

    return chunks
