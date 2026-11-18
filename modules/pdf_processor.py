from pathlib import Path
import pymupdf

def extract_pdf_pages(file_path: str) -> list[dict]:
    path = Path(file_path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise ValueError("A valid PDF file is required.")

    pages = []
    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "page_number": page_number,
                    "content": text
                })

    if not pages:
        raise ValueError("No extractable text found in the PDF.")

    return pages
