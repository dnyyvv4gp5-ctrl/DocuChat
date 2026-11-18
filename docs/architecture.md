# DocuChat Architecture

## Overview

DocuChat uses a Retrieval-Augmented Generation (RAG) pipeline. Uploaded PDF content is extracted, split into chunks, embedded, and indexed for semantic retrieval. Retrieved passages are then provided to an AI model to generate an answer grounded in the uploaded documents.

## Processing Pipeline

1. **PDF ingestion:** The Streamlit interface accepts PDF uploads.
2. **Text extraction:** PyMuPDF extracts text page by page and preserves page numbers.
3. **Chunking:** The chunker divides page text into overlapping segments and records document and page metadata.
4. **Embedding:** Sentence Transformers (`all-MiniLM-L6-v2`) converts chunks and user questions into vector representations.
5. **Indexing and retrieval:** FAISS uses normalized vectors and inner-product similarity to retrieve the most relevant chunks.
6. **Answer generation:** Retrieved evidence is sent to Groq. If Groq fails, Gemini is attempted as a fallback.
7. **Source display:** The interface displays the answer and the source document names and page numbers.

## Main Components

| Component | Responsibility |
|---|---|
| `app/main.py` | Streamlit interface, uploads, workspace and chat |
| `modules/pdf_processor.py` | PDF text extraction |
| `modules/chunker.py` | Overlapping text chunks and metadata |
| `modules/retriever.py` | Embeddings, FAISS index and semantic search |
| `modules/answer_generator.py` | Evidence prompt, Groq/Gemini generation and source metadata |
| `modules/schemas.py` | Pydantic data models |
| `modules/evaluation.py` | Hit Rate@K and MRR calculations |
| `modules/evaluation_ui.py` | Evaluation workflow and CSV export |

## Grounding Approach

The answer-generation prompt instructs the model to use only the retrieved document evidence, avoid inventing facts, state when evidence is insufficient, and cite document names and page numbers. Retrieved evidence and citations support verification but do not guarantee that every generated statement is correct.

## Scope

The current implementation processes text-based PDFs. OCR for scanned documents and persistent vector storage across application sessions are not included in the current MVP.