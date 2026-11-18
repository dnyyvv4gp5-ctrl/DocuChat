# DocuChat – RAG-Based AI Chatbot

DocuChat is a Streamlit application that lets users upload PDF documents, ask questions in natural language, and receive AI-generated answers supported by document citations.

## Features

- Upload and process multiple PDFs.
- Extract text and split it into overlapping chunks.
- Generate semantic embeddings with Sentence Transformers.
- Retrieve relevant passages using FAISS.
- Generate evidence-based answers using Grox, with Gemini fallback.
- Display source document names and page numbers.
- Handle questions when documents lack sufficient evidence.
- Evaluate retrieval using Hit Rate@K and Mean Reciprocal Rank (MRR).
- Export evaluation results as CSV.
- Light and dark interface themes.

## Technology Stack

Python, Streamlit, PyMuPDF, Sentence Transformers, FAISS, Groq, Google Gemini, Pydantic and Pytest.

## How It Works

1. Users upload PDF documents.
2. PyMuPDF extracts text and page numbers.
3. Text is divided into overlapping chunks.
4. Sentence Transformers generates embeddings.
5. FAISS retrieves passages relevant to the question.
6. Groq or Gemini generates an answer using the retrieved evidence.
7. The application displays the answer with document and page citations.

## Project Structure

- `app/` – Streamlit application
- `modules/` – PDF processing, chunking, retrieval, AI generation and evaluation
- `tests/` – automated tests
- `docs/` – architecture and evaluation documentation

## Setup

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd DocuChat
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Never commit API keys or the `.env` file.

Run the application:

```bash
streamlit run app/main.py
```

## Testing

Run `python -m pytest -q`.

Latest local result: **15 tests passed**.

## Retrieval Evaluation

The evaluation interface reports:

- **Hit Rate@K:** Whether an expected source appears within the top K retrieved results.
- **MRR:** How highly the first relevant result is ranked.

Evaluation questions and expected document pages are manually defined. Results can be exported as CSV. These metrics assess retrieval, not the factual correctness of generated answers.

## Limitations

- Scanned PDFs requiring OCR are not supported.
- Retrieval evaluation uses manually defined expected sources.
- Citations help users verify answers but do not guarantee factual correctness.
- AI generation depends on external provider availability and usage limits.

## Project Goal

DocuChat demonstrates a practical Retrieval-Augmented Generation workflow by combining document processing, semantic search, vector retrieval and large language models.
