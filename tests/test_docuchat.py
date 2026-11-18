import pytest
import pymupdf

from modules.schemas import DocumentChunk
from modules.pdf_processor import extract_pdf_pages
from modules.chunker import chunk_document_pages
from modules import answer_generator


def make_pdf(path, text):
    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_pdf_extracts_text_and_page_number(tmp_path):
    path = tmp_path / "sample.pdf"
    make_pdf(path, "Refunds are available within 30 days.")
    pages = extract_pdf_pages(str(path))
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "30 days" in pages[0]["content"]


def test_pdf_rejects_non_pdf(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("not a PDF")
    with pytest.raises(ValueError, match="valid PDF"):
        extract_pdf_pages(str(path))


def test_pdf_rejects_empty_pdf(tmp_path):
    path = tmp_path / "empty.pdf"
    make_pdf(path, "")
    with pytest.raises(ValueError, match="No extractable text"):
        extract_pdf_pages(str(path))


def test_chunking_preserves_page_and_overlap():
    chunks = chunk_document_pages(
        [{"page_number": 3, "content": "A" * 1000}],
        "doc-1", "policy.pdf", chunk_size=400, overlap=100
    )
    assert len(chunks) == 3
    assert all(chunk.page_number == 3 for chunk in chunks)
    assert chunks[0].content[-100:] == chunks[1].content[:100]


def test_chunking_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="Invalid chunk size"):
        chunk_document_pages(
            [{"page_number": 1, "content": "Some text"}],
            "doc-1", "policy.pdf", chunk_size=100, overlap=100
        )


def test_schema_rejects_page_zero():
    with pytest.raises(ValueError):
        DocumentChunk(
            chunk_id="c1", document_id="d1", document_name="policy.pdf",
            page_number=0, content="Evidence"
        )


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="Please enter a question"):
        answer_generator.generate_answer("  ", [])


def test_no_evidence_returns_ungrounded_answer():
    result = answer_generator.generate_answer("What is the refund period?", [])
    assert result.grounded is False
    assert result.sources == []
    assert "could not find relevant information" in result.answer


def sample_result():
    chunk = DocumentChunk(
        chunk_id="doc-1-p1-c1", document_id="doc-1",
        document_name="policy.pdf", page_number=1,
        content="Customers can request a refund within 30 days."
    )
    return [{"chunk": chunk, "score": 0.9}]


def test_groq_success_skips_gemini(monkeypatch):
    monkeypatch.setattr(
        answer_generator, "_generate_with_groq",
        lambda prompt: "Customers have 30 days. [policy.pdf, page 1]"
    )
    monkeypatch.setattr(
        answer_generator, "_generate_with_gemini",
        lambda prompt: pytest.fail("Gemini should not run")
    )
    result = answer_generator.generate_answer("Refund period?", sample_result())
    assert result.grounded is True
    assert "30 days" in result.answer
    assert result.sources[0].page_number == 1


def test_gemini_fallback_runs_when_groq_fails(monkeypatch):
    def fail_groq(prompt):
        raise RuntimeError("Groq unavailable")

    monkeypatch.setattr(answer_generator, "_generate_with_groq", fail_groq)
    monkeypatch.setattr(
        answer_generator, "_generate_with_gemini",
        lambda prompt: "Customers have 30 days. [policy.pdf, page 1]"
    )
    result = answer_generator.generate_answer("Refund period?", sample_result())
    assert result.grounded is True
    assert "30 days" in result.answer
    assert result.sources[0].document_name == "policy.pdf"


def test_both_provider_failures_raise_error(monkeypatch):
    def fail(prompt):
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr(answer_generator, "_generate_with_groq", fail)
    monkeypatch.setattr(answer_generator, "_generate_with_gemini", fail)
    with pytest.raises(answer_generator.AnswerGenerationError, match="Both AI providers failed"):
        answer_generator.generate_answer("Refund period?", sample_result())
