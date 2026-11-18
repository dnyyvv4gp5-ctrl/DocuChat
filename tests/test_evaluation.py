import pytest

from modules.evaluation import evaluate_retrieval
from modules.schemas import DocumentChunk


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=5):
        return self.results.get(query, [])[:top_k]


def make_result(document, page, score=0.9):
    chunk = DocumentChunk(
        chunk_id=f"{document}-p{page}",
        document_id=document,
        document_name=document,
        page_number=page,
        content="Test evidence",
    )
    return {"chunk": chunk, "score": score}


def test_retrieval_metrics_for_correct_sources():
    retriever = FakeRetriever({
        "refund period": [
            make_result("other.pdf", 1),
            make_result("policy.pdf", 2),
        ],
        "shipping time": [
            make_result("shipping.pdf", 3),
        ],
    })

    cases = [
        {
            "question": "refund period",
            "expected_sources": [
                {"document_name": "policy.pdf", "page_number": 2}
            ],
        },
        {
            "question": "shipping time",
            "expected_sources": [
                {"document_name": "shipping.pdf", "page_number": 3}
            ],
        },
    ]

    report = evaluate_retrieval(retriever, cases, top_k=3)

    assert report["total_questions"] == 2
    assert report["hits"] == 2
    assert report["hit_rate"] == 1.0
    assert report["mean_reciprocal_rank"] == 0.75


def test_retrieval_metrics_include_misses():
    retriever = FakeRetriever({
        "refund period": [make_result("other.pdf", 1)]
    })

    cases = [{
        "question": "refund period",
        "expected_sources": [
            {"document_name": "policy.pdf", "page_number": 2}
        ],
    }]

    report = evaluate_retrieval(retriever, cases)

    assert report["hits"] == 0
    assert report["hit_rate"] == 0.0
    assert report["mean_reciprocal_rank"] == 0.0


def test_evaluation_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        evaluate_retrieval(FakeRetriever({}), [], top_k=0)


def test_evaluation_rejects_empty_test_cases():
    with pytest.raises(ValueError, match="At least one"):
        evaluate_retrieval(FakeRetriever({}), [])
