def evaluate_retrieval(retriever, test_cases: list[dict], top_k: int = 5) -> dict:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not test_cases:
        raise ValueError("At least one evaluation case is required.")

    details = []
    total_reciprocal_rank = 0.0
    total_hits = 0

    for case in test_cases:
        question = case["question"]
        expected_sources = case["expected_sources"]

        results = retriever.search(question, top_k=top_k)
        reciprocal_rank = 0.0
        retrieved_sources = []

        for rank, result in enumerate(results, start=1):
            chunk = result["chunk"]

            retrieved_sources.append({
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
            })

            is_relevant = any(
                chunk.document_name == expected["document_name"]
                and chunk.page_number == expected["page_number"]
                for expected in expected_sources
            )

            if is_relevant and reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank

        hit = reciprocal_rank > 0.0
        total_hits += int(hit)
        total_reciprocal_rank += reciprocal_rank

        details.append({
            "question": question,
            "hit": hit,
            "reciprocal_rank": reciprocal_rank,
            "retrieved_sources": retrieved_sources,
        })

    total = len(test_cases)

    return {
        "total_questions": total,
        "hits": total_hits,
        "hit_rate": total_hits / total,
        "mean_reciprocal_rank": total_reciprocal_rank / total,
        "top_k": top_k,
        "details": details,
    }
