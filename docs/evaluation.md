# Retrieval Evaluation

## Purpose

DocuChat includes a basic evaluation workflow to check whether semantic retrieval returns the expected document pages for manually written questions.

## Metrics

- **Hit Rate@K:** Number of test questions with at least one expected source in the top K results, divided by the total number of questions.
- **Mean Reciprocal Rank (MRR):** The average reciprocal rank of the first expected source for each question. A higher value means relevant sources tend to appear nearer the top. A question with no expected source in the retrieved results contributes zero.

## Evaluation Workflow

1. Upload and process the PDF documents.
2. Open the Evaluation tab.
3. Add a test question and select its expected document and page.
4. Choose K and run the evaluation.
5. Review aggregate and question-level results.
6. Download the report as a CSV file.

Expected document pages are manually specified. The evaluation measures retrieval against those labels; it does not measure answer factuality, citation entailment, or completeness.

## Initial Manual Checks

The portfolio PDF was used for initial manual checks. Some test questions retrieved their expected pages, while one evaluation row showed an inconsistent result even though the expected page appeared among the listed top-five results. A near-identical question without a question mark was marked as a hit, with the same retrieved page list.

This discrepancy should be treated as a limitation of the initial manual evaluation run, not as a reliable benchmark score. The evaluation dataset is small and should not be presented as a statistically meaningful performance result.

## Test Suite

The automated unit tests cover evaluation calculations, including successful retrieval, misses, invalid K values, and empty test datasets. The latest recorded local test run completed with **15 passed**.

## Limitations

- Small, manually created test dataset.
- Expected sources are manually selected document pages.
- Results depend on the uploaded documents, question wording, embedding model and K.
- Metrics assess retrieval only, not the quality or factual correctness of generated answers.
