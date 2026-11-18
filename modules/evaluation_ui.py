import csv
import io

import streamlit as st

from modules.evaluation import evaluate_retrieval


def render_evaluation(retriever, documents):
    st.subheader("Retrieval Evaluation")
    st.caption(
        "Measure whether DocuChat retrieves the document pages "
        "expected for a set of test questions."
    )

    document_names = [doc["name"] for doc in documents]

    if not document_names:
        st.info("Upload and process PDFs before running an evaluation.")
        return

    st.markdown("**Create evaluation questions**")

    if "evaluation_cases" not in st.session_state:
        st.session_state.evaluation_cases = []

    with st.form("evaluation_case_form", clear_on_submit=True):
        question = st.text_input("Test question")
        expected_document = st.selectbox(
            "Expected document",
            document_names,
        )
        expected_page = st.number_input(
            "Expected page number",
            min_value=1,
            step=1,
        )
        submitted = st.form_submit_button("Add test case")

        if submitted:
            if not question.strip():
                st.error("Enter a test question.")
            else:
                st.session_state.evaluation_cases.append({
                    "question": question.strip(),
                    "expected_sources": [{
                        "document_name": expected_document,
                        "page_number": int(expected_page),
                    }],
                })
                st.success("Evaluation question added.")

    cases = st.session_state.evaluation_cases

    if cases:
        st.markdown("**Evaluation dataset**")
        st.dataframe(
            [
                {
                    "Question": case["question"],
                    "Expected document": case["expected_sources"][0]["document_name"],
                    "Expected page": case["expected_sources"][0]["page_number"],
                }
                for case in cases
            ],
            use_container_width=True,
        )

        if st.button("Clear evaluation dataset"):
            st.session_state.evaluation_cases = []
            st.session_state.pop("evaluation_report", None)
            st.rerun()

        top_k = st.slider("Number of retrieved results (K)", 1, 10, 5)

        if st.button("Run retrieval evaluation", type="primary"):
            with st.spinner("Evaluating retrieval..."):
                report = evaluate_retrieval(retriever, cases, top_k)
                st.session_state.evaluation_report = report

    report = st.session_state.get("evaluation_report")

    if not report:
        return

    st.divider()
    st.markdown("**Evaluation results**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Questions", report["total_questions"])
    col2.metric("Hit Rate", f'{report["hit_rate"]:.1%}')
    col3.metric("MRR", f'{report["mean_reciprocal_rank"]:.3f}')

    st.caption(
        f'Hit Rate@{report["top_k"]}: {report["hits"]} of '
        f'{report["total_questions"]} questions retrieved at least one expected source.'
    )

    st.markdown("**Question-level results**")

    rows = []
    for detail in report["details"]:
        rows.append({
            "Question": detail["question"],
            "Expected source found": "Yes" if detail["hit"] else "No",
            "Reciprocal rank": round(detail["reciprocal_rank"], 3),
            "Retrieved sources": "; ".join(
                f'{source["document_name"]}, page {source["page_number"]}'
                for source in detail["retrieved_sources"]
            ),
        })

    st.dataframe(rows, use_container_width=True)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

    st.download_button(
        "Download evaluation report (CSV)",
        data=output.getvalue(),
        file_name="docuchat_evaluation.csv",
        mime="text/csv",
    )
