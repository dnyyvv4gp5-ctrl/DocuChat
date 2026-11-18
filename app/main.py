import hashlib
import tempfile
from pathlib import Path
from uuid import uuid4

import streamlit as st

from modules.answer_generator import AnswerGenerationError, generate_answer
from modules.chunker import chunk_document_pages
from modules.pdf_processor import extract_pdf_pages
from modules.retriever import DocumentRetriever
from modules.evaluation_ui import render_evaluation

st.set_page_config(page_title="DocuChat", page_icon="✦", layout="wide")

st.markdown("""
<style>
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--background-color) !important;
    color: var(--text-color) !important;
}

[data-testid="stSidebar"] {
    background: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    border-right: 1px solid var(--border-color);
}

[data-testid="stSidebar"] * {
    color: var(--text-color);
}

.block-container {
    max-width: 1160px;
    padding-top: 6rem !important;
    padding-bottom: 5rem;
}

.kicker {
    color: var(--text-color) !important;
    opacity: .8;
    font-size: .72rem;
    font-weight: 750;
    letter-spacing: .16em;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.hero {
    color: var(--text-color) !important;
    font-size: clamp(2.2rem, 4vw, 3.3rem);
    font-weight: 780;
    letter-spacing: -.05em;
    line-height: 1.15;
    margin-top: .5rem;
    margin-bottom: 1.2rem;
}

.copy {
    color: var(--text-color) !important;
    opacity: .85;
    font-size: 1rem;
    line-height: 1.7;
    max-width: 700px;
    margin-bottom: 1.5rem;
}

[data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"],
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: var(--text-color);
}

[data-testid="stChatMessage"],
[data-testid="stMetric"],
[data-testid="stAlert"] {
    background: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color);
    border-radius: 12px;
}

[data-testid="stChatMessage"] {
    padding: .8rem 1rem;
}

[data-testid="stMetric"] {
    padding: .7rem 1rem;
}

div.stButton>button {
    background: #0B1325;
    color: #FFFFFF;
    border: 1px solid #A9CDBE;
    border-radius: 9px;
    font-weight: 650;
}

div.stButton>button:hover {
    background: #263850;
    color: #FFFFFF;
    border-color: #F2B53A;
}
</style>
""", unsafe_allow_html=True)

for key, default in {
    "chunks": [], "documents": [], "hashes": set(),
    "retriever": None, "messages": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.markdown("## ✦ DocuChat")
    st.caption("Your documents. One grounded conversation.")
    st.divider()
    st.markdown("**DOCUMENT WORKSPACE**")
    uploads = st.file_uploader(
        "Add PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process documents", use_container_width=True, disabled=not uploads):
        added_chunks, added_docs = [], []
        progress = st.progress(0, text="Preparing PDFs...")

        for i, uploaded in enumerate(uploads):
            data = uploaded.getvalue()
            digest = hashlib.sha256(data).hexdigest()

            if digest in st.session_state.hashes:
                progress.progress((i + 1) / len(uploads))
                continue

            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(data)
                    temp_path = f.name

                pages = extract_pdf_pages(temp_path)
                doc_id = uuid4().hex[:12]
                chunks = chunk_document_pages(pages, doc_id, uploaded.name)

                if not chunks:
                    raise ValueError("No searchable text was extracted.")

                added_chunks.extend(chunks)
                added_docs.append({
                    "name": uploaded.name,
                    "pages": len(pages),
                    "chunks": len(chunks),
                    "hash": digest
                })

            except Exception as exc:
                st.error(f"{uploaded.name}: {exc}")

            finally:
                if temp_path and Path(temp_path).exists():
                    Path(temp_path).unlink()

            progress.progress((i + 1) / len(uploads), text="Processing PDFs...")

        if added_chunks:
            combined = st.session_state.chunks + added_chunks

            try:
                with st.spinner("Building semantic search index..."):
                    retriever = DocumentRetriever()
                    retriever.build_index(combined)

                st.session_state.chunks = combined
                st.session_state.documents.extend(added_docs)
                st.session_state.hashes.update(d["hash"] for d in added_docs)
                st.session_state.retriever = retriever
                st.session_state.messages = []
                st.success(f"Added {len(added_docs)} document(s).")

            except Exception as exc:
                st.error(f"Index creation failed: {exc}")

        elif uploads:
            st.info("No new documents to add. Already processed files were skipped.")

        progress.empty()

    st.divider()
    st.markdown("**YOUR LIBRARY**")

    if st.session_state.documents:
        for doc in st.session_state.documents:
            st.markdown(f"**{doc['name']}**")
            st.caption(f"{doc['pages']} pages · {doc['chunks']} chunks")

        if st.button("Clear workspace", use_container_width=True):
            for key in ("chunks", "documents", "hashes", "messages"):
                st.session_state[key] = [] if key != "hashes" else set()

            st.session_state.retriever = None
            st.rerun()
    else:
        st.caption("Uploaded PDFs will appear here.")

    st.divider()
    st.caption("AI answers may be incomplete. Verify important details in the original documents.")

st.markdown(
    '<p class="kicker">DOCUMENT INTELLIGENCE WORKSPACE</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<h1 class="hero">Ask your documents<br>better questions.</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="copy">Search across your PDFs and receive answers grounded in retrieved '
    'passages, with document names and page references to help you verify the source.</p>',
    unsafe_allow_html=True
)

a, b, c = st.columns(3)
a.metric("Documents", len(st.session_state.documents))
b.metric("Evidence chunks", len(st.session_state.chunks))
c.metric("Workspace", "Ready" if st.session_state.retriever else "Awaiting PDFs")

st.divider()

if not st.session_state.retriever:
    st.subheader("Start with your first document")
    st.info(
        "Upload one or more text-based PDFs in the sidebar and select **Process documents**."
    )
    st.markdown("**Try asking:**")
    st.write("- What are the main recommendations?")
    st.write("- Which page explains the key deadlines?")
    st.write("- Compare the policies described in these documents.")

else:
    conversation_tab, evaluation_tab = st.tabs(["Conversation", "Evaluation"])
    with evaluation_tab:
        render_evaluation(st.session_state.retriever, st.session_state.documents)
    with conversation_tab:
        st.markdown("**CONVERSATION**")
        st.subheader("Ask a question")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg["role"] == "assistant":
                    st.caption(
                        "Evidence retrieved"
                        if msg["grounded"]
                        else "No relevant evidence retrieved"
                    )

                    for n, source in enumerate(msg["sources"], 1):
                        with st.expander(
                            f"Source {n} · {source.document_name} · Page {source.page_number}"
                        ):
                            st.write(source.content)

        question = st.chat_input("Ask something about your uploaded documents...")

        if question:
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching documents and preparing an answer..."):
                        results = st.session_state.retriever.search(question, top_k=5)
                        response = generate_answer(question, results)

                    st.markdown(response.answer)
                    st.caption(
                        "Evidence retrieved"
                        if response.grounded
                        else "No relevant evidence retrieved"
                    )

                    for n, source in enumerate(response.sources, 1):
                        with st.expander(
                            f"Source {n} · {source.document_name} · Page {source.page_number}"
                        ):
                            st.write(source.content)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "grounded": response.grounded,
                        "sources": response.sources
                    })

                except AnswerGenerationError as exc:
                    message = (
                        "The AI providers could not generate an answer. "
                        "Please try again shortly."
                    )
                    st.error(message)
                    st.caption(str(exc))
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": message,
                        "grounded": False,
                        "sources": []
                    })

                except Exception as exc:
                    message = (
                        "Something went wrong while searching or answering. "
                        "Please try again."
                    )
                    st.error(message)
                    st.caption(str(exc))
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": message,
                        "grounded": False,
                        "sources": []
                    })
