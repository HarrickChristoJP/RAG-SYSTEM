"""
app.py
------
Streamlit user interface for the Document RAG Assistant.

Sections:
A. Document Management — upload and ingest documents
B. Ask Questions — RAG-powered question answering
C. RAG vs Without RAG — side-by-side comparison

Usage:
    streamlit run app.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import streamlit as st

# Add project root to path so src imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Imports (after page config) ───────────────────────────────────────────────
from src.vector_store import (
    initialize_vector_store,
    get_collection_info,
    delete_collection,
    COLLECTION_NAME,
    CHROMA_DB_PATH,
)
from src.retriever import retrieve_relevant_chunks, DEFAULT_TOP_K
from src.rag_pipeline import ingest_directory, ingest_single_file, ask_with_rag, ask_without_rag
from src.generator import is_gemini_available, GEMINI_MODEL
from src.text_chunker import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.embeddings import EMBEDDING_MODEL_NAME

# ─── Session state initialization ──────────────────────────────────────────────
if "collection" not in st.session_state:
    st.session_state.collection = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingestion_done" not in st.session_state:
    st.session_state.ingestion_done = False


@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def get_vector_store():
    """Cache the ChromaDB connection — only connects once per session."""
    client, collection = initialize_vector_store()
    return client, collection


def get_collection_stats() -> dict:
    """Get current stats from ChromaDB."""
    try:
        _, collection = get_vector_store()
        info = get_collection_info(collection)
        return info
    except Exception as e:
        return {"name": COLLECTION_NAME, "count": 0, "error": str(e)}


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Document RAG")
    st.markdown("---")

    # Configuration display
    st.subheader("⚙️ Configuration")
    st.code(f"""Model   : {GEMINI_MODEL}
Embedding: all-MiniLM-L6-v2
Chunk    : {DEFAULT_CHUNK_SIZE} tokens
Overlap  : {DEFAULT_CHUNK_OVERLAP} tokens
Top-K    : {DEFAULT_TOP_K} chunks
ChromaDB : {CHROMA_DB_PATH}/""", language="text")

    # API key status
    st.markdown("---")
    st.subheader("🔑 API Status")
    if is_gemini_available():
        st.success("✅ Gemini API key configured")
    else:
        st.error("❌ Gemini API key missing")
        st.caption("Add GEMINI_API_KEY to .env file to enable answer generation.")

    # ChromaDB status
    stats = get_collection_stats()
    st.markdown("---")
    st.subheader("🗄️ ChromaDB Status")
    st.metric("Chunks in DB", stats.get("count", 0))
    st.caption(f"Collection: `{stats.get('name', COLLECTION_NAME)}`")

    # Reset button
    st.markdown("---")
    if st.button("🗑️ Reset Database", type="secondary", use_container_width=True):
        with st.spinner("Resetting ChromaDB..."):
            client, _ = get_vector_store()
            delete_collection(client)
            st.cache_resource.clear()
        st.success("Database reset! Reload the page.")
        st.rerun()


# ─── Main Content ──────────────────────────────────────────────────────────────
st.title("📚 Document RAG Assistant")
st.markdown("*Ask questions about your documents using Retrieval-Augmented Generation*")

tab_a, tab_b, tab_c = st.tabs([
    "📂 A. Document Management",
    "💬 B. Ask Questions",
    "⚖️ C. RAG vs Without RAG",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB A: Document Management
# ══════════════════════════════════════════════════════════════════════════════
with tab_a:
    st.header("📂 Document Management")
    st.markdown("Upload documents or use the pre-loaded Chroma documentation dataset.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Documents")
        st.markdown("Supported formats: **PDF**, **Markdown (.md)**, **Text (.txt)**")

        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=["pdf", "md", "txt"],
            help="Files will be saved to data/source_docs/ and ingested into ChromaDB.",
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) selected")
            for f in uploaded_files:
                st.write(f"  📄 {f.name}")

            if st.button("💾 Save Uploaded Files", use_container_width=True):
                save_dir = Path("data/source_docs")
                save_dir.mkdir(parents=True, exist_ok=True)
                saved = []
                for uploaded_file in uploaded_files:
                    save_path = save_dir / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved.append(uploaded_file.name)
                st.success(f"✅ Saved {len(saved)} file(s) to data/source_docs/")

    with col2:
        st.subheader("📋 Available Documents")
        source_dir = Path("data/source_docs")
        if source_dir.exists():
            doc_files = list(source_dir.glob("*.pdf")) + \
                        list(source_dir.glob("*.md")) + \
                        list(source_dir.glob("*.txt"))
            if doc_files:
                st.metric("Documents found", len(doc_files))
                for f in sorted(doc_files):
                    size_kb = f.stat().st_size / 1024
                    icon = {"pdf": "📕", "md": "📗", "txt": "📄"}.get(f.suffix.lstrip("."), "📄")
                    st.write(f"{icon} `{f.name}` ({size_kb:.1f} KB)")
            else:
                st.info("No documents found. Upload files or run the download script.")
                st.code("python scripts/download_docs.py", language="bash")
        else:
            st.warning("data/source_docs/ directory not found.")

    st.divider()

    # Ingestion controls
    st.subheader("⚙️ Ingestion Settings")

    adv_col1, adv_col2, adv_col3 = st.columns(3)
    with adv_col1:
        chunk_size = st.slider(
            "Chunk Size (tokens)", 200, 1000, DEFAULT_CHUNK_SIZE, 50,
            help="Target number of tokens per chunk. 500 is a good balance."
        )
    with adv_col2:
        chunk_overlap = st.slider(
            "Chunk Overlap (tokens)", 20, 200, DEFAULT_CHUNK_OVERLAP, 10,
            help="Tokens of overlap between chunks to preserve context."
        )
    with adv_col3:
        st.metric("Approx words/chunk", f"~{int(chunk_size * 0.75)}")

    st.markdown("---")
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("🚀 Ingest All Documents", type="primary", use_container_width=True):
            if not source_dir.exists() or not any(source_dir.iterdir()):
                st.error("No documents to ingest! Upload files or run: python scripts/download_docs.py")
            else:
                with st.spinner("⏳ Ingesting documents (this may take a few minutes)..."):
                    progress_bar = st.progress(0, text="Loading documents...")
                    try:
                        progress_bar.progress(10, text="Loading documents...")
                        result = ingest_directory(
                            directory="data/source_docs",
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        progress_bar.progress(100, text="Done!")

                        if "error" in result:
                            st.error(f"❌ Ingestion failed: {result['error']}")
                        else:
                            st.cache_resource.clear()
                            st.session_state.ingestion_done = True

                            st.success("✅ Ingestion complete!")
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            metric_col1.metric("Documents loaded", result.get("documents_loaded", 0))
                            metric_col2.metric("Chunks created", result.get("chunks_created", 0))
                            metric_col3.metric("Total in DB", result.get("collection_count", 0))
                    except Exception as e:
                        st.error(f"❌ Error during ingestion: {e}")

    with btn_col2:
        if st.button("👁️ Preview Chunks", use_container_width=True):
            if not source_dir.exists() or not any(source_dir.iterdir()):
                st.warning("No documents found to preview.")
            else:
                from src.document_loader import load_documents_from_directory
                from src.text_chunker import chunk_documents, preview_chunks

                with st.spinner("Generating chunk preview..."):
                    docs = load_documents_from_directory("data/source_docs")
                    chunks = chunk_documents(docs, chunk_size, chunk_overlap)

                st.subheader(f"📊 Chunk Preview ({len(chunks)} total chunks)")
                for i, chunk in enumerate(chunks[:5]):
                    with st.expander(f"Chunk {i}: {chunk['title']} (~{chunk['token_estimate']} tokens)"):
                        st.text(chunk["text"][:800] + ("..." if len(chunk["text"]) > 800 else ""))
                        st.caption(f"Source: {chunk['source']} | File: {chunk['file_type']}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB B: Ask Questions (RAG-powered)
# ══════════════════════════════════════════════════════════════════════════════
with tab_b:
    st.header("💬 Ask Questions")
    st.markdown("Ask anything about your ingested documents. Answers are grounded in retrieved content.")

    # Quick stats
    stats = get_collection_stats()
    if stats.get("count", 0) == 0:
        st.warning("⚠️ No documents ingested yet. Go to **Document Management** to ingest documents first.")
    else:
        st.info(f"📊 Knowledge base has **{stats['count']}** chunks from your documents.")

    # Sample questions
    with st.expander("💡 Sample questions to try"):
        sample_questions = [
            "How do I create a Chroma collection?",
            "What is the difference between add() and upsert() in ChromaDB?",
            "How do I filter results by metadata in Chroma?",
            "How does persistent storage work in ChromaDB?",
            "What are the exact chunk size, overlap, embedding model, and ChromaDB collection name configured in this project?",
        ]
        for q in sample_questions:
            if st.button(f"→ {q}", key=f"sample_{q[:20]}"):
                st.session_state.pending_question = q

    # Question input
    question = st.text_input(
        "Your question:",
        placeholder="e.g. How do I query ChromaDB for similar documents?",
        value=st.session_state.get("pending_question", ""),
        key="question_input",
    )
    if "pending_question" in st.session_state:
        del st.session_state.pending_question

    top_k = st.slider("Number of chunks to retrieve", 1, 10, DEFAULT_TOP_K, key="topk_b")

    if st.button("🔍 Get Answer", type="primary", use_container_width=True, disabled=not question):
        if question.strip():
            with st.spinner("🔍 Retrieving relevant chunks and generating answer..."):
                try:
                    result = ask_with_rag(question.strip(), top_k=top_k)
                    st.session_state.last_rag_result = result
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.last_rag_result = None

    # Display results
    if hasattr(st.session_state, "last_rag_result") and st.session_state.last_rag_result:
        result = st.session_state.last_rag_result
        retrieved_chunks = result.get("retrieved_chunks", [])

        st.markdown("---")
        st.subheader("📝 Answer")

        if is_gemini_available():
            st.markdown(result["answer"])
        else:
            st.info("💡 Gemini API key not configured. Showing retrieved context instead.")
            if retrieved_chunks:
                for i, chunk in enumerate(retrieved_chunks[:3], 1):
                    st.markdown(f"**[Context {i}]** {chunk['text'][:400]}...")

        # Sources
        if result.get("sources"):
            st.subheader("📚 Sources Used")
            for src in result["sources"]:
                col_a, col_b = st.columns([2, 3])
                col_a.write(f"📄 **{src.get('title', 'Unknown')}**")
                if src.get("url"):
                    col_b.write(f"🔗 {src['url']}")

        # Retrieved chunks (expandable)
        if retrieved_chunks:
            with st.expander(f"🔎 View {len(retrieved_chunks)} Retrieved Chunks"):
                st.markdown(
                    "*Note: ChromaDB cosine distance — 0 = perfect match, 2 = completely different. "
                    "Relevance score = 1 - distance (higher is better).*"
                )
                for i, chunk in enumerate(retrieved_chunks, 1):
                    meta = chunk["metadata"]
                    with st.container():
                        st.markdown(f"**Chunk {i}** | 📄 {meta.get('title', '?')} | "
                                    f"Distance: `{chunk['distance']}` | Relevance: `{chunk['relevance_score']}`")
                        st.text_area(
                            f"Content (chunk {i})",
                            value=chunk["text"],
                            height=150,
                            key=f"chunk_text_{i}",
                            disabled=True,
                        )
                        if meta.get("url"):
                            st.caption(f"Source: {meta['url']}")
                        st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB C: RAG vs Without RAG
# ══════════════════════════════════════════════════════════════════════════════
with tab_c:
    st.header("⚖️ RAG vs Without RAG Comparison")
    st.markdown(
        "Compare how Gemini answers the **same question** in two ways:\n"
        "- **Without RAG**: Gemini uses only its training knowledge — no access to your documents\n"
        "- **With RAG**: Gemini uses retrieved chunks from your document collection"
    )

    # Explanation
    with st.expander("📖 Why does this comparison matter?"):
        st.markdown("""
        **The key RAG demonstration:**

        Some questions can only be answered correctly if the LLM has access to specific, private documents.
        For example:
        - What are the exact configuration values used in THIS project?
        - What does a specific internal document say?

        **Without RAG**: Gemini can only guess or say "I don't know" because this information
        was never in its training data.

        **With RAG**: We retrieve the relevant chunks from ChromaDB and include them in the prompt.
        Gemini can now answer accurately using the retrieved content.

        This is the fundamental value of RAG: grounding LLM responses in YOUR private documents.
        """)

    # Pre-loaded demo question
    st.subheader("🧪 Demonstration Question")
    st.markdown(
        "This question can **only** be answered correctly with RAG, "
        "because the answer is in `project_facts.md` — a document that was never in Gemini's training data."
    )

    demo_question = "What are the exact chunk size, overlap, embedding model, and ChromaDB collection name configured in this project?"
    st.info(f"🔎 **Demo Question:** {demo_question}")

    if st.button("▶️ Run RAG Demonstration", type="primary", use_container_width=True):
        _, collection = get_vector_store()
        if collection.count() == 0:
            st.error("⚠️ No documents ingested! Please ingest documents in Tab A first.")
        else:
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("❌ Without RAG")
                st.caption("Gemini only knows its training data — not your project config.")
                with st.spinner("Asking Gemini without any context..."):
                    if is_gemini_available():
                        direct_result = ask_without_rag(demo_question)
                        st.markdown(direct_result["answer"])
                    else:
                        st.warning("Gemini API key not configured.")
                        st.markdown(
                            "*Without RAG, Gemini would say something like:*\n\n"
                            "> I don't have access to your project's specific configuration values. "
                            "I cannot tell you the exact chunk size, overlap, embedding model name, "
                            "or ChromaDB collection name without seeing your code or configuration files."
                        )

            with col_right:
                st.subheader("✅ With RAG")
                st.caption("Gemini uses retrieved content from project_facts.md.")
                with st.spinner("Retrieving from ChromaDB and generating answer..."):
                    rag_result = ask_with_rag(demo_question)
                    retrieved_chunks = rag_result.get("retrieved_chunks", [])

                    if is_gemini_available():
                        st.markdown(rag_result["answer"])
                    else:
                        st.info("Gemini API key not configured. Here are the retrieved chunks:")
                        for chunk in retrieved_chunks[:2]:
                            st.markdown(f"**Retrieved from**: {chunk['metadata'].get('title', '?')}")
                            st.text(chunk["text"][:600])

                    if retrieved_chunks:
                        with st.expander(f"📋 Retrieved chunks ({len(retrieved_chunks)})"):
                            for i, chunk in enumerate(retrieved_chunks, 1):
                                st.write(f"**Chunk {i}**: {chunk['metadata'].get('title', '?')} "
                                         f"(relevance: {chunk['relevance_score']})")
                                st.text(chunk["text"][:300])
                                st.divider()

    st.divider()

    # Custom comparison question
    st.subheader("💬 Ask Your Own Comparison Question")
    custom_question = st.text_input(
        "Enter any question:",
        placeholder="e.g. How do I install ChromaDB?",
        key="compare_question",
    )
    custom_top_k = st.slider("Chunks to retrieve", 1, 8, DEFAULT_TOP_K, key="topk_c")

    if st.button("⚖️ Compare", type="primary", disabled=not custom_question, use_container_width=True):
        _, collection = get_vector_store()
        if collection.count() == 0:
            st.error("⚠️ No documents ingested! Go to Document Management first.")
        elif custom_question.strip():
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("❌ Without RAG (Direct LLM)")
                with st.spinner("Asking Gemini directly..."):
                    if is_gemini_available():
                        direct = ask_without_rag(custom_question)
                        st.markdown(direct["answer"])
                    else:
                        st.warning("Gemini API not configured.")

            with col_right:
                st.subheader("✅ With RAG (Retrieved Context)")
                with st.spinner("Retrieving and generating..."):
                    rag = ask_with_rag(custom_question, top_k=custom_top_k)
                    retrieved = rag.get("retrieved_chunks", [])

                    if is_gemini_available():
                        st.markdown(rag["answer"])
                    else:
                        st.info("Gemini API not configured. Retrieved chunks:")
                        for c in retrieved[:2]:
                            st.text(c["text"][:300])

                    if retrieved:
                        with st.expander(f"🔎 Retrieved chunks ({len(retrieved)})"):
                            for i, chunk in enumerate(retrieved, 1):
                                st.write(f"**{i}.** {chunk['metadata'].get('title','?')} "
                                         f"| Relevance: {chunk['relevance_score']}")
                                st.text(chunk["text"][:300])
                                st.divider()
