import streamlit as st

from config import (
    GROQ_MODEL,
    TOP_K,
    get_groq_client,
)

from chunk import create_chunks, load_markdown_files
from embed import get_embedding

from store import (
    search,
    get_index_stats,
    delete_namespace,
    index_exists,
)

from ingest import ingest_documents


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def build_context(matches):
    """Convert Pinecone matches into an LLM context."""
    context_parts = []

    for index, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {})
        source = metadata.get("source", "Unknown")
        text = metadata.get("text", "")

        context_parts.append(
            f"""
SOURCE {index}
File: {source}

{text}
"""
        )

    return "\n".join(context_parts)


def generate_answer(question: str, matches):
    """Generate answer using Groq LLM."""
    if not matches:
        return "I could not find relevant information in the provided documents."

    context = build_context(matches)

    system_prompt = """
You are a document question-answering assistant.

Your job is to answer questions ONLY using the
provided document context.

Rules:
1. Do not invent information.
2. If the answer is not available in the context,
   clearly say that the information was not found
   in the documents.
3. Keep the answer clear and concise.
4. Cite the source number for factual claims.
5. Use citations like [Source 1], [Source 2].
6. Do not cite sources that do not support the answer.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Answer the question using only the document context.
Include source citations such as [Source 1].
"""

    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1000,
    )

    return response.choices[0].message.content


def retrieve_documents(question, top_k):
    """Convert question into an embedding and search Pinecone."""
    query_embedding = get_embedding(question)
    results = search(query_embedding=query_embedding, top_k=top_k)
    return results.matches


def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "retrieved_docs" not in st.session_state:
        st.session_state.retrieved_docs = TOP_K
    if "conversation_cleared" not in st.session_state:
        st.session_state.conversation_cleared = False


@st.cache_data(show_spinner=False)
def get_document_stats():
    """Return document and chunk counts for the dashboard."""
    try:
        documents = load_markdown_files()
        chunks = create_chunks()
        return len(documents), len(chunks)
    except Exception:
        return 0, 0


def get_pinecone_status():
    """Return a Pinecone connection status string."""
    try:
        return "Connected" if index_exists() else "Not connected"
    except Exception as exc:
        return f"Error: {exc}"


def render_chat_history():
    if not st.session_state.history:
        st.info("Ask a question about your documents to begin the conversation.")
        return

    for message in st.session_state.history:
        role = message["role"]
        content = message["content"]
        sources = message.get("sources")

        if hasattr(st, "chat_message"):
            with st.chat_message(role):
                st.write(content)
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.markdown(
                                f"**{source['source']}** (score: {source['score']:.4f})"
                            )
                            st.write(source["text"])
        else:
            prefix = "You:" if role == "user" else "Assistant:"
            st.markdown(f"**{prefix}** {content}")
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(
                            f"**{source['source']}** (score: {source['score']:.4f})"
                        )
                        st.write(source["text"])


# ---------------------------------------------------------
# STATE

init_session_state()


# ---------------------------------------------------------
# SIDEBAR

with st.sidebar:
    st.title("Settings")
    st.write("Manage document ingestion, retrieval settings, and conversation state.")

    st.markdown("---")

    st.session_state.retrieved_docs = st.slider(
        "Retrieved documents",
        min_value=1,
        max_value=10,
        value=st.session_state.retrieved_docs,
        step=1,
    )

    st.markdown("---")
    st.subheader("Model")
    st.write(f"**Groq:** {GROQ_MODEL}")
    st.write("**Embeddings:** Gemini")
    st.write("**Vector Store:** Pinecone")

    st.markdown("---")
    if st.button("🧩 Ingest Documents", use_container_width=True):
        try:
            with st.spinner("Reading, chunking, embedding and storing documents..."):
                count = ingest_documents()
            st.success(f"Successfully ingested {count} chunks.")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")

    if st.button("🗑️ Clear Vector Store", use_container_width=True):
        try:
            delete_namespace()
            st.success("Vector store cleared.")
        except Exception as e:
            st.error(f"Unable to clear vector store: {e}")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.conversation_cleared = True

    st.markdown("---")
    if st.button("📊 Refresh Statistics", use_container_width=True):
        try:
            stats = get_index_stats()
            st.write(stats)
        except Exception as e:
            st.error(f"Unable to retrieve statistics: {e}")


# ---------------------------------------------------------
# MAIN UI

st.header("😄 RAG Assistant")
st.write("Ask questions about the documents stored in the knowledge base.")

stats_col1, stats_col2, stats_col3 = st.columns([1, 1, 1])

with stats_col1:
    docs_count, chunks_count = get_document_stats()
    st.metric("📄 Documents", docs_count)

with stats_col2:
    st.metric("🧩 Chunks", chunks_count)

with stats_col3:
    st.metric("🌲 Pinecone", get_pinecone_status())

st.markdown("---")

render_chat_history()

with st.form("query_form", clear_on_submit=True):
    question = st.text_input(
        "Ask a question about your documents...",
        placeholder="Example: Is Nandhini a negative or positive character?",
    )
    submit = st.form_submit_button("Send")

if submit:
    if not question or not question.strip():
        st.warning("Please enter a question.")
    else:
        st.session_state.history.append({"role": "user", "content": question})
        try:
            with st.spinner("Searching documents..."):
                matches = retrieve_documents(question, st.session_state.retrieved_docs)

            if not matches:
                assistant_text = (
                    "I could not find relevant information in the provided documents."
                )
                st.session_state.history.append({"role": "assistant", "content": assistant_text})
            else:
                with st.spinner("Generating answer..."):
                    answer = generate_answer(question, matches)

                sources = [
                    {
                        "source": match.get("metadata", {}).get("source", "Unknown"),
                        "score": match.get("score", 0),
                        "text": match.get("metadata", {}).get("text", ""),
                    }
                    for match in matches
                ]

                st.session_state.history.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
        except Exception as e:
            st.session_state.history.append(
                {"role": "assistant", "content": f"Application error: {e}"}
            )
        # Rerun is not required here; Streamlit will refresh on state changes.
