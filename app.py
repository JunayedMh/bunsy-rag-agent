import os
import time
import streamlit as st

try:
    if "API_KEY" in st.secrets:
        os.environ["API_KEY"] = st.secrets["API_KEY"]
    if "BASE_URL" in st.secrets:
        os.environ["BASE_URL"] = st.secrets["BASE_URL"]
except Exception:
    pass

from ragkit import RAG
from ragkit.exceptions import GenerationError

st.set_page_config(page_title="Bunsy Doc Q&A", page_icon="📄")
st.title("📄 Ask the Bunsy")
st.caption("Answers are grounded strictly in the uploaded documents.")


@st.cache_resource
def load_rag():
    doc_paths = sorted(
        os.path.join("docs", f)
        for f in os.listdir("docs")
        if f.lower().endswith(".pdf")
    )
    return RAG(
        document=doc_paths,
        model="deepseek/deepseek-v4.1-flash",
        base_url="https://openrouter.ai/api/v1",
        persist_directory="./rag_index",
        top_k=4,
        rerank=False,
        retrieval="vector",
    )


rag = load_rag()

question = st.text_input(
    "Ask a question about the documents:",
    placeholder="e.g. What are the models?",
)

if question:
    with st.spinner("Searching the documents and generating an answer..."):
        response = None
        for attempt in range(3):
            try:
                response = rag.query(question)
                break
            except GenerationError as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                st.error(f"Generation failed: {e}")
                st.stop()

    if response is None:
        st.stop()

    st.markdown("### Answer")
    st.write(response.answer)

    with st.expander("📚 Sources"):
        for i, source in enumerate(response.sources, 1):
            name = os.path.basename(source.document)
            st.write(f"**[{i}]** {name}")

    with st.expander("⏱️ Timing"):
        st.json(getattr(response, "metadata", {}) or {})