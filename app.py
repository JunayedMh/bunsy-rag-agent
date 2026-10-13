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
st.caption("Answers are grounded strictly in the uploaded document, with page citations.")


@st.cache_resource
def load_rag():
    return RAG(
        document="Bunsy Product Reference.pdf",
        model="poolside/laguna-s-2.1:free",
        base_url="https://openrouter.ai/api/v1",
        persist_directory="./rag_index",
    )

rag = load_rag()

question = st.text_input(
    "Ask a question about the document:",
    placeholder="e.g. What are the models?",
)

if question:
    with st.spinner("Searching the document and generating an answer..."):
        for attempt in range(3):
            try:
                response = rag.query(question)
                break
            except GenerationError as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                st.error(f"Generation failed: {e}")
                st.stop()

    st.markdown("### Answer")
    st.write(response.answer)

    with st.expander("📚 Sources"):
        for i, source in enumerate(response.sources, 1):
            st.write(f"**[{i}]** {source.document} — page {source.page}")