import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agents.generation import generate_answer
from agents.verification import verify_answer
from agents.reflection import reflect_answer


# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="TrustLayer AI",
    # page_icon="🔐",
    layout="wide"
)

st.title("TrustLayer AI")
st.subheader("Self-Reflective Multi-Agent RAG System")


# -------------------------
# Load Environment
# -------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found. Please set it in environment variables.")
    st.stop()


# -------------------------
# Load Default Vector Store
# -------------------------

@st.cache_resource
def load_default_vectorstore():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db


# -------------------------
# Build Temporary Index (Upload)
# -------------------------

def build_temp_index(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    return db


# -------------------------
# UI Controls
# -------------------------

uploaded_file = st.file_uploader("Upload your own PDF (optional)", type=["pdf"])
evaluation_mode = st.toggle(" Evaluation Mode", value=True)
query = st.text_input("Ask a question:")


# -------------------------
# Main Logic
# -------------------------

if query:

    start_time = time.time()

    # Select index
    if uploaded_file is not None:
        db = build_temp_index(uploaded_file)
        st.info(f"Using uploaded document: {uploaded_file.name}")
    else:
        db = load_default_vectorstore()
        st.info("Using default fairness research documents")

    # Retrieve documents
    docs_with_scores = db.similarity_search_with_score(query, k=5)
    docs = [doc for doc, score in docs_with_scores]

    # -------------------------
    # Retrieval Score
    # -------------------------

    if docs_with_scores:
        raw_scores = [score for doc, score in docs_with_scores]
        avg_score = sum(raw_scores) / len(raw_scores)
        retrieval_score = max(0, min(100, int(100 - avg_score)))
    else:
        retrieval_score = 0

    # -------------------------
    # Generation
    # -------------------------

    answer = generate_answer(query, docs)

    # -------------------------
    # Verification (FIXED — No JSON parsing)
    # -------------------------

    verification = verify_answer(query, answer, docs)

    alignment_score = verification.get("alignment_score", 0)
    supported = verification.get("supported", False)
    issues = verification.get("issues", "")

    # -------------------------
    # Reflection Loop
    # -------------------------

    if alignment_score < 60:

        improved_answer = reflect_answer(query, answer, docs)
        answer = improved_answer

        verification = verify_answer(query, answer, docs)

        alignment_score = verification.get("alignment_score", 0)
        supported = verification.get("supported", False)
        issues = verification.get("issues", "")

    # -------------------------
    # Final Confidence Score
    # -------------------------

    final_confidence = int((retrieval_score * 0.4) + (alignment_score * 0.6))

    if final_confidence >= 80:
        risk = "Low"
        risk_color = "#2ecc71"
    elif final_confidence >= 50:
        risk = "Medium"
        risk_color = "#f39c12"
    else:
        risk = "High"
        risk_color = "#e74c3c"

    response_time = round(time.time() - start_time, 2)

    # -------------------------
    # Display Output
    # -------------------------

    st.markdown("##  Final Answer")
    st.write(answer)

    if evaluation_mode:

        st.markdown("## Confidence Breakdown")

        col1, col2, col3 = st.columns(3)

        col1.metric("Retrieval Score", f"{retrieval_score}%")
        col2.metric("Alignment Score", f"{alignment_score}%")
        col3.metric("Final Confidence", f"{final_confidence}%")

        st.progress(final_confidence / 100)

        st.markdown(
            f"<div style='background-color:{risk_color}; padding:12px; border-radius:6px;'>"
            f"<b>Risk Level: {risk}</b>"
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown(f"⏱ Response Time: {response_time}s")

        # -------------------------
        # Sources
        # -------------------------

        st.markdown("## Sources Used")

        for i, doc in enumerate(docs):
            with st.expander(f"Source {i+1}"):
                st.write(doc.page_content)

        # -------------------------
        # Verification Report
        # -------------------------

        st.markdown("## Verification Report")

        st.json({
            "supported": supported,
            "alignment_score": alignment_score,
            "issues": issues
        })