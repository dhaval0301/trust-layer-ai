import os
import json
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from agents.generation import generate_answer
from agents.verification import verify_answer
from agents.reflection import reflect_answer

# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="TrustLayer AI",
    # page_icon="",
    layout="wide"
)

# -------------------------
# Load Environment
# -------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Set it in environment variables.")
    st.stop()

# -------------------------
# Load Vector Store
# -------------------------

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db

db = load_vectorstore()
retriever = db.as_retriever(search_kwargs={"k": 5})

# -------------------------
# UI Header
# -------------------------

st.title("TrustLayer AI")
st.subheader("Self-Reflective Multi-Agent RAG System")

mode = st.selectbox(
    "Select Mode",
    ["Strict Grounded Mode"]
)

evaluation_mode = st.toggle("Evaluation Mode", value=True)

query = st.text_input("Ask a question:")

# -------------------------
# Main Logic
# -------------------------

if query:

    start_time = time.time()

    # Retrieve documents with scores
    docs_with_scores = db.similarity_search_with_score(query, k=5)
    docs = [doc for doc, score in docs_with_scores]

    # Compute Retrieval Score (normalized)
    if docs_with_scores:
        raw_scores = [score for doc, score in docs_with_scores]
        avg_score = sum(raw_scores) / len(raw_scores)
        retrieval_score = max(0, min(100, int(100 - avg_score)))
    else:
        retrieval_score = 0

    # Generate answer
    answer = generate_answer(query, docs)

    # Verify answer
    verification_raw = verify_answer(query, answer, docs)

    try:
        verification = json.loads(verification_raw)
        alignment_score = verification.get("alignment_score", 0)
        supported = verification.get("supported", False)
        issues = verification.get("issues", "")
    except:
        alignment_score = 0
        supported = False
        issues = "Verification parsing failed"

    # Reflection if low alignment
    if alignment_score < 60:
        improved_answer = reflect_answer(query, answer, docs)
        answer = improved_answer

        verification_raw = verify_answer(query, answer, docs)
        try:
            verification = json.loads(verification_raw)
            alignment_score = verification.get("alignment_score", 0)
            supported = verification.get("supported", False)
            issues = verification.get("issues", "")
        except:
            pass

    # Final confidence calculation
    final_confidence = int((retrieval_score * 0.4) + (alignment_score * 0.6))

    # Risk classification
    if final_confidence >= 80:
        risk = "Low"
        risk_color = "green"
    elif final_confidence >= 50:
        risk = "Medium"
        risk_color = "orange"
    else:
        risk = "High"
        risk_color = "red"

    response_time = round(time.time() - start_time, 2)

    # -------------------------
    # Display Output
    # -------------------------

    st.markdown("## Final Answer")
    st.write(answer)

    if evaluation_mode:

        st.markdown("## Confidence Breakdown")

        col1, col2, col3 = st.columns(3)

        col1.metric("Retrieval Score", f"{retrieval_score}%")
        col2.metric("Alignment Score", f"{alignment_score}%")
        col3.metric("Final Confidence", f"{final_confidence}%")

        st.progress(final_confidence / 100)

        st.markdown(
            f"<div style='background-color:{risk_color}; padding:10px; border-radius:5px;'>"
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

        verification_display = {
            "supported": supported,
            "alignment_score": alignment_score,
            "issues": issues
        }

        st.json(verification_display)