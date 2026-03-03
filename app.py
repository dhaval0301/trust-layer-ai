import os
import time
import tempfile
import gradio as gr
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agents.generation import generate_answer
from agents.verification import verify_answer
from agents.reflection import reflect_answer


# -------------------------
# Load Environment
# -------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Please set it.")


# -------------------------
# Build Vector Store (Default)
# -------------------------

def build_default_index():

    loader = PyPDFLoader("data/your_default_pdf.pdf")  # Change this file
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
# Main Pipeline
# -------------------------

def trustlayer_pipeline(query, uploaded_file, evaluation_mode):

    if not query:
        return "Please enter a question.", None, None, None, None, None

    start_time = time.time()

    # Select index
    if uploaded_file is not None:
        db = build_temp_index(uploaded_file)
    else:
        db = build_default_index()

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
    # Verification
    # -------------------------

    verification = verify_answer(query, answer, docs)

    alignment_score = verification.get("alignment_score", 0)
    supported = verification.get("supported", False)
    issues = verification.get("issues", "")

    # -------------------------
    # Reflection Loop
    # -------------------------

    if alignment_score < 60:
        answer = reflect_answer(query, answer, docs)
        verification = verify_answer(query, answer, docs)

        alignment_score = verification.get("alignment_score", 0)
        supported = verification.get("supported", False)
        issues = verification.get("issues", "")

    # -------------------------
    # Final Confidence
    # -------------------------

    final_confidence = int((retrieval_score * 0.4) + (alignment_score * 0.6))

    if final_confidence >= 80:
        risk = "Low"
    elif final_confidence >= 50:
        risk = "Medium"
    else:
        risk = "High"

    response_time = round(time.time() - start_time, 2)

    sources = "\n\n".join([doc.page_content[:800] for doc in docs])

    if not evaluation_mode:
        return answer, None, None, None, None, None

    return (
        answer,
        f"{retrieval_score}%",
        f"{alignment_score}%",
        f"{final_confidence}%",
        f"{risk}",
        f"{response_time}s\n\nSources:\n{sources}"
    )


# -------------------------
# Gradio UI
# -------------------------

with gr.Blocks(title="TrustLayer AI") as demo:

    gr.Markdown("# TrustLayer AI")
    gr.Markdown("Self-Reflective Multi-Agent RAG System")

    query = gr.Textbox(label="Ask a question")
    uploaded_file = gr.File(label="Upload PDF (optional)", file_types=[".pdf"])
    evaluation_mode = gr.Checkbox(label="Evaluation Mode", value=True)

    run_btn = gr.Button("Run")

    answer = gr.Textbox(label="Final Answer", lines=6)
    retrieval = gr.Textbox(label="Retrieval Score")
    alignment = gr.Textbox(label="Alignment Score")
    confidence = gr.Textbox(label="Final Confidence")
    risk = gr.Textbox(label="Risk Level")
    details = gr.Textbox(label="Details", lines=10)

    run_btn.click(
        trustlayer_pipeline,
        inputs=[query, uploaded_file, evaluation_mode],
        outputs=[answer, retrieval, alignment, confidence, risk, details]
    )


if __name__ == "__main__":
    demo.launch()