---

# TrustLayer AI

TrustLayer AI is a self-reflective multi-agent RAG system that evaluates its own answers before presenting a final confidence score.
Instead of stopping at retrieval and generation, it adds verification and reflection layers to reduce hallucinations and improve reliability.

---

## Overview

Most RAG systems retrieve documents and generate an answer.
TrustLayer AI extends this pipeline by introducing:

* A verification agent that checks factual support
* A reflection agent that improves weak answers
* A weighted confidence scoring system
* Risk-level classification

The goal is to build a reliability layer on top of standard Retrieval-Augmented Generation.

Demo Link : https://huggingface.co/spaces/dhaval0003/trustlayer_ai

---

## Architecture

The system consists of five main stages:

1. Query Input
   User submits a question.

2. Retrieval Layer
   Top-k relevant chunks are retrieved using FAISS vector similarity search.

3. Generation Agent
   An answer is generated using retrieved context.

4. Verification Agent
   The answer is evaluated for factual alignment with the retrieved sources.

5. Reflection Agent
   If alignment is weak, the system rewrites and re-verifies the answer.

---

## Scoring Model

### Retrieval Score

FAISS returns similarity distance scores (lower is better).
The system converts distance into a 0–100 quality score:

R = clip(100 − average_distance, 0, 100)

---

### Alignment Score

The verification agent evaluates whether:

* The answer is supported by retrieved documents
* Claims are grounded in evidence
* There are factual inconsistencies

This produces:

* alignment_score (0–100)
* supported (True/False)
* issues (explanation)

---

### Final Confidence

Final confidence is computed using weighted averaging:

C = 0.4R + 0.6A

Where:

* R = Retrieval Score
* A = Alignment Score
* C = Final Confidence

Alignment is weighted higher to prioritize factual correctness over similarity.

---

### Risk Classification

* High Risk: C < 50
* Medium Risk: 50 ≤ C < 80
* Low Risk: C ≥ 80

This prevents overconfident hallucinated answers.

---

## Features

* Multi-agent RAG architecture
* Self-verification loop
* Reflection-based answer improvement
* Confidence scoring system
* Risk-level estimation
* Optional PDF upload for custom document indexing
* Gradio-based interactive UI

---

## Tech Stack

* Python
* Gradio
* OpenAI API
* LangChain
* FAISS
* PyPDF
* RecursiveCharacterTextSplitter

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/trustlayer-ai.git
cd trustlayer-ai
```

Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_key_here
```

Run the app:

```bash
python app.py
```

---

## Example Test Prompt

What are the two training stages in Constitutional AI and why are they important?

---

## Why This Project

Large language models often produce confident but unsupported answers.
TrustLayer AI demonstrates how multi-agent evaluation and reflection can create more reliable AI systems.

This project focuses on building a structured reliability layer on top of RAG pipelines.

---

## Future Improvements

* Better calibrated retrieval scoring
* Cross-document contradiction detection
* Citation highlighting
* Adaptive weighting for confidence
* Research-grade evaluation metrics

---
