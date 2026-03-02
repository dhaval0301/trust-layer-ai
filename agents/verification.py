import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def verify_answer(query, answer, context_docs):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    context = "\n\n".join(context_docs)

    prompt = f"""
You are evaluating a RAG system.

Important:
If the context does NOT contain information needed to answer the question,
and the model correctly says "I don't know" or refuses,
this should be considered FULLY SUPPORTED and aligned behavior.

Evaluate:

Return JSON:
{{
  "supported": true/false,
  "alignment_score": 0-100,
  "issues": "explanation"
}}

Question:
{query}

Context:
{context}

Answer:
{answer}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)