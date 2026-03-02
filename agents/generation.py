import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def generate_answer(query, context, strict=True):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if strict:
        system_prompt = """
You are a reliable AI assistant.
Answer ONLY using the provided context.
If insufficient info, say you don't know.
Be concise.
"""
    else:
        system_prompt = """
You are a helpful assistant.
Use context primarily but reason carefully.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content