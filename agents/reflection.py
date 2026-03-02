from openai import OpenAI
import os

client = OpenAI()

def reflect_answer(query, answer, context_docs):

    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""
You are a reflection agent.

The original question:
{query}

The current answer:
{answer}

Retrieved context:
{context_text}

If the answer is weak, unsupported, or incomplete,
rewrite it to be fully grounded in the retrieved context.

If the context does not support the question,
respond with: "I don't know."

Return only the improved answer.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()