import json
from openai import OpenAI

print("🔥 NEW VERIFICATION FILE LOADED 🔥")

client = OpenAI()


def verify_answer(query, answer, docs):
    """
    Verifies whether the generated answer is supported by retrieved documents.
    Returns a structured JSON dictionary.
    """

    # Convert LangChain documents into plain text
    try:
        context = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print("Context build error:", e)
        context = ""

    system_prompt = """
You are a strict factual verification system.

You must return ONLY valid JSON in this exact format:

{
  "supported": true or false,
  "alignment_score": number between 0 and 100,
  "issues": "short explanation"
}

Rules:
- No markdown
- No explanation outside JSON
- No extra text
- Must be valid parsable JSON
"""

    user_prompt = f"""
Question:
{query}

Answer:
{answer}

Context:
{context}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )

        raw_content = response.choices[0].message.content

        print("========== RAW VERIFIER OUTPUT ==========")
        print(raw_content)
        print("=========================================")

        # Parse JSON safely
        parsed = json.loads(raw_content)

        # Safety fallback in case model returns incomplete keys
        return {
            "supported": parsed.get("supported", False),
            "alignment_score": parsed.get("alignment_score", 0),
            "issues": parsed.get("issues", "No explanation provided")
        }

    except Exception as e:
        print("❌ VERIFICATION ERROR:", str(e))

        return {
            "supported": False,
            "alignment_score": 0,
            "issues": f"Verification parsing failed: {str(e)}"
        }