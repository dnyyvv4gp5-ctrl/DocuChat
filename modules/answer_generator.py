import os
from dotenv import load_dotenv
from groq import Groq
from google import genai
from modules.schemas import ChatResponse, RetrievedSource

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
TIMEOUT_SECONDS = 45


class AnswerGenerationError(Exception):
    pass


def _build_evidence(retrieved_results):
    sources = []
    evidence = []

    for index, result in enumerate(retrieved_results, start=1):
        chunk = result["chunk"]
        sources.append(
            RetrievedSource(
                chunk_id=chunk.chunk_id,
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                content=chunk.content,
            )
        )
        evidence.append(
            "[SOURCE " + str(index) + "]\n"
            + "Document: " + chunk.document_name + "\n"
            + "Page: " + str(chunk.page_number) + "\n"
            + "Content: " + chunk.content
        )

    return sources, evidence


def _build_prompt(question, evidence):
    return (
        "Answer using ONLY the document evidence below. Do not use outside "
        "knowledge or invent facts. If evidence is insufficient, say so. "
        "Cite claims using [Document name, page X].\n\n"
        "QUESTION:\n" + question
        + "\n\nDOCUMENT EVIDENCE:\n" + "\n\n".join(evidence)
    )


def _generate_with_groq(prompt):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise AnswerGenerationError("Groq API key is not configured.")

    client = Groq(api_key=api_key, timeout=TIMEOUT_SECONDS, max_retries=1)
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a document-grounded assistant. Follow the supplied evidence strictly.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=1500,
        )
        answer = response.choices[0].message.content
        if not answer or not answer.strip():
            raise AnswerGenerationError("Groq returned an empty answer.")
        return answer.strip()
    finally:
        client.close()


def _generate_with_gemini(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise AnswerGenerationError("Gemini API key is not configured.")

    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": TIMEOUT_SECONDS * 1000},
    )
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        if not response.text or not response.text.strip():
            raise AnswerGenerationError("Gemini returned an empty answer.")
        return response.text.strip()
    finally:
        client.close()


def generate_answer(question: str, retrieved_results: list[dict]) -> ChatResponse:
    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    if not retrieved_results:
        return ChatResponse(
            answer="I could not find relevant information in the uploaded documents.",
            sources=[],
            grounded=False,
        )

    sources, evidence = _build_evidence(retrieved_results)
    prompt = _build_prompt(question, evidence)

    groq_error = None

    try:
        answer = _generate_with_groq(prompt)
        return ChatResponse(answer=answer, sources=sources, grounded=True)
    except Exception as exc:
        groq_error = exc

    try:
        answer = _generate_with_gemini(prompt)
        return ChatResponse(answer=answer, sources=sources, grounded=True)
    except Exception as gemini_error:
        raise AnswerGenerationError(
            "Both AI providers failed. Groq: "
            + str(groq_error)
            + " | Gemini: "
            + str(gemini_error)
        ) from gemini_error
