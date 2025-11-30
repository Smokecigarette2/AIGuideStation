import os
from typing import List, Dict, Any, Tuple

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# ======================
#  1. INIT
# ======================

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")

genai.configure(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.5-flash"

# ======================
#  2. IN-MEMORY STORE
# ======================

# org_id -> list of docs
# doc: {"id": int, "text": str, "embedding": List[float]}
ORG_STORE: Dict[str, List[Dict[str, Any]]] = {}
DOC_ID_COUNTER = 1

# ниже порога — предположительный ответ
SIMILARITY_THRESHOLD = 0.35

# ======================
#  3. MODELS (Pydantic)
# ======================


class AddDocumentRequest(BaseModel):
    org_id: str
    text: str


class AskRequest(BaseModel):
    org_id: str
    question: str
    language: str = "ru"   # пока просто лежит, на будущее
    top_k: int = 5


class AskResponse(BaseModel):
    answer_text: str
    mode: str
    results: List[Dict[str, Any]]


# ======================
#  4. EMBEDDINGS
# ======================


def get_embedding(text: str) -> List[float]:
    """Получаем embedding через Gemini."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document",
    )
    return result["embedding"]


# ======================
#  5. DOCUMENT MANAGEMENT
# ======================


def add_document(org_id: str, text: str) -> Dict[str, Any]:
    """Добавляем документ в память для конкретной организации."""
    global DOC_ID_COUNTER

    emb = get_embedding(text)

    doc = {
        "id": DOC_ID_COUNTER,
        "text": text,
        "embedding": emb,
    }

    ORG_STORE.setdefault(org_id, []).append(doc)
    DOC_ID_COUNTER += 1

    return doc


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def search(org_id: str, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
    """Поиск ближайших документов по org_id."""
    docs = ORG_STORE.get(org_id)
    if not docs:
        return []

    query_emb = np.array(get_embedding(query), dtype=np.float32)

    scored: List[Tuple[Dict[str, Any], float]] = []

    for d in docs:
        doc_vec = np.array(d["embedding"], dtype=np.float32)
        sim = cosine_similarity(query_emb, doc_vec)
        scored.append((d, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ======================
#  6. GEMINI ANSWERING
# ======================


def build_answer(question: str,
                 docs_with_scores: List[Tuple[Dict[str, Any], float]]) -> Tuple[str, str]:
    """
    Возвращает (answer_text, mode), где mode:
    - "strict"  — ответ только по базе
    - "approx"  — предположительный ответ с дисклеймером
    """

    if not docs_with_scores:
        best_similarity = 0.0
    else:
        best_similarity = docs_with_scores[0][1]

    # ==== MODE B: APPROXIMATE (нет релевантного контекста) ====
    if best_similarity < SIMILARITY_THRESHOLD:
        mode = "approx"

        system_prompt = """
You are Guide Station — an internal assistant for an organization.

There is NO exact internal information about this question in the system.

You may give a helpful general answer based on common institutional practices,
BUT you MUST clearly state that:
1) exact information for this organization is not present in the system;
2) your answer is an assumption based on typical cases.

Use soft language such as:
"usually", "typically", "in most institutions"0, "commonly", "presumably".

Never present your answer as a guaranteed fact.
Never invent specific details about this particular organization.
        """.strip()

        context_text = "No reliable internal context is available for this question."

    # ==== MODE A: STRICT FACTUAL (есть нормальный контекст) ====
    else:
        mode = "strict"
        context_lines = [f"- {d['text']}" for (d, s) in docs_with_scores]

        context_text = "\n".join(context_lines)

        system_prompt = """
You are Guide Station — an internal assistant for an organization.

Answer ONLY using the information in the CONTEXT below.

If the information needed to answer the question is NOT present in the context,
you MUST reply exactly:
"This information is not in the system. Please contact the administrator."

Do NOT:
- invent facts;
- guess;
- use outside knowledge;
- add extra assumptions.

Do NOT mention the word "context" explicitly. Just answer naturally.
        """.strip()

    model = genai.GenerativeModel(GENERATION_MODEL)

    full_prompt = f"""{system_prompt}

CONTEXT:
{context_text}

USER QUESTION:
{question}
"""

    response = model.generate_content(full_prompt)
    answer = response.text
    return answer, mode


# ======================
#  7. FASTAPI
# ======================

app = FastAPI(title="Guide Station (Gemini RAG)")

# --- CORS, чтобы React на http://localhost:3000 мог ходить к бэку ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #allows all origins (React local + Render UI + anything)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Guide Station backend is alive"}


@app.post("/add", response_model=Dict[str, Any])
def api_add_document(req: AddDocumentRequest):
    """
    Добавляет текст в базу конкретной организации.
    org_id — любой идентификатор (например, "my_university").
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    doc = add_document(req.org_id, req.text.strip())
    return {
        "status": "ok",
        "org_id": req.org_id,
        "document": {
            "id": doc["id"],
            "text": doc["text"],
        },
    }


@app.post("/ask", response_model=AskResponse)
def api_ask(req: AskRequest):
    """
    Задает вопрос ассистенту по конкретной организации.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")

    docs_with_scores = search(req.org_id, req.question, top_k=req.top_k)

    answer, mode = build_answer(req.question, docs_with_scores)

    results = [
        {
            "id": d["id"],
            "text": d["text"],
            "similarity": score,
        }
        for (d, score) in docs_with_scores
    ]

    return AskResponse(
        answer_text=answer,
        mode=mode,
        results=results,
    )


@app.get("/orgs/{org_id}/docs", response_model=List[Dict[str, Any]])
def api_list_docs(org_id: str):
    """
    Получить список документов для организации (для дебага).
    """
    return [
        {"id": d["id"], "text": d["text"]}
        for d in ORG_STORE.get(org_id, [])
    ]