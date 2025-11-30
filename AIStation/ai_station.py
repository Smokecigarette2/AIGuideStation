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

ORG_STORE: Dict[str, List[Dict[str, Any]]] = {}
DOC_ID_COUNTER = 1

SIMILARITY_THRESHOLD = 0.35

# ======================
#  2.1 PRELOADED UNIVERSITY DATA (prototype)
# ======================

UNIVERSITY_ORG_ID = "my_university"

UNIVERSITY_DOCS = [
    "Главный вход находится со стороны Манаса.",
    "На первом этаже расположен ресепшн.",
    "Гардероб находится рядом с входом.",
    "Камеры хранения доступны на первом этаже.",
    "Буфет работает с 9:00 до 17:00.",
    "Столовая открыта с 9:00 до 18:00.",
    "Библиотека находится на 3 этаже.",
    "Библиотека открыта с 9:00 до 18:00.",
    "Учебная часть находится в кабинете 108.",
    "Справки выдаются в кабинете 105 с 10:00 до 16:00.",
    "Документы принимают в кабинете 204 с 9:00 до 17:00.",
    "Деканат факультета CS расположен в кабинете 215.",
    "Деканат факультета IT расположен в кабинете 220.",
    "Актовый зал находится на 2 этаже.",
    "IT-лаборатория находится в кабинете 210.",
    "Wi-Fi доступен по студенческому логину.",
    "Термополка для разогрева еды стоит на первом этаже.",
    "Лифт расположен справа от ресепшена.",
    "Лестницы находятся по обе стороны коридора.",
    "Коворкинг открыт с 9:00 до 20:00.",
    "Коворкинг находится на втором этаже.",
    "Медицинский кабинет расположен в аудитории 112.",
    "Аудитории начинаются с первого по четвертый этаж.",
    "Кабинеты с 100-х номеров на первом этаже.",
    "Кабинеты с 200-х номеров на втором этаже.",
    "Кабинеты с 300-х номеров на третьем этаже.",
    "Кабинеты с 400-х номеров на четвертом этаже.",
    "Кафедра программирования находится на 3 этаже.",
    "Спортзал расположен в подвальном этаже.",
    "В спортзале есть раздевалки.",
    "Автоматы с кофе стоят на каждом этаже.",
    "Автоматы с перекусами стоят у лестницы.",
    "Университет работает с 8:00 до 20:00.",
    "Охрана находится у входа.",
    "Студенческий билет проверяют на входе.",
    "В университете есть небольшая зона отдыха.",
    "Принтер для студентов расположен в библиотеке.",
    "Ксерокс доступен в библиотеке.",
    "Фотозона IITU стоит на первом этаже.",
    "На втором этаже есть мягкие диваны для отдыха.",
    "На третьем этаже есть столы для групповой работы.",
    "В университете есть питьевые фонтанчики.",
    "Рядом с университетом есть несколько кофеен.",
    "Парковка для студентов находится за зданием.",
    "Парковка для преподавателей расположена сбоку.",
    "Окна большинства кабинетов выходят на Манаса.",
    "В здании работает система видеонаблюдения.",
    "В университете регулярно проходят мероприятия.",
    "На первом этаже можно получить карту посетителя.",
    "Утерянные вещи можно оставить на ресепшене."
]


# ======================
#  3. MODELS (Pydantic)
# ======================

class AddDocumentRequest(BaseModel):
    org_id: str
    text: str

class AskRequest(BaseModel):
    org_id: str
    question: str
    language: str = "ru"
    top_k: int = 5

class AskResponse(BaseModel):
    answer_text: str
    mode: str
    results: List[Dict[str, Any]]

# ======================
#  4. EMBEDDINGS
# ======================

def get_embedding(text: str) -> List[float]:
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

def search(org_id: str, query: str, top_k: int = 5):
    docs = ORG_STORE.get(org_id)
    if not docs:
        return []

    query_emb = np.array(get_embedding(query), dtype=np.float32)

    scored = []
    for d in docs:
        sim = cosine_similarity(query_emb, np.array(d["embedding"], dtype=np.float32))
        scored.append((d, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# ======================
#  6. GENERATION
# ======================

def build_answer(question: str, docs_with_scores):
    if not docs_with_scores:
        best_similarity = 0.0
    else:
        best_similarity = docs_with_scores[0][1]

    if best_similarity < SIMILARITY_THRESHOLD:
        mode = "approx"

        system_prompt = """
You are Guide Station — an internal assistant for an organization.
There is NO exact internal information about this question in the system.
You may give a helpful general answer BUT you must clearly say this is an assumption.
        """

        context_text = "No reliable internal context is available."
    else:
        mode = "strict"

        context_text = "\n".join(
            f"- {d['text']}" for (d, s) in docs_with_scores
        )

        system_prompt = """
You are Guide Station — an internal assistant.
Answer ONLY using the information in the context.
If information is missing, reply:
"This information is not in the system. Please contact the administrator."
        """

    model = genai.GenerativeModel(GENERATION_MODEL)

    full_prompt = f"""{system_prompt}

CONTEXT:
{context_text}

QUESTION:
{question}
"""

    response = model.generate_content(full_prompt)
    return response.text, mode

# ======================
#  7. FASTAPI
# ======================

app = FastAPI(title="Guide Station (Gemini RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/add")
def api_add_document(req: AddDocumentRequest):
    if not req.text.strip():
        raise HTTPException(400, "Text must not be empty")

    doc = add_document(req.org_id, req.text.strip())

    return {
        "status": "ok",
        "org_id": req.org_id,
        "document": {"id": doc["id"], "text": doc["text"]},
    }

@app.post("/ask", response_model=AskResponse)
def api_ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question must not be empty")

    docs_with_scores = search(req.org_id, req.question, top_k=req.top_k)
    answer, mode = build_answer(req.question, docs_with_scores)

    results = [
        {"id": d["id"], "text": d["text"], "similarity": score}
        for (d, score) in docs_with_scores
    ]

    return AskResponse(answer_text=answer, mode=mode, results=results)

@app.get("/orgs/{org_id}/docs")
def api_list_docs(org_id: str):
    return [
        {"id": d["id"], "text": d["text"]}
        for d in ORG_STORE.get(org_id, [])
    ]

# ======================
#  8. STARTUP — preload data
# ======================

@app.on_event("startup")
async def preload_university_docs():
    """
    Загружаем тестовую базу при старте сервера.
    Работает и локально, и на Render.
    """
    global DOC_ID_COUNTER
    DOC_ID_COUNTER = 1

    ORG_STORE[UNIVERSITY_ORG_ID] = []

    for text in UNIVERSITY_DOCS:
        add_document(UNIVERSITY_ORG_ID, text)

    print(f"[startup] Loaded {len(UNIVERSITY_DOCS)} docs for {UNIVERSITY_ORG_ID}")
