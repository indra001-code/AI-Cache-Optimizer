import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import chromadb
from google import genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv
except ImportError:  # Keeps the app importable before dependencies are installed.
    load_dotenv = None

if load_dotenv:
    load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("ai-cache")

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CACHE_DISTANCE_THRESHOLD = float(os.getenv("CACHE_DISTANCE_THRESHOLD", "0.45"))
ESTIMATED_COST_PER_QUERY = float(os.getenv("ESTIMATED_COST_PER_QUERY", "0.002"))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "4000"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None
    logger.warning("GEMINI_API_KEY is not configured; AI requests use the local fallback.")

logger.info("Loading embedding model: %s", MODEL_NAME)
embedding_model = SentenceTransformer(MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="ai_cache")

stats_lock = threading.Lock()
stats = {
    "total_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "total_dollars_saved": 0.0,
}


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)


class AskResponse(BaseModel):
    status: str
    latency_ms: float
    cache_distance: float | None = None
    cost_saved: float = 0.0
    answer: str


def increment_stat(name: str, amount: int | float = 1) -> None:
    with stats_lock:
        stats[name] += amount


def current_stats() -> dict[str, Any]:
    with stats_lock:
        snapshot = dict(stats)
    total = snapshot["total_requests"]
    snapshot["cache_hit_rate"] = round(snapshot["cache_hits"] / total * 100, 2) if total else 0.0
    snapshot["estimated_money_saved"] = round(snapshot["total_dollars_saved"], 4)
    return snapshot


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("AI cache service started with %s cached entries", collection.count())
    yield
    logger.info("AI cache service stopped")


app = FastAPI(
    title="AI API Semantic Cache & Cost Optimizer",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "gemini_configured": gemini_client is not None,
        "embedding_model": MODEL_NAME,
        "cached_entries": collection.count(),
    }


@app.post("/ask", response_model=AskResponse)
def ask_ai(payload: AskRequest) -> AskResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query cannot be blank.")

    increment_stat("total_requests")
    started_at = time.perf_counter()
    query_vector = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=1)

    distances = results.get("distances") or [[]]
    documents = results.get("documents") or [[]]
    if distances[0] and documents[0] and distances[0][0] < CACHE_DISTANCE_THRESHOLD:
        distance = float(distances[0][0])
        increment_stat("cache_hits")
        increment_stat("total_dollars_saved", ESTIMATED_COST_PER_QUERY)
        return AskResponse(
            status="CACHE_HIT",
            latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
            cache_distance=round(distance, 4),
            cost_saved=ESTIMATED_COST_PER_QUERY,
            answer=documents[0][0],
        )

    increment_stat("cache_misses")
    if gemini_client:
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=query,
            )
            answer = (response.text or "").strip()
            if not answer:
                raise RuntimeError("Gemini returned an empty response.")
        except Exception:
            logger.exception("Gemini request failed")
            raise HTTPException(
                status_code=502,
                detail="The AI provider could not answer this request. Please try again.",
            )
    else:
        answer = f"Simulated response for: {query}"

    collection.add(
        embeddings=[query_vector],
        documents=[answer],
        metadatas=[{"query": query, "model": GEMINI_MODEL_NAME}],
        ids=[str(uuid.uuid4())],
    )
    return AskResponse(
        status="CACHE_MISS",
        latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
        answer=answer,
    )


@app.get("/stats")
def get_analytics() -> dict[str, Any]:
    return current_stats()


@app.delete("/cache")
def clear_cache() -> dict[str, Any]:
    cached = collection.get()
    ids = cached.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return {"status": "ok", "deleted_entries": len(ids)}
