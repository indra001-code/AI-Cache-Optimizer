# AI Cache Optimizer

A FastAPI backend and Streamlit frontend that uses semantic caching to answer repeated or similar AI questions faster while reducing API usage and cost.

## How it works

1. The frontend sends a question to the FastAPI `/ask` endpoint.
2. The backend creates an embedding for the question with Sentence Transformers.
3. ChromaDB searches the persistent semantic cache for a similar question.
4. If the distance is below `CACHE_DISTANCE_THRESHOLD`, the cached answer is returned as a cache hit.
5. Otherwise, Gemini generates a new answer, which is stored in ChromaDB for future requests.

When `GEMINI_API_KEY` is not configured, the application uses a local simulated response for development.

## Features

- Semantic similarity matching instead of exact-text matching
- Persistent ChromaDB cache stored in `chroma_db/`
- Gemini-powered answers with a local fallback
- Streamlit chat interface
- Cache hit/miss status and request latency
- Request statistics, cache hit rate, and estimated savings
- FastAPI health check and interactive API docs
- Cache management through the UI and API

## Project structure

| File | Purpose |
| --- | --- |
| `mainvd.py` | FastAPI backend, embeddings, Gemini calls, and cache management |
| `app.py` | Streamlit chat frontend and live statistics |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template |
| `run` | Local startup commands |
| `chroma_db/` | Local persistent cache data (ignored by Git) |

## Setup

### 1. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and set your Gemini key:

```env
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=all-MiniLM-L6-v2
CACHE_DISTANCE_THRESHOLD=0.45
ESTIMATED_COST_PER_QUERY=0.002
MAX_QUERY_LENGTH=4000
API_URL=http://127.0.0.1:8000
```

> **Never commit `.env` or API keys to source control.**

## Run locally

Start the backend in one terminal:

```bash
python -m uvicorn mainvd:app --reload
```

Start the Streamlit frontend in another terminal:

```bash
python -m streamlit run app.py
```

Open the frontend at **http://127.0.0.1:8501**.

- Backend documentation: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/health

## API endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service status, embedding model, and cache count |
| `POST` | `/ask` | Answer a question and report cache status |
| `GET` | `/stats` | Request counts, hit rate, and estimated savings |
| `DELETE` | `/cache` | Clear all cached answers |

### Example request

```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is semantic caching?\"}"
```

## Configuration

| Variable | Description |
| --- | --- |
| `CACHE_DISTANCE_THRESHOLD` | Controls how similar a question must be to reuse a cached answer. Lower values are stricter. |
| `CHROMA_PATH` | Changes the location of the persistent ChromaDB data. |
| `CORS_ORIGINS` | Configures allowed frontend origins as a comma-separated list. |
| `MAX_QUERY_LENGTH` | Limits the length of incoming questions. |

## License

No license has been specified yet.
