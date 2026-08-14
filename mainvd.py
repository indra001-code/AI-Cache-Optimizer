import os
import time
import uuid
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai

# ==================== CONFIGURATION ====================
# 1. Gemini API Key Setup
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")  # Set your key here or via env variable

if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-3.5-flash')
else:
    gemini_model = None

# ==================== APP & DATABASE INIT ====================
app = FastAPI(title="AI API Semantic Cache & Cost Optimizer ⚡")

print("Loading Embedding Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Persistent Database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ai_cache")

# Analytics Tracker (Cost calculation based on ~$0.002 per 1k tokens)
ESTIMATED_COST_PER_QUERY = 0.002  # $0.002 saved per cache hit

stats = {
    "total_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "total_dollars_saved": 0.0
}

# ==================== API ENDPOINTS ====================

@app.post("/ask")
async def ask_ai(query: str):
    global stats
    stats["total_requests"] += 1
    start_time = time.time()
    
    # Step 1: convert Query into a vector
    query_vector = model.encode(query).tolist()
    
    # Step 2: Cache search 
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=1
    )
    
    # Step 3: Check for Similarity Match
    if results['distances'] and len(results['distances'][0]) > 0:
        distance = results['distances'][0][0]
        
        # Threshold: Distance < 0.45 means high similarity (~90%+)
        if distance < 0.45:
            stats["cache_hits"] += 1
            stats["total_dollars_saved"] += ESTIMATED_COST_PER_QUERY
            latency_ms = round((time.time() - start_time) * 1000, 2)
            
            return {
                "status": "CACHE HIT ⚡",
                "latency": f"{latency_ms} ms",
                "cost_saved": f"${ESTIMATED_COST_PER_QUERY}",
                "distance": round(distance, 4),
                "answer": results['documents'][0][0]
            }

    # Step 4: Cache Miss - Real AI API Call
    stats["cache_misses"] += 1
    latency_start = time.time()
    
    if gemini_model:
        try:
            response = gemini_model.generate_content(query)
            ai_generated_answer = response.text
        except Exception as e:
            ai_generated_answer = f"API Error: {str(e)}"
    else:
        # Fallback if API Key is not set yet
        ai_generated_answer = f"[Simulated AI Response for: '{query}'] (Add your Gemini API Key in mainvd.py to get real AI answers!)"

    latency_ms = round((time.time() - latency_start) * 1000, 2)

    # Step 5: Save new Q&A pair into Vector DB
    doc_id = str(uuid.uuid4())
    collection.add(
        embeddings=[query_vector],
        documents=[ai_generated_answer],
        metadatas=[{"query": query}],
        ids=[doc_id]
    )
    
    return {
        "status": "CACHE MISS 🐢 (API Billed)",
        "latency": f"{latency_ms} ms",
        "answer": ai_generated_answer
    }


@app.get("/stats")
async def get_analytics():
    """Live Dashboard Metrics Endpoint"""
    hit_rate = 0
    if stats["total_requests"] > 0:
        hit_rate = round((stats["cache_hits"] / stats["total_requests"]) * 100, 2)
        
    return {
        "total_requests_processed": stats["total_requests"],
        "cache_hits": stats["cache_hits"],
        "cache_misses": stats["cache_misses"],
        "cache_hit_rate": f"{hit_rate}%",
        "estimated_money_saved": f"${round(stats['total_dollars_saved'], 4)}"
    }
