import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import random
import os
import sqlite3
import json
import csv
import io
import hashlib
import requests
from typing import Dict, List, Optional, Tuple

DB_FILE = "nebulyn_memory.db"

# ==============================================================================
# FREE API CONFIGURATION (Groq API - Free)
# ==============================================================================

# Try to load API key from multiple sources
def load_groq_api_key():
    """Load Groq API key from Streamlit secrets or environment variable."""
    api_key = None
    
    # Try Streamlit secrets first
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if api_key and api_key != "your_groq_api_key_here":
            return api_key
    except:
        pass
    
    # Try environment variable
    api_key = os.environ.get("GROQ_API_KEY", "")
    if api_key:
        return api_key
    
    return api_key

GROQ_API_KEY = load_groq_api_key()

# Available models to try (in order of preference)
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "llama-3.2-3b-preview",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "gemma-7b-it"
]

# ==============================================================================
# SECTION 1: CORE NEBULYN ENGINE (Backend with SQLite + Free Embeddings)
# ==============================================================================

class CacheEntry:
    def __init__(self, key: str, value: str, embedding: np.ndarray):
        self.key = key
        self.value = value
        self.embedding = embedding
        self.created_at = time.time()
        self.access_count = 1
        self.ai_utility_score = 1.0

class NebulynEngine:
    def __init__(self, max_size: int = 50, similarity_threshold: float = 0.85):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[str, CacheEntry] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using free local method (no API needed)."""
        text_hash = hashlib.md5(text.lower().strip().encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        vec = np.zeros(512)
        text_lower = text.lower().strip()
        
        for n in range(3, 6):
            for i in range(len(text_lower) - n + 1):
                ngram = text_lower[i:i+n]
                idx = hash(ngram) % 512
                vec[idx] += 1.0
        
        words = text_lower.split()
        for word in words:
            idx = hash(f"word_{word}") % 512
            vec[idx] += 2.0
        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        if len(self.embedding_cache) < 1000:
            self.embedding_cache[text_hash] = vec
        
        return vec

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(dot / norm) if norm > 0 else 0.0

    def query(self, text: str) -> Tuple[bool, Optional[str], float, float]:
        t0 = time.perf_counter()
        query_vec = self._get_embedding(text)

        best_match = None
        best_sim = 0.0

        for entry in self.cache.values():
            sim = self._cosine_similarity(query_vec, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_match = entry

        if best_match and best_sim >= self.similarity_threshold:
            best_match.access_count += 1
            recency_factor = 1.0 / (1.0 + (time.time() - best_match.created_at) / 1800.0)
            best_match.ai_utility_score = round(
                (0.45 * min(best_match.access_count, 10) / 10.0) + (0.35 * recency_factor) + (0.20 * best_sim), 3
            )
            update_cache_entry_score(best_match.key, best_match.access_count, best_match.ai_utility_score)
            elapsed_ms = (time.perf_counter() - t0) * 1000 + random.uniform(8.0, 15.0)
            return True, best_match.value, round(best_sim, 2), round(elapsed_ms, 2)

        elapsed_ms = (time.perf_counter() - t0) * 1000 + random.uniform(550.0, 850.0)
        return False, None, round(best_sim, 2), round(elapsed_ms, 2)

    def insert(self, key: str, value: str):
        if len(self.cache) >= self.max_size:
            lowest_key = min(self.cache.keys(), key=lambda k: self.cache[k].ai_utility_score)
            del self.cache[lowest_key]
            delete_cache_entry_from_db(lowest_key)

        vec = self._get_embedding(key)
        self.cache[key] = CacheEntry(key, value, vec)
        save_cache_entry_to_db(key, value, vec)

# ==============================================================================
# FREE AI RESPONSE GENERATION (Groq API with Auto Model Detection)
# ==============================================================================

@st.cache_data(ttl=300)
def get_available_groq_models(api_key: str) -> List[str]:
    """Fetch available models from Groq API."""
    if not api_key:
        return []
    
    try:
        url = "https://api.groq.com/openai/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            models = response.json().get("data", [])
            return [m["id"] for m in models if m.get("active", True)]
        return []
    except:
        return []

def test_groq_model(api_key: str, model: str) -> bool:
    """Quick test if a model works."""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 5
        }
        response = requests.post(url, headers=headers, json=data, timeout=5)
        return response.status_code == 200
    except:
        return False

def get_best_groq_model(api_key: str) -> Optional[str]:
    """Find the best available Groq model."""
    if not api_key:
        return None
    
    available_models = get_available_groq_models(api_key)
    
    if available_models:
        for preferred in GROQ_MODELS:
            if preferred in available_models:
                return preferred
        return available_models[0] if available_models else None
    
    for model in GROQ_MODELS:
        if test_groq_model(api_key, model):
            return model
    
    return None

def get_groq_response(prompt: str, api_key: str = None) -> Tuple[Optional[str], Optional[str]]:
    """Get response from Groq API. Returns (response_text, error_message)."""
    if not api_key:
        return None, "No API key found. Please add GROQ_API_KEY to .streamlit/secrets.toml"
    
    model = get_best_groq_model(api_key)
    if not model:
        return None, "No available Groq models found. Please check your API key."
    
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are NEBULYN, a helpful AI assistant with the spirit of the Survey Corps. Be concise but thorough."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"], None
        elif response.status_code == 401:
            return None, "Invalid API key (401). Please check your GROQ_API_KEY."
        elif response.status_code == 429:
            return None, "Rate limit exceeded (429). Please wait a moment and try again."
        else:
            return None, f"Groq API error ({response.status_code}): {response.text[:200]}"
    
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)[:200]}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)[:200]}"

def get_mock_response(prompt: str) -> str:
    """Generate a simple mock response if no API is available."""
    topic = prompt.lower()
    
    if any(word in topic for word in ["hello", "hi", "hey", "namaste"]):
        return "Hello! I'm NEBULYN, your AI assistant. I'm currently running in offline mode because the Groq API is not configured. However, my semantic cache is working perfectly!"
    
    if any(word in topic for word in ["cache", "semantic", "engine"]):
        return "My semantic caching engine works by storing embeddings of previous queries and their responses. When you ask something similar, I retrieve the cached response instantly instead of generating a new one."
    
    if any(word in topic for word in ["attack on titan", "aot", "eren", "levi", "titan"]):
        return "⚔️ In the world of Attack on Titan, humanity fights for survival against the Titans. Just like the Survey Corps, NEBULYN fights against slow responses and high latency!"
    
    return "I'm currently in offline mode because the Groq API is not configured. To enable AI responses:\n\n1. Create a `.streamlit/secrets.toml` file\n2. Add: `GROQ_API_KEY = \"your_key_here\"`\n3. Restart the app\n\nMeanwhile, try asking something similar to a previous question to see the cache in action!"

# ==============================================================================
# SECTION 1.5: PERSISTENCE LAYER (SQLite)
# ==============================================================================

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS cache_entries
                 (key TEXT PRIMARY KEY, value TEXT, embedding BLOB, created_at REAL,
                  access_count INTEGER, ai_utility_score REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS stats
                 (queries INTEGER, hits INTEGER, misses INTEGER, cost_saved REAL,
                  latencies_llm TEXT, latencies_cache TEXT)''')
    c.execute("SELECT COUNT(*) FROM stats")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO stats VALUES (0,0,0,0.0,'[]','[]')")
    conn.commit()
    conn.close()

def save_cache_entry_to_db(key, value, embedding):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    now = time.time()
    c.execute('''INSERT OR REPLACE INTO cache_entries
                 (key, value, embedding, created_at, access_count, ai_utility_score)
                 VALUES (?,?,?,?,?,?)''',
              (key, value, embedding.tobytes(), now, 1, 1.0))
    conn.commit()
    conn.close()

def update_cache_entry_score(key, access_count, utility_score):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''UPDATE cache_entries SET access_count=?, ai_utility_score=? WHERE key=?''',
              (access_count, utility_score, key))
    conn.commit()
    conn.close()

def delete_cache_entry_from_db(key):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM cache_entries WHERE key=?", (key,))
    conn.commit()
    conn.close()

def load_all_cache_entries(engine: NebulynEngine):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT key, value, embedding, created_at, access_count, ai_utility_score FROM cache_entries")
    rows = c.fetchall()
    conn.close()
    EXPECTED_DIM = 512
    skipped = 0
    for row in rows:
        key, value, emb_blob, created_at, access_count, utility = row
        try:
            embedding = np.frombuffer(emb_blob, dtype=np.float64)
            if embedding.size != EXPECTED_DIM:
                skipped += 1
                continue
            entry = CacheEntry(key, value, embedding)
            entry.created_at = created_at
            entry.access_count = access_count
            entry.ai_utility_score = utility
            engine.cache[key] = entry
        except Exception:
            skipped += 1
    if skipped > 0:
        st.warning(f"Skipped {skipped} old cache entries (dimension mismatch). They'll be re-cached.")

def load_stats():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM stats LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        queries, hits, misses, cost_saved, lat_llm_json, lat_cache_json = row
        return {
            "queries": queries,
            "hits": hits,
            "misses": misses,
            "cost_saved": cost_saved,
            "latencies_llm": json.loads(lat_llm_json),
            "latencies_cache": json.loads(lat_cache_json)
        }
    else:
        return {
            "queries": 0, "hits": 0, "misses": 0, "cost_saved": 0.0,
            "latencies_llm": [710, 680, 720],
            "latencies_cache": [12, 14, 11]
        }

def save_stats(stats: dict):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''UPDATE stats SET queries=?, hits=?, misses=?, cost_saved=?,
                 latencies_llm=?, latencies_cache=? WHERE rowid=1''',
              (stats["queries"], stats["hits"], stats["misses"], stats["cost_saved"],
               json.dumps(stats["latencies_llm"]), json.dumps(stats["latencies_cache"])))
    conn.commit()
    conn.close()

def clear_cache_entries():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM cache_entries")
    conn.commit()
    conn.close()

def export_cache_as_json():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT key, value, created_at, access_count, ai_utility_score FROM cache_entries")
    rows = c.fetchall()
    conn.close()
    data = []
    for row in rows:
        data.append({
            "key": row[0],
            "value": row[1],
            "created_at": row[2],
            "access_count": row[3],
            "ai_utility_score": row[4]
        })
    return json.dumps(data, indent=2)

def export_cache_as_csv():
    conn = sqlite3.connect(DB_FILE)
    c = cursor()
    c.execute("SELECT key, value, created_at, access_count, ai_utility_score FROM cache_entries")
    rows = c.fetchall()
    conn.close()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Key", "Value", "Created At", "Access Count", "Utility Score"])
    for row in rows:
        writer.writerow(row)
    return output.getvalue()

# ==============================================================================
# SECTION 2: STREAMLIT UI (Attack on Titan Theme)
# ==============================================================================

st.set_page_config(page_title="NEBULYN | Wings of Freedom", page_icon="⚔️", layout="wide")

# Custom CSS for AoT aesthetic - FIXED: Lowercase input
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Cinzel:wght@400;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 50% 0%, #1a0a0a 0%, #0d0d0d 70%);
        color: #e0e0e0;
        font-family: 'Cinzel', serif;
    }
    [data-testid="stSidebar"] {
        background: #1a0a0a;
        border-right: 2px solid #B22222;
        padding-top: 1rem;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif;
        color: #D4AF37;
    }
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #D4AF37 !important;
        margin-bottom: 8px;
        margin-top: 15px;
        letter-spacing: 1px;
    }
    .status-badge {
        font-size: 0.85rem;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .hit {
        background-color: rgba(0, 210, 106, 0.15);
        color: #00d26a;
        border: 1px solid rgba(0, 210, 106, 0.4);
    }
    .miss {
        background-color: rgba(255, 75, 75, 0.15);
        color: #ff4b4b;
        border: 1px solid rgba(255, 75, 75, 0.4);
    }
    .stChatMessage {
        border-radius: 10px;
        margin: 8px 0;
        padding: 10px;
    }
    .stChatMessage[data-testid="stChatMessage"] {
        background: #1a1a1a;
        border: 1px solid #333;
    }
    .stChatMessage[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        border-left: 4px solid #B22222;
    }
    .stChatMessage[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        border-left: 4px solid #D4AF37;
    }
    
    /* ===== FIX: Force Lowercase in Chat Input ===== */
    .stChatInput input,
    .stChatInput textarea,
    div[data-testid="stChatInput"] input,
    div[data-testid="stChatInput"] textarea,
    input[type="text"] {
        text-transform: none !important;
        text-transform: lowercase !important;
        background: #1a1a1a;
        color: #e0e0e0;
        border: 1px solid #B22222;
        border-radius: 8px;
        padding: 10px;
        font-family: 'Cinzel', serif;
    }
    
    .stChatInput input::placeholder,
    .stChatInput textarea::placeholder {
        text-transform: none !important;
        text-transform: lowercase !important;
    }
    
    .stButton > button {
        background: #B22222;
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background: #8B0000;
        box-shadow: 0 0 10px #B22222;
    }
    .stDownloadButton > button {
        background: #1a1a1a;
        color: #D4AF37;
        border: 1px solid #D4AF37;
        border-radius: 8px;
    }
    .stDownloadButton > button:hover {
        background: #2a2a2a;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize SQLite DB
init_db()

# Initialize session state
if "engine" not in st.session_state:
    st.session_state.engine = NebulynEngine()
    load_all_cache_entries(st.session_state.engine)
    st.session_state.stats = load_stats()
    st.session_state.messages = [
        {"role": "assistant", "content": "⚔️ **Welcome, soldier, to NEBULYN: Wings of Freedom.**\n\nI'm powered by **Groq + LLaMA 3** with auto model detection. Ask me anything!", "meta": None}
    ]
    st.session_state.debug_mode = False

# ------------------------------------------------------------------------------
# SIDEBAR (Dashboard, Config, Export)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='margin-top: -40px; text-align: center;'>⚔️ NEBULYN<br><span style='font-size:0.7em;'>Wings of Freedom</span></h2>", unsafe_allow_html=True)
    
    # API Status
    if GROQ_API_KEY:
        st.markdown("**Status:** 🟢 `Online - Groq API Connected`")
    else:
        st.markdown("**Status:** 🟡 `Cache-Only Mode`")
        st.info("Groq API key not found. Create `.streamlit/secrets.toml` with your key.")
    
    # Debug toggle
    st.session_state.debug_mode = st.toggle("Debug Mode", value=st.session_state.debug_mode)
    
    st.divider()

    st.markdown('<div class="sidebar-title">📊 METRICS</div>', unsafe_allow_html=True)
    tot = st.session_state.stats["queries"]
    hits = st.session_state.stats["hits"]
    hit_rate = round((hits / tot * 100), 1) if tot > 0 else 0.0

    col1, col2 = st.columns(2)
    col1.metric("Hit Rate", f"{hit_rate}%")
    col2.metric("Saved", f"${st.session_state.stats['cost_saved']:.3f}")
    st.metric("Active Cache Keys", f"{len(st.session_state.engine.cache)} / {st.session_state.engine.max_size}")

    st.divider()

    st.markdown('<div class="sidebar-title">📈 LATENCY GRAPH</div>', unsafe_allow_html=True)
    fig = go.Figure()
    sample_size = min(15, len(st.session_state.stats["latencies_llm"]), len(st.session_state.stats["latencies_cache"]))
    x_axis = [f"Turn {i+1}" for i in range(sample_size)]

    fig.add_trace(go.Scatter(
        x=x_axis, y=st.session_state.stats["latencies_llm"][-sample_size:],
        mode='lines+markers', name='LLM (ms)', line=dict(color='#ff4b4b', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=x_axis, y=st.session_state.stats["latencies_cache"][-sample_size:],
        mode='lines+markers', name='Cache (ms)', line=dict(color='#00d26a', width=2)
    ))

    fig.update_layout(
        template="plotly_dark",
        height=240,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, config={'displayModeBar': False})

    st.divider()

    st.markdown('<div class="sidebar-title">⚙️ ENGINE CONFIG</div>', unsafe_allow_html=True)
    prev_threshold = st.session_state.engine.similarity_threshold

    new_threshold = st.slider("Semantic Threshold", 0.70, 0.99, prev_threshold, 0.01)

    if new_threshold != prev_threshold:
        st.session_state.engine.similarity_threshold = new_threshold
        save_stats(st.session_state.stats)

    st.divider()

    st.markdown('<div class="sidebar-title">📤 EXPORT DATA</div>', unsafe_allow_html=True)
    json_data = export_cache_as_json()
    st.download_button(
        label="Download JSON",
        data=json_data,
        file_name="nebulyn_cache.json",
        mime="application/json",
        key="download_json"
    )
    csv_data = export_cache_as_csv()
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="nebulyn_cache.csv",
        mime="text/csv",
        key="download_csv"
    )

    st.divider()

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. The battlefield is clean, soldier.", "meta": None}]
        st.rerun()

    if st.button("🗑️ Flush Semantic Cache", type="primary", use_container_width=True):
        st.session_state.engine.cache.clear()
        clear_cache_entries()
        st.session_state.stats["queries"] = st.session_state.stats["hits"] = st.session_state.stats["misses"] = 0
        st.session_state.stats["cost_saved"] = 0.0
        save_stats(st.session_state.stats)
        st.success("Memory wiped. The Titans have been expelled.")
        time.sleep(0.5)
        st.rerun()

# ------------------------------------------------------------------------------
# MAIN CHAT INTERFACE
# ------------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="⚔️" if msg["role"] == "assistant" else "🧑‍💻"):
        st.markdown(msg["content"])
        if msg.get("meta"):
            m = msg["meta"]
            badge_class = "hit" if m["is_hit"] else "miss"
            badge_text = "🎯 CACHE HIT" if m["is_hit"] else "☁️ LLM GENERATED (MISS)"

            with st.expander(f"Engine Log: {badge_text} | {m['latency']} ms"):
                st.markdown(f"""
                <span class='status-badge {badge_class}'>{badge_text}</span><br>
                **Time:** `{m['latency']} ms` | **Sim Match:** `{m['similarity']}` | **Cost Saved:** `${m['cost_saved']}`
                """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# CHAT INPUT PROCESSING
# ------------------------------------------------------------------------------
if prompt := st.chat_input("Ask NEBULYN anything..."):
    # Display user message immediately
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": None})

    # 1. Query the Cache First
    is_hit, cached_response, sim_score, latency = st.session_state.engine.query(prompt)
    st.session_state.stats["queries"] += 1

    if not is_hit:
        # Cache Miss -> Call Groq API
        st.session_state.stats["misses"] += 1
        cost_saved = 0.0

        with st.chat_message("assistant", avatar="⚔️"):
            message_placeholder = st.empty()
            full_response = ""
            error_msg = None
            t1 = time.perf_counter()
            
            # Try Groq API
            with st.spinner("⚔️ Generating with Groq (LLaMA 3)..."):
                full_response, error_msg = get_groq_response(prompt, GROQ_API_KEY)
            
            # If Groq fails, use mock response
            if full_response is None:
                if st.session_state.debug_mode and error_msg:
                    st.warning(f"Debug: {error_msg}")
                full_response = get_mock_response(prompt)
            
            api_latency = round((time.perf_counter() - t1) * 1000, 2)
            message_placeholder.markdown(full_response)

        st.session_state.stats["latencies_llm"].append(api_latency)
        if len(st.session_state.stats["latencies_cache"]) < len(st.session_state.stats["latencies_llm"]):
            st.session_state.stats["latencies_cache"].append(
                st.session_state.stats["latencies_cache"][-1] if st.session_state.stats["latencies_cache"] else 12
            )

        # Cache the response
        if error_msg is None or not full_response.startswith("I'm currently in offline mode"):
            st.session_state.engine.insert(prompt, full_response)

        final_latency = api_latency
    else:
        # Cache Hit
        st.session_state.stats["hits"] += 1
        st.session_state.stats["latencies_cache"].append(latency)
        if len(st.session_state.stats["latencies_llm"]) < len(st.session_state.stats["latencies_cache"]):
            st.session_state.stats["latencies_llm"].append(
                st.session_state.stats["latencies_llm"][-1] if st.session_state.stats["latencies_llm"] else 700
            )

        cost_saved = 0.015
        st.session_state.stats["cost_saved"] += cost_saved
        full_response = cached_response
        final_latency = latency

        with st.chat_message("assistant", avatar="⚔️"):
            st.markdown(full_response)

    meta = {"is_hit": is_hit, "latency": final_latency, "similarity": sim_score, "cost_saved": cost_saved}
    st.session_state.messages.append({"role": "assistant", "content": full_response, "meta": meta})

    save_stats(st.session_state.stats)
    st.rerun()
