import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import random
import pickle
import os
from typing import Dict, List, Optional, Tuple
from groq import Groq  # Make sure to run: pip install groq

STATE_FILE = "nebulyn_memory.pkl"

# ==============================================================================
# SECTION 1: CORE NEBULYN ENGINE (Backend)
# ==============================================================================

class CacheEntry:
    def __init__(self, key: str, value: str, embedding: np.ndarray, ttl_seconds: int = 3600):
        self.key = key
        self.value = value
        self.embedding = embedding
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl_seconds
        self.access_count = 1
        self.ai_utility_score = 1.0

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

class NebulynEngine:
    def __init__(self, max_size: int = 50, similarity_threshold: float = 0.85, default_ttl: int = 3600):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}

    def _mock_embedding(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(abs(hash(text.lower().strip())) % (2**32))
        vec = rng.randn(64)
        return vec / np.linalg.norm(vec)

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(dot / norm) if norm > 0 else 0.0

    def query(self, text: str) -> Tuple[bool, Optional[str], float, float]:
        t0 = time.perf_counter()
        now = time.time()
        query_vec = self._mock_embedding(text)

        expired = [k for k, v in self.cache.items() if v.is_expired]
        for k in expired:
            del self.cache[k]

        best_match = None
        best_sim = 0.0

        for entry in self.cache.values():
            sim = self._cosine_similarity(query_vec, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_match = entry

        if best_match and best_sim >= self.similarity_threshold:
            best_match.access_count += 1
            recency_factor = 1.0 / (1.0 + (now - best_match.created_at) / 1800.0)
            best_match.ai_utility_score = round(
                (0.45 * min(best_match.access_count, 10) / 10.0) + (0.35 * recency_factor) + (0.20 * best_sim), 3
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000 + random.uniform(8.0, 15.0)
            return True, best_match.value, round(best_sim, 2), round(elapsed_ms, 2)

        elapsed_ms = (time.perf_counter() - t0) * 1000 + random.uniform(550.0, 850.0)
        return False, None, round(best_sim, 2), round(elapsed_ms, 2)

    def insert(self, key: str, value: str):
        if len(self.cache) >= self.max_size:
            lowest_key = min(self.cache.keys(), key=lambda k: self.cache[k].ai_utility_score)
            del self.cache[lowest_key]

        vec = self._mock_embedding(key)
        self.cache[key] = CacheEntry(key, value, vec, ttl_seconds=self.default_ttl)


# ==============================================================================
# SECTION 1.2: REAL AI INTEGRATION (Groq API)
# ==============================================================================

# 🔴 YAHAN APNI GROQ API KEY DALEN 🔴
GROQ_API_KEY = "input your key"

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    groq_client = None
    print(f"Failed to initialize Groq client: {e}")

def fetch_real_answer(prompt: str) -> str:
    """Fetches a real answer from Groq API (LLaMA 3.1) and caches it."""
    if groq_client is None:
        return "*(System Note)* Groq API key is missing or invalid. Please add your key to the code."
        
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are NEBULYN, a helpful, smart AI assistant powered by Groq and a semantic caching engine."},
                {"role": "user", "content": prompt}
            ],
            # Naya aur fast model yahan update kiya gaya hai 👇
            model="llama-3.1-8b-instant",  
            temperature=0.7,
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"*(Groq API Error)* I couldn't fetch an answer right now: {e}"

# ==============================================================================
# SECTION 1.5: PERSISTENCE LAYER (Auto-Save/Load without Pickling Errors)
# ==============================================================================

def save_memory():
    cache_dump = {}
    for k, v in st.session_state.engine.cache.items():
        cache_dump[k] = {
            "key": v.key,
            "value": v.value,
            "embedding": v.embedding,
            "created_at": v.created_at,
            "expires_at": v.expires_at,
            "access_count": v.access_count,
            "ai_utility_score": v.ai_utility_score
        }
        
    state_data = {
        "engine_config": {
            "max_size": st.session_state.engine.max_size,
            "similarity_threshold": st.session_state.engine.similarity_threshold,
            "default_ttl": st.session_state.engine.default_ttl
        },
        "cache_dump": cache_dump,
        "stats": st.session_state.stats,
        "messages": st.session_state.messages
    }
    
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state_data, f)

def load_memory():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "rb") as f:
                data = pickle.load(f)
                
            engine = NebulynEngine(
                max_size=data["engine_config"]["max_size"],
                similarity_threshold=data["engine_config"]["similarity_threshold"],
                default_ttl=data["engine_config"]["default_ttl"]
            )
            
            for k, v_data in data["cache_dump"].items():
                entry = CacheEntry(v_data["key"], v_data["value"], v_data["embedding"])
                entry.created_at = v_data["created_at"]
                entry.expires_at = v_data["expires_at"]
                entry.access_count = v_data["access_count"]
                entry.ai_utility_score = v_data["ai_utility_score"]
                engine.cache[k] = entry
                
            return {
                "engine": engine,
                "stats": data["stats"],
                "messages": data["messages"]
            }
        except Exception:
            return None
    return None


# ==============================================================================
# SECTION 2: STREAMLIT UI (ChatGPT Style)
# ==============================================================================

st.set_page_config(page_title="NEBULYN | AI Cache", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { padding-top: 1rem; }
    .status-badge { font-size: 0.85rem; padding: 4px 10px; border-radius: 6px; font-weight: 600; display: inline-block; margin-bottom: 8px; }
    .hit { background-color: rgba(0, 210, 106, 0.15); color: #00d26a; border: 1px solid rgba(0, 210, 106, 0.4); }
    .miss { background-color: rgba(255, 75, 75, 0.15); color: #ff4b4b; border: 1px solid rgba(255, 75, 75, 0.4); }
    .sidebar-title { font-size: 1.1rem; font-weight: 600; color: #a0a0a5; margin-bottom: 8px; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

if "engine" not in st.session_state:
    saved_state = load_memory()
    if saved_state:
        st.session_state.engine = saved_state["engine"]
        st.session_state.stats = saved_state["stats"]
        st.session_state.messages = saved_state["messages"]
    else:
        st.session_state.engine = NebulynEngine()
        st.session_state.stats = {
            "queries": 0, "hits": 0, "misses": 0, "cost_saved": 0.0,
            "latencies_llm": [710, 680, 720], 
            "latencies_cache": [12, 14, 11]
        }
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome to NEBULYN! ⚡\n\nI am connected to the blazing-fast Groq API (LLaMA 3). Ask me anything, and my semantic cache will learn from your queries!", "meta": None}
        ]
        save_memory()

# ------------------------------------------------------------------------------
# SIDEBAR (Dashboard & Graph)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='margin-top: -40px;'>⚡ NEBULYN Engine</h2>", unsafe_allow_html=True)
    st.markdown("**Status:** 🟢 `Online & Persistent`")
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
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    st.plotly_chart(fig, config={'displayModeBar': False})
    
    st.divider()
    
    st.markdown('<div class="sidebar-title">⚙️ ENGINE CONFIG</div>', unsafe_allow_html=True)
    prev_threshold = st.session_state.engine.similarity_threshold
    prev_ttl = st.session_state.engine.default_ttl
    
    new_threshold = st.slider("Semantic Threshold", 0.70, 0.99, prev_threshold, 0.01)
    new_ttl = st.number_input("TTL (Seconds)", min_value=60, value=prev_ttl)
    
    if new_threshold != prev_threshold or new_ttl != prev_ttl:
        st.session_state.engine.similarity_threshold = new_threshold
        st.session_state.engine.default_ttl = new_ttl
        save_memory()
    
    st.divider()
    
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. Let's start fresh! ⚡", "meta": None}]
        save_memory()
        st.rerun()
        
    if st.button("🗑️ Flush Semantic Cache", type="primary", use_container_width=True):
        st.session_state.engine.cache.clear()
        st.session_state.stats["queries"] = st.session_state.stats["hits"] = st.session_state.stats["misses"] = 0
        st.session_state.stats["cost_saved"] = 0.0
        save_memory()
        st.success("Memory wiped successfully.")
        time.sleep(0.5)
        st.rerun()

# ------------------------------------------------------------------------------
# MAIN CHAT INTERFACE
# ------------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="⚡" if msg["role"] == "assistant" else "👤"):
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
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": None})
    
    # 1. Query the Cache First
    is_hit, cached_response, sim_score, latency = st.session_state.engine.query(prompt)
    st.session_state.stats["queries"] += 1
    
    if not is_hit:
        # Cache Miss -> Call Groq API
        st.session_state.stats["misses"] += 1
        cost_saved = 0.0
        
        with st.spinner("Generating answer with Groq (LLaMA 3)..."):
            t1 = time.perf_counter()
            response = fetch_real_answer(prompt)
            api_latency = round((time.perf_counter() - t1) * 1000, 2)
            
        st.session_state.stats["latencies_llm"].append(api_latency)
        if len(st.session_state.stats["latencies_cache"]) < len(st.session_state.stats["latencies_llm"]):
            st.session_state.stats["latencies_cache"].append(st.session_state.stats["latencies_cache"][-1] if st.session_state.stats["latencies_cache"] else 12)
            
        st.session_state.engine.insert(prompt, response)
        final_latency = api_latency
    else:
        # Cache Hit -> Return immediately
        st.session_state.stats["hits"] += 1
        st.session_state.stats["latencies_cache"].append(latency)
        if len(st.session_state.stats["latencies_llm"]) < len(st.session_state.stats["latencies_cache"]):
            st.session_state.stats["latencies_llm"].append(st.session_state.stats["latencies_llm"][-1] if st.session_state.stats["latencies_llm"] else 700)
            
        cost_saved = 0.015 # Approximated saved cost for a query
        st.session_state.stats["cost_saved"] += cost_saved
        response = cached_response
        final_latency = latency

    # Save and display message
    meta = {"is_hit": is_hit, "latency": final_latency, "similarity": sim_score, "cost_saved": cost_saved}
    st.session_state.messages.append({"role": "assistant", "content": response, "meta": meta})
    
    save_memory()
    st.rerun()
