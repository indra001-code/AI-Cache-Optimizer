import streamlit as st
import time
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

st.set_page_config(page_title="AI Cache Optimizer", page_icon="⚡", layout="wide")

@st.cache_resource
def load_ai_system():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.PersistentClient(path="./chroma_db_live")
    collection = chroma_client.get_or_create_collection(name="ai_cache")
    return model, collection

embed_model, db_collection = load_ai_system()

# Groq API Key Secrets se uthayenge
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

if "stats" not in st.session_state:
    st.session_state.stats = {"total_req": 0, "hits": 0, "misses": 0, "saved": 0.0}
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("⚡ AI Chat with Semantic Caching (Groq Powered)")
st.markdown("High-speed, quota-free AI caching system deployed on Streamlit Cloud!")

st.sidebar.header("📊 Live Optimization Stats")
st.sidebar.metric("Total Requests Processed", st.session_state.stats["total_req"])
st.sidebar.metric("Cache Hit Rate 🎯", f"{st.session_state.stats['hits']} Hits")
st.sidebar.metric("Estimated Money Saved 💰", f"${st.session_state.stats['saved']:.4f}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "status" in msg:
            st.caption(msg['status'])

if prompt := st.chat_input("Ask a question to the AI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not GROQ_API_KEY:
            st.error("⚠️ Groq API Key missing! Please add it in Streamlit Secrets.")
            st.stop()
            
        with st.spinner("Processing your query..."):
            start_time = time.time()
            st.session_state.stats["total_req"] += 1
            
            query_vector = embed_model.encode(prompt).tolist()
            results = db_collection.query(query_embeddings=[query_vector], n_results=1)
            
            ai_answer = None
            status_msg = ""
            
            # Check Cache Hit
            if results['distances'] and len(results['distances'][0]) > 0 and results['distances'][0][0] < 0.45:
                cached_doc = results['documents'][0][0]
                if "API Error" not in cached_doc:
                    ai_answer = cached_doc
                    latency = round((time.time() - start_time) * 1000, 2)
                    st.session_state.stats["hits"] += 1
                    st.session_state.stats["saved"] += 0.002
                    status_msg = f"⚡ CACHE HIT | Latency: {latency} ms"

            # Cache Miss - Real Groq API Call
            if not ai_answer:
                try:
                    chat_completion = groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.3-70b-versatile", # Extremely fast and smart model
                    )
                    ai_answer = chat_completion.choices[0].message.content
                    
                    db_collection.add(
                        embeddings=[query_vector],
                        documents=[ai_answer],
                        metadatas=[{"query": prompt}],
                        ids=[str(uuid.uuid4())]
                    )
                    
                    latency = round((time.time() - start_time) * 1000, 2)
                    st.session_state.stats["misses"] += 1
                    status_msg = f"🐢 CACHE MISS | Latency: {latency} ms"
                    
                except Exception as e:
                    ai_answer = f"⚠️ API Error: {str(e)}"
                    status_msg = "❌ API Error"

            st.markdown(ai_answer)
            st.caption(status_msg)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ai_answer, 
                "status": status_msg
            })
            
            st.rerun()
