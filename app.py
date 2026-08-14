import streamlit as st
import time
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai

st.set_page_config(page_title="AI Cache Optimizer", page_icon="⚡", layout="wide")

@st.cache_resource
def load_ai_system():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.PersistentClient(path="./chroma_db_live")
    collection = chroma_client.get_or_create_collection(name="ai_cache")
    return model, collection

embed_model, db_collection = load_ai_system()

API_KEY = st.secrets.get("GEMINI_API_KEY", "")

if API_KEY:
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
else:
    gemini_model = None

if "stats" not in st.session_state:
    st.session_state.stats = {"total_req": 0, "hits": 0, "misses": 0, "saved": 0.0}
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("⚡ AI Chat with Semantic Caching")
st.markdown("All-in-One App deployed on Streamlit Cloud!")

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
        if not API_KEY:
            st.error("⚠️ Gemini API Key missing!")
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
                # Agar cache mein pehle se error save hai, toh use ignore karke fresh call karenge
                if "API Error" not in cached_doc and "quota" not in cached_doc.lower():
                    ai_answer = cached_doc
                    latency = round((time.time() - start_time) * 1000, 2)
                    st.session_state.stats["hits"] += 1
                    st.session_state.stats["saved"] += 0.002
                    status_msg = f"⚡ CACHE HIT | Latency: {latency} ms"

            # Agar Cache Hit nahi hua ya error tha, toh real API call karenge
            if not ai_answer:
                try:
                    response = gemini_model.generate_content(prompt)
                    ai_answer = response.text
                    
                    # Sirf successful answer hi database mein save hoga (Error save nahi hoga)
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
                    ai_answer = f"⚠️ API Limit/Error: {str(e)}. Please wait 30 seconds and try again."
                    status_msg = "❌ API Error"

            st.markdown(ai_answer)
            st.caption(status_msg)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ai_answer, 
                "status": status_msg
            })
            
            st.rerun()
