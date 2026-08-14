import streamlit as st
import requests

# Aapka FastAPI Backend URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Cache Optimizer", page_icon="⚡", layout="wide")

st.title("⚡ AI Chat with Semantic Caching")
st.markdown("ChatGPT-style interface connected to your FastAPI backend!")

# ================= CHAT HISTORY MEMORY =================
# Streamlit reload hone par memory clear na ho, isliye session_state use karte hain
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= SIDEBAR (LIVE STATS) =================
st.sidebar.header("📊 Live Optimization Stats")
try:
    stats_response = requests.get(f"{API_URL}/stats")
    if stats_response.status_code == 200:
        stats = stats_response.json()
        
        st.sidebar.metric(label="Total Requests Processed", value=stats["total_requests_processed"])
        st.sidebar.metric(label="Cache Hit Rate 🎯", value=stats["cache_hit_rate"])
        st.sidebar.metric(label="Estimated Money Saved 💰", value=stats["estimated_money_saved"])
        
        st.sidebar.divider()
        st.sidebar.markdown(f"**Cache Hits:** {stats['cache_hits']}")
        st.sidebar.markdown(f"**Cache Misses:** {stats['cache_misses']}")
except Exception as e:
    st.sidebar.warning("⚠️ FastAPI Backend is not running! Please start Uvicorn first.")

# ================= PICHALI CHAT HISTORY DIKHAYEIN =================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Agar AI ka reply hai toh uske niche Latency aur Hit/Miss status bhi dikhayein
        if "status" in message:
            if "HIT" in message["status"]:
                st.caption(f"⚡ {message['status']} | Latency: {message.get('latency', 'N/A')}")
            else:
                st.caption(f"🐢 {message['status']} | Latency: {message.get('latency', 'N/A')}")

# ================= USER INPUT BUBBLE =================
# st.chat_input screen ke sabse niche ek input box banata hai
if prompt := st.chat_input("Ask a question to the AI (e.g., What is Quantum Computing?)..."):
    
    # 1. User ka message UI par dikhayein
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. User ke message ko memory mein save karein
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Backend (FastAPI) se reply maangein
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(f"{API_URL}/ask", params={"query": prompt})
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    status = data["status"]
                    latency = data.get("latency", "N/A")
                    
                    # AI ka answer UI par dikhayein
                    st.markdown(answer)
                    
                    # Status (Cache Hit/Miss) color aur icon ke sath dikhayein
                    if "HIT" in status:
                        st.caption(f"⚡ {status} | Latency: {latency}")
                    else:
                        st.caption(f"🐢 {status} | Latency: {latency}")
                    
                    # AI ke reply ko memory mein save karein
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "status": status,
                        "latency": latency
                    })
                    
                    # Page ko turant refresh karein taaki Sidebar ke Stats update ho jayein
                    st.rerun()
                    
                else:
                    st.error("❌ Failed to get response from Backend.")
            except Exception as e:
                st.error("❌ Cannot connect to FastAPI server. Ensure it is running on port 8000.")