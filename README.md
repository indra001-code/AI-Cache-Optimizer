# ⚡ NEBULYN — AI-Powered Semantic Cache

NEBULYN is a smart caching engine built for AI and LLM applications. It speeds up your API responses, reduces latency (delay), and saves API costs and tokens when similar questions are asked repeatedly.

---

## 🚀 How It Works

1. **User Query**: When you ask a question in the chat, NEBULYN first checks its local memory (cache).
2. **Cache Hit (<15ms)**: If a similar question was asked before, it instantly returns the answer without calling the AI API.
3. **Cache Miss (Groq API)**: If it's a new question, it calls the **Groq LLaMA 3.1** API, fetches the real answer, and saves it in the cache for future use.

---

## ✨ Features

- **ChatGPT-Style UI**: A completely clean and modern chat interface.
- **Persistent Memory**: Your chat history and cache are not lost when you refresh the page (saved safely in `nebulyn_memory.pkl`).
- **Live Latency Graph**: Real-time graph showing how fast the Cache (Green) is compared to direct LLM calls (Red).
- **Groq LLaMA 3.1 Powered**: Uses the lightning-fast LLaMA 3.1 model for rapid responses.

---

## 🛠️ Installation & Setup Guide

### 1. Install Requirements
Run the following command in your terminal:
```bash
pip install -r requirements.txt
2. Set Your API Key
Insert your Groq API Key inside the app.py file in this variable:
GROQ_API_KEY = "gsk_Your_Real_API_Key_Here"

3. Run the App
streamlit run app.py
