# ⚡ AI API Semantic Cache & Cost Optimizer

A full-stack AI web application that significantly reduces LLM API costs and response latency using **Semantic Caching**. 

Instead of sending every user query to the paid AI API (like Google Gemini or OpenAI), this system first converts the query into vector embeddings and searches a local Vector Database (ChromaDB) for similar past questions. If a similar question (e.g., >90% match) was asked before, it instantly returns the cached answer.

## 🚀 Key Features
* **Semantic Caching:** Understands the *meaning* of a question, not just exact keywords. ("What is AI?" and "Explain Artificial Intelligence?" will hit the same cache).
* **Cost Tracking:** Live dashboard showing how many API requests were bypassed and the estimated money saved.
* **Ultra-Fast Responses:** Cache hits return answers in milliseconds (~40ms) compared to actual API calls (~1500ms).
* **ChatGPT-Style UI:** Beautiful, continuous chat interface built with Streamlit.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Backend:** FastAPI, Uvicorn
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`)
* **LLM:** Google Gemini 3.5 Flash

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME
