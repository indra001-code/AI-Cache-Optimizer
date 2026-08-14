# ⚡ NEBULYN — AI-Powered Semantic Cache

NEBULYN ek smart caching engine hai jo AI/LLM apps ke liye banaya gaya hai. Yeh aapke API calls ko fast banata hai, latency (delay) ko kam karta hai, aur bar-bar same sawal poochhne par paise/tokens bachata hai.

---

## 🚀 Yeh Kaam Kaise Karta hai?

1. **User Query**: Jab aap chat mein koi sawal poochhte hain, toh NEBULYN sabse pehle apni local memory (cache) mein check karta hai.
2. **Cache Hit (<15ms)**: Agar waisa milta-julta sawal pehle poochha gaya tha, toh yeh bina AI API ko call kiye turant wahi jawab de deta hai.
3. **Cache Miss (Groq API)**: Agar naya sawal hai, toh yeh **Groq LLaMA 3.1** API ko call karta hai, real answer fetch karta hai, aur aage ke liye cache mein save kar leta hai.

---

## ✨ Features

- **ChatGPT-Style UI**: Ekdum clean aur modern chat interface.
- **Persistent Memory**: Page refresh karne par bhi aapka chat history aur cache delete nahi hota (`nebulyn_memory.pkl` mein save rehta hai).
- **Live Latency Graph**: Real-time graph dikhata hai ki Cache (Green) kitna fast hai direct LLM (Red) ke mukable.
- **Groq LLaMA 3.1 Powered**: Lightning-fast speed ke liye LLaMA 3.1 model ka use.

---

## 🛠️ Kaise Run Karein (Installation Guide)

### 1. Requirements Install Karein
Apne terminal mein yeh command chalayein:
```bash

pip install -r requirements.txt
2. Set Your API Key
Insert your Groq API Key inside the app.py file in this variable:
GROQ_API_KEY = "gsk_Your_Real_API_Key_Here"


3. Run the App
streamlit run app.py
