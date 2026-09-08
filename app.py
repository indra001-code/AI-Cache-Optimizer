import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 90

st.set_page_config(
    page_title="AI Cache Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def get_json(path: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=5)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as error:
        return None, str(error)


def render_stats() -> None:
    stats, error = get_json("/stats")
    if error:
        st.sidebar.error("Backend offline")
        st.sidebar.caption("Start FastAPI with the command in run.")
        return
    st.sidebar.success("Backend connected")
    st.sidebar.metric("Requests", stats["total_requests"])
    st.sidebar.metric("Cache hit rate", f'{stats["cache_hit_rate"]}%')
    st.sidebar.metric("Estimated savings", f'${stats["estimated_money_saved"]:.4f}')
    st.sidebar.divider()
    st.sidebar.caption(f'Cache hits: {stats["cache_hits"]}')
    st.sidebar.caption(f'Cache misses: {stats["cache_misses"]}')


st.title("⚡ AI Cache Optimizer")
st.caption("Ask questions faster and reduce repeated AI API costs with semantic caching.")

with st.sidebar:
    st.header("Live optimization")
    render_stats()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("Clear server cache", use_container_width=True):
        try:
            response = requests.delete(f"{API_URL}/cache", timeout=10)
            response.raise_for_status()
            st.success("Server cache cleared.")
        except requests.RequestException:
            st.error("Could not clear the server cache.")

if not st.session_state.messages:
    st.info("Ask a question to get started. Repeated or similar questions are served from cache.")
    st.markdown("**Try:** quantum computing · explain REST APIs · what is semantic caching?")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("status"):
            badge = "⚡ Cache hit" if message["status"] == "CACHE_HIT" else "🌐 API call"
            st.caption(f'{badge} · {message.get("latency_ms", 0):.0f} ms')

if prompt := st.chat_input("Ask the AI a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"query": prompt},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                if response.status_code == 422:
                    st.error("Please enter a non-empty question under 4,000 characters.")
                else:
                    response.raise_for_status()
                    data = response.json()
                    st.markdown(data["answer"])
                    st.caption(
                        f'{"⚡ Cache hit" if data["status"] == "CACHE_HIT" else "🌐 API call"}'
                        f' · {data["latency_ms"]:.0f} ms'
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": data["answer"],
                            "status": data["status"],
                            "latency_ms": data["latency_ms"],
                        }
                    )
            except requests.Timeout:
                st.error("The request timed out. Please try again.")
            except requests.ConnectionError:
                st.error("Cannot connect to FastAPI. Start the backend and try again.")
            except requests.HTTPError:
                st.error("The backend could not complete this request. Please try again.")
            except (KeyError, ValueError):
                st.error("The backend returned an invalid response.")
    st.rerun()
