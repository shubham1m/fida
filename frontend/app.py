"""
Streamlit application entrypoint.

Composes the upload and chat components and owns the sidebar settings
(top_k, temperature) that both depend on. Each component is responsible
for its own rendering logic; this file only wires them together.
"""
import os

import streamlit as st
from dotenv import load_dotenv

from frontend.components.chat import ChatComponent
from frontend.components.upload import UploadComponent

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Financial Document Intelligence Assistant", layout="wide")
st.title("📊 Financial Document Intelligence Assistant")

upload_component = UploadComponent(backend_url=BACKEND_URL)
upload_component.render()

st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top K chunks", min_value=1, max_value=10, value=5)
factual_mode = st.sidebar.toggle("Factual mode (temperature=0)", value=True)
temperature = 0.0 if factual_mode else 0.7
source_filter = st.sidebar.text_input("Filter by source filename (optional)") or None

chat_component = ChatComponent(backend_url=BACKEND_URL)
chat_component.render(top_k=top_k, temperature=temperature, source_filter=source_filter)
