"""
Main chat interface: message history, the question input box, and
dispatching queries to the backend.
"""
import requests
import streamlit as st

from frontend.components.citations import CitationsComponent


class ChatComponent:
    """Owns the conversation state and renders the chat history.

    Conversation history is kept in st.session_state so it survives
    Streamlit reruns (which happen on every widget interaction).
    """

    SESSION_KEY = "chat_history"

    def __init__(self, backend_url: str):
        self._backend_url = backend_url
        self._citations = CitationsComponent()
        if self.SESSION_KEY not in st.session_state:
            st.session_state[self.SESSION_KEY] = []

    def render(self, top_k: int, temperature: float, source_filter: str | None) -> None:
        """Render past messages, then handle a new question if one was submitted."""
        for message in st.session_state[self.SESSION_KEY]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and "citations" in message:
                    self._citations.render(message["citations"], message["confidence"])

        if st.button("Clear chat"):
            st.session_state[self.SESSION_KEY] = []
            st.rerun()

        question = st.chat_input("Ask a question about your financial documents...")
        if question:
            self._handle_question(question, top_k, temperature, source_filter)

    def _handle_question(
        self, question: str, top_k: int, temperature: float, source_filter: str | None
    ) -> None:
        """Append the user's question, call the backend, and append the grounded answer."""
        st.session_state[self.SESSION_KEY].append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            response = requests.post(
                f"{self._backend_url}/query",
                json={
                    "question": question,
                    "top_k": top_k,
                    "temperature": temperature,
                    "source_filter": source_filter,
                },
                timeout=120,
            )

        if response.ok:
            data = response.json()
            st.session_state[self.SESSION_KEY].append(
                {
                    "role": "assistant",
                    "content": data["answer"],
                    "citations": data["citations"],
                    "confidence": data["confidence"],
                }
            )
        else:
            detail = response.json().get("detail", response.text)
            st.session_state[self.SESSION_KEY].append(
                {"role": "assistant", "content": f"Error: {detail}"}
            )
        st.rerun()
