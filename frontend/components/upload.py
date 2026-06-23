"""
Sidebar document-upload and document-list widget.
"""
import requests
import streamlit as st


class UploadComponent:
    """Renders the PDF upload control and the list of currently indexed documents.

    Wrapping this in a class (rather than a bare function) lets it own the
    backend_url it talks to, so it can be unit-tested or pointed at a
    different backend without touching global state.
    """

    def __init__(self, backend_url: str):
        self._backend_url = backend_url

    def render(self) -> None:
        """Draw the upload button, the document list, and delete controls."""
        st.sidebar.header("Documents")

        uploaded_file = st.sidebar.file_uploader("Upload a PDF filing", type=["pdf"])
        if uploaded_file is not None and st.sidebar.button("Ingest document"):
            self._ingest(uploaded_file)

        self._render_document_list()

    def _ingest(self, uploaded_file) -> None:
        """POST the uploaded file to the backend /ingest endpoint."""
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            response = requests.post(
                f"{self._backend_url}/ingest",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                timeout=300,
            )
        if response.ok:
            data = response.json()
            st.sidebar.success(f"Indexed {data['chunks_created']} chunks from {data['filename']}.")
        else:
            st.sidebar.error(f"Ingestion failed: {response.json().get('detail', response.text)}")

    def _render_document_list(self) -> None:
        """Fetch and display indexed documents with a per-document delete button."""
        response = requests.get(f"{self._backend_url}/docs", timeout=30)
        if not response.ok:
            st.sidebar.warning("Could not load document list.")
            return

        data = response.json()
        st.sidebar.caption(
            f"{data['total_documents']} document(s), {data['total_chunks']} chunk(s) indexed"
        )
        for doc in data["documents"]:
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(f"{doc['filename']} ({doc['chunk_count']} chunks)")
            if col2.button("Delete", key=f"delete_{doc['filename']}"):
                requests.delete(f"{self._backend_url}/docs/{doc['filename']}", timeout=30)
                st.rerun()
