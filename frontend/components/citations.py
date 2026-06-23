"""
Renders the expandable "Sources" section and confidence badge under an
assistant message.
"""
import streamlit as st

_CONFIDENCE_ICONS = {"high": "🟢", "medium": "🟡", "low": "🔴"}


class CitationsComponent:
    """Displays citations and a confidence indicator for one query response."""

    def render(self, citations: list[dict], confidence: str) -> None:
        """Draw a confidence badge followed by an expander listing each citation."""
        icon = _CONFIDENCE_ICONS.get(confidence, "⚪")
        st.caption(f"{icon} Confidence: {confidence.capitalize()}")

        if not citations:
            return

        with st.expander(f"Sources ({len(citations)})"):
            for citation in citations:
                st.markdown(
                    f"**{citation['source_file']}**, page {citation['page_number']} "
                    f"(relevance: {citation['similarity_score']:.2f})"
                )
                st.markdown(f"> {citation['excerpt']}")
