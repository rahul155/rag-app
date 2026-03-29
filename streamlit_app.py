import streamlit as st
import requests
import os
from pathlib import Path

st.set_page_config(page_title="RAG PDF App", page_icon="📄", layout="centered")

BASE_URL = os.getenv("BACKEND_URL")

# ---------- Upload PDF ----------
st.title("📄 Upload a PDF")

uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded is not None:
    with st.spinner("Uploading and processing..."):

        # Save locally (temporary)
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        file_path = uploads_dir / uploaded.name

        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # Call backend
        response = requests.post(
            f"{BASE_URL}/rag/ingest_pdf",
            json={
                "pdf_path": str(file_path),
                "source_id": uploaded.name
            }
        )

    st.success("PDF processed successfully!")

# ---------- Ask Question ----------
st.divider()
st.title("💬 Ask a question")

question = st.text_input("Your question")
top_k = st.number_input("Top K", min_value=1, max_value=10, value=5)

if st.button("Ask") and question.strip():
    with st.spinner("Thinking..."):

        response = requests.post(
            f"{BASE_URL}/rag/query_pdf_ai",
            json={
                "question": question,
                "top_k": int(top_k)
            }
        )

        data = response.json()

        answer = data.get("answer", "")
        sources = data.get("sources", [])

    st.subheader("Answer")
    st.write(answer)

    if sources:
        st.caption("Sources")
        for s in sources:
            st.write(f"- {s}")