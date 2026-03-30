import streamlit as st
import requests
import os

st.set_page_config(page_title="RAG PDF App", page_icon="📄")

BASE_URL = os.getenv("BACKEND_URL")

st.write("Backend URL:", BASE_URL)

# 🔥 Store current file
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# ---------- Upload ----------
st.title("📄 Upload PDF")

uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded:
    with st.spinner("Uploading and processing..."):

        files = {
            "file": (uploaded.name, uploaded.getvalue(), "application/pdf")
        }

        response = requests.post(
            f"{BASE_URL}/rag/ingest_pdf",
            files=files
        )

        st.write("Status:", response.status_code)
        st.write("Response:", response.text)

    if response.status_code == 200:
        st.session_state.current_file = uploaded.name  # 🔥 save file
        st.success("PDF processed successfully!")
    else:
        st.error("Upload failed")


# ---------- Query ----------
st.divider()
st.title("💬 Ask Question")

question = st.text_input("Your question")
top_k = st.number_input("Top K", 1, 10, 5)

if st.button("Ask") and question:
    if not st.session_state.current_file:
        st.warning("Please upload a PDF first")
    else:
        with st.spinner("Thinking..."):

            response = requests.post(
                f"{BASE_URL}/rag/query_pdf_ai",
                json={
                    "question": question,
                    "top_k": int(top_k),
                    "source_id": st.session_state.current_file  # 🔥 isolation fix
                }
            )

            st.write("Status:", response.status_code)
            st.write("Raw:", response.text)

            if response.status_code == 200:
                data = response.json()

                st.subheader("Answer")
                st.write(data.get("answer", ""))

                sources = data.get("sources", [])
                if sources:
                    st.caption("Sources")
                    for s in sources:
                        st.write(f"- {s}")
            else:
                st.error("Query failed")