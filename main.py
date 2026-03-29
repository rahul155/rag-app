import os
import uuid
from fastapi import FastAPI
from dotenv import load_dotenv

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage

from openai import OpenAI

load_dotenv()

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------- INGEST PDF ----------------
@app.post("/rag/ingest_pdf")
def ingest_pdf(data: dict):
    pdf_path = data["pdf_path"]
    source_id = data.get("source_id", pdf_path)

    # Load & chunk
    chunks = load_and_chunk_pdf(pdf_path)

    # Embed
    vecs = embed_texts(chunks)

    # Prepare IDs + payload
    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
        for i in range(len(chunks))
    ]

    payloads = [
        {"source": source_id, "text": chunks[i]}
        for i in range(len(chunks))
    ]

    # Store
    QdrantStorage().upsert(ids, vecs, payloads)

    return {"ingested": len(chunks)}


# ---------------- QUERY ----------------
@app.post("/rag/query_pdf_ai")
def query_pdf(data: dict):
    question = data["question"]
    top_k = int(data.get("top_k", 5))

    # Embed question
    query_vec = embed_texts([question])[0]

    # Search
    store = QdrantStorage()
    found = store.search(query_vec, top_k)

    contexts = found["contexts"]
    sources = found["sources"]

    if not contexts:
        return {
            "answer": "No relevant context found.",
            "sources": [],
            "num-contexts": 0
        }

    # Build prompt
    context_block = "\n\n".join(f"- {c}" for c in contexts)

    prompt = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    # Call OpenAI
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer using only provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    answer = res.choices[0].message.content.strip()

    return {
        "answer": answer,
        "sources": sources,
        "num-contexts": len(contexts),
    }