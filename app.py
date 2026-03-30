import os
import uuid
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from openai import OpenAI

load_dotenv()

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------- TEST ----------------
@app.get("/")
def root():
    return {"message": "API WORKING"}


# ---------------- RERANK ----------------
def rerank_contexts(question: str, contexts: list[str]) -> list[str]:
    if not contexts:
        return []

    joined_contexts = "\n\n".join(
        [f"{i}. {c}" for i, c in enumerate(contexts)]
    )

    prompt = (
        "Select the 3 most relevant contexts for the question.\n\n"
        f"Question: {question}\n\n"
        f"Contexts:\n{joined_contexts}\n\n"
        "Return only indices like: 0,2,4"
    )

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        output = res.choices[0].message.content.strip()

        indices = [
            int(x.strip())
            for x in output.split(",")
            if x.strip().isdigit()
        ]

        selected = [contexts[i] for i in indices if i < len(contexts)]

        return selected if selected else contexts[:3]

    except Exception as e:
        print("Rerank error:", str(e))
        return contexts[:3]


# ---------------- INGEST ----------------
@app.post("/rag/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        temp_path = f"temp_{file.filename}"

        with open(temp_path, "wb") as f:
            f.write(await file.read())

        source_id = file.filename

        chunks = load_and_chunk_pdf(temp_path)[:200]
        vecs = embed_texts(chunks)

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {"source": source_id, "text": chunks[i]}
            for i in range(len(chunks))
        ]

        QdrantStorage().upsert(ids, vecs, payloads)

        try:
            os.remove(temp_path)
        except:
            pass

        return {"ingested": len(chunks)}

    except Exception as e:
        return {"error": str(e)}


# ---------------- QUERY ----------------
@app.post("/rag/query_pdf_ai")
def query_pdf(data: dict):
    try:
        question = data["question"]
        source_id = data.get("source_id")
        top_k = int(data.get("top_k", 10))

        query_vec = embed_texts([question])[0]

        store = QdrantStorage()
        found = store.search(
            query_vec,
            top_k=15,
            keyword=question,
            source_id=source_id   # 🔥 isolation fix
        )

        # 🔥 reranking
        contexts = rerank_contexts(question, found["contexts"])
        sources = found["sources"]

        if not contexts:
            return {
                "answer": "No relevant context found.",
                "sources": [],
                "num_contexts": 0
            }

        # 🔥 limit context size
        MAX_CHARS = 4000
        context_block = ""

        for c in contexts:
            if len(context_block) + len(c) < MAX_CHARS:
                context_block += f"- {c}\n\n"
            else:
                break

        prompt = (
            "Answer using the context below.\n\n"
            "If context is not enough, use general knowledge.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}"
        )

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Give clear answers."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=400
        )

        answer = res.choices[0].message.content.strip()

        return {
            "answer": answer,
            "sources": sources,
            "num_contexts": len(contexts),
        }

    except Exception as e:
        return {"error": str(e)}