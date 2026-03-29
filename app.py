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

# ✅ CORS (needed for Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------- TEST ROUTES ----------------
@app.get("/")
def root():
    return {"message": "API WORKING"}

@app.get("/test123")
def test():
    return {"status": "THIS IS NEW CODE"}


# ---------------- INGEST PDF ----------------
@app.post("/rag/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        temp_path = f"temp_{file.filename}"

        # Save file
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        source_id = file.filename

        # Load + chunk
        chunks = load_and_chunk_pdf(temp_path)[:200]

        # Embed (batched inside function)
        vecs = embed_texts(chunks)

        # IDs + payload
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

        # Cleanup
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
        top_k = int(data.get("top_k", 10))

        # Step 1: Embed query
        query_vec = embed_texts([question])[0]

        # Step 2: Search
        store = QdrantStorage()
        found = store.search(query_vec, top_k,keyword=question)

        contexts = found["contexts"][:5]   
        sources = found["sources"]

        if not contexts:
            return {
                "answer": "No relevant context found.",
                "sources": [],
                "num_contexts": 0
            }

        # 🔥 Limit context size
        MAX_CHARS = 4000
        context_block = ""

        for c in contexts:
            if len(context_block) + len(c) < MAX_CHARS:
                context_block += f"- {c}\n\n"
            else:
                break

        # 🔥 Better prompt
        prompt = (
            "You are a precise assistant.\n\n"
            "Answer ONLY from the given context.\n"
            "Extract exact information if present.\n"
            "If not found, say: 'Not found in context'.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}"
        )

        # Step 3: LLM call
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You answer strictly using context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300
        )

        answer = res.choices[0].message.content.strip()

        return {
            "answer": answer,
            "sources": sources,
            "num_contexts": len(contexts),
        }

    except Exception as e:
        return {"error": str(e)}