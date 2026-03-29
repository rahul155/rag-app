from openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import time

load_dotenv()

client = OpenAI()

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

# 🔥 Better chunking (balanced for speed + accuracy)
splitter = SentenceSplitter(chunk_size=500, chunk_overlap=100)


# -------- LOAD + CHUNK PDF --------
def load_and_chunk_pdf(path: str):
    reader = SimpleDirectoryReader(input_files=[path])
    docs = reader.load_data()

    texts = [d.text for d in docs if getattr(d, "text", None)]

    chunks = []
    for t in texts:
        cleaned = t.strip().replace("\n", " ")
        chunks.extend([
            c.strip()
            for c in splitter.split_text(cleaned)
            if len(c.strip()) > 30
        ])

    # 🔥 limit to avoid overload on large PDFs
    return chunks[:500]


# -------- EMBEDDINGS (with batching + retry) --------
def embed_texts(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    batch_size = 50
    max_retries = 3

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                )

                all_embeddings.extend(
                    [item.embedding for item in response.data]
                )
                break  # success → exit retry loop

            except Exception as e:
                print(f"Embedding error (attempt {attempt+1}):", str(e))
                time.sleep(2)  # wait before retry

        else:
            print("Skipping batch due to repeated failures")

    return all_embeddings