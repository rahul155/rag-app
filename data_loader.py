from openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=500, chunk_overlap=100)


# -------- LOAD + CHUNK PDF --------
def load_and_chunk_pdf(path: str):
    reader = SimpleDirectoryReader(input_files=[path])
    docs = reader.load_data()

    texts = [d.text for d in docs if getattr(d, "text", None)]

    chunks = []
    for t in texts:
        chunks.extend([c for c in splitter.split_text(t) if len(c.strip()) > 30])

    return chunks[:500]


# -------- EMBEDDINGS --------
def embed_texts(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    batch_size = 50  # safe for OpenAI

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        try:
            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in response.data])

        except Exception as e:
            print("Embedding error:", str(e))
            continue

    return all_embeddings