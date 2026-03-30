import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(self, collection="docs", dim=3072):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=30
        )
        self.collection = collection
        self.dim = dim

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                ),
            )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i]
            )
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)

    # 🔥 FIXED SEARCH WITH FILTER
    def search(self, query_vector, top_k: int = 5, keyword: str = None, source_id: str = None):
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector,
            with_payload=True,
            limit=top_k * 3,
            query_filter={
                "must": [
                    {
                        "key": "source",
                        "match": {"value": source_id}
                    }
                ]
            } if source_id else None
        )

        results = response.points

        contexts = []
        sources = set()

        keywords = []
        if keyword:
            keywords = [w.lower() for w in keyword.split() if len(w) > 3]

        for r in results:
            payload = r.payload or {}
            text = payload.get("text", "")
            source = payload.get("source", "")

            if not text:
                continue

            text_lower = text.lower()

            score = sum(1 for k in keywords if k in text_lower)

            if score > 0:
                contexts.insert(0, text)
            else:
                contexts.append(text)

            if source:
                sources.add(source)

        return {
            "contexts": contexts[:top_k],
            "sources": list(sources)
        }