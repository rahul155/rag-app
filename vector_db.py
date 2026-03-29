import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(self, collection="docs", dim=3072):
        self.client = QdrantClient(
            url=os.getenv("https://c639a804-9609-427e-b807-c45bb36bcc9d.eu-west-1-0.aws.cloud.qdrant.io:6333"),          
            api_key=os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.zV2W-mk16LOA7AvflBuTPGcMQzmt_w6aLyWYlJHg2bQ"),   
            timeout=30
        )
        self.collection = collection
        self.dim = dim

        # Create collection if it doesn't exist
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

    def search(self, query_vector, top_k: int = 5):
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector,
            with_payload=True,
            limit=top_k
        )

        results = response.points

        contexts = []
        sources = set()

        for r in results:
            payload = r.payload or {}
            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                contexts.append(text)
                sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources)
        }