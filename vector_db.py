def search(self, query_vector, top_k: int = 5, keyword: str = None):
    response = self.client.query_points(
        collection_name=self.collection,
        query=query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector,
        with_payload=True,
        limit=top_k * 2   # 🔥 fetch more for better filtering
    )

    results = response.points

    contexts = []
    sources = set()

    for r in results:
        payload = r.payload or {}
        text = payload.get("text", "")
        source = payload.get("source", "")

        if not text:
            continue

        # 🔥 KEYWORD BOOST (important)
        if keyword and keyword.lower() in text.lower():
            contexts.insert(0, text)   # move relevant chunk to top
        else:
            contexts.append(text)

        if source:
            sources.add(source)

    return {
        "contexts": contexts[:top_k],   # return best top_k
        "sources": list(sources)
    }