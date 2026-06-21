from src.embedding import create_embeddings
from src.vector_store import store_chunks, search

chunks = [
    "Artificial Intelligence is transforming industries.",
    "Generative AI can generate text and code.",
    "YOLO is used for object detection."
]

embeddings = create_embeddings(chunks)

store_chunks(chunks, embeddings)

query = "What is generative AI?"

query_embedding = create_embeddings([query])[0]

results = search(query_embedding)

print(results["documents"][0])