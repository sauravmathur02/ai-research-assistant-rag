import os
os.environ["CHROMA_DB_PATH"] = "./chroma_db_test"
os.environ["CHROMA_COLLECTION_NAME"] = "test_documents"

from src.embedding import create_embeddings
from src.vector_store import store_chunks, search

chunks = [
    "Artificial Intelligence is transforming industries.",
    "Generative AI can generate text and code.",
    "YOLO is used for object detection."
]

embeddings = create_embeddings(chunks)

store_chunks(chunks, embeddings, source_name="test_doc.txt")

query = "What is generative AI?"

query_embedding = create_embeddings([query])[0]

results = search(query_embedding)

print(results["documents"][0])