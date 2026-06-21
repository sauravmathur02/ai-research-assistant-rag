from src.embedding import create_embeddings
from src.vector_store import (
    store_chunks,
    reset_collection
)
from src.rag import ask_question


# Clear old test data
reset_collection()

chunks = [
    "Generative AI can generate text and code.",
    "YOLO is used for object detection in computer vision.",
    "RAG stands for Retrieval Augmented Generation. It combines retrieval and generation."
]

embeddings = create_embeddings(chunks)

store_chunks(
    chunks,
    embeddings
)

question = "What is RAG?"

answer, sources = ask_question(question)

print("\n" + "="*50)
print("ANSWER")
print("="*50)
print(answer)

print("\n" + "="*50)
print("SOURCES")
print("="*50)

for i, source in enumerate(sources, start=1):
    print(f"{i}. {source}")