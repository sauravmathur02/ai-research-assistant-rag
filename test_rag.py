import os
os.environ["CHROMA_DB_PATH"] = "./chroma_db_test"
os.environ["CHROMA_COLLECTION_NAME"] = "test_documents"

from src.embedding import create_embeddings
from src.vector_store import (
    store_chunks,
    reset_collection
)
from src.rag import ask_question


reset_collection()

chunks = [
    "Generative AI can generate text and code.",
    "YOLO is used for object detection in computer vision.",
    "RAG stands for Retrieval Augmented Generation. It combines retrieval and generation."
]

embeddings = create_embeddings(chunks)

store_chunks(
    chunks,
    embeddings,
    source_name="test_rag_doc.txt"
)

question = "What is RAG?"

answer, sources, ret_time, gen_time, standalone, stats = ask_question(question)

print("\n" + "="*50)
print("ANSWER")
print("="*50)
print(answer)
print(f"\n[Retrieval Time: {ret_time:.3f}s | Generation Time: {gen_time:.3f}s]")
print(f"[Standalone Query: {standalone}]")

print("\n" + "="*50)
print("STATS")
print("="*50)
print(f"Documents Involved Count: {stats.get('docs_involved_count')}")
print(f"Chunks per Document: {stats.get('chunks_per_doc')}")
print(f"Contributing Documents: {stats.get('contributing_docs')}")

print("\n" + "="*50)
print("SOURCES")
print("="*50)

for i, source in enumerate(sources, start=1):
    print(f"{i}. Document: {source['source']} (Chunk {source['chunk_index']})")
    print(f"   Content: {source['content']}\n")