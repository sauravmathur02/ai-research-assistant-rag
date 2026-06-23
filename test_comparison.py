import os
os.environ["CHROMA_DB_PATH"] = "./chroma_db_test"
os.environ["CHROMA_COLLECTION_NAME"] = "test_documents"

from src.embedding import create_embeddings
from src.vector_store import store_chunks, reset_collection
from src.rag import ask_question

reset_collection()

doc_a_chunks = [
    "Python is an interpreted, high-level, general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant whitespace.",
    "Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented, and functional programming."
]
store_chunks(doc_a_chunks, create_embeddings(doc_a_chunks), source_name="Python_Specs.txt")

doc_b_chunks = [
    "C++ is a high-performance, compiled, general-purpose programming language. It was created by Bjarne Stroustrup as an extension of the C programming language.",
    "C++ is statically-typed, supports object-oriented, generic, and functional programming. It provides low-level memory manipulation features."
]
store_chunks(doc_b_chunks, create_embeddings(doc_b_chunks), source_name="Cpp_Specs.txt")

question = "How does Python relate to C++?"
print(f"Querying: '{question}'")

answer, sources, ret_time, gen_time, standalone, stats = ask_question(question)

print("\n" + "="*50)
print("COMPARISON TEST RESULTS")
print("="*50)
print(f"Comparison Mode Flag: {stats.get('comparison_mode')}")
print(f"Documents Involved Count: {stats.get('docs_involved_count')}")
print(f"Chunks per Document: {stats.get('chunks_per_doc')}")
print(f"Contributing Documents: {stats.get('contributing_docs')}")

print("\n" + "="*50)
print("CITED SOURCES (Guaranteed Multi-Document representation)")
print("="*50)
for i, source in enumerate(sources, start=1):
    print(f"{i}. {source['source']} (Chunk {source['chunk_index']})")
    print(f"   Excerpt: {source['content'][:80]}...\n")

print("\n" + "="*50)
print("GENERATED ANSWER")
print("="*50)
print(answer)
