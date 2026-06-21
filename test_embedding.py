from src.embedding import create_embeddings

chunks = [
    "Artificial Intelligence is transforming industries.",
    "Generative AI can generate text and code."
]

embeddings = create_embeddings(chunks)

print(f"Chunks: {len(chunks)}")
print(f"Embeddings: {len(embeddings)}")
print(f"Dimension: {len(embeddings[0])}")