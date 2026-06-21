import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)


def store_chunks(chunks, embeddings):

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )


def search(query_embedding, k=3):

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results


def reset_collection():

    global collection

    try:
        client.delete_collection("documents")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="documents"
    )