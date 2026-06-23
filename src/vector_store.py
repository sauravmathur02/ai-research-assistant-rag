import os
import chromadb

db_path = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "documents")

client = chromadb.PersistentClient(path=db_path)


def get_collection():

    return client.get_or_create_collection(
        name=collection_name
    )


def store_chunks(chunks, embeddings, source_name):

    coll = get_collection()

    try:
        coll.delete(where={"source": source_name})
    except Exception:
        pass

    ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    coll.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )


def search(query_embedding, k=5, targeted_docs=[]):

    coll = get_collection()

    where_filter = None
    if targeted_docs:
        if len(targeted_docs) == 1:
            where_filter = {"source": targeted_docs[0]}
        else:
            where_filter = {"source": {"$in": targeted_docs}}

    results = coll.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_filter
    )

    return results


def delete_document(source_name: str):
    try:
        coll = get_collection()
        coll.delete(where={"source": source_name})
    except Exception:
        pass


def reset_collection():

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


def get_uploaded_documents():

    try:
        coll = get_collection()
        results = coll.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        if not metadatas:
            return []
        sources = {meta.get("source") for meta in metadatas if meta and meta.get("source")}
        return sorted(list(sources))
    except Exception:
        return []


def get_stored_chunks_count():

    try:
        return get_collection().count()
    except Exception:
        return 0