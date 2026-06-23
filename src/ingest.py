from src.document_loader import extract_text
from src.chunker import chunk_text
from src.embedding import create_embeddings
from src.vector_store import store_chunks

def ingest_document(file):

    text = extract_text(file)

    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)

    store_chunks(
        chunks,
        embeddings,
        source_name=file.name
    )

    return len(chunks)