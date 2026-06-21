from src.embedding import create_embeddings
from src.vector_store import search
from src.llm import generate_answer


def ask_question(question):

    query_embedding = create_embeddings([question])[0]

    results = search(query_embedding)

    retrieved_docs = results["documents"][0]

    context = "\n\n".join(retrieved_docs)

    answer = generate_answer(
        question,
        context
    )

    return answer, retrieved_docs