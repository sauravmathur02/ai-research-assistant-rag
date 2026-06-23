import os
import re
import sys
import json
import time
import logging

os.environ["CHROMA_DB_PATH"] = "./chroma_db_eval"
os.environ["CHROMA_COLLECTION_NAME"] = "eval_documents"

logging.basicConfig(level=logging.ERROR)

from src.embedding import create_embeddings
from src.vector_store import store_chunks, reset_collection
from src.rag import ask_question

BENCHMARK = [
    {
        "doc_name": "rag_overview.txt",
        "doc_text": (
            "Retrieval-Augmented Generation (RAG) is an AI framework that enhances "
            "large language model (LLM) responses by grounding them in retrieved "
            "external documents. The pipeline consists of three stages: ingestion "
            "(chunking and embedding documents into a vector store), retrieval "
            "(fetching the most relevant chunks via semantic search), and generation "
            "(prompting the LLM with retrieved context to produce a grounded answer). "
            "RAG reduces hallucination by ensuring the model cites real source material."
        ),
        "question": "What is RAG and what are its main stages?",
        "expected_keywords": ["retrieval", "generation", "ingestion", "vector", "hallucination"],
    },
    {
        "doc_name": "embeddings_guide.txt",
        "doc_text": (
            "Text embeddings are dense vector representations of text that capture "
            "semantic meaning. Sentence-Transformers models like all-MiniLM-L6-v2 "
            "encode text into 384-dimensional vectors. Cosine similarity between "
            "these vectors measures semantic closeness. Embeddings enable semantic "
            "search, where queries retrieve conceptually similar documents even "
            "without exact keyword matches."
        ),
        "question": "How do text embeddings enable semantic search?",
        "expected_keywords": ["vector", "semantic", "cosine", "similarity", "sentence"],
    },
    {
        "doc_name": "chromadb_intro.txt",
        "doc_text": (
            "ChromaDB is an open-source, embedded vector database designed for "
            "AI applications. It stores document chunks alongside their vector "
            "embeddings and metadata. ChromaDB supports filtering queries by "
            "metadata fields such as source filename. Its PersistentClient stores "
            "the index on disk, ensuring data survives application restarts. "
            "ChromaDB uses approximate nearest-neighbour (ANN) search for fast retrieval."
        ),
        "question": "What is ChromaDB and why is it used in RAG systems?",
        "expected_keywords": ["chromadb", "vector", "embeddings", "metadata", "persistent"],
    },
    {
        "doc_name": "chunking_strategy.txt",
        "doc_text": (
            "Text chunking splits long documents into smaller, overlapping segments "
            "before embedding. A chunk size of 600 characters with 120-character "
            "overlap is commonly used. Overlap ensures that context spanning chunk "
            "boundaries is not lost. RecursiveCharacterTextSplitter from LangChain "
            "splits on paragraph breaks, newlines, and spaces in order of preference, "
            "preserving sentence structure wherever possible."
        ),
        "question": "What chunking strategy is used and why is overlap important?",
        "expected_keywords": ["chunk", "overlap", "600", "split", "context"],
    },
    {
        "doc_name": "hybrid_reranking.txt",
        "doc_text": (
            "Hybrid reranking combines semantic vector similarity with keyword-based "
            "Jaccard overlap to score retrieved chunks. The combined score formula is: "
            "score = 0.6 * semantic_similarity + 0.4 * keyword_overlap. "
            "Semantic similarity is computed from the L2 distance returned by ChromaDB. "
            "Keyword overlap counts matching non-stop-word tokens between query and chunk. "
            "This hybrid approach improves precision over pure vector search alone."
        ),
        "question": "How does hybrid reranking work and what are the weights used?",
        "expected_keywords": ["hybrid", "semantic", "keyword", "0.6", "0.4", "jaccard"],
    },
    {
        "doc_name": "comparison_mode.txt",
        "doc_text": (
            "Comparison mode is triggered when the user's query contains comparison "
            "intent keywords such as 'compare', 'difference', 'versus', or 'contrast'. "
            "In comparison mode, the system routes the query to all targeted documents, "
            "increases retrieval depth to 20 chunks, and guarantees at least one chunk "
            "from each targeted document. The LLM is explicitly instructed to produce "
            "a structured Markdown table summarising differences and similarities."
        ),
        "question": "What happens when comparison mode is activated?",
        "expected_keywords": ["comparison", "table", "chunks", "documents", "routing"],
    },
    {
        "doc_name": "gemini_api.txt",
        "doc_text": (
            "The Google Gemini 2.5 Flash model is used for text generation. The LLM "
            "receives a structured prompt containing retrieved context segments and the "
            "user's question. It is instructed to cite sources using the format "
            "[Source: filename (Chunk N)]. A retry mechanism with exponential backoff "
            "handles ResourceExhausted (rate limit) errors, retrying up to 3 times "
            "with delays of 2, 4, and 8 seconds respectively."
        ),
        "question": "How does the system handle Gemini API rate limit errors?",
        "expected_keywords": ["retry", "exponential", "backoff", "rate", "resourceexhausted"],
    },
    {
        "doc_name": "query_memory.txt",
        "doc_text": (
            "Query memory allows the assistant to resolve follow-up questions that "
            "reference prior conversation context. The last 3 exchanges are compiled "
            "into a history string and sent to Gemini with a rephrasing prompt. "
            "Gemini returns a standalone question with pronouns resolved, e.g., "
            "'What is its architecture?' becomes 'What is RAG's architecture?' "
            "This standalone query is then used for vector retrieval."
        ),
        "question": "How does the system resolve follow-up questions with pronouns?",
        "expected_keywords": ["rephrase", "standalone", "history", "pronouns", "context"],
    },
    {
        "doc_name": "document_formats.txt",
        "doc_text": (
            "The document loader supports three file formats: PDF, DOCX, and TXT. "
            "PDF files are parsed using pypdf's PdfReader, extracting text page by page. "
            "DOCX files are parsed using python-docx, iterating over paragraphs. "
            "TXT files are decoded as UTF-8 strings directly. After extraction, the "
            "raw text is passed to the chunker regardless of the original format."
        ),
        "question": "What document formats are supported and how are they parsed?",
        "expected_keywords": ["pdf", "docx", "txt", "pypdf", "paragraph"],
    },
    {
        "doc_name": "offline_fallback.txt",
        "doc_text": (
            "If the Gemini API is unavailable or rate-limited and retries are exhausted, "
            "the system generates a structured offline fallback response. The fallback "
            "presents the top retrieved chunks with their source names, chunk indices, "
            "similarity scores, and 250-character excerpts. This ensures users always "
            "receive partial answers from retrieved context even without LLM connectivity."
        ),
        "question": "What does the system return when the Gemini API is offline?",
        "expected_keywords": ["fallback", "offline", "chunks", "similarity", "excerpts"],
    },
]


def tokenize(text: str) -> set:
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "and", "or", "in", "on", "at", "to", "for", "of", "with",
        "it", "its", "this", "that", "by", "as", "if", "how", "what",
        "does", "do", "used", "use", "using", "from", "such", "when",
    }
    tokens = set(re.findall(r'\b[a-z0-9]+\b', text.lower()))
    return tokens - stop_words


def context_relevance(question: str, sources: list) -> float:
    q_tokens = tokenize(question)
    if not q_tokens or not sources:
        return 0.0
    context_text = " ".join(s["content"] for s in sources)
    c_tokens = tokenize(context_text)
    overlap = len(q_tokens & c_tokens)
    return round(overlap / len(q_tokens), 3)


def answer_faithfulness(answer: str, sources: list) -> float:
    a_tokens = tokenize(answer)
    if not a_tokens or not sources:
        return 0.0
    context_text = " ".join(s["content"] for s in sources)
    c_tokens = tokenize(context_text)
    grounded = len(a_tokens & c_tokens)
    return round(grounded / len(a_tokens), 3)


def retrieval_recall(sources: list, expected_keywords: list) -> float:
    if not expected_keywords or not sources:
        return 0.0
    context_text = " ".join(s["content"] for s in sources).lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in context_text)
    return round(found / len(expected_keywords), 3)


def setup_benchmark_db(benchmark: list):
    print("📦  Setting up benchmark vector store...")
    reset_collection()
    for entry in benchmark:
        chunks = [entry["doc_text"]]
        embeddings = create_embeddings(chunks)
        store_chunks(chunks, embeddings, source_name=entry["doc_name"])
    print(f"    ✅  {len(benchmark)} documents ingested.\n")


def run_evaluation(benchmark: list) -> list:
    results = []

    print("🔬  Running evaluation...\n")
    print(f"{'#':<4} {'Question':<52} {'CtxRel':>7} {'Faith':>7} {'Recall':>7} {'Latency':>8}")
    print("─" * 90)

    for i, entry in enumerate(benchmark, start=1):
        question = entry["question"]

        try:
            answer, sources, ret_time, gen_time, standalone, stats = ask_question(question)
            total_latency = ret_time + gen_time

            ctx_rel  = context_relevance(question, sources)
            faith    = answer_faithfulness(answer, sources)
            recall   = retrieval_recall(sources, entry["expected_keywords"])

            result = {
                "id": i,
                "question": question,
                "expected_keywords": entry["expected_keywords"],
                "answer_excerpt": answer[:200].replace("\n", " "),
                "sources_retrieved": len(sources),
                "context_relevance": ctx_rel,
                "answer_faithfulness": faith,
                "retrieval_recall": recall,
                "retrieval_time_s": round(ret_time, 3),
                "generation_time_s": round(gen_time, 3),
                "total_latency_s": round(total_latency, 3),
                "comparison_mode": stats.get("comparison_mode", False),
                "standalone_query": standalone,
                "error": None,
            }

        except Exception as e:
            result = {
                "id": i,
                "question": question,
                "error": str(e),
                "context_relevance": 0.0,
                "answer_faithfulness": 0.0,
                "retrieval_recall": 0.0,
                "total_latency_s": 0.0,
            }

        results.append(result)

        q_short = question[:50] + ".." if len(question) > 50 else question
        if result["error"]:
            print(f"{i:<4} {q_short:<52} {'ERROR':>7} {'ERROR':>7} {'ERROR':>7} {'N/A':>8}")
        else:
            print(
                f"{i:<4} {q_short:<52} "
                f"{ctx_rel:>7.2f} "
                f"{faith:>7.2f} "
                f"{recall:>7.2f} "
                f"{total_latency:>7.2f}s"
            )

    return results


def compute_summary(results: list) -> dict:
    valid = [r for r in results if not r.get("error")]
    if not valid:
        return {}

    def avg(key):
        return round(sum(r[key] for r in valid) / len(valid), 3)

    return {
        "total_questions": len(results),
        "successful_runs": len(valid),
        "failed_runs": len(results) - len(valid),
        "avg_context_relevance": avg("context_relevance"),
        "avg_answer_faithfulness": avg("answer_faithfulness"),
        "avg_retrieval_recall": avg("retrieval_recall"),
        "avg_total_latency_s": avg("total_latency_s"),
        "avg_retrieval_time_s": avg("retrieval_time_s"),
        "avg_generation_time_s": avg("generation_time_s"),
    }


def save_results(results: list, summary: dict):
    os.makedirs("eval", exist_ok=True)

    json_path = "eval/evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    txt_path = "eval/evaluation_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  AI Research Assistant — Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total Questions Evaluated : {summary.get('total_questions', 0)}\n")
        f.write(f"Successful Runs           : {summary.get('successful_runs', 0)}\n")
        f.write(f"Failed Runs               : {summary.get('failed_runs', 0)}\n\n")

        f.write("Aggregate Metrics\n")
        f.write("-" * 40 + "\n")
        f.write(f"Avg Context Relevance     : {summary.get('avg_context_relevance', 0):.3f}\n")
        f.write(f"Avg Answer Faithfulness   : {summary.get('avg_answer_faithfulness', 0):.3f}\n")
        f.write(f"Avg Retrieval Recall      : {summary.get('avg_retrieval_recall', 0):.3f}\n")
        f.write(f"Avg Retrieval Latency     : {summary.get('avg_retrieval_time_s', 0):.3f}s\n")
        f.write(f"Avg Generation Latency    : {summary.get('avg_generation_time_s', 0):.3f}s\n")
        f.write(f"Avg End-to-End Latency    : {summary.get('avg_total_latency_s', 0):.3f}s\n\n")

        f.write("Per-Question Breakdown\n")
        f.write("-" * 40 + "\n")
        for r in results:
            f.write(f"\nQ{r['id']}: {r['question']}\n")
            if r.get("error"):
                f.write(f"  ERROR: {r['error']}\n")
            else:
                f.write(f"  Context Relevance   : {r['context_relevance']:.3f}\n")
                f.write(f"  Answer Faithfulness : {r['answer_faithfulness']:.3f}\n")
                f.write(f"  Retrieval Recall    : {r['retrieval_recall']:.3f}\n")
                f.write(f"  Total Latency       : {r['total_latency_s']:.3f}s\n")

    return json_path, txt_path


def main():
    print("\n" + "=" * 60)
    print("  AI Research Assistant — RAG Evaluation Framework")
    print("=" * 60 + "\n")

    setup_benchmark_db(BENCHMARK)
    results = run_evaluation(BENCHMARK)
    summary = compute_summary(results)

    print("\n" + "─" * 90)
    print("📊  AGGREGATE METRICS")
    print("─" * 90)
    print(f"  Avg Context Relevance   : {summary.get('avg_context_relevance', 0):.3f}")
    print(f"  Avg Answer Faithfulness : {summary.get('avg_answer_faithfulness', 0):.3f}")
    print(f"  Avg Retrieval Recall    : {summary.get('avg_retrieval_recall', 0):.3f}")
    print(f"  Avg Retrieval Latency   : {summary.get('avg_retrieval_time_s', 0):.3f}s")
    print(f"  Avg Generation Latency  : {summary.get('avg_generation_time_s', 0):.3f}s")
    print(f"  Avg End-to-End Latency  : {summary.get('avg_total_latency_s', 0):.3f}s")

    json_path, txt_path = save_results(results, summary)
    print(f"\n✅  Results saved:")
    print(f"    JSON  → {json_path}")
    print(f"    Text  → {txt_path}\n")

    failed = summary.get("failed_runs", 0)
    if failed > 0:
        print(f"⚠️  {failed} question(s) failed. Check {json_path} for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
