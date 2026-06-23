import os
import re
import time
import logging
from src.embedding import create_embeddings
from src.vector_store import search, get_uploaded_documents
from src.llm import generate_answer, rephrase_query, analyze_query_routing, detect_comparison_intent

logger = logging.getLogger("AI-Research-Assistant.RAG")


def compute_keyword_overlap(query, text):

    stop_words = {
        "what", "is", "are", "the", "a", "an", "and", "or", "in", "on", "at", 
        "to", "for", "with", "of", "about", "its", "it", "my", "your", "his", 
        "her", "their", "this", "that", "these", "those", "how", "why", "where", 
        "who", "when", "which", "compare", "difference", "between"
    }

    query_tokens = set(re.findall(r'\b\w+\b', query.lower())) - stop_words
    text_tokens = set(re.findall(r'\b\w+\b', text.lower()))

    if not query_tokens:
        return 0.0

    overlap = len(query_tokens.intersection(text_tokens))
    return overlap / len(query_tokens)


def extract_targeted_documents(question, document_list):

    if not document_list:
        return []

    question_lower = question.lower()
    targeted = []

    for doc in document_list:
        base_name = os.path.splitext(doc)[0].lower()
        if base_name in question_lower or doc.lower() in question_lower:
            targeted.append(doc)

    if targeted:
        return targeted

    try:
        return analyze_query_routing(question, document_list)
    except Exception as e:
        logger.warning(f"Semantic query routing failed: {e}. Searching all documents.")
        return []


def ask_question(question, chat_history=[]):

    try:
        standalone_question = rephrase_query(question, chat_history)
    except Exception as e:
        logger.warning(f"Rephrasing follow-up query failed: {e}. Falling back to original query.")
        standalone_question = question

    try:
        comparison_mode = detect_comparison_intent(standalone_question) or detect_comparison_intent(question)
    except Exception as e:
        logger.warning(f"Failed to detect comparison intent: {e}")
        comparison_mode = False

    uploaded_docs = get_uploaded_documents()
    try:
        targeted_docs = extract_targeted_documents(standalone_question, uploaded_docs)
    except Exception as e:
        logger.warning(f"Failed to route query: {e}. Searching all documents.")
        targeted_docs = []

    if comparison_mode and not targeted_docs:
        targeted_docs = uploaded_docs

    start_retrieval = time.perf_counter()
    query_embedding = create_embeddings([standalone_question])[0]
    
    retrieved_docs = []
    retrieved_metadatas = []
    retrieved_distances = []
    
    k_total = 20 if comparison_mode else 10
    
    if len(targeted_docs) > 1:
        k_per_doc = max(5 if comparison_mode else 3, k_total // len(targeted_docs))
        logger.info(f"Partitioned search active. Fetching {k_per_doc} chunks per targeted document: {targeted_docs}")
        
        for doc in targeted_docs:
            try:
                res = search(query_embedding, k=k_per_doc, targeted_docs=[doc])
                if res.get("documents") and res["documents"][0]:
                    retrieved_docs.extend(res["documents"][0])
                    retrieved_metadatas.extend(res["metadatas"][0])
                    retrieved_distances.extend(res["distances"][0])
            except Exception as e:
                logger.error(f"Error querying ChromaDB for document '{doc}': {e}")
    else:
        results = search(query_embedding, k=k_total, targeted_docs=targeted_docs)
        retrieved_docs = results["documents"][0] if results.get("documents") else []
        retrieved_metadatas = results["metadatas"][0] if results.get("metadatas") else []
        retrieved_distances = results["distances"][0] if results.get("distances") else []
        
    retrieval_time = time.perf_counter() - start_retrieval

    candidates = []

    for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metadatas)):
        source_name = meta.get("source", "Unknown Document")
        chunk_idx = meta.get("chunk_index", 0)
        
        distance = retrieved_distances[i] if i < len(retrieved_distances) else None
        if distance is not None:
            similarity = 1.0 - (distance / 2.0)
            similarity = max(0.0, min(1.0, similarity))
        else:
            similarity = 0.0

        keyword_score = compute_keyword_overlap(standalone_question, doc)
        
        alpha = 0.4
        combined_score = similarity * (1.0 - alpha) + keyword_score * alpha

        candidates.append({
            "source": source_name,
            "chunk_index": chunk_idx,
            "content": doc,
            "similarity": int(similarity * 100),
            "combined_score": combined_score
        })

    candidates.sort(key=lambda x: x["combined_score"], reverse=True)

    context_limit = 8 if comparison_mode else 5

    if comparison_mode and targeted_docs:
        top_candidates = []
        seen_docs = set()
        
        for doc_name in targeted_docs:
            best_cand = None
            for cand in candidates:
                if cand["source"] == doc_name:
                    if best_cand is None or cand["combined_score"] > best_cand["combined_score"]:
                        best_cand = cand
            if best_cand:
                top_candidates.append(best_cand)
                seen_docs.add(doc_name)
                
        added_ids = {f"{c['source']}_chunk_{c['chunk_index']}" for c in top_candidates}
        for cand in candidates:
            if len(top_candidates) >= context_limit:
                break
            cand_id = f"{cand['source']}_chunk_{cand['chunk_index']}"
            if cand_id not in added_ids:
                top_candidates.append(cand)
                added_ids.add(cand_id)
                
        top_candidates.sort(key=lambda x: x["combined_score"], reverse=True)
    else:
        top_candidates = candidates[:context_limit]

    formatted_segments = []
    sources = []

    for i, cand in enumerate(top_candidates):
        formatted_segments.append(
            f"[Segment {i+1} | Source: {cand['source']} (Chunk {cand['chunk_index']})]\n{cand['content']}"
        )
        sources.append({
            "source": cand["source"],
            "chunk_index": cand["chunk_index"],
            "content": cand["content"],
            "similarity": cand["similarity"]
        })

    context = "\n\n".join(formatted_segments)

    start_generation = time.perf_counter()
    try:
        answer = generate_answer(
            standalone_question,
            context,
            comparison_mode=comparison_mode
        )
    except Exception as e:
        logger.error(f"Failed to generate answer from Gemini API: {e}. Compiling local fallback response.")
        
        answer = (
            "[Warning] **Gemini API Error (Quota/Rate Limit Exceeded or Offline)**\n\n"
            "I was unable to contact the Gemini API to synthesize an answer. However, "
            "I successfully retrieved the most relevant matching text segments from your "
            "documents locally. Please review the summaries below:\n\n"
        )
        for idx, cand in enumerate(top_candidates, start=1):
            excerpt = cand['content'][:250].strip() + "..."
            answer += f"**{idx}. From `{cand['source']}` (Chunk {cand['chunk_index']}, Similarity: {cand['similarity']}%):**\n"
            answer += f"> {excerpt}\n\n"
        answer += "\n*For the full text, check the 'Retrieved Sources' panel below.*"

    generation_time = time.perf_counter() - start_generation

    try:
        cited_files = re.findall(r'\[Source:\s*([^\]]+?)\s*\(Chunk\s*\d+\)\]', answer)
        contributing_docs = sorted(list(set([f.strip() for f in cited_files if f.strip()])))
    except Exception:
        contributing_docs = []

    chunks_per_doc = {}
    for cand in top_candidates:
        src = cand["source"]
        chunks_per_doc[src] = chunks_per_doc.get(src, 0) + 1

    retrieval_stats = {
        "chunks_per_doc": chunks_per_doc,
        "contributing_docs": contributing_docs,
        "docs_involved_count": len(chunks_per_doc),
        "comparison_mode": comparison_mode
    }

    return answer, sources, retrieval_time, generation_time, standalone_question, retrieval_stats