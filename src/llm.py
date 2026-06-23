import os
import re
import time
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

logger = logging.getLogger("AI-Research-Assistant.LLM")

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")


def generate_content_with_retry(prompt):

    max_retries = 3
    delay = 2.0
    
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt)
        except ResourceExhausted as e:
            logger.warning(
                f"Gemini API ResourceExhausted (rate-limit/quota) encountered on attempt {attempt+1}/{max_retries}. "
                f"Retrying in {delay} seconds..."
            )
            if attempt == max_retries - 1:
                logger.error("Gemini API rate-limit retry attempts exhausted.")
                raise e
            time.sleep(delay)
            delay *= 2.0


def detect_comparison_intent(question):
    comparison_keywords = {
        "compare", "difference", "similarities", "versus", "vs", "contrast", 
        "relate", "relationship", "comparison", "similar", "differ", 
        "similarities", "match up", "comparison of"
    }
    
    tokens = set(re.findall(r'\b\w+\b', question.lower()))
    if tokens.intersection(comparison_keywords):
        return True
        
    q_low = question.lower()
    for phrase in ["different from", "relate to", "relates to"]:
        if phrase in q_low:
            return True
            
    try:
        prompt = f"""Analyze the user's question and determine if it has a comparison intent (i.e. the user is asking to compare, contrast, find differences/similarities, check relationships, or map concepts between different entities, files, terms, or technologies).
        
User Question: "{question}"

Return ONLY "YES" if it has comparison intent, or "NO" if it does not. Do not include any other text.

Output:"""
        response = generate_content_with_retry(prompt)
        return response.text.strip().upper() == "YES"
    except Exception as e:
        logger.warning(f"Failed semantic comparison check: {e}")
        return False


def generate_answer(question, context, comparison_mode=False):

    comparison_instruction = ""
    if comparison_mode:
        comparison_instruction = "\nIMPORTANT: The user has explicit comparison intent. You MUST construct a structured Markdown table summarizing the differences, similarities, and relationships across the documents. Use clear row and column headers corresponding to each file's content."

    prompt = f"""You are a professional research assistant. Answer the user's question using ONLY the provided text segments.{comparison_instruction}

For every fact or piece of information you mention, you MUST cite the source document name and chunk number by using square brackets, e.g., [Source: filename.docx (Chunk 3)].

Strict Guidelines:
1. Base your answer strictly on the provided Context. Do NOT use any outside knowledge, assumptions, or extrapolate.
2. If the context does not contain the answer, state clearly: "I cannot find the answer to this question in the uploaded documents." Do not try to make up an answer.
3. Be detailed, structured, and comprehensive in your response if the context provides sufficient details.
4. Always attribute information to its specific source.
5. When comparing terms, skills, technologies, or concepts across multiple files, construct a structured Markdown table summarizing the differences and similarities. Use clear row/column headers corresponding to each file's content.

Context Segments:
{context}

Question:
{question}

Answer:"""

    response = generate_content_with_retry(prompt)

    return response.text


def rephrase_query(question, chat_history):

    if not chat_history:
        return question

    history_str = ""
    for chat in chat_history[-3:]:
        history_str += f"User: {chat['question']}\nAssistant: {chat['answer']}\n"

    prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that can be answered independently of the history (e.g., by resolving pronouns like "it", "its", "their", "they", "that", etc., back to the entities referenced).

Do NOT answer the question, and do NOT add any new info. Just return the standalone rephrased question. If the follow-up question is already self-contained, return it exactly as is.

Conversation History:
{history_str}
Follow-up Question: {question}

Standalone Question:"""

    response = generate_content_with_retry(prompt)

    return response.text.strip()


def analyze_query_routing(question, document_list):

    if not document_list:
        return []

    document_names_str = ", ".join([f'"{doc}"' for doc in document_list])

    prompt = f"""You are a query router for a multi-document research assistant. 
Given the list of uploaded documents and the user's question, determine if the question is specifically targeting one or more of these documents.

Available Documents:
[{document_names_str}]

User Question:
"{question}"

Instructions:
1. Identify if the user's question specifically references or targets one or more of the available documents (either by name, acronym, file extension, or semantic reference like "my resume", "the research paper", "Satyam case study", etc.).
2. Return ONLY a comma-separated list of the matching document names from the list.
3. If the question is general, asks for comparisons across all documents, or does not specify a document, return "ALL".
4. Do NOT include any code block, markdown, formatting, or extra text. Just return the comma-separated names or "ALL".

Output:"""

    response = generate_content_with_retry(prompt)
    output = response.text.strip()

    if "ALL" in output or not output:
        return []

    targeted_docs = []
    for doc in document_list:
        if doc.lower() in output.lower():
            targeted_docs.append(doc)

    return targeted_docs