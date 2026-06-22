# AI Research Assistant

A Multi-Document Retrieval-Augmented Generation (RAG) System built using Streamlit, ChromaDB, Sentence Transformers, and Google Gemini.

## Overview

AI Research Assistant enables users to upload multiple documents (PDF, DOCX, TXT), perform semantic search, ask natural language questions, compare information across documents, and receive grounded answers with source citations.

The system combines Retrieval-Augmented Generation (RAG), hybrid retrieval, conversation memory, and multi-document reasoning to provide accurate and explainable responses.

---

## Features

### Document Processing

* PDF Upload
* DOCX Upload
* TXT Upload
* Automatic Text Extraction
* Intelligent Chunking

### Retrieval Pipeline

* Sentence Transformer Embeddings
* ChromaDB Vector Storage
* Hybrid Retrieval
* Query Rewriting
* Multi-Document Search

### Generative AI

* Google Gemini Integration
* Context-Aware Answers
* Source Citations
* Conversation Memory
* Comparison Query Support

### Evaluation Dashboard

* Retrieval Latency
* Generation Latency
* Total Response Time
* Document Statistics
* Chunk Statistics

### Production Features

* Dockerized Deployment
* Rate Limit Handling
* Offline Fallback Responses
* Duplicate Document Detection

---

## Tech Stack

* Python
* Streamlit
* ChromaDB
* Sentence Transformers
* LangChain Text Splitters
* Google Gemini
* Docker

---

## System Architecture

Document Upload
↓
Text Extraction
↓
Chunking
↓
Embedding Generation
↓
ChromaDB Storage
↓
Semantic Retrieval
↓
Hybrid Reranking
↓
Gemini LLM
↓
Answer + Citations

---

## Installation

```bash
git clone <repo-url>

cd AI-Research-Assistant

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

Create:

```env
GOOGLE_API_KEY=your_api_key
```

Run:

```bash
streamlit run app.py
```

---

## Docker Deployment

Build:

```bash
docker build -t ai-research-assistant .
```

Run:

```bash
docker run -p 8501:8501 \
-e GOOGLE_API_KEY=YOUR_KEY \
ai-research-assistant
```

---

## Example Queries

* What is RAG?
* Summarize this research paper.
* Compare Resume.pdf and ResearchPaper.pdf.
* What skills in my resume are related to this project?
* What are the key findings of the uploaded documents?

---

## Future Work

* Cross-Encoder Reranking
* OCR Support
* Knowledge Graph Generation
* Multi-Agent Workflows
* Authentication System
