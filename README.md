# **RAG Engine — Advanced Retrieval-Augmented Generation Pipeline**

A high-performance, modular, production-ready **Retrieval-Augmented Generation (RAG)** engine designed for real-world AI applications such as enterprise document QA, knowledge mining, intelligent assistants, and automated insights generation.

This RAG engine is built with:

- **Sentence-Transformers** (async batching for maximum speed)  
- **FAISS Vector Search (GPU/CPU fallback)**  
- **Sliding-Window Smart Chunking**  
- **Local LLM Inference (HuggingFace models)**  
- **Real-Time Streaming Token Output**  
- **PDF Text Ingestion (PyMuPDF)**  
- **UI Callback Event Hooks**  
- **Metadata Export for Traceability**  

## **Key Features**

### 1. High-Fidelity PDF Extraction
- Powered by PyMuPDF
- Cleans formatting and normalizes whitespace

### 2. Intelligent Sentence Splitting
- Using SpaCy Sentencizer

### 3. Sliding Window Chunking
- Configurable window + overlap

### 4. Async Embedding Pipeline
- Massive performance boost  
- GPU accelerated, CPU fallback

### 5. FAISS Vector Store
- GPU → CPU fallback  
- Inner product optimized search  

### 6. LLM Answer Generation
Supports: LLaMA, Mistral, Gemma, Phi, Qwen, etc.

### 7. Real-Time Streaming Output
- Token-by-token generation

### 8. UI Callback Hooks
- reading_pdf  
- splitting  
- chunking  
- embedding_batch  
- indexing  
- answer_stream  

### 9. Metadata Export
- Chunks  
- Dimensions  
- Pages  
- Model info  

---

# Architecture Overview
                ┌──────────────┐
                │     PDF      │
                └──────┬───────┘
                       │
                       ▼
         ┌────────────────────────────┐
         │  Text Extraction (PyMuPDF) │
         └─────────────┬──────────────┘
                       │
                       ▼
      ┌────────────────────────────────┐
      │ Sentence Splitting (SpaCy)     │
      └─────────────────┬──────────────┘
                        │
                        ▼
     ┌──────────────────────────────────┐
     │ Sliding Window Chunking (Overlap)│
     └───────────────────┬──────────────┘
                         │
                         ▼
     ┌─────────────────────────────────────┐
     │ Async Embedding (Sentence-BERT)     │
     └────────────────────┬────────────────┘
                          │
                          ▼
        ┌────────────────────────────────┐
        │   FAISS Vector Index (GPU/CPU) │
        └────────────────┬───────────────┘
                         │
                         ▼
           ┌──────────────────────────┐
           │  LLM Prompt + Streaming  │
           └──────────────────────────┘


---

# Installation

```bash
pip install -r requirements.txt
```

(Optional)
```bash
pip install faiss-gpu
```

---

# Usage

Process a PDF
```python
await rag.process_pdf("document.pdf")
```
Ask a Question (Streaming Output)
```python
stream, ctx = rag.ask("What is this document?")
for token in stream:
    print(token, end="")
```
Run the application
```bash
python main.py
```
# Project structure
```
rag-engine/
│
├── rag_inference.py
├── main.py
├── requirements.txt
└── README.md

```

---

# 👨‍💻 Developer

**Sanju Sarkar**  
📧 **sanjusarkar44@hotmail.com**
