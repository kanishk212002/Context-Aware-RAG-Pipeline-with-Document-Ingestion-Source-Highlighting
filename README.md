# Context-Aware RAG Pipeline with Source Attribution

A production-ready Retrieval-Augmented Generation (RAG) system that ingests documents, performs intelligent chunking, creates embeddings, and generates accurate answers with complete source attribution.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Pipeline Flow](#pipeline-flow)

---

## 🎯 Overview

This RAG pipeline provides an end-to-end solution for document processing and question-answering with precise source attribution. The system accepts multiple document formats, processes them intelligently, and generates contextually accurate answers while citing exact sources.

### Key Capabilities

1. **Multi-Format Document Ingestion**: PDF, DOCX, DOC, CSV
2. **Intelligent Chunking**: Agentic chunking using Gemini 2.5 Flash
3. **Vector Storage**: ChromaDB with HuggingFace embeddings
4. **Semantic Retrieval**: Context-aware document retrieval
5. **Source Attribution**: Complete traceability of answers to source chunks

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER UPLOADS DOCUMENT                     │
│                     (PDF / DOCX / DOC / CSV)                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DOCUMENT INGESTION LAYER                      │
│                    (/upload-path endpoint)                       │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  File Type Detection & Validation                          │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
   ┌─────────┐    ┌──────────┐    ┌──────────┐
   │   PDF   │    │   DOCX   │    │   CSV    │
   └────┬────┘    └─────┬────┘    └─────┬────┘
        │               │               │
        ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│              DOCUMENT PROCESSING MODULE                          │
│              (document_processor.py)                             │
│                                                                   │
│  PDF Processing:                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1. PDF → Images (via Poppler)                              │ │
│  │ 2. Images → Mistral OCR (Pixtral)                          │ │
│  │ 3. Extract text in LaTeX format                            │ │
│  │ 4. Preprocess & clean text                                 │ │
│  │ 5. Save as: data/<pdf_name>/pages/page1.md, page2.md...   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  DOCX Processing:                                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1. Extract text from DOCX                                  │ │
│  │ 2. Process with Mistral OCR                                │ │
│  │ 3. Save as: data/<docx_name>/pages/page1.md               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  CSV Processing:                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1. Read CSV with Pandas                                    │ │
│  │ 2. Convert to Markdown table                               │ │
│  │ 3. Save as: data/<csv_name>/page1.md                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  INTELLIGENT CHUNKING LAYER                      │
│                      (chunker.py)                                │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Agentic Chunker (Gemini 2.5 Flash)                        │ │
│  │                                                              │ │
│  │  1. Read all .md files from data/<doc_name>/pages/         │ │
│  │  2. Combine page content                                    │ │
│  │  3. Send to Gemini for semantic analysis:                  │ │
│  │     • Identify topic boundaries                            │ │
│  │     • Suggest optimal split points                         │ │
│  │     • Extract topics & reasoning                           │ │
│  │  4. Split text at suggested positions                      │ │
│  │  5. Validate chunk sizes (300-800 tokens)                  │ │
│  │  6. Adjust if needed (merge/split)                         │ │
│  │  7. Token counting via tiktoken                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Output: data/<doc_name>/chunks.json                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  {                                                          │ │
│  │    "document_info": {                                       │ │
│  │      "filename": "document.pdf",                            │ │
│  │      "total_chunks": 12,                                    │ │
│  │      "chunking_method": "agentic_gemini",                  │ │
│  │      "token_range": "300-800"                              │ │
│  │    },                                                        │ │
│  │    "chunks": [                                             │ │
│  │      {                                                      │ │
│  │        "chunk_id": "doc_chunk_001",                        │ │
│  │        "chunk_number": 1,                                  │ │
│  │        "content": "...",                                   │ │
│  │        "token_count": 510,                                 │ │
│  │        "source_info": {...},                               │ │
│  │        "gemini_analysis": {"topic": "..."}                 │ │
│  │      }                                                      │ │
│  │    ]                                                        │ │
│  │  }                                                          │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING GENERATION LAYER                    │
│              (/create-embeddings endpoint)                       │
│                    (embedder.py)                                 │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Embedding Manager                                          │ │
│  │                                                              │ │
│  │  1. Load chunks.json                                        │ │
│  │  2. Prepare documents & metadata                           │ │
│  │  3. Generate embeddings:                                    │ │
│  │     • Model: sentence-transformers/all-MiniLM-L6-v2        │ │
│  │     • Batch processing (100 chunks/batch)                  │ │
│  │  4. Store in ChromaDB:                                     │ │
│  │     • Collection: <doc_name>_embeddings                    │ │
│  │     • Persist to: ./chroma_db/                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Stored Data per Chunk:                                         │
│  • chunk_text (content)                                         │
│  • embedding_vector (384-dim)                                   │
│  • source_filename                                              │
│  • chunk_id                                                     │
│  • chunk_number                                                 │
│  • document_name                                                │
│  • topic (from Gemini analysis)                                 │
│  • token_count                                                  │
│  • created_at                                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                         QUERY LAYER                              │
│                    (User asks question)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL MODULE                              │
│                (/api/retrieve endpoint)                          │
│                   (retrieval.py)                                 │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Semantic Search Process:                                   │ │
│  │                                                              │ │
│  │  1. Convert user query to embedding                         │ │
│  │  2. Load ChromaDB collection                               │ │
│  │  3. Perform similarity search (cosine distance)            │ │
│  │  4. Retrieve top-k chunks (default: 5)                     │ │
│  │  5. Return ranked results with scores                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Output Format:                                                  │
│  {                                                               │
│    "query": "What does chromadb say",                           │
│    "collection": "embeddings",                                   │
│    "total_results": 2,                                          │
│    "results": [                                                  │
│      {                                                           │
│        "rank": 1,                                               │
│        "text": "chunk content...",                             │
│        "score": 1.826,                                         │
│        "source_filename": "doc.pdf",                           │
│        "chunk_number": 1,                                       │
│        "chunk_id": "doc_chunk_001",                            │
│        "topic": "Project Overview"                             │
│      }                                                           │
│    ]                                                             │
│  }                                                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ANSWER GENERATION LAYER                         │
│                   (/answer endpoint)                             │
│                (answer_generator.py)                             │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  LLM Answer Generator (Gemini 2.5 Flash)                   │ │
│  │                                                              │ │
│  │  1. Receive query + retrieved chunks                        │ │
│  │  2. Format prompt with context                             │ │
│  │  3. Send to Gemini with strict instructions:               │ │
│  │     • Answer ONLY from provided context                    │ │
│  │     • Cite sources in format: [Source: file | chunk N]    │ │
│  │  4. Parse LLM response                                     │ │
│  │  5. Extract citations                                       │ │
│  │  6. Format final output                                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Final Output:                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  {                                                          │ │
│  │    "final_answer": "The answer text...",                   │ │
│  │    "sources_used": [                                       │ │
│  │      {                                                      │ │
│  │        "display": "document.pdf — Chunk 3",                │ │
│  │        "source_filename": "document.pdf",                  │ │
│  │        "chunk_number": 3,                                  │ │
│  │        "chunk_id": "doc_chunk_003",                        │ │
│  │        "score": 0.85                                       │ │
│  │      }                                                      │ │
│  │    ],                                                       │ │
│  │    "retrieved_context": [                                  │ │
│  │      {                                                      │ │
│  │        "label": "[Source: doc.pdf | chunk 3]",            │ │
│  │        "text": "Full text of chunk..."                    │ │
│  │      }                                                      │ │
│  │    ]                                                        │ │
│  │  }                                                          │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Document Processing
- **Multi-format Support**: PDF, DOCX, DOC, CSV
- **Advanced OCR**: Mistral Pixtral for PDF text extraction with LaTeX support
- **CSV Intelligence**: Automatic conversion to markdown tables
- **Structured Storage**: Organized file system with `data/<document_name>/` structure

### Intelligent Chunking
- **Agentic Chunking**: Gemini 2.5 Flash analyzes semantic boundaries
- **Topic Identification**: Automatic topic extraction for each chunk
- **Size Optimization**: 300-800 token chunks with smart merging/splitting
- **Token Counting**: Precise token calculation using tiktoken

### Embedding & Storage
- **Free Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **Vector Database**: ChromaDB with persistent storage
- **Rich Metadata**: Complete source tracking with timestamps

### Retrieval & Generation
- **Semantic Search**: Vector similarity with cosine distance scoring
- **Source Attribution**: Exact chunk and document tracking
- **LLM Generation**: Gemini 2.5 Flash for answer generation
- **Citation Format**: Clean, readable source citations

---

## 🛠️ Tech Stack

### Core Framework
- **FastAPI**: REST API framework
- **Python 3.8+**: Core language
- **Uvicorn**: ASGI server

### Document Processing
- **Poppler**: PDF to image conversion
- **Mistral AI (Pixtral)**: OCR and text extraction
- **python-docx**: DOCX processing
- **Pandas**: CSV handling

### Chunking
- **LangChain**: Framework integration
- **Google Gemini 2.5 Flash**: Semantic analysis
- **tiktoken**: Token counting

### Embeddings & Storage
- **HuggingFace Transformers**: sentence-transformers/all-MiniLM-L6-v2
- **ChromaDB**: Vector database
- **LangChain Chroma**: ChromaDB integration

### LLM
- **Google Gemini 2.5 Flash**: Answer generation

---

## 📁 Project Structure

```
rag-pipeline/
│
├── main.py                      # FastAPI application & endpoints
├── document_processor.py        # PDF/DOCX/CSV processing logic
├── chunker.py                   # Agentic chunking with Gemini
├── embedder.py                  # Embedding generation & ChromaDB storage
├── retrieval.py                 # Semantic search module
├── answer_generator.py          # LLM answer generation
│
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (API keys)
├── README.md                    # This file
│
├── data/                        # Processed documents storage
│   └── <document_name>/
│       ├── pages/              # Extracted markdown pages
│       │   ├── page1.md
│       │   ├── page2.md
│       │   └── ...
│       └── chunks.json         # Generated chunks
│
└── chroma_db/                   # ChromaDB persistent storage
    └── <collection_name>/
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Poppler (for PDF processing)
- Git (optional)

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install Poppler

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
Download from: https://blog.alivate.com.au/poppler-windows/

### Step 3: Set Up Environment Variables
Create a `.env` file in the project root:
```env
# Required API Keys
MISTRAL_API_KEY=your_mistral_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional Configuration
CHROMA_DB_PATH=./chroma_db
DATA_FOLDER=./data
```

### Step 4: Run the Server
```bash
python main.py
```

Or with Uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://127.0.0.1:8000`

Interactive API documentation: `http://127.0.0.1:8000/docs`

---

## ⚙️ Configuration

### API Keys Required

1. **Mistral API Key**: For OCR processing
   - Get it from: https://console.mistral.ai/

2. **Google Gemini API Key**: For chunking and answer generation
   - Get it from: https://makersuite.google.com/app/apikey

### Embedding Model
The system uses `sentence-transformers/all-MiniLM-L6-v2` which is:
- Free and open-source
- 384-dimensional embeddings
- Lightweight and fast
- No API key required

To change the embedding model, modify `embedder.py`:
```python
self.embeddings = HuggingFaceEmbeddings(
    model_name="your-preferred-model"
)
```

---

## 📡 API Documentation

### 1. Upload Document
**Endpoint:** `POST /upload-path`

Upload a document for processing (PDF, DOCX, DOC, CSV).

**Request Body:**
```json
{
  "file_path": "/path/to/your/document.pdf"
}
```

**Response:**
```json
{
  "message": "PDF processed and chunked successfully",
  "original_path": "/path/to/document.pdf",
  "normalized_path": "/normalized/path/document.pdf",
  "filename": "document.pdf",
  "file_size": 245760,
  "file_extension": ".pdf",
  "mistral_processing": {
    "data_folder": "data/document",
    "total_pages": 5,
    "processed_pages": [...]
  },
  "chunking_result": {
    "chunks_file": "data/document/chunks.json",
    "total_chunks": 12,
    "gemini_analysis": {...}
  }
}
```

**Process Flow:**
1. Validates file path and type
2. Converts PDF pages to images (Poppler)
3. Extracts text using Mistral OCR (Pixtral)
4. Saves markdown pages to `data/<doc_name>/pages/`
5. Performs agentic chunking with Gemini
6. Saves chunks to `data/<doc_name>/chunks.json`

---

### 2. Create Embeddings
**Endpoint:** `POST /create-embeddings`

Generate and store embeddings for chunked documents.

**Request Body:**
```json
{
  "chunks_file_path": "data/document/chunks.json",
  "collection_name": "my_document_embeddings"
}
```

**Response:**
```json
{
  "message": "Embeddings created and stored successfully",
  "chunks_file": "data/document/chunks.json",
  "result": {
    "status": "success",
    "chunks_processed": 12,
    "collection_name": "my_document_embeddings",
    "storage_result": {
      "total_embeddings": 12,
      "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    }
  },
  "status": "success"
}
```

**Process Flow:**
1. Loads chunks from JSON file
2. Generates embeddings using HuggingFace model
3. Stores in ChromaDB with metadata
4. Creates persistent collection

---

### 3. Query Documents (Simple Retrieval)
**Endpoint:** `POST /query-documents`

Search for similar chunks without generating an answer.

**Request Body:**
```json
{
  "query_text": "What are the safety precautions?",
  "collection_name": "my_document_embeddings",
  "n_results": 5
}
```

**Response:**
```json
{
  "message": "Query completed successfully",
  "query": "What are the safety precautions?",
  "collection": "my_document_embeddings",
  "results": {
    "documents": [[...]],
    "metadatas": [[...]],
    "distances": [[...]]
  },
  "status": "success"
}
```

---

### 4. Retrieve Documents (Advanced)
**Endpoint:** `POST /api/retrieve`

Advanced retrieval with formatted results and scoring.

**Request Body:**
```json
{
  "query": "What does chromadb say",
  "collection_name": "embeddings",
  "top_k": 5
}
```

**Real Response Example:**
```json
{
  "query": "What does chromadb say",
  "collection": "embeddings",
  "total_results": 2,
  "results": [
    {
      "rank": 1,
      "text": "# AI Developer Assignment: Build a RAG Pipeline...",
      "score": 1.826218605041504,
      "source_filename": "AI Developer Assignment_ Build a RAG Pipeline With Source Attribution.pdf",
      "chunk_number": 1,
      "document_name": "AI Developer Assignment_ Build a RAG Pipeline With Source Attribution",
      "chunk_id": "AI Developer Assignment_ Build a RAG Pipeline With Source Attribution_chunk_001",
      "topic": "Project Overview and Core RAG Pipeline Requirements",
      "token_count": 510,
      "created_at": "2025-11-16T22:39:29.184041",
      "chunking_method": "agentic_gemini"
    }
  ]
}
```

---

### 5. Answer Question (Full RAG)
**Endpoint:** `POST /answer`

Complete RAG pipeline with answer generation and source attribution.

**Request Body:**
```json
{
  "question": "What are the safety precautions for machine X?",
  "collection_name": "embeddings",
  "top_k": 3
}
```

**Real Response Example:**
```json
{
  "final_answer": "**Final Answer:**\nI don't know the safety precautions for **machine X**—the provided context does not contain any information about it.\n\n---\n**Sources Used:**\nNone (no relevant context found).\n\n---\n**Retrieved Context:**\nNo applicable chunks were retrieved. The available context only describes RAG pipeline requirements and formatting guidelines.",
  
  "sources_used": [
    {
      "display": "AI Developer Assignment_ Build a RAG Pipeline With Source Attribution.pdf — Chunk 2",
      "source_filename": "AI Developer Assignment_ Build a RAG Pipeline With Source Attribution.pdf",
      "chunk_number": 2,
      "chunk_id": "AI Developer Assignment_ Build a RAG Pipeline With Source Attribution_chunk_002",
      "score": 1.7131905555725098
    },
    {
      "display": "AI Developer Assignment_ Build a RAG Pipeline With Source Attribution.pdf — Chunk 1",
      "source_filename": "AI Developer Assignment_ Build a RAG Pipeline With Source Attribution.pdf",
      "chunk_number": 1,
      "chunk_id": "AI Developer Assignment_ Build a RAG Pipeline With Source Attribution_chunk_001",
      "score": 1.9266726970672607
    }
  ],
  
  "retrieved_context": [
    {
      "label": "[Source: AI Developer Assignment_ Build a RAG Pipeline With Source Attribution.pdf | chunk 2]",
      "text": "\\end{itemize}\n\\end{itemize}\n\n\\section{5. Final Output Format (Mandatory)}..."
    },
    {
      "label": "[Source: AI Developer Assignment_ Build a RAG Pipeline With Source Attribution.pdf | chunk 1]",
      "text": "# AI Developer Assignment: Build a RAG Pipeline..."
    }
  ]
}
```

**Process Flow:**
1. Retrieves top-k relevant chunks from ChromaDB
2. Formats context for LLM with proper structure
3. Sends to Gemini with strict grounding instructions
4. LLM generates answer ONLY from provided context
5. Extracts and formats citations
6. Returns comprehensive response with sources and context

**Note:** The LLM correctly refuses to hallucinate when the context doesn't contain the requested information, as shown in the example above.

---

### 6. List Collections
**Endpoint:** `GET /list-collections`

Get all available ChromaDB collections.

**Response:**
```json
{
  "message": "Collections retrieved successfully",
  "collections": [
    {
      "name": "embeddings",
      "count": 2,
      "metadata": {...}
    }
  ],
  "total_collections": 1
}
```

---

### 7. Get Collection Stats
**Endpoint:** `GET /collection-stats/{collection_name}`

Get statistics for a specific collection.

**Response:**
```json
{
  "message": "Collection stats retrieved successfully",
  "stats": {
    "collection_name": "embeddings",
    "total_documents": 2,
    "metadata": {
      "description": "Embeddings for embeddings"
    }
  }
}
```

---

### 8. Debug Collection
**Endpoint:** `GET /debug/collection/{collection_name}`

Debug endpoint to inspect collection internals.

**Response:**
```json
{
  "collection_name": "embeddings",
  "total_documents": 2,
  "metadata": {},
  "sample_ids": ["chunk_001", "chunk_002"],
  "sample_metadata": [...],
  "has_embeddings": true
}
```

---

## 💡 Usage Examples

### Example 1: Complete Workflow with cURL

```bash
# Step 1: Upload and process PDF
curl -X POST "http://127.0.0.1:8000/upload-path" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/home/user/documents/employee_handbook.pdf"
  }'

# Step 2: Create embeddings
curl -X POST "http://127.0.0.1:8000/create-embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "chunks_file_path": "data/employee_handbook/chunks.json",
    "collection_name": "employee_handbook_embeddings"
  }'

# Step 3: Test retrieval
curl -X POST "http://127.0.0.1:8000/api/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the vacation policy?",
    "collection_name": "employee_handbook_embeddings",
    "top_k": 5
  }'

# Step 4: Ask a question with full RAG
curl -X POST "http://127.0.0.1:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the vacation policy?",
    "collection_name": "employee_handbook_embeddings",
    "top_k": 5
  }'
```

---

### Example 2: Processing CSV File

```bash
# Upload CSV
curl -X POST "http://127.0.0.1:8000/upload-path" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/home/user/data/sales_report.csv"
  }'

# Create embeddings
curl -X POST "http://127.0.0.1:8000/create-embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "chunks_file_path": "data/sales_report/chunks.json",
    "collection_name": "sales_embeddings"
  }'

# Query
curl -X POST "http://127.0.0.1:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What were the top selling products in Q3?",
    "collection_name": "sales_embeddings",
    "top_k": 5
  }'
```

---

### Example 3: Python Client Script

```python
import requests
import json

# Configuration
BASE_URL = "http://127.0.0.1:8000"
DOCUMENT_PATH = "/path/to/your/document.pdf"

def upload_document(file_path):
    """Upload and process document"""
    response = requests.post(
        f"{BASE_URL}/upload-path",
        json={"file_path": file_path}
    )
    return response.json()

def create_embeddings(chunks_file, collection_name):
    """Generate and store embeddings"""
    response = requests.post(
        f"{BASE_URL}/create-embeddings",
        json={
            "chunks_file_path": chunks_file,
            "collection_name": collection_name
        }
    )
    return response.json()

def ask_question(question, collection_name, top_k=5):
    """Ask question and get answer with sources"""
    response = requests.post(
        f"{BASE_URL}/answer",
        json={
            "question": question,
            "collection_name": collection_name,
            "top_k": top_k
        }
    )
    return response.json()

# Main workflow
if __name__ == "__main__":
    # Step 1: Upload document
    print("Step 1: Uploading document...")
    upload_result = upload_document(DOCUMENT_PATH)
    print(f"✓ Document processed: {upload_result['filename']}")
    chunks_file = upload_result["chunking_result"]["chunks_file"]
    print(f"✓ Created {upload_result['chunking_result']['total_chunks']} chunks")
    
    # Step 2: Create embeddings
    print("\nStep 2: Creating embeddings...")
    collection_name = "my_collection"
    embed_result = create_embeddings(chunks_file, collection_name)
    print(f"✓ Embeddings created: {embed_result['result']['chunks_processed']} chunks")
    
    # Step 3: Ask questions
    print("\nStep 3: Asking questions...")
    questions = [
        "What is the main topic of this document?",
        "What are the key requirements mentioned?",
        "Are there any safety guidelines?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        answer_data = ask_question(question, collection_name, top_k=3)
        
        print(f"\n💡 Answer:\n{answer_data['final_answer']}\n")
        
        print("📚 Sources:")
        for source in answer_data['sources_used']:
            print(f"  • {source['display']} (score: {source['score']:.4f})")
        
        print("-" * 80)
```

---

### Example 4: Testing the System

```python
# test_rag_system.py
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_system():
    """Test the complete RAG pipeline"""
    
    # 1. Check available collections
    print("1. Checking available collections...")
    response = requests.get(f"{BASE_URL}/list-collections")
    collections = response.json()
    print(f"   Found {collections['total_collections']} collections")
    for col in collections['collections']:
        print(f"   - {col['name']}: {col['count']} documents")
    
    # 2. Test retrieval
    print("\n2. Testing retrieval...")
    collection_name = "embeddings"  # Use your collection name
    response = requests.post(
        f"{BASE_URL}/api/retrieve",
        json={
            "query": "What does chromadb say",
            "collection_name": collection_name,
            "top_k": 5
        }
    )
    retrieval_result = response.json()
    print(f"   Retrieved {retrieval_result['total_results']} results")
    
    # 3. Test full RAG
    print("\n3. Testing full RAG pipeline...")
    response = requests.post(
        f"{BASE_URL}/answer",
        json={
            "question": "What are the requirements for the RAG pipeline?",
            "collection_name": collection_name,
            "top_k": 3
        }
    )
    answer_data = response.json()
    
    print(f"\n   Answer:\n   {answer_data['final_answer'][:200]}...")
    print(f"\n   Sources used: {len(answer_data['sources_used'])}")
    
    return True

if __name__ == "__main__":
    test_system()
```

---

## 🔄 Pipeline Flow

### Complete End-to-End Flow

```
User Input → Document Processing → Chunking → Embedding → Storage → Query → Retrieval → LLM → Answer
```

### Detailed Step-by-Step Process

#### Phase 1: Document Ingestion

1. **Upload Document** (`POST /upload-path`)
   ```
   User provides: /path/to/document.pdf
   ↓
   System validates: File exists? Supported format?
   ↓
   Routes to: PDF/DOCX/CSV processor
   ```

2. **PDF Processing** (`document_processor.py`)
   ```
   PDF File
   ↓ [Poppler]
   Images (page1.jpg, page2.jpg...)
   ↓ [Mistral Pixtral OCR]
   LaTeX Text Extraction
   ↓ [Preprocessing]
   Clean Markdown Files
   ↓ [Save]
   data/document_name/pages/page1.md
   data/document_name/pages/page2.md
   ```

3. **DOCX Processing** (`document_processor.py`)
   ```
   DOCX File
   ↓ [python-docx]
   Text Extraction
   ↓ [Mistral Processing]
   Formatted Content
   ↓ [Save]
   data/document_name/pages/page1.md
   ```

4. **CSV Processing** (`document_processor.py`)
   ```
   CSV File
   ↓ [Pandas]
   DataFrame
   ↓ [to_markdown()]
   Markdown Table
   ↓ [Save]
   data/document_name/page1.md
   ```

#### Phase 2: Intelligent Chunking

5. **Agentic Chunking** (`chunker.py`)
   ```
   Read all .md files from pages/
   ↓
   Combine into single text
   ↓ [Count tokens with tiktoken]
   Total: 2,450 tokens
   ↓ [Send to Gemini 2.5 Flash]
   Prompt: "Analyze this text and suggest optimal chunk boundaries..."
   ↓ [Gemini Analysis]
   {
     "suggested_splits": [800, 1600, 2200],
     "topics": ["Introduction", "Requirements", "Implementation"],
     "reasoning": ["Natural section break after objectives..."]
   }
   ↓ [Split text at positions]
   Chunks: ["Intro text...", "Requirements text...", "Implementation..."]
   ↓ [Validate sizes]
   Chunk 1: 510 tokens ✓
   Chunk 2: 473 tokens ✓
   ↓ [Save]
   data/document_name/chunks.json
   ```

6. **chunks.json Structure**
   ```json
   {
     "document_info": {
       "filename": "document.pdf",
       "document_name": "document",
       "total_chunks": 2,
       "processed_date": "2025-11-16T22:39:29.183373",
       "chunking_method": "agentic_gemini",
       "token_range": "300-800"
     },
     "chunks": [
       {
         "chunk_id": "document_chunk_001",
         "chunk_number": 1,
         "content": "Full text content...",
         "token_count": 510,
         "source_info": {
           "filename": "document.pdf",
           "document_name": "document"
         },
         "gemini_analysis": {
           "topic": "Introduction and Overview",
           "chunk_reasoning": "Semantic boundary identified"
         },
         "created_at": "2025-11-16T22:39:29.184041"
       }
     ]
   }
   ```

#### Phase 3: Embedding Generation

7. **Create Embeddings** (`POST /create-embeddings`)
   ```
   Load chunks.json
   ↓
   Extract chunks: 12 total
   ↓ [Prepare for embedding]
   Chunk 1: "Full text content..."
   Metadata 1: {filename, chunk_id, topic...}
   ↓ [HuggingFace Model]
   Model: sentence-transformers/all-MiniLM-L6-v2
   ↓ [Batch processing: 100 chunks/batch]
   Batch 1: Generate embeddings for chunks 1-12
   ↓ [384-dimensional vectors]
   Chunk 1 embedding: [0.023, -0.145, 0.089, ...]
   ↓ [Store in ChromaDB]
   Collection: "document_embeddings"
   ↓ [Persist]
   ./chroma_db/document_embeddings/
   ```

8. **ChromaDB Storage Structure**
   ```
   ChromaDB Collection: "document_embeddings"
   ├── Documents: ["chunk text 1", "chunk text 2", ...]
   ├── Embeddings: [[0.023, -0.145, ...], [0.156, 0.023, ...], ...]
   ├── Metadata: [
   │     {
   │       "chunk_id": "doc_chunk_001",
   │       "chunk_number": 1,
   │       "source_filename": "document.pdf",
   │       "document_name": "document",
   │       "topic": "Introduction",
   │       "token_count": 510,
   │       "created_at": "2025-11-16T22:39:29.184041"
   │     }
   │   ]
   └── IDs: ["doc_chunk_001", "doc_chunk_002", ...]
   ```

#### Phase 4: Query & Retrieval

9. **User Query** (`POST /answer`)
   ```
   User asks: "What are the safety precautions for machine X?"
   ↓
   Convert query to embedding
   ↓ [HuggingFace Model]
   Query embedding: [0.156, -0.023, 0.234, ...]
   ```

10. **Semantic Search** (`retrieval.py`)
    ```
    Query embedding: [0.156, -0.023, 0.234, ...]
    ↓ [Load ChromaDB collection]
    Collection: "embeddings"
    Total documents: 2
    ↓ [Cosine similarity search]
    Calculate distance between query and all chunk embeddings
    ↓ [Rank by similarity]
    Chunk 2: distance = 1.713 (most similar)
    Chunk 1: distance = 1.927
    ↓ [Return top-k]
    Top 3 results with metadata
    ```

11. **Retrieval Response**
    ```json
    {
      "query": "What are the safety precautions for machine X?",
      "collection": "embeddings",
      "total_results": 2,
      "results": [
        {
          "rank": 1,
          "text": "Full chunk text...",
          "score": 1.713,
          "source_filename": "AI Developer Assignment.pdf",
          "chunk_number": 2,
          "chunk_id": "assignment_chunk_002",
          "topic": "Output Format Requirements"
        }
      ]
    }
    ```

#### Phase 5: Answer Generation

12. **Format Context** (`answer_generator.py`)
    ```python
    # Prepare context for LLM
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"\n\n[Source: {chunk['source_filename']} | chunk {chunk['chunk_number']}]\n"
        context += chunk['text']
    
    # Build prompt
    prompt = f"""
    You are a helpful assistant. Answer the question based ONLY on the context provided below.
    
    Context:
    {context}
    
    Question: {user_question}
    
    Instructions:
    - Answer ONLY using information from the context
    - If the answer is not in the context, say "I don't know"
    - Cite sources using format: [Source: filename.pdf | chunk N]
    """
    ```

13. **LLM Processing** (Gemini 2.5 Flash)
    ```
    Prompt sent to Gemini
    ↓
    LLM analyzes context
    ↓
    Determines: Context does NOT contain info about "machine X"
    ↓
    Generates honest response:
    "I don't know the safety precautions for machine X—
     the provided context does not contain any information about it."
    ↓
    Returns structured response
    ```

14. **Final Response Assembly**
    ```json
    {
      "final_answer": "I don't know the safety precautions for machine X—the provided context does not contain any information about it.",
      
      "sources_used": [
        {
          "display": "AI Developer Assignment.pdf — Chunk 2",
          "source_filename": "AI Developer Assignment.pdf",
          "chunk_number": 2,
          "chunk_id": "assignment_chunk_002",
          "score": 1.713
        }
      ],
      
      "retrieved_context": [
        {
          "label": "[Source: AI Developer Assignment.pdf | chunk 2]",
          "text": "Full text of the chunk..."
        }
      ]
    }
    ```

---

## 🎯 Key Features Demonstrated

### 1. Document Processing
- ✅ Multi-format support (PDF, DOCX, CSV)
- ✅ OCR with Mistral Pixtral
- ✅ Structured storage with proper organization

### 2. Intelligent Chunking
- ✅ Semantic analysis with Gemini
- ✅ Topic identification
- ✅ Token-aware splitting (300-800 tokens)
- ✅ Metadata-rich chunks

### 3. Embedding & Storage
- ✅ Free HuggingFace embeddings
- ✅ ChromaDB vector storage
- ✅ Persistent collections
- ✅ Complete metadata tracking

### 4. Retrieval
- ✅ Semantic similarity search
- ✅ Scored results
- ✅ Full metadata in responses

### 5. Answer Generation
- ✅ Context-grounded responses
- ✅ Source attribution
- ✅ Citation format
- ✅ Honest "I don't know" when context insufficient

---

## 🔍 Understanding the Scores

The **score** field in retrieval results represents the **distance** between the query embedding and chunk embedding:

- **Lower scores = More similar** (closer in vector space)
- **Higher scores = Less similar** (farther in vector space)

Example from real response:
```json
{
  "score": 1.826218605041504  // Distance metric (lower is better)
}
```

**Note:** ChromaDB uses **L2 (Euclidean) distance** by default. A score of 0 would mean perfect match.

---

## 📊 Sample Output Formats

### Successful Answer (Context Available)
```json
{
  "final_answer": "The document specifies that the RAG pipeline must support multiple file formats including PDF, TXT, DOCX, and CSV. The ingestion module should extract text cleanly and split it into chunks of 300-800 tokens.",
  
  "sources_used": [
    {
      "display": "AI Developer Assignment.pdf — Chunk 1",
      "source_filename": "AI Developer Assignment.pdf",
      "chunk_number": 1,
      "chunk_id": "assignment_chunk_001",
      "score": 0.523
    }
  ],
  
  "retrieved_context": [
    {
      "label": "[Source: AI Developer Assignment.pdf | chunk 1]",
      "text": "Document Ingestion Module: Accepts multiple file formats..."
    }
  ]
}
```

### No Answer (Insufficient Context)
```json
{
  "final_answer": "I don't know the safety precautions for machine X—the provided context does not contain any information about it.",
  
  "sources_used": [
    {
      "display": "AI Developer Assignment.pdf — Chunk 2",
      "score": 1.713
    }
  ],
  
  "retrieved_context": [
    {
      "label": "[Source: AI Developer Assignment.pdf | chunk 2]",
      "text": "Sample Input: What does the document say about inventory..."
    }
  ]
}
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "File not found" error**
```bash
# Solution: Use absolute paths
curl -X POST "http://127.0.0.1:8000/upload-path" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/absolute/path/to/document.pdf"}'
```

**2. "Collection not found" error**
```bash
# Solution: Check available collections first
curl -X GET "http://127.0.0.1:8000/list-collections"

# Use exact collection name from the response
```

**3. Poppler not installed**
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler

# Verify installation
pdftoppm -v
```

**4. API keys not loaded**
```bash
# Check .env file exists
ls -la .env

# Verify environment variables
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('GEMINI_API_KEY'))"
```

**5. ChromaDB permission errors**
```bash
# Fix permissions
chmod -R 755 ./chroma_db

# Or remove and recreate
rm -rf ./chroma_db
```

---

## 📝 Requirements.txt

```txt
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
python-dotenv==1.0.0
pandas==2.1.3
python-docx==1.1.0
pdf2image==1.16.3
Pillow==10.1.0
tiktoken==0.5.1
langchain==0.1.0
langchain-google-genai==1.0.0
langchain-community==0.0.10
langchain-core==0.1.10
langchain-chroma==0.1.0
langchain-huggingface==0.0.1
sentence-transformers==2.2.2
transformers==4.35.2
chromadb==0.4.18
mistralai==0.0.11
torch==2.1.0
numpy==1.26.2
```

---

## 🚀 Production Considerations

### Performance Optimization
- **Batch Processing**: Process multiple documents in parallel
- **Caching**: Cache embeddings for frequently queried content
- **Connection Pooling**: Reuse ChromaDB connections
- **Async Processing**: Use FastAPI's async capabilities fully

### Security
- **API Key Management**: Use secure secret management
- **Input Validation**: Validate file paths and sizes
- **Rate Limiting**: Implement rate limiting for API endpoints
- **Authentication**: Add authentication for production use

### Scalability
- **Vector Database**: Consider Pinecone or Weaviate for larger scale
- **Distributed Processing**: Use Celery for background tasks
- **Load Balancing**: Deploy with multiple Uvicorn workers
- **Monitoring**: Add logging and monitoring (Prometheus, Grafana)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Support

For issues, questions, or contributions, please contact the development team.

---

**Built with ❤️ using FastAPI, ChromaDB, and Google Gemini**