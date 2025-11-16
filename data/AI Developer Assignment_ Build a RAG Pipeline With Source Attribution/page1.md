# AI Developer Assignment: Build a RAG Pipeline With Source Attribution

**Project Title: Context-Aware RAG Pipeline with Document Ingestion + Source Highlighting**

---

**Objective**

Build a Retrieval-Augmented Generation (RAG) pipeline that can:

1. Ingest any documents (PDF, TXT, DOCX, CSV).
2. Chunk, embed, and store them in a vector database.
3. Accept a user query and retrieve the most relevant context.
4. Generate an accurate answer strictly based on the retrieved context.
5. Show which document + which chunk was used in the final answer.

---

**Requirements**

1. **Document Ingestion Module**

   Candidate must build a module that:

   - Accepts multiple file formats
     - .pdf, .txt, .docx, .csv
   - Extracts text cleanly
   - Cleans & preprocesses text (remove boilerplate)
   - Splits text into chunks of 300–800 tokens
   - Generates embeddings using:
     - OpenAI Embeddings (preferred)
     - OR any embedding model (sentence-transformers)

---

2. **Vector Store (Choose Any One)**

   - FAISS (local)
   - ChromaDB
   - Pinecone
   - Weaviate
   - Qdrant

   Each stored chunk must include:

   - chunk_text