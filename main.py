from fastapi import FastAPI, HTTPException
from pydantic import BaseModel , Field 
from typing import Dict, Any
import os
import pandas as pd
import os

import uvicorn
from document_processor import process_pdf_with_mistral, process_docx_with_mistral
from chunker import AgenticChunker
from embedder import EmbeddingManager
from retrieval import RetrievalModule
from answer_generator import LLMAnswerGenerator

app = FastAPI(title="Context-Aware RAG Pipeline API")

class DocumentPath(BaseModel):
    file_path: str

class EmbeddingRequest(BaseModel):
    chunks_file_path: str
    collection_name: str = None

class QueryRequest(BaseModel):
    query_text: str
    collection_name: str
    n_results: int = 5
class AskRequest(BaseModel):
    question: str
    collection_name: str
    top_k: int = 5
# Request Model
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    collection_name: str = Field(..., description="Name of the collection to search")
    top_k: int = Field(5, description="Number of top results to return", ge=1, le=50)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "collection_name": "my_documents_embeddings",
                "top_k": 5
            }
        }

# Initialize components
chunker = AgenticChunker(min_tokens=300, max_tokens=800)
embedding_manager = EmbeddingManager()
# Initialize Retrieval Module
retrieval_module = RetrievalModule(chroma_db_path="./chroma_db")
answer_generator = LLMAnswerGenerator()

@app.post("/upload-path")
async def upload_document_path(doc_path: DocumentPath):
    """
    Accept a local file path for document processing
    """
    import pandas as pd

    # Normalize the path to handle different OS formats
    normalized_path = os.path.normpath(doc_path.file_path)
    
    # Check if file exists
    if not os.path.exists(normalized_path):
        raise HTTPException(status_code=404, detail=f"File not found: {normalized_path}")
    
    # Check if it's a file (not directory)
    if not os.path.isfile(normalized_path):
        raise HTTPException(status_code=400, detail="Path is not a file")
    
    # Get file info
    file_size = os.path.getsize(normalized_path)
    file_extension = os.path.splitext(normalized_path)[1].lower()
    filename = os.path.basename(normalized_path)

    response = {
        "message": "File path received successfully",
        "original_path": doc_path.file_path,
        "normalized_path": normalized_path,
        "filename": filename,
        "file_size": file_size,
        "file_extension": file_extension
    }

    # ============================
    # CASE 1: PDF / DOC / DOCX
    # ============================
    if file_extension in ['.pdf', '.docx', '.doc', '.docs']:
        try:
            if file_extension == '.pdf':
                print(f"Processing PDF: {filename}")
                processing_result = await process_pdf_with_mistral(normalized_path, filename)
                response["mistral_processing"] = processing_result

                print(f"Starting agentic chunking...")
                chunking_result = await chunker.process_document_pages(
                    processing_result["data_folder"],
                    filename
                )
                response["chunking_result"] = chunking_result
                response["message"] = "PDF processed and chunked successfully"

            elif file_extension in ['.docx', '.doc', '.docs']:
                print(f"Processing {file_extension.upper()}: {filename}")
                processing_result = await process_docx_with_mistral(normalized_path, filename)
                response["mistral_processing"] = processing_result

                print(f"Starting agentic chunking...")
                chunking_result = await chunker.process_document_pages(
                    processing_result["data_folder"],
                    filename
                )
                response["chunking_result"] = chunking_result
                response["message"] = f"{file_extension.upper()} processed and chunked successfully"

        except Exception as e:
            response["error"] = str(e)
            response["message"] = f"Error processing {file_extension} file"

        return response

    # ============================
    # CASE 2: CSV (DIRECT CHUNKING)
    # ============================
    if file_extension == ".csv":
        try:
            print(f"Processing CSV: {filename}")

            # 1. Read CSV
            import pandas as pd
            df = pd.read_csv(normalized_path)

            # 2. Create data folder (same structure used for PDF/DOC)
            base_name = os.path.splitext(filename)[0]
            data_folder = os.path.join("data", base_name)
            os.makedirs(data_folder, exist_ok=True)

            # 3. Convert CSV into readable markdown
            csv_markdown = df.to_markdown(index=False)

            page_path = os.path.join(data_folder, "page1.md")
            with open(page_path, "w", encoding="utf-8") as f:
                f.write(f"# CSV Content: {filename}\n\n")
                f.write(csv_markdown)

            # Fake processing result to match PDF/DOC format
            processing_result = {
                "data_folder": data_folder,
                "total_pages": 1,
                "processed_pages": [
                    {
                        "page_number": 1,
                        "file_path": page_path,
                        "status": "completed"
                    }
                ]
            }
            response["csv_processing"] = processing_result

            print("Starting agentic chunking on CSV...")
            chunking_result = await chunker.process_document_pages(
                data_folder,
                filename
            )
            response["chunking_result"] = chunking_result
            response["message"] = "CSV processed and chunked successfully"

        except Exception as e:
            response["error"] = str(e)
            response["message"] = "Error processing CSV file"

        return response

    # If extension is not supported
    raise HTTPException(status_code=400, detail="Unsupported file type.")



@app.post("/create-embeddings")
async def create_embeddings(embedding_request: EmbeddingRequest):
    """
    Generate embeddings from chunks.json and store in ChromaDB
    """
    try:
        # Check if chunks file exists
        if not os.path.exists(embedding_request.chunks_file_path):
            raise HTTPException(status_code=404, detail=f"Chunks file not found: {embedding_request.chunks_file_path}")
        
        print(f"Creating embeddings from: {embedding_request.chunks_file_path}")
        
        # Process embeddings
        result = await embedding_manager.process_document_embeddings(
            chunks_file_path=embedding_request.chunks_file_path,
            collection_name=embedding_request.collection_name
        )
        
        return {
            "message": "Embeddings created and stored successfully",
            "chunks_file": embedding_request.chunks_file_path,
            "result": result,
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")

@app.post("/query-documents")
async def query_documents(query_request: QueryRequest):
    """
    Query documents using similarity search in ChromaDB
    """
    try:
        print(f"Querying: '{query_request.query_text}' in collection: {query_request.collection_name}")
        
        # Search for similar chunks
        results = embedding_manager.search_similar_chunks(
            query_text=query_request.query_text,
            collection_name=query_request.collection_name,
            n_results=query_request.n_results
        )
        
        return {
            "message": "Query completed successfully",
            "query": query_request.query_text,
            "collection": query_request.collection_name,
            "results": results,
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

@app.get("/list-collections")
async def list_collections():
    """
    List all ChromaDB collections
    """
    try:
        collections = embedding_manager.list_collections()
        return {
            "message": "Collections retrieved successfully",
            "collections": collections,
            "total_collections": len(collections)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")

@app.get("/collection-stats/{collection_name}")
async def get_collection_stats(collection_name: str):
    """
    Get statistics for a specific collection
    """
    try:
        stats = embedding_manager.get_collection_stats(collection_name)
        return {
            "message": "Collection stats retrieved successfully",
            "stats": stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collection stats: {str(e)}")
    

# Single API Endpoint
@app.post("/api/retrieve", response_model=Dict[str, Any])
async def retrieve_documents(request: SearchRequest):
    """
    Retrieve relevant document chunks for a given query
    
    Process:
    1. Convert query to embedding
    2. Perform vector similarity search
    3. Return top relevant chunks
    4. Show metadata (file name + chunk numbers)
    """
    try:
        results = retrieval_module.retrieve(
            query=request.query,
            collection_name=request.collection_name,
            top_k=request.top_k
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/debug/collection/{collection_name}")
async def debug_collection(collection_name: str):
    """
    Debug endpoint to check collection details
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get collection
        collection = client.get_collection(collection_name)
        
        # Get all data
        all_data = collection.get()
        
        return {
            "collection_name": collection_name,
            "total_documents": collection.count(),
            "metadata": collection.metadata,
            "sample_ids": all_data['ids'][:5] if all_data['ids'] else [],
            "sample_metadata": all_data['metadatas'][:5] if all_data['metadatas'] else [],
            "has_embeddings": len(all_data['embeddings']) > 0 if all_data.get('embeddings') else False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")
@app.post("/answer")
async def answer_question(req: AskRequest):
    """
    Full RAG flow:
    1. Retrieve top-k relevant chunks from Chroma
    2. Generate LLM answer grounded on those chunks
    3. Return Final Answer + Sources Used + Retrieved Context
    """
    try:
        # 1. Use your RetrievalModule to get chunks
        retrieval_result = retrieval_module.retrieve(
            query=req.question,
            collection_name=req.collection_name,
            top_k=req.top_k,
        )
        # Adjust this depending on how RetrievalModule returns data
        # Assume retrieval_result["results"] is a list of chunk dicts:
        retrieved_chunks = retrieval_result.get("results", retrieval_result)

        # 2. Generate LLM answer based on retrieved chunks
        answer_payload = answer_generator.generate_answer(
            question=req.question,
            retrieved_chunks=retrieved_chunks,
        )

        # 3. Format exactly as assignment wants
        final_answer = answer_payload["final_answer"]
        sources_used = answer_payload["sources_used"]
        retrieved_context = answer_payload["retrieved_context"]

        # Build pretty structure
        return {
            "final_answer": final_answer,
            "sources_used": [
                {
                    "display": f"{src['source_filename']} — Chunk {src['chunk_number']}",
                    "source_filename": src["source_filename"],
                    "chunk_number": src["chunk_number"],
                    "chunk_id": src["chunk_id"],
                    "score": src["score"],
                }
                for src in sources_used
            ],
            "retrieved_context": retrieved_context,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)