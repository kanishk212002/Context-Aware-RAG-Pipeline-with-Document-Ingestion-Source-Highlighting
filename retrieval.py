import os
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class RetrievalModule:
    """
    Simple Retrieval Module for semantic search over document chunks
    """
    
    def __init__(self, chroma_db_path: str = "./chroma_db"):
        """
        Initialize the Retrieval Module
        
        Args:
            chroma_db_path: Path to ChromaDB storage
        """
        self.chroma_db_path = chroma_db_path
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def retrieve(
        self, 
        query: str, 
        collection_name: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for a given query
        
        Steps:
        1. Convert query to embedding
        2. Perform vector similarity search
        3. Return top relevant chunks
        4. Show metadata (file name + chunk numbers)
        """
        try:
            # Step 1 & 2: Initialize vectorstore and perform similarity search
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.chroma_db_path
            )
            
            # similarity_search_with_score returns List[(Document, score)]
            results = vectorstore.similarity_search_with_score(
                query,
                k=top_k
            )
            
            # Step 3 & 4: Format and return results with metadata
            formatted_results = []
            
            for idx, (doc, score) in enumerate(results):
                meta = doc.metadata or {}
                
                result = {
                    "rank": idx + 1,
                    # 👇 This key name MUST match what LLMAnswerGenerator expects
                    "text": doc.page_content,
                    "score": float(score),
                    # 👇 Use source_filename key expected downstream
                    "source_filename": meta.get("source_filename", meta.get("document_name", "unknown")),
                    "chunk_number": meta.get("chunk_number", 0),
                    "document_name": meta.get("document_name", "unknown"),
                    "chunk_id": meta.get("chunk_id", "unknown"),
                    "topic": meta.get("topic", "unknown"),
                    "token_count": meta.get("token_count"),
                    "created_at": meta.get("created_at"),
                    "chunking_method": meta.get("chunking_method"),
                }
                formatted_results.append(result)
            
            return {
                "query": query,
                "collection": collection_name,
                "total_results": len(formatted_results),
                "results": formatted_results
            }
            
        except Exception as e:
            raise Exception(f"Error during retrieval: {str(e)}")
