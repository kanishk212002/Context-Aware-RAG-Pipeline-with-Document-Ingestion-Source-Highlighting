import os
import json
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from typing import List, Dict, Any
import uuid
from datetime import datetime

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

class EmbeddingManager:
    def __init__(self, chroma_db_path="./chroma_db"):
        # Initialize Gemini embeddings through LangChain
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
           
        )
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.chroma_db_path = chroma_db_path
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        
    def get_gemini_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using Gemini"""
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch"""
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            raise Exception(f"Error generating batch embeddings: {str(e)}")
    
    def create_or_get_collection(self, collection_name: str):
        """Create or get existing ChromaDB collection"""
        try:
            # Try to get existing collection
            collection = self.chroma_client.get_collection(collection_name)
            print(f"Found existing collection: {collection_name}")
            

        except Exception:
            # Create new collection if doesn't exist
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": f"Embeddings for {collection_name}"}
            )
            print(f"Created new collection: {collection_name}")
        
        return collection
    
    def create_langchain_vectorstore(self, collection_name: str):
        """Create or get LangChain Chroma vectorstore"""
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.chroma_db_path
            )
            print(f"Initialized LangChain vectorstore: {collection_name}")
            return vectorstore
        except Exception as e:
            raise Exception(f"Error creating vectorstore: {str(e)}")
    
    def load_chunks_from_json(self, chunks_file_path: str) -> Dict:
        """Load chunks from JSON file"""
        if not os.path.exists(chunks_file_path):
            raise Exception(f"Chunks file not found: {chunks_file_path}")
        
        with open(chunks_file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        return chunks_data
    
    def prepare_chunk_data(self, chunks_data: Dict) -> tuple:
        """Prepare chunk data for ChromaDB storage"""
        documents = []
        metadatas = []
        ids = []
        
        document_info = chunks_data.get("document_info", {})
        chunks = chunks_data.get("chunks", [])
        
        for chunk in chunks:
            # Document content
            documents.append(chunk["content"])
            
            # Metadata
            metadata = {
                "chunk_id": chunk["chunk_id"],
                "chunk_number": chunk["chunk_number"],
                "token_count": chunk["token_count"],
                "source_filename": document_info.get("filename", "unknown"),
                "document_name": document_info.get("document_name", "unknown"),
                "topic": chunk.get("gemini_analysis", {}).get("topic", "unknown"),
                "created_at": chunk["created_at"],
                "chunking_method": document_info.get("chunking_method", "unknown")
            }
            metadatas.append(metadata)
            
            # Unique ID
            ids.append(chunk["chunk_id"])
        
        return documents, metadatas, ids
    
    def prepare_langchain_documents(self, chunks_data: Dict) -> List[Document]:
        """Prepare LangChain Document objects from chunks"""
        document_info = chunks_data.get("document_info", {})
        chunks = chunks_data.get("chunks", [])
        
        langchain_docs = []
        for chunk in chunks:
            metadata = {
                "chunk_id": chunk["chunk_id"],
                "chunk_number": chunk["chunk_number"],
                "token_count": chunk["token_count"],
                "source_filename": document_info.get("filename", "unknown"),
                "document_name": document_info.get("document_name", "unknown"),
                "topic": chunk.get("gemini_analysis", {}).get("topic", "unknown"),
                "created_at": chunk["created_at"],
                "chunking_method": document_info.get("chunking_method", "unknown")
            }
            
            doc = Document(
                page_content=chunk["content"],
                metadata=metadata
            )
            langchain_docs.append(doc)
        
        return langchain_docs
    
    def store_embeddings_in_chroma(self, collection, documents: List[str], 
                                 metadatas: List[Dict], ids: List[str]) -> Dict:
        """Store documents with embeddings in ChromaDB"""
        try:
            print(f"Generating embeddings for {len(documents)} chunks...")
            
            # Generate embeddings in batches
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(documents), batch_size):
                batch_texts = documents[i:i + batch_size]
                batch_embeddings = self.get_batch_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)
                print(f"Generated embeddings for batch {i//batch_size + 1}")
            
            # Store in ChromaDB
            collection.add(
                documents=documents,
                embeddings=all_embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Successfully stored {len(documents)} embeddings in ChromaDB")
            
            return {
                "total_embeddings": len(documents),
                "collection_name": collection.name,
                "embedding_model": self.embedding_model,
                "storage_path": self.chroma_db_path
            }
            
        except Exception as e:
            raise Exception(f"Error storing embeddings: {str(e)}")
    
    def store_embeddings_with_langchain(self, collection_name: str, 
                                       documents: List[Document]) -> Dict:
        """Store documents using LangChain Chroma vectorstore"""
        try:
            print(f"Storing {len(documents)} documents with LangChain...")
            
            # Create vectorstore and add documents
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=self.chroma_db_path
            )
            
            print(f"Successfully stored {len(documents)} documents in ChromaDB")
            
            return {
                "total_embeddings": len(documents),
                "collection_name": collection_name,
                "embedding_model": self.embedding_model,
                "storage_path": self.chroma_db_path
            }
            
        except Exception as e:
            raise Exception(f"Error storing embeddings with LangChain: {str(e)}")
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """Get statistics about a collection"""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            count = collection.count()
            
            return {
                "collection_name": collection_name,
                "total_documents": count,
                "metadata": collection.metadata
            }
        except Exception as e:
            return {"error": f"Collection not found: {str(e)}"}
    
    def list_collections(self) -> List[Dict]:
        """List all ChromaDB collections"""
        try:
            collections = self.chroma_client.list_collections()
            return [
                {
                    "name": col.name,
                    "count": col.count(),
                    "metadata": col.metadata
                }
                for col in collections
            ]
        except Exception as e:
            raise Exception(f"Error listing collections: {str(e)}")
    
    async def process_document_embeddings(self, chunks_file_path: str, 
                                        collection_name: str = None,
                                        use_langchain: bool = True) -> Dict:
        """Main function to process document chunks and create embeddings"""
        try:
            # Load chunks data
            chunks_data = self.load_chunks_from_json(chunks_file_path)
            
            # Generate collection name if not provided
            if not collection_name:
                doc_name = chunks_data.get("document_info", {}).get("document_name", "unknown")
                collection_name = f"{doc_name}_embeddings".replace(" ", "_").lower()
            
            if use_langchain:
                # Use LangChain approach
                langchain_docs = self.prepare_langchain_documents(chunks_data)
                storage_result = self.store_embeddings_with_langchain(
                    collection_name, langchain_docs
                )
            else:
                # Use direct ChromaDB approach
                collection = self.create_or_get_collection(collection_name)
                documents, metadatas, ids = self.prepare_chunk_data(chunks_data)
                storage_result = self.store_embeddings_in_chroma(
                    collection, documents, metadatas, ids
                )
            
            # Get final stats
            stats = self.get_collection_stats(collection_name)
            
            return {
                "status": "success",
                "chunks_processed": len(chunks_data.get("chunks", [])),
                "collection_name": collection_name,
                "storage_result": storage_result,
                "collection_stats": stats,
                "embedding_model": self.embedding_model,
                "method": "langchain" if use_langchain else "direct"
            }
            
        except Exception as e:
            raise Exception(f"Error processing document embeddings: {str(e)}")
    
    def search_similar_chunks(self, query_text: str, collection_name: str, 
                            n_results: int = 5, use_langchain: bool = True) -> Dict:
        """Search for similar chunks using query text"""
        try:
            if use_langchain:
                # Use LangChain vectorstore for search
                vectorstore = self.create_langchain_vectorstore(collection_name)
                
                # Perform similarity search
                results = vectorstore.similarity_search_with_score(
                    query_text, 
                    k=n_results
                )
                
                # Format results
                formatted_results = {
                    "documents": [[doc.page_content for doc, score in results]],
                    "metadatas": [[doc.metadata for doc, score in results]],
                    "distances": [[score for doc, score in results]]
                }
                
                return {
                    "query": query_text,
                    "collection": collection_name,
                    "results": formatted_results,
                    "total_found": len(results),
                    "method": "langchain"
                }
            else:
                # Use direct ChromaDB approach
                collection = self.chroma_client.get_collection(collection_name)
                query_embedding = self.get_gemini_embedding(query_text)
                
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=['documents', 'metadatas', 'distances']
                )
                
                return {
                    "query": query_text,
                    "collection": collection_name,
                    "results": results,
                    "total_found": len(results['documents'][0]),
                    "method": "direct"
                }
            
        except Exception as e:
            raise Exception(f"Error searching similar chunks: {str(e)}")
    
    def search_with_retriever(self, query_text: str, collection_name: str,
                            k: int = 5, search_type: str = "similarity") -> List[Document]:
        """Search using LangChain retriever interface"""
        try:
            vectorstore = self.create_langchain_vectorstore(collection_name)
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type=search_type,  # "similarity", "mmr", or "similarity_score_threshold"
                search_kwargs={"k": k}
            )
            
            # Retrieve documents
            docs = retriever.get_relevant_documents(query_text)
            
            return docs
            
        except Exception as e:
            raise Exception(f"Error with retriever search: {str(e)}")