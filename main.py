from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, Any
import os
import pandas as pd

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
retrieval_module = RetrievalModule(chroma_db_path="./chroma_db")
answer_generator = LLMAnswerGenerator()

# ============================================================================
# UI ENDPOINT - Serve the Simple Web Interface
# ============================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """
    Serve a simple web UI for the RAG pipeline
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Pipeline - Document Q&A System</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            
            .header p {
                font-size: 1.1em;
                opacity: 0.9;
            }
            
            .content {
                padding: 40px;
            }
            
            .section {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 25px;
                border-left: 5px solid #667eea;
            }
            
            .section h2 {
                color: #333;
                margin-bottom: 20px;
                font-size: 1.5em;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .step-number {
                background: #667eea;
                color: white;
                width: 35px;
                height: 35px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
            }
            
            .input-group {
                margin-bottom: 20px;
            }
            
            label {
                display: block;
                font-weight: 600;
                color: #555;
                margin-bottom: 8px;
            }
            
            input[type="text"],
            input[type="number"],
            textarea {
                width: 100%;
                padding: 12px 15px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 1em;
                transition: all 0.3s;
            }
            
            input[type="text"]:focus,
            input[type="number"]:focus,
            textarea:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 14px 30px;
                border-radius: 8px;
                font-size: 1em;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                width: 100%;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            
            .status {
                margin-top: 15px;
                padding: 15px;
                border-radius: 8px;
                display: none;
            }
            
            .status.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
                display: block;
            }
            
            .status.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
                display: block;
            }
            
            .status.info {
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
                display: block;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .result-box {
                background: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
                max-height: 500px;
                overflow-y: auto;
            }
            
            .result-box h3 {
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.2em;
            }
            
            .result-box pre {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                overflow-x: auto;
                font-size: 0.9em;
            }
            
            .source-item {
                background: #f8f9fa;
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 10px;
                border-left: 3px solid #667eea;
            }
            
            .collections-list {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            
            .collection-card {
                background: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 15px;
                cursor: pointer;
                transition: all 0.3s;
            }
            
            .collection-card:hover {
                border-color: #667eea;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
            }
            
            .collection-card.selected {
                border-color: #667eea;
                background: #f0f4ff;
            }
            
            .collection-name {
                font-weight: 600;
                color: #333;
                margin-bottom: 5px;
            }
            
            .collection-count {
                color: #666;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 RAG Pipeline</h1>
                <p>Document Question-Answering System with Source Attribution</p>
            </div>
            
            <div class="content">
                <!-- Step 1: Upload Document -->
                <div class="section">
                    <h2>
                        <span class="step-number">1</span>
                        Upload Document
                    </h2>
                    <div class="input-group">
                        <label for="file-path">📁 File Path (PDF, DOCX, DOC, CSV)</label>
                        <input type="text" id="file-path" placeholder="/path/to/your/document.pdf">
                    </div>
                    <button onclick="uploadDocument()">Upload & Process Document</button>
                    <div id="upload-loading" class="loading">
                        <div class="spinner"></div>
                        <p>Processing document...</p>
                    </div>
                    <div id="upload-status" class="status"></div>
                </div>
                
                <!-- Step 2: Create Embeddings -->
                <div class="section">
                    <h2>
                        <span class="step-number">2</span>
                        Create Embeddings
                    </h2>
                    <div class="input-group">
                        <label for="chunks-path">📄 Chunks File Path</label>
                        <input type="text" id="chunks-path" placeholder="data/document_name/chunks.json" readonly>
                    </div>
                    <div class="input-group">
                        <label for="collection-name">🗂️ Collection Name </label>
                        <input type="text" id="collection-name" placeholder="Leave empty for auto-generated name">
                    </div>
                    <button id="embed-btn" onclick="createEmbeddings()" disabled>Create Embeddings</button>
                    <div id="embed-loading" class="loading">
                        <div class="spinner"></div>
                        <p>Generating embeddings...</p>
                    </div>
                    <div id="embed-status" class="status"></div>
                </div>
                
                <!-- Step 3: View Collections -->
                <div class="section">
                    <h2>
                        <span class="step-number">3</span>
                        Available Collections
                    </h2>
                    <button onclick="loadCollections()">🔄 Refresh Collections</button>
                    <div id="collections-container" class="collections-list"></div>
                </div>
                
                <!-- Step 4: Retrieve Documents -->
                <div class="section">
                    <h2>
                        <span class="step-number">4</span>
                        Retrieve Documents
                    </h2>
                    <div class="input-group">
                        <label for="retrieve-query">🔍 Search Query</label>
                        <input type="text" id="retrieve-query" placeholder="What does the document say about...">
                    </div>
                    <div class="input-group">
                        <label for="retrieve-collection">🗂️ Collection Name</label>
                        <input type="text" id="retrieve-collection" placeholder="your_collection_name">
                    </div>
                    <div class="input-group">
                        <label for="retrieve-topk">📊 Top K Results</label>
                        <input type="number" id="retrieve-topk" value="5" min="1" max="20">
                    </div>
                    <button id="retrieve-btn" onclick="retrieveDocuments()">Retrieve Documents</button>
                    <div id="retrieve-loading" class="loading">
                        <div class="spinner"></div>
                        <p>Searching documents...</p>
                    </div>
                    <div id="retrieve-status" class="status"></div>
                    <div id="retrieve-results" class="result-box" style="display:none;"></div>
                </div>
                
                <!-- Step 5: Ask Question -->
                <div class="section">
                    <h2>
                        <span class="step-number">5</span>
                        Ask Question (Full RAG)
                    </h2>
                    <div class="input-group">
                        <label for="question">❓ Your Question</label>
                        <textarea id="question" rows="3" placeholder="Ask anything about your documents..."></textarea>
                    </div>
                    <div class="input-group">
                        <label for="answer-collection">🗂️ Collection Name</label>
                        <input type="text" id="answer-collection" placeholder="your_collection_name">
                    </div>
                    <div class="input-group">
                        <label for="answer-topk">📊 Top K Results</label>
                        <input type="number" id="answer-topk" value="5" min="1" max="20">
                    </div>
                    <button id="answer-btn" onclick="askQuestion()">Get Answer</button>
                    <div id="answer-loading" class="loading">
                        <div class="spinner"></div>
                        <p>Generating answer...</p>
                    </div>
                    <div id="answer-status" class="status"></div>
                    <div id="answer-results" class="result-box" style="display:none;"></div>
                </div>
            </div>
        </div>
        
        <script>
            let selectedCollection = '';
            
            // Upload Document
            async function uploadDocument() {
                const filePath = document.getElementById('file-path').value;
                const statusDiv = document.getElementById('upload-status');
                const loadingDiv = document.getElementById('upload-loading');
                
                if (!filePath) {
                    showStatus('upload-status', 'Please enter a file path', 'error');
                    return;
                }
                
                statusDiv.style.display = 'none';
                loadingDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/upload-path', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ file_path: filePath })
                    });
                    
                    const data = await response.json();
                    loadingDiv.style.display = 'none';
                    
                    if (response.ok) {
                        const chunksFile = data.chunking_result?.chunks_file || '';
                        document.getElementById('chunks-path').value = chunksFile;
                        document.getElementById('embed-btn').disabled = false;
                        
                        showStatus('upload-status', 
                            `✅ Success! Processed ${data.chunking_result?.total_chunks || 0} chunks`, 
                            'success');
                    } else {
                        showStatus('upload-status', `❌ Error: ${data.detail || 'Unknown error'}`, 'error');
                    }
                } catch (error) {
                    loadingDiv.style.display = 'none';
                    showStatus('upload-status', `❌ Error: ${error.message}`, 'error');
                }
            }
            
            // Create Embeddings
            async function createEmbeddings() {
                const chunksPath = document.getElementById('chunks-path').value;
                const collectionName = document.getElementById('collection-name').value;
                const statusDiv = document.getElementById('embed-status');
                const loadingDiv = document.getElementById('embed-loading');
                
                if (!chunksPath) {
                    showStatus('embed-status', 'Please upload a document first', 'error');
                    return;
                }
                
                statusDiv.style.display = 'none';
                loadingDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/create-embeddings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            chunks_file_path: chunksPath,
                            collection_name: collectionName || null
                        })
                    });
                    
                    const data = await response.json();
                    loadingDiv.style.display = 'none';
                    
                    if (response.ok) {
                        const finalCollection = data.result?.collection_name || collectionName;
                        document.getElementById('retrieve-collection').value = finalCollection;
                        document.getElementById('answer-collection').value = finalCollection;
                        
                        showStatus('embed-status', 
                            `✅ Embeddings created! Collection: ${finalCollection}`, 
                            'success');
                        
                        // Auto-refresh collections
                        loadCollections();
                    } else {
                        showStatus('embed-status', `❌ Error: ${data.detail || 'Unknown error'}`, 'error');
                    }
                } catch (error) {
                    loadingDiv.style.display = 'none';
                    showStatus('embed-status', `❌ Error: ${error.message}`, 'error');
                }
            }
            
            // Load Collections
            async function loadCollections() {
                const container = document.getElementById('collections-container');
                container.innerHTML = '<p>Loading collections...</p>';
                
                try {
                    const response = await fetch('/list-collections');
                    const data = await response.json();
                    
                    if (response.ok && data.collections.length > 0) {
                        container.innerHTML = '';
                        data.collections.forEach(col => {
                            const card = document.createElement('div');
                            card.className = 'collection-card';
                            if (col.name === selectedCollection) {
                                card.classList.add('selected');
                            }
                            card.innerHTML = `
                                <div class="collection-name">📚 ${col.name}</div>
                                <div class="collection-count">${col.count} documents</div>
                            `;
                            card.onclick = () => selectCollection(col.name);
                            container.appendChild(card);
                        });
                    } else {
                        container.innerHTML = '<p>No collections found. Upload and process a document first.</p>';
                    }
                } catch (error) {
                    container.innerHTML = `<p style="color: red;">Error loading collections: ${error.message}</p>`;
                }
            }
            
            // Select Collection
            function selectCollection(collectionName) {
                selectedCollection = collectionName;
                document.getElementById('retrieve-collection').value = collectionName;
                document.getElementById('answer-collection').value = collectionName;
                loadCollections();
            }
            
            // Retrieve Documents
            async function retrieveDocuments() {
                const query = document.getElementById('retrieve-query').value;
                const collection = document.getElementById('retrieve-collection').value;
                const topK = parseInt(document.getElementById('retrieve-topk').value);
                const statusDiv = document.getElementById('retrieve-status');
                const loadingDiv = document.getElementById('retrieve-loading');
                const resultsDiv = document.getElementById('retrieve-results');
                
                if (!query || !collection) {
                    showStatus('retrieve-status', 'Please enter both query and collection name', 'error');
                    return;
                }
                
                statusDiv.style.display = 'none';
                resultsDiv.style.display = 'none';
                loadingDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/api/retrieve', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            query: query,
                            collection_name: collection,
                            top_k: topK
                        })
                    });
                    
                    const data = await response.json();
                    loadingDiv.style.display = 'none';
                    
                    if (response.ok) {
                        displayRetrievalResults(data);
                        showStatus('retrieve-status', 
                            `✅ Found ${data.total_results} results`, 
                            'success');
                    } else {
                        showStatus('retrieve-status', `❌ Error: ${data.detail || 'Unknown error'}`, 'error');
                    }
                } catch (error) {
                    loadingDiv.style.display = 'none';
                    showStatus('retrieve-status', `❌ Error: ${error.message}`, 'error');
                }
            }
            
            // Display Retrieval Results
            function displayRetrievalResults(data) {
                const resultsDiv = document.getElementById('retrieve-results');
                resultsDiv.style.display = 'block';
                
                let html = '<h3>📋 Retrieved Chunks</h3>';
                
                if (data.results && data.results.length > 0) {
                    data.results.forEach((result, index) => {
                        html += `
                            <div class="source-item">
                                <strong>Rank ${result.rank}:</strong> ${result.source_filename} — Chunk ${result.chunk_number}<br>
                                <strong>Score:</strong> ${result.score.toFixed(4)}<br>
                                <strong>Topic:</strong> ${result.topic}<br>
                                <strong>Preview:</strong> ${result.text.substring(0, 200)}...
                            </div>
                        `;
                    });
                } else {
                    html += '<p>No results found.</p>';
                }
                
                resultsDiv.innerHTML = html;
            }
            
            // Ask Question
            async function askQuestion() {
                const question = document.getElementById('question').value;
                const collection = document.getElementById('answer-collection').value;
                const topK = parseInt(document.getElementById('answer-topk').value);
                const statusDiv = document.getElementById('answer-status');
                const loadingDiv = document.getElementById('answer-loading');
                const resultsDiv = document.getElementById('answer-results');
                
                if (!question || !collection) {
                    showStatus('answer-status', 'Please enter both question and collection name', 'error');
                    return;
                }
                
                statusDiv.style.display = 'none';
                resultsDiv.style.display = 'none';
                loadingDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/answer', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            question: question,
                            collection_name: collection,
                            top_k: topK
                        })
                    });
                    
                    const data = await response.json();
                    loadingDiv.style.display = 'none';
                    
                    if (response.ok) {
                        displayAnswer(data);
                        showStatus('answer-status', '✅ Answer generated successfully!', 'success');
                    } else {
                        showStatus('answer-status', `❌ Error: ${data.detail || 'Unknown error'}`, 'error');
                    }
                } catch (error) {
                    loadingDiv.style.display = 'none';
                    showStatus('answer-status', `❌ Error: ${error.message}`, 'error');
                }
            }
            
            // Display Answer
            function displayAnswer(data) {
                const resultsDiv = document.getElementById('answer-results');
                resultsDiv.style.display = 'block';
                
                let html = `
                    <h3>💡 Final Answer</h3>
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; white-space: pre-wrap;">
                        ${data.final_answer}
                    </div>
                    
                    <h3>📚 Sources Used</h3>
                `;
                
                if (data.sources_used && data.sources_used.length > 0) {
                    data.sources_used.forEach(source => {
                        html += `
                            <div class="source-item">
                                <strong>${source.display}</strong><br>
                                Score: ${source.score.toFixed(4)} | Chunk ID: ${source.chunk_id}
                            </div>
                        `;
                    });
                } else {
                    html += '<p>No sources used.</p>';
                }
                
                html += '<h3>📄 Retrieved Context</h3>';
                
                if (data.retrieved_context && data.retrieved_context.length > 0) {
                    data.retrieved_context.forEach((context, index) => {
                        html += `
                            <div class="source-item">
                                <strong>${context.label}</strong><br>
                                <pre style="white-space: pre-wrap; margin-top: 10px;">${context.text.substring(0, 300)}...</pre>
                            </div>
                        `;
                    });
                }
                
                resultsDiv.innerHTML = html;
            }
            
            // Helper function to show status messages
            function showStatus(elementId, message, type) {
                const statusDiv = document.getElementById(elementId);
                statusDiv.className = `status ${type}`;
                statusDiv.innerHTML = message;
                statusDiv.style.display = 'block';
            }
            
            // Load collections on page load
            window.onload = function() {
                loadCollections();
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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