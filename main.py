from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uvicorn
from document_processor import process_pdf_with_mistral, process_docx_with_mistral
from chunker import AgenticChunker

app = FastAPI(title="Document Path API")

class DocumentPath(BaseModel):
    file_path: str


# Initialize the agentic chunker
chunker = AgenticChunker(min_tokens=300, max_tokens=800)

@app.post("/upload-path")
async def upload_document_path(doc_path: DocumentPath):
    """
    Accept a local file path for document processing
    """
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
    
    # Process PDF or DOCX files with Mistral
    if file_extension in ['.pdf', '.docx', '.doc', '.docs']:
        try:
            if file_extension == '.pdf':
                print(f"Processing PDF: {filename}")
                processing_result = await process_pdf_with_mistral(normalized_path, filename)
                response["mistral_processing"] = processing_result
                
                # Perform agentic chunking
                print(f"Starting agentic chunking with Gemini...")
                chunking_result = await chunker.process_document_pages(
                    processing_result["data_folder"], 
                    filename
                )
                response["chunking_result"] = chunking_result
                response["message"] = f"PDF processed and chunked successfully"
            
            elif file_extension in ['.docx', '.doc', '.docs']:
                print(f"Processing {file_extension.upper()}: {filename}")
                processing_result = await process_docx_with_mistral(normalized_path, filename)
                response["mistral_processing"] = processing_result
                
                # Perform agentic chunking
                print(f"Starting agentic chunking with Gemini...")
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)