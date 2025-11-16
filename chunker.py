import os
import json
import tiktoken
from datetime import datetime
from dotenv import load_dotenv
import glob
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

class AgenticChunker:
    def __init__(self, min_tokens=300, max_tokens=800):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1
        )
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def get_gemini_analysis(self, text: str) -> Dict:
        """Get Gemini's analysis for optimal chunking using LangChain"""
        prompt = f"""
        Analyze the following text and provide intelligent chunking guidance.
        
        Requirements:
        - Each chunk should be {self.min_tokens}-{self.max_tokens} tokens
        - Identify natural semantic boundaries (topics, concepts, sections)
        - Maintain context and coherence within chunks
        - Avoid splitting related concepts
        
        Text to analyze:
        {text}
        
        Please provide a JSON response with:
        1. "suggested_splits": List of character positions where to split
        2. "topics": List of main topics identified
        3. "reasoning": Explanation for each split decision
        
        Format your response as valid JSON only.
        """
        
        try:
            # Create message using LangChain
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            
            # Extract JSON from response
            response_text = response.content.strip()
            
            # Clean up response to extract JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.rfind("```")
                response_text = response_text[json_start:json_end].strip()
            
            return json.loads(response_text)
        
        except Exception as e:
            print(f"Gemini analysis error: {e}")
            # Fallback to simple splitting
            return self.fallback_splitting(text)
    
    def fallback_splitting(self, text: str) -> Dict:
        """Fallback splitting if Gemini fails"""
        words = text.split()
        avg_chunk_size = (self.min_tokens + self.max_tokens) // 2
        splits = []
        
        current_pos = 0
        for i in range(0, len(words), avg_chunk_size):
            if i > 0:
                splits.append(current_pos)
            current_pos += len(' '.join(words[i:i+avg_chunk_size])) + 1
        
        return {
            "suggested_splits": splits,
            "topics": ["Content section"],
            "reasoning": ["Fallback automatic splitting"]
        }
    
    def split_text_by_positions(self, text: str, positions: List[int]) -> List[str]:
        """Split text at given character positions"""
        chunks = []
        start = 0
        
        for pos in positions:
            if start < len(text):
                chunk = text[start:pos].strip()
                if chunk:
                    chunks.append(chunk)
                start = pos
        
        # Add remaining text
        if start < len(text):
            remaining = text[start:].strip()
            if remaining:
                chunks.append(remaining)
        
        return chunks
    
    def validate_and_adjust_chunks(self, chunks: List[str]) -> List[str]:
        """Validate chunk sizes and adjust if needed"""
        validated_chunks = []
        
        for chunk in chunks:
            token_count = self.count_tokens(chunk)
            
            if token_count < self.min_tokens:
                # Try to merge with next chunk if possible
                if validated_chunks and self.count_tokens(validated_chunks[-1] + " " + chunk) <= self.max_tokens:
                    validated_chunks[-1] = validated_chunks[-1] + " " + chunk
                else:
                    validated_chunks.append(chunk)
            
            elif token_count > self.max_tokens:
                # Split large chunk
                sub_chunks = self.split_large_chunk(chunk)
                validated_chunks.extend(sub_chunks)
            
            else:
                validated_chunks.append(chunk)
        
        return validated_chunks
    
    def split_large_chunk(self, chunk: str) -> List[str]:
        """Split a chunk that's too large"""
        sentences = chunk.split('. ')
        sub_chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
            
            if self.count_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    sub_chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            sub_chunks.append(current_chunk.strip())
        
        return sub_chunks
    
    def create_chunk_json(self, chunks: List[str], filename: str, gemini_analysis: Dict) -> Dict:
        """Create JSON structure for chunks"""
        document_name = os.path.splitext(filename)[0]
        
        chunk_data = {
            "document_info": {
                "filename": filename,
                "document_name": document_name,
                "total_chunks": len(chunks),
                "processed_date": datetime.now().isoformat(),
                "chunking_method": "agentic_gemini",
                "token_range": f"{self.min_tokens}-{self.max_tokens}"
            },
            "chunks": []
        }
        
        for i, chunk_text in enumerate(chunks, 1):
            chunk = {
                "chunk_id": f"{document_name}_chunk_{i:03d}",
                "chunk_number": i,
                "content": chunk_text,
                "token_count": self.count_tokens(chunk_text),
                "source_info": {
                    "filename": filename,
                    "document_name": document_name
                },
                "gemini_analysis": {
                    "topic": gemini_analysis.get("topics", ["Unknown"])[min(i-1, len(gemini_analysis.get("topics", [])) - 1)] if gemini_analysis.get("topics") else "Unknown",
                    "chunk_reasoning": f"Chunk {i} based on semantic analysis"
                },
                "created_at": datetime.now().isoformat()
            }
            chunk_data["chunks"].append(chunk)
        
        return chunk_data
    
    def save_chunks_json(self, chunk_data: Dict, data_folder: str):
        """Save chunks as JSON file"""
        chunks_file = os.path.join(data_folder, "chunks.json")
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        print(f"Chunks saved to: {chunks_file}")
        return chunks_file
    
    async def process_document_pages(self, data_folder: str, filename: str):
        """Process all pages from a document and create chunks"""
        pages_folder = os.path.join(data_folder, "pages")
        
        # Check if pages folder exists
        if not os.path.exists(pages_folder):
            # If no pages folder, look for .md files directly in data_folder
            md_files = glob.glob(os.path.join(data_folder, "*.md"))
        else:
            md_files = glob.glob(os.path.join(pages_folder, "*.md"))
        
        if not md_files:
            raise Exception("No markdown files found for chunking")
        
        # Sort files by page number
        md_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or '0'))
        
        # Combine all page content
        combined_text = ""
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    combined_text += content + "\n\n"
        
        if not combined_text.strip():
            raise Exception("No content found in markdown files")
        
        print(f"Processing {len(md_files)} pages with total {self.count_tokens(combined_text)} tokens")
        
        # Get Gemini analysis
        gemini_analysis = self.get_gemini_analysis(combined_text)
        
        # Split text based on Gemini's suggestions
        suggested_splits = gemini_analysis.get("suggested_splits", [])
        if suggested_splits:
            chunks = self.split_text_by_positions(combined_text, suggested_splits)
        else:
            # Fallback to sentence-based splitting
            chunks = self.split_large_chunk(combined_text)
        
        # Validate and adjust chunk sizes
        final_chunks = self.validate_and_adjust_chunks(chunks)
        
        print(f"Created {len(final_chunks)} chunks")
        
        # Create JSON structure
        chunk_data = self.create_chunk_json(final_chunks, filename, gemini_analysis)
        
        # Save chunks JSON
        chunks_file = self.save_chunks_json(chunk_data, data_folder)
        
        return {
            "chunks_file": chunks_file,
            "total_chunks": len(final_chunks),
            "gemini_analysis": gemini_analysis,
            "chunk_data": chunk_data
        }