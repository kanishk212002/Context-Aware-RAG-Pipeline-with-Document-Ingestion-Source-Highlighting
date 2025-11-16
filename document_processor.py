import os
import base64
import io
from pdf2image import convert_from_path
from docx import Document
from PIL import Image
from mistralai.client import MistralClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Mistral client
mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

def convert_pdf_to_images(pdf_path):
    """Convert PDF pages to images"""
    try:
        # Try with default poppler path first
        try:
            images = convert_from_path(pdf_path, dpi=200)
            return images
        except:
            # Try with common Windows poppler paths
            poppler_paths = [
                # r".\poppler-25.11.0\bin",  # Your local poppler path
                # r"C:\Program Files\poppler\bin",
                # r"C:\poppler-25.11.0\bin",
                # r"C:\Program Files (x86)\poppler\bin",
                # r"C:\poppler\bin",
                r".\poppler\bin"
            ]
            
            for poppler_path in poppler_paths:
                if os.path.exists(poppler_path):
                    images = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)
                    return images
            
            # If no poppler found, raise the original error
            raise Exception("Poppler not found in common paths. Please install poppler and add to PATH.")
            
    except Exception as e:
        raise Exception(f"Error converting PDF to images: {str(e)}")

def convert_docx_to_images(docx_path):
    """Convert DOCX pages to images (placeholder - requires more complex implementation)"""
    # For now, we'll extract text directly from DOCX
    # In a real implementation, you'd need to render DOCX pages as images
    try:
        doc = Document(docx_path)
        pages_text = []
        current_page = ""
        
        for paragraph in doc.paragraphs:
            current_page += paragraph.text + "\n"
            # Simple page break detection (you might need more sophisticated logic)
            if len(current_page) > 2000:  # Approximate page break
                pages_text.append(current_page)
                current_page = ""
        
        if current_page:
            pages_text.append(current_page)
        
        return pages_text
    except Exception as e:
        raise Exception(f"Error processing DOCX: {str(e)}")

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

async def extract_text_with_mistral(image_base64):
    """Extract text from image using Mistral Vision API"""
    try:
        # Create the message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all text from this image and format it as LaTeX. Include any mathematical formulas, tables, and preserve the structure."
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{image_base64}"
                    }
                ]
            }
        ]
        
        # Call Mistral API
        response = mistral_client.chat(
            model="pixtral-12b-2409",
            messages=messages
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def create_data_folder(filename):
    """Create data folder structure"""
    base_name = os.path.splitext(filename)[0]
    data_folder = os.path.join("data", base_name)
    os.makedirs(data_folder, exist_ok=True)
    return data_folder

async def process_pdf_with_mistral(pdf_path, filename):
    """Process PDF with Mistral OCR"""
    try:
        # Convert PDF to images
        images = convert_pdf_to_images(pdf_path)
        
        # Create data folder
        data_folder = create_data_folder(filename)
        
        processed_pages = []
        
        for i, image in enumerate(images, 1):
            print(f"Processing page {i}...")
            
            # Convert image to base64
            image_base64 = image_to_base64(image)
            
            # Extract text with Mistral
            extracted_text = await extract_text_with_mistral(image_base64)
            
            # Save as markdown file
            page_file = os.path.join(data_folder, f"page{i}.md")
            with open(page_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            
            processed_pages.append({
                "page_number": i,
                "file_path": page_file,
                "status": "completed"
            })
        
        return {
            "total_pages": len(images),
            "processed_pages": processed_pages,
            "data_folder": data_folder
        }
    
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

async def process_docx_with_mistral(docx_path, filename):
    """Process DOCX file"""
    try:
        pages_text = convert_docx_to_images(docx_path)
        data_folder = create_data_folder(filename)
        
        processed_pages = []
        for i, text in enumerate(pages_text, 1):
            page_file = os.path.join(data_folder, f"page{i}.md")
            with open(page_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            processed_pages.append({
                "page_number": i,
                "file_path": page_file,
                "status": "completed"
            })
        
        return {
            "total_pages": len(pages_text),
            "processed_pages": processed_pages,
            "data_folder": data_folder
        }
    
    except Exception as e:
        raise Exception(f"Error processing DOCX: {str(e)}")