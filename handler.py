#!/usr/bin/env python3

import runpod
import os
import json
import base64
from PIL import Image
import io
import torch
from transformers import AutoModel, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
model = None
tokenizer = None

def load_model():
    """Load the OCR model"""
    global model, tokenizer
    
    if model is None:
        try:
            logger.info("Loading OCR model...")
            # Using GOT-OCR2.0 which is excellent for documents
            model_name = "ucaslcl/GOT-OCR2_0"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map='cuda' if torch.cuda.is_available() else 'cpu',
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            model.eval()
            logger.info("Model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            # Fallback to simpler model
            try:
                logger.info("Trying alternative: TrOCR model...")
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                
                processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
                model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
                tokenizer = processor
                logger.info("TrOCR model loaded as fallback")
                return True
            except Exception as e2:
                logger.error(f"Fallback model also failed: {str(e2)}")
                return False
    return True

def process_pdf(pdf_bytes):
    """Process PDF document"""
    try:
        import fitz  # PyMuPDF
        
        # Open PDF from bytes
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        all_text = []
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Method 1: Direct text extraction
            text = page.get_text()
            if text.strip():
                all_text.append(f"Page {page_num + 1} (Text Layer):\n{text}")
            
            # Method 2: OCR on page image if no text
            if not text.strip() or len(text.strip()) < 50:
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better OCR
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                # OCR the image
                if model:
                    ocr_text = process_image_with_model(image)
                    all_text.append(f"Page {page_num + 1} (OCR):\n{ocr_text}")
        
        pdf_document.close()
        return "\n\n".join(all_text)
        
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        return f"Error processing PDF: {str(e)}"

def process_image_with_model(image):
    """Process image with the loaded model"""
    try:
        if model is None:
            return "Model not loaded"
        
        # Save image temporarily (some models need file path)
        temp_path = "/tmp/temp_image.png"
        image.save(temp_path)
        
        # Try GOT-OCR style processing
        if hasattr(model, 'chat'):
            result = model.chat(tokenizer, temp_path, ocr_type='ocr')
        # Try TrOCR style processing
        elif hasattr(tokenizer, 'processor'):
            pixel_values = tokenizer(image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            # Generic transformers approach
            inputs = tokenizer(images=image, return_tensors="pt")
            outputs = model.generate(**inputs)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return result
        
    except Exception as e:
        logger.error(f"Model inference error: {str(e)}")
        # Fallback to basic OCR
        try:
            import pytesseract
            result = pytesseract.image_to_string(image)
            return f"Basic OCR: {result}"
        except:
            return f"OCR failed: {str(e)}"

def handler(job):
    """
    RunPod handler for OCR processing
    """
    logger.info(f"Received job: {job.get('id', 'unknown')}")
    
    try:
        job_input = job.get("input", {})
        
        # Handle test requests without images
        if "prompt" in job_input and "image" not in job_input and "pdf" not in job_input:
            result = f"Echo test: {job_input['prompt']}"
            logger.info(f"Returning test result: {result}")
            return {"output": result}
        
        # Try to load model (but don't fail if it doesn't work)
        model_loaded = load_model()
        
        # Handle PDF documents
        if "pdf" in job_input:
            logger.info("Processing PDF document...")
            try:
                pdf_bytes = base64.b64decode(job_input["pdf"])
                result = process_pdf(pdf_bytes)
                
                return {
                    "output": {
                        "type": "pdf",
                        "text": result,
                        "pages": result.count("Page "),
                        "model_used": "GOT-OCR2" if model_loaded else "PyMuPDF text extraction"
                    }
                }
            except Exception as e:
                logger.error(f"PDF processing failed: {str(e)}")
                return {"error": f"PDF processing error: {str(e)}"}
        
        # Handle image OCR requests
        if "image" in job_input:
            logger.info("Processing image...")
            try:
                # Decode the image
                image_bytes = base64.b64decode(job_input["image"])
                image = Image.open(io.BytesIO(image_bytes))
                
                if model_loaded:
                    # Use actual OCR
                    ocr_result = process_image_with_model(image)
                    result = {
                        "type": "image",
                        "text": ocr_result,
                        "image_size": image.size,
                        "model_used": "GOT-OCR2" if model else "TrOCR"
                    }
                else:
                    # Return basic info if model not loaded
                    result = {
                        "type": "image",
                        "message": "Model still loading, returning image info only",
                        "image_size": image.size,
                        "image_mode": image.mode
                    }
                
                return {"output": result}
                
            except Exception as e:
                error_msg = f"Error processing image: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg}
        
        # Default response
        return {"output": "Handler ready. Send 'image' or 'pdf' in base64 format."}
        
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

# Start the RunPod serverless worker
if __name__ == "__main__":
    logger.info("Starting RunPod serverless worker with OCR support...")
    runpod.serverless.start({"handler": handler})