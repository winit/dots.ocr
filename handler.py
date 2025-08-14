import runpod
import torch
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from PIL import Image
import base64
import io
import os

model = None
tokenizer = None
image_processor = None

def load_model():
    global model, tokenizer, image_processor
    if model is None:
        model_name = os.environ.get("MODEL_NAME", "rednote-ai/dotsocr-v1.0")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        model.eval()

def handler(job):
    """RunPod handler for dots.ocr inference"""
    try:
        load_model()
        job_input = job["input"]
        
        if "image" not in job_input:
            return {"error": "No image provided"}
        
        # Decode base64 image
        image_bytes = base64.b64decode(job_input["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Process image
        prompt = job_input.get("prompt", "Parse the text, table, and formula in the image.")
        inputs = image_processor(images=image, return_tensors="pt").to(model.device)
        text_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        inputs.update(text_inputs)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"status": "success", "result": result}
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

runpod.serverless.start({"handler": handler})