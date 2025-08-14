#!/usr/bin/env python3

import runpod
import os
import json
import base64
from PIL import Image
import io

def handler(job):
    """
    Simple RunPod handler for testing
    """
    print(f"Received job: {json.dumps(job)}")
    
    try:
        job_input = job.get("input", {})
        
        # Handle test requests without images
        if "prompt" in job_input and "image" not in job_input:
            # Echo back for testing
            result = f"Echo test: {job_input['prompt']}"
            print(f"Returning test result: {result}")
            return {"output": result}
        
        # Handle image OCR requests
        if "image" in job_input:
            try:
                # Decode the image
                image_bytes = base64.b64decode(job_input["image"])
                image = Image.open(io.BytesIO(image_bytes))
                
                # For now, just return image info (model loading takes too long)
                result = {
                    "message": "Image received successfully",
                    "image_size": image.size,
                    "image_mode": image.mode,
                    "prompt": job_input.get("prompt", "No prompt provided")
                }
                
                print(f"Returning image result: {result}")
                return {"output": result}
                
            except Exception as e:
                error_msg = f"Error processing image: {str(e)}"
                print(error_msg)
                return {"error": error_msg}
        
        # Default response
        return {"output": "Handler is working but no valid input provided"}
        
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

# Start the RunPod serverless worker
print("Starting RunPod serverless worker...")
runpod.serverless.start({"handler": handler})