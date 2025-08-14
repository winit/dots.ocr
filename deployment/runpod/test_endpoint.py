#!/usr/bin/env python3
"""
Test script for dots.ocr RunPod endpoint deployment.
This script tests the deployed model endpoint with various inputs.
"""

import requests
import json
import time
import base64
from pathlib import Path
import argparse
import sys
from typing import Dict, Any, Optional

class RunPodTester:
    def __init__(self, endpoint_url: str, api_key: Optional[str] = None):
        """Initialize the tester with endpoint URL and optional API key."""
        self.endpoint_url = endpoint_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        # Set headers
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
        else:
            self.session.headers.update({'Content-Type': 'application/json'})

    def test_health_check(self) -> bool:
        """Test the health endpoint."""
        print("🏥 Testing health check...")
        try:
            response = self.session.get(f"{self.endpoint_url}/health", timeout=30)
            if response.status_code == 200:
                print("✅ Health check passed")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False

    def test_model_info(self) -> bool:
        """Test the model info endpoint."""
        print("📋 Testing model info...")
        try:
            response = self.session.get(f"{self.endpoint_url}/v1/models", timeout=30)
            if response.status_code == 200:
                models = response.json()
                print("✅ Model info retrieved:")
                print(json.dumps(models, indent=2))
                return True
            else:
                print(f"❌ Model info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Model info error: {e}")
            return False

    def encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def test_text_completion(self) -> bool:
        """Test basic text completion."""
        print("📝 Testing text completion...")
        
        payload = {
            "model": "rednote-hilab/dots.ocr",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?"
                }
            ],
            "max_tokens": 100,
            "temperature": 0.1
        }
        
        try:
            response = self.session.post(
                f"{self.endpoint_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Text completion successful:")
                if 'choices' in result and len(result['choices']) > 0:
                    print(f"   Response: {result['choices'][0]['message']['content']}")
                return True
            else:
                print(f"❌ Text completion failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Text completion error: {e}")
            return False

    def test_ocr_capability(self, image_path: Optional[str] = None) -> bool:
        """Test OCR capability with image input."""
        print("🔍 Testing OCR capability...")
        
        # If no image provided, create a simple test prompt
        if not image_path or not Path(image_path).exists():
            print("📝 Testing OCR with text prompt (no image provided)...")
            payload = {
                "model": "rednote-hilab/dots.ocr",
                "messages": [
                    {
                        "role": "user",
                        "content": "Please extract text from the following image description: A document with the text 'Hello World' written in Arial font."
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.1
            }
        else:
            print(f"📷 Testing OCR with image: {image_path}")
            try:
                image_b64 = self.encode_image_base64(image_path)
                payload = {
                    "model": "rednote-hilab/dots.ocr",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Please extract all text from this image."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 500,
                    "temperature": 0.1
                }
            except Exception as e:
                print(f"❌ Error preparing image: {e}")
                return False

        try:
            response = self.session.post(
                f"{self.endpoint_url}/v1/chat/completions",
                json=payload,
                timeout=120  # OCR might take longer
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ OCR test successful:")
                if 'choices' in result and len(result['choices']) > 0:
                    print(f"   Extracted text: {result['choices'][0]['message']['content']}")
                return True
            else:
                print(f"❌ OCR test failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ OCR test error: {e}")
            return False

    def test_performance(self, num_requests: int = 3) -> bool:
        """Test performance with multiple requests."""
        print(f"⚡ Testing performance with {num_requests} requests...")
        
        total_time = 0
        successful_requests = 0
        
        for i in range(num_requests):
            print(f"   Request {i+1}/{num_requests}...")
            
            payload = {
                "model": "rednote-hilab/dots.ocr",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Test request {i+1}: Please respond with a simple acknowledgment."
                    }
                ],
                "max_tokens": 50,
                "temperature": 0.1
            }
            
            start_time = time.time()
            try:
                response = self.session.post(
                    f"{self.endpoint_url}/v1/chat/completions",
                    json=payload,
                    timeout=60
                )
                
                end_time = time.time()
                request_time = end_time - start_time
                total_time += request_time
                
                if response.status_code == 200:
                    successful_requests += 1
                    print(f"   ✅ Request {i+1} completed in {request_time:.2f}s")
                else:
                    print(f"   ❌ Request {i+1} failed: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Request {i+1} error: {e}")
        
        if successful_requests > 0:
            avg_time = total_time / successful_requests
            print(f"✅ Performance test completed:")
            print(f"   Successful requests: {successful_requests}/{num_requests}")
            print(f"   Average response time: {avg_time:.2f}s")
            return successful_requests == num_requests
        else:
            print("❌ No successful requests")
            return False

def main():
    parser = argparse.ArgumentParser(description='Test dots.ocr RunPod endpoint')
    parser.add_argument('endpoint_url', help='RunPod endpoint URL')
    parser.add_argument('--api-key', help='API key for authentication')
    parser.add_argument('--image', help='Path to test image for OCR')
    parser.add_argument('--skip-performance', action='store_true', 
                       help='Skip performance testing')
    parser.add_argument('--performance-requests', type=int, default=3,
                       help='Number of requests for performance test')
    
    args = parser.parse_args()
    
    print("🧪 Starting dots.ocr RunPod endpoint tests...")
    print("=" * 60)
    
    tester = RunPodTester(args.endpoint_url, args.api_key)
    
    tests_passed = 0
    total_tests = 4 if not args.skip_performance else 3
    
    # Test 1: Health Check
    if tester.test_health_check():
        tests_passed += 1
    
    # Test 2: Model Info
    if tester.test_model_info():
        tests_passed += 1
    
    # Test 3: OCR Capability
    if tester.test_ocr_capability(args.image):
        tests_passed += 1
    
    # Test 4: Performance (optional)
    if not args.skip_performance:
        if tester.test_performance(args.performance_requests):
            tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Endpoint is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())