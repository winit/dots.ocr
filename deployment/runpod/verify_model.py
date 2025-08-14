#!/usr/bin/env python3
"""
Model verification script for dots.ocr deployment on RunPod.
This script verifies that the model has been downloaded and is accessible.
"""

import os
import sys
import json
from pathlib import Path

def verify_model_files():
    """Verify that required model files exist."""
    model_path = Path(os.environ.get('MODEL_PATH', '/models/DotsOCR'))
    
    print(f"Verifying model files in: {model_path}")
    
    if not model_path.exists():
        print(f"❌ Model directory does not exist: {model_path}")
        return False
    
    # Check for required files
    required_files = [
        'config.json',
        'tokenizer.json',
        'tokenizer_config.json'
    ]
    
    # Check for model weights (could be .bin, .safetensors, or sharded)
    model_weight_patterns = [
        'pytorch_model.bin',
        'model.safetensors',
        'pytorch_model-00001-of-*.bin',
        'model-00001-of-*.safetensors'
    ]
    
    missing_files = []
    found_files = []
    
    # Check required files
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            print(f"✅ Found: {file}")
            found_files.append(file)
        else:
            print(f"⚠️  Missing: {file}")
            missing_files.append(file)
    
    # Check for model weights
    weight_found = False
    for pattern in model_weight_patterns:
        if '*' in pattern:
            # Handle wildcard patterns
            pattern_parts = pattern.split('*')
            for file in model_path.glob(pattern_parts[0] + '*' + pattern_parts[1]):
                print(f"✅ Found model weights: {file.name}")
                weight_found = True
                break
        else:
            file_path = model_path / pattern
            if file_path.exists():
                print(f"✅ Found model weights: {pattern}")
                weight_found = True
                break
    
    if not weight_found:
        print("❌ No model weights found!")
        return False
    
    return True

def verify_model_config():
    """Verify model configuration."""
    model_path = Path(os.environ.get('MODEL_PATH', '/models/DotsOCR'))
    config_path = model_path / 'config.json'
    
    if not config_path.exists():
        print("❌ config.json not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("✅ Model configuration loaded successfully")
        print(f"   Model type: {config.get('model_type', 'unknown')}")
        print(f"   Architecture: {config.get('architectures', ['unknown'])[0]}")
        
        if 'vocab_size' in config:
            print(f"   Vocabulary size: {config['vocab_size']}")
        
        return True
    
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse config.json: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False

def verify_environment():
    """Verify environment variables."""
    required_vars = [
        'MODEL_NAME',
        'MODEL_PATH',
        'TOKENIZER_PATH'
    ]
    
    print("Verifying environment variables:")
    missing_vars = []
    
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)
    
    return len(missing_vars) == 0

def main():
    """Main verification function."""
    print("🔍 Starting dots.ocr model verification...")
    print("=" * 50)
    
    success = True
    
    # Verify environment
    print("\n1. Environment Variables:")
    if not verify_environment():
        success = False
    
    # Verify model files
    print("\n2. Model Files:")
    if not verify_model_files():
        success = False
    
    # Verify model configuration
    print("\n3. Model Configuration:")
    if not verify_model_config():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All verifications passed!")
        print("🚀 Model is ready for deployment")
        return 0
    else:
        print("❌ Some verifications failed!")
        print("🛑 Model deployment may not work correctly")
        return 1

if __name__ == "__main__":
    sys.exit(main())