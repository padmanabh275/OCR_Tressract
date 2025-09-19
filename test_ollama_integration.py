"""
Test script for Ollama integration with document extraction system
"""

import requests
import json
import time
from pathlib import Path

def test_ollama_status():
    """Test Ollama service status"""
    print("🔍 Testing Ollama Status...")
    
    try:
        response = requests.get("http://localhost:8001/ollama/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Ollama Status: {status['status']}")
            print(f"📱 Model: {status.get('model', 'N/A')}")
            print(f"🌐 Base URL: {status.get('base_url', 'N/A')}")
            print(f"💬 Message: {status.get('message', 'N/A')}")
            return status['status'] == 'available'
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return False

def test_available_models():
    """Test available Ollama models"""
    print("\n📋 Testing Available Models...")
    
    try:
        response = requests.get("http://localhost:8001/ollama/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            print(f"✅ Models Status: {models_data['status']}")
            print(f"🎯 Current Model: {models_data.get('current_model', 'N/A')}")
            print("📚 Available Models:")
            for model in models_data.get('models', []):
                print(f"   - {model['name']} ({model.get('size', 'Unknown size')})")
            return True
        else:
            print(f"❌ Models check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False

def test_ollama_processing():
    """Test Ollama document processing"""
    print("\n🚀 Testing Ollama Document Processing...")
    
    # Create a sample document for testing
    sample_text = """
    INCOME TAX DEPARTMENT
    GOVT. OF INDIA
    PERMANENT ACCOUNT NUMBER
    P.A.N. : ABCDE1234F
    NAME: RAJESH KUMAR SHARMA
    FATHER'S NAME: RAMESH KUMAR SHARMA
    DATE OF BIRTH: 15/01/1990
    SIGNATURE: RAJESH KUMAR SHARMA
    """
    
    # Create a simple test image
    import cv2
    import numpy as np
    
    # Create test image
    img = np.ones((300, 500, 3), dtype=np.uint8) * 255
    cv2.putText(img, "INCOME TAX DEPARTMENT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(img, "GOVT. OF INDIA", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(img, "PERMANENT ACCOUNT NUMBER", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    cv2.putText(img, "P.A.N. : ABCDE1234F", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    cv2.putText(img, "NAME: RAJESH KUMAR SHARMA", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(img, "FATHER'S NAME: RAMESH KUMAR SHARMA", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(img, "DATE OF BIRTH: 15/01/1990", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(img, "SIGNATURE: RAJESH KUMAR SHARMA", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    
    # Save test image
    test_image_path = "test_pan_card_ollama.png"
    cv2.imwrite(test_image_path, img)
    print(f"📄 Created test image: {test_image_path}")
    
    try:
        # Test Ollama processing
        with open(test_image_path, "rb") as f:
            files = {"file": ("test_pan_card_ollama.png", f, "image/png")}
            response = requests.post("http://localhost:8001/upload/ollama", files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Ollama Processing Successful!")
            print(f"📊 Document Type: {result.get('document_type', 'N/A')}")
            print(f"🎯 Confidence: {result.get('confidence_score', 0.0):.3f}")
            print(f"🤖 Model Used: {result.get('model_used', 'N/A')}")
            print(f"⏱️ Processing Time: {result.get('processing_time', 0.0):.2f}s")
            
            # Print extracted data
            extracted_data = result.get('extracted_data', {})
            personal_info = extracted_data.get('personal_info', {})
            print(f"\n📋 Extracted Information:")
            print(f"   👤 Name: {personal_info.get('full_name', 'N/A')}")
            print(f"   📅 DOB: {personal_info.get('date_of_birth', 'N/A')}")
            print(f"   🏠 Birth City: {personal_info.get('birth_city', 'N/A')}")
            
            identification = extracted_data.get('identification', {})
            print(f"   🆔 Document Number: {identification.get('document_number', 'N/A')}")
            print(f"   📄 Document Type: {identification.get('document_type', 'N/A')}")
            
            # Print validation results
            validation = result.get('validation', {})
            print(f"\n✅ Validation Results:")
            print(f"   Valid: {validation.get('is_valid', False)}")
            print(f"   Confidence: {validation.get('confidence', 0.0):.3f}")
            
            # Print text enhancement info
            text_enhancement = result.get('text_enhancement', {})
            print(f"\n✨ Text Enhancement:")
            print(f"   Original Length: {text_enhancement.get('original_length', 0)}")
            print(f"   Enhanced Length: {text_enhancement.get('enhanced_length', 0)}")
            print(f"   Improvement Ratio: {text_enhancement.get('improvement_ratio', 1.0):.2f}")
            
            return True
        else:
            print(f"❌ Processing failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return False
    finally:
        # Clean up test image
        if Path(test_image_path).exists():
            Path(test_image_path).unlink()
            print(f"🧹 Cleaned up test image: {test_image_path}")

def test_ollama_vs_traditional():
    """Compare Ollama processing with traditional processing"""
    print("\n⚖️ Comparing Ollama vs Traditional Processing...")
    
    # Create test image
    import cv2
    import numpy as np
    
    img = np.ones((300, 500, 3), dtype=np.uint8) * 255
    cv2.putText(img, "UNITED STATES OF AMERICA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(img, "DRIVER LICENSE", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(img, "NAME: JOHN DOE", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(img, "DATE OF BIRTH: 01/15/1990", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(img, "ADDRESS: 123 MAIN ST", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(img, "CITY: NEW YORK, NY 10001", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(img, "LICENSE NO: D123456789", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(img, "EXPIRES: 01/15/2025", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    
    test_image_path = "test_driver_license_ollama.png"
    cv2.imwrite(test_image_path, img)
    
    try:
        # Test traditional processing
        print("🔄 Testing Traditional Processing...")
        with open(test_image_path, "rb") as f:
            files = {"file": ("test_driver_license_ollama.png", f, "image/png")}
            traditional_response = requests.post("http://localhost:8001/upload/dual", files=files, timeout=30)
        
        # Test Ollama processing
        print("🤖 Testing Ollama Processing...")
        with open(test_image_path, "rb") as f:
            files = {"file": ("test_driver_license_ollama.png", f, "image/png")}
            ollama_response = requests.post("http://localhost:8001/upload/ollama", files=files, timeout=60)
        
        # Compare results
        if traditional_response.status_code == 200 and ollama_response.status_code == 200:
            traditional_result = traditional_response.json()
            ollama_result = ollama_response.json()
            
            print("\n📊 Comparison Results:")
            print(f"{'Metric':<25} {'Traditional':<15} {'Ollama':<15} {'Difference':<15}")
            print("-" * 70)
            
            traditional_conf = traditional_result.get('confidence_score', 0.0)
            ollama_conf = ollama_result.get('confidence_score', 0.0)
            conf_diff = ollama_conf - traditional_conf
            
            print(f"{'Confidence Score':<25} {traditional_conf:<15.3f} {ollama_conf:<15.3f} {conf_diff:+.3f}")
            
            traditional_time = traditional_result.get('processing_time', 0.0)
            ollama_time = ollama_result.get('processing_time', 0.0)
            time_diff = ollama_time - traditional_time
            
            print(f"{'Processing Time (s)':<25} {traditional_time:<15.2f} {ollama_time:<15.2f} {time_diff:+.2f}")
            
            traditional_type = traditional_result.get('document_type', 'unknown')
            ollama_type = ollama_result.get('document_type', 'unknown')
            
            print(f"{'Document Type':<25} {traditional_type:<15} {ollama_type:<15} {'Same' if traditional_type == ollama_type else 'Different'}")
            
            # Show extracted data comparison
            print(f"\n📋 Extracted Data Comparison:")
            traditional_data = traditional_result.get('extracted_data', {})
            ollama_data = ollama_result.get('extracted_data', {})
            
            traditional_name = traditional_data.get('first_name', '') + ' ' + traditional_data.get('last_name', '')
            ollama_name = ollama_data.get('personal_info', {}).get('full_name', '')
            
            print(f"   Traditional Name: {traditional_name.strip()}")
            print(f"   Ollama Name: {ollama_name}")
            
            return True
        else:
            print("❌ One or both processing methods failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        return False
    finally:
        # Clean up test image
        if Path(test_image_path).exists():
            Path(test_image_path).unlink()

def main():
    """Main test function"""
    print("🧪 Ollama Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Check Ollama status
    ollama_available = test_ollama_status()
    
    if not ollama_available:
        print("\n❌ Ollama is not available. Please ensure Ollama is running.")
        print("   To start Ollama, run: ollama serve")
        return
    
    # Test 2: Check available models
    test_available_models()
    
    # Test 3: Test Ollama processing
    test_ollama_processing()
    
    # Test 4: Compare with traditional processing
    test_ollama_vs_traditional()
    
    print("\n🎉 Ollama integration tests completed!")
    print("\n📚 Available Endpoints:")
    print("   - POST /upload/ollama - Process documents with Ollama LLM")
    print("   - GET /ollama/status - Check Ollama service status")
    print("   - GET /ollama/models - List available models")

if __name__ == "__main__":
    main()
