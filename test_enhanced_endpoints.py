"""
Test Enhanced Accuracy Endpoints
Simple test to verify the enhanced accuracy endpoints work correctly
"""

import requests
import json
import time
import numpy as np
import cv2
from pathlib import Path

def create_test_image():
    """Create a test image for testing"""
    # Create a white background
    image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    
    # Add text
    cv2.putText(image, "PERMANENT ACCOUNT NUMBER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "P.A.N. : ABCDE1234F", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "NAME: RAJESH KUMAR SHARMA", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "FATHER'S NAME: RAMESH KUMAR", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "DATE OF BIRTH: 15/01/1990", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "GOVT. OF INDIA", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    
    return image

def test_enhanced_endpoints():
    """Test the enhanced accuracy endpoints"""
    
    print("🧪 Testing Enhanced Accuracy Endpoints...")
    
    # Test server availability
    try:
        response = requests.get("http://localhost:8001/accuracy/modes", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and accessible")
        else:
            print("❌ Server is not responding properly")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running. Please start the server first.")
        print("   Run: python app_with_database.py")
        return
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return
    
    # Test accuracy modes endpoint
    print("\n📊 Testing accuracy modes endpoint...")
    try:
        response = requests.get("http://localhost:8001/accuracy/modes")
        if response.status_code == 200:
            modes = response.json()
            print("✅ Accuracy modes endpoint working")
            print(f"Available modes: {list(modes['accuracy_modes'].keys())}")
        else:
            print("❌ Accuracy modes endpoint failed")
    except Exception as e:
        print(f"❌ Error testing accuracy modes: {e}")
    
    # Test accuracy stats endpoint
    print("\n📈 Testing accuracy stats endpoint...")
    try:
        response = requests.get("http://localhost:8001/accuracy/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✅ Accuracy stats endpoint working")
            print(f"Enhanced documents: {stats['total_enhanced_documents']}")
        else:
            print("❌ Accuracy stats endpoint failed")
    except Exception as e:
        print(f"❌ Error testing accuracy stats: {e}")
    
    # Test enhanced upload endpoint
    print("\n📤 Testing enhanced upload endpoint...")
    try:
        # Create a test image
        test_image = create_test_image()
        test_file = "test_enhanced_document.png"
        cv2.imwrite(test_file, test_image)
        
        # Test with balanced accuracy mode
        print("🔬 Testing balanced_accuracy mode...")
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file, f, "image/png")}
            data = {
                "accuracy_mode": "balanced_accuracy",
                "document_type": "pan_card"
            }
            
            start_time = time.time()
            response = requests.post(
                "http://localhost:8001/upload/enhanced",
                files=files,
                data=data,
                timeout=30
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Enhanced upload working")
                print(f"   Processing time: {processing_time:.2f}s")
                print(f"   Confidence: {result['confidence_score']:.3f}")
                print(f"   Accuracy boost: {result['accuracy_boost']:.3f}")
                print(f"   Techniques applied: {len(result['techniques_applied'])}")
                print(f"   Document type: {result['document_type']}")
                
                # Check extracted data
                extracted = result['extracted_data']
                print(f"   Extracted fields: {list(extracted.keys())}")
                
            else:
                print(f"❌ Enhanced upload failed: {response.status_code}")
                print(f"   Error: {response.text}")
        
        # Clean up test file
        Path(test_file).unlink()
        
    except Exception as e:
        print(f"❌ Error testing enhanced upload: {e}")
    
    # Test processing stats
    print("\n📊 Testing processing stats...")
    try:
        response = requests.get("http://localhost:8001/processing/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✅ Processing stats endpoint working")
            print(f"Total documents: {stats['total_documents']}")
            print(f"Average confidence: {stats['average_confidence']:.3f}")
        else:
            print("❌ Processing stats endpoint failed")
    except Exception as e:
        print(f"❌ Error testing processing stats: {e}")
    
    print("\n🎉 Enhanced accuracy endpoints test completed!")

if __name__ == "__main__":
    test_enhanced_endpoints()
