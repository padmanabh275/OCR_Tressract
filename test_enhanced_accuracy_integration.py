"""
Test Enhanced Accuracy Integration
Test the comprehensive accuracy system integration with app_with_database.py
"""

import requests
import json
import time
from pathlib import Path
import numpy as np
import cv2

def test_enhanced_accuracy_integration():
    """Test the enhanced accuracy integration"""
    
    print("🧪 Testing Enhanced Accuracy Integration...")
    
    # Test server availability
    try:
        response = requests.get("http://localhost:8001/accuracy/modes")
        if response.status_code == 200:
            print("✅ Server is running and accessible")
        else:
            print("❌ Server is not responding properly")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running. Please start the server first.")
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
        
        # Test with different accuracy modes
        for mode in ["fast_accuracy", "balanced_accuracy", "maximum_accuracy"]:
            print(f"\n🔬 Testing {mode} mode...")
            
            with open(test_file, "rb") as f:
                files = {"file": (test_file, f, "image/png")}
                data = {
                    "accuracy_mode": mode,
                    "document_type": "pan_card"
                }
                
                start_time = time.time()
                response = requests.post(
                    "http://localhost:8001/upload/enhanced",
                    files=files,
                    data=data
                )
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ {mode} mode working")
                    print(f"   Processing time: {processing_time:.2f}s")
                    print(f"   Confidence: {result['confidence_score']:.2f}")
                    print(f"   Accuracy boost: {result['accuracy_boost']:.2f}")
                    print(f"   Techniques applied: {len(result['techniques_applied'])}")
                else:
                    print(f"❌ {mode} mode failed: {response.status_code}")
                    print(f"   Error: {response.text}")
        
        # Clean up test file
        Path(test_file).unlink()
        
    except Exception as e:
        print(f"❌ Error testing enhanced upload: {e}")
    
    # Test enhanced batch upload
    print("\n📦 Testing enhanced batch upload...")
    try:
        # Create multiple test images
        test_files = []
        for i in range(3):
            test_image = create_test_image(f"Test Document {i+1}")
            test_file = f"test_batch_{i+1}.png"
            cv2.imwrite(test_file, test_image)
            test_files.append(test_file)
        
        # Test batch upload
        files = []
        for test_file in test_files:
            files.append(("files", (test_file, open(test_file, "rb"), "image/png")))
        
        data = {"accuracy_mode": "balanced_accuracy"}
        
        start_time = time.time()
        response = requests.post(
            "http://localhost:8001/upload/enhanced/batch",
            files=files,
            data=data
        )
        processing_time = time.time() - start_time
        
        # Close files
        for _, (_, file_obj, _) in files:
            file_obj.close()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Enhanced batch upload working")
            print(f"   Total documents: {result['total_documents']}")
            print(f"   Processed: {result['processed_documents']}")
            print(f"   Failed: {result['failed_documents']}")
            print(f"   Processing time: {processing_time:.2f}s")
        else:
            print(f"❌ Enhanced batch upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
        
        # Clean up test files
        for test_file in test_files:
            Path(test_file).unlink()
        
    except Exception as e:
        print(f"❌ Error testing enhanced batch upload: {e}")
    
    # Test processing stats
    print("\n📊 Testing processing stats...")
    try:
        response = requests.get("http://localhost:8001/processing/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✅ Processing stats endpoint working")
            print(f"Total documents: {stats['total_documents']}")
            print(f"Average confidence: {stats['average_confidence']:.2f}")
        else:
            print("❌ Processing stats endpoint failed")
    except Exception as e:
        print(f"❌ Error testing processing stats: {e}")
    
    print("\n🎉 Enhanced accuracy integration test completed!")

def create_test_image(text="TEST DOCUMENT"):
    """Create a test image for testing"""
    # Create a white background
    image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    
    # Add text
    cv2.putText(image, "PERMANENT ACCOUNT NUMBER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "P.A.N. : ABCDE1234F", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, f"NAME: {text}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "FATHER'S NAME: RAMESH KUMAR", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "DATE OF BIRTH: 15/01/1990", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "GOVT. OF INDIA", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    
    return image

def test_accuracy_comparison():
    """Test accuracy comparison between different modes"""
    
    print("\n🔬 Testing Accuracy Comparison...")
    
    try:
        # Create a test image
        test_image = create_test_image()
        test_file = "test_accuracy_comparison.png"
        cv2.imwrite(test_file, test_image)
        
        results = {}
        
        # Test each accuracy mode
        for mode in ["fast_accuracy", "balanced_accuracy", "maximum_accuracy"]:
            print(f"\n📊 Testing {mode}...")
            
            with open(test_file, "rb") as f:
                files = {"file": (test_file, f, "image/png")}
                data = {
                    "accuracy_mode": mode,
                    "document_type": "pan_card"
                }
                
                start_time = time.time()
                response = requests.post(
                    "http://localhost:8001/upload/enhanced",
                    files=files,
                    data=data
                )
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    results[mode] = {
                        "confidence": result['confidence_score'],
                        "accuracy_boost": result['accuracy_boost'],
                        "processing_time": processing_time,
                        "techniques_count": len(result['techniques_applied']),
                        "techniques": result['techniques_applied']
                    }
                    print(f"   Confidence: {result['confidence_score']:.3f}")
                    print(f"   Accuracy boost: {result['accuracy_boost']:.3f}")
                    print(f"   Processing time: {processing_time:.2f}s")
                    print(f"   Techniques: {len(result['techniques_applied'])}")
                else:
                    print(f"   ❌ Failed: {response.status_code}")
        
        # Clean up
        Path(test_file).unlink()
        
        # Print comparison
        print("\n📈 Accuracy Comparison Results:")
        print("=" * 80)
        print(f"{'Mode':<20} {'Confidence':<12} {'Boost':<10} {'Time':<8} {'Techniques':<12}")
        print("=" * 80)
        
        for mode, result in results.items():
            print(f"{mode:<20} {result['confidence']:<12.3f} {result['accuracy_boost']:<10.3f} {result['processing_time']:<8.2f} {result['techniques_count']:<12}")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error in accuracy comparison: {e}")

if __name__ == "__main__":
    test_enhanced_accuracy_integration()
    test_accuracy_comparison()
