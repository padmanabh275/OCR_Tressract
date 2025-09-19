"""
Test Dual Processing
Test the dual processing functionality (Indian + Standard) with confidence comparison
"""

import requests
import json
import time
import numpy as np
import cv2
from pathlib import Path

def create_test_pan_image():
    """Create a test PAN card image"""
    # Create a white background
    image = np.ones((300, 500, 3), dtype=np.uint8) * 255
    
    # Add PAN card text
    cv2.putText(image, "INCOME TAX DEPARTMENT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(image, "GOVT. OF INDIA", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(image, "PERMANENT ACCOUNT NUMBER", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    cv2.putText(image, "P.A.N. : ABCDE1234F", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    cv2.putText(image, "NAME: RAJESH KUMAR SHARMA", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "FATHER'S NAME: RAMESH KUMAR SHARMA", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "DATE OF BIRTH: 15/01/1990", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "SIGNATURE: RAJESH KUMAR SHARMA", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    
    return image

def create_test_us_document():
    """Create a test US document image"""
    # Create a white background
    image = np.ones((300, 500, 3), dtype=np.uint8) * 255
    
    # Add US document text
    cv2.putText(image, "UNITED STATES OF AMERICA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(image, "DRIVER LICENSE", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(image, "NAME: JOHN DOE", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "DATE OF BIRTH: 01/15/1990", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "ADDRESS: 123 MAIN ST", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "CITY: NEW YORK, NY 10001", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "LICENSE NO: D123456789", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(image, "EXPIRES: 01/15/2025", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    
    return image

def test_dual_processing():
    """Test the dual processing functionality"""
    
    print("🧪 Testing Dual Processing (Indian + Standard)...")
    
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
    
    # Test 1: Indian Document (PAN Card)
    print("\n🇮🇳 Test 1: Indian Document (PAN Card)")
    print("=" * 50)
    
    try:
        # Create test PAN card image
        test_image = create_test_pan_image()
        test_file = "test_pan_card.png"
        cv2.imwrite(test_file, test_image)
        
        # Upload with dual processing
        with open(test_file, "rb") as f:
            files = {"file": (test_file, f, "image/png")}
            
            start_time = time.time()
            response = requests.post(
                "http://localhost:8001/upload/dual",
                files=files,
                timeout=60
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Dual processing completed successfully!")
                print(f"   📄 Document Type: {result['document_type']}")
                print(f"   📊 Document Category: {result.get('document_category', 'unknown')}")
                print(f"   🏆 Best Method: {result.get('dual_processing', {}).get('best_method', 'unknown')}")
                print(f"   📈 Confidence Score: {result['confidence_score']:.3f}")
                print(f"   ⏱️ Total Processing Time: {processing_time:.2f}s")
                
                # Show confidence comparison
                dual_info = result['dual_processing']
                print(f"\n   📊 Confidence Comparison:")
                print(f"      🇮🇳 Indian Method: {dual_info['indian_confidence']:.3f}")
                print(f"      🌍 Standard Method: {dual_info['standard_confidence']:.3f}")
                print(f"      📏 Difference: {dual_info['confidence_difference']:.3f}")
                
                # Show processing times
                print(f"\n   ⏱️ Processing Times:")
                print(f"      🇮🇳 Indian: {dual_info['indian_processing_time']:.2f}s")
                print(f"      🌍 Standard: {dual_info['standard_processing_time']:.2f}s")
                print(f"      📊 Total: {dual_info['total_processing_time']:.2f}s")
                
                # Show extracted data
                extracted = result['extracted_data']
                print(f"\n   📋 Extracted Data:")
                if extracted['first_name']:
                    print(f"      Name: {extracted['first_name']} {extracted['last_name']}")
                if extracted['date_of_birth']:
                    print(f"      DOB: {extracted['date_of_birth']}")
                if extracted['ssn']:
                    print(f"      ID: {extracted['ssn']}")
                if extracted['current_address']:
                    print(f"      Address: {extracted['current_address']}")
                
            else:
                print(f"❌ Dual processing failed: {response.status_code}")
                print(f"   Error: {response.text}")
        
        # Clean up
        Path(test_file).unlink()
        
    except Exception as e:
        print(f"❌ Error testing Indian document: {e}")
    
    # Test 2: US Document
    print("\n🌍 Test 2: US Document (Driver License)")
    print("=" * 50)
    
    try:
        # Create test US document image
        test_image = create_test_us_document()
        test_file = "test_us_document.png"
        cv2.imwrite(test_file, test_image)
        
        # Upload with dual processing
        with open(test_file, "rb") as f:
            files = {"file": (test_file, f, "image/png")}
            
            start_time = time.time()
            response = requests.post(
                "http://localhost:8001/upload/dual",
                files=files,
                timeout=60
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Dual processing completed successfully!")
                print(f"   📄 Document Type: {result['document_type']}")
                print(f"   📊 Document Category: {result.get('document_category', 'unknown')}")
                print(f"   🏆 Best Method: {result.get('dual_processing', {}).get('best_method', 'unknown')}")
                print(f"   📈 Confidence Score: {result['confidence_score']:.3f}")
                print(f"   ⏱️ Total Processing Time: {processing_time:.2f}s")
                
                # Show confidence comparison
                dual_info = result['dual_processing']
                print(f"\n   📊 Confidence Comparison:")
                print(f"      🇮🇳 Indian Method: {dual_info['indian_confidence']:.3f}")
                print(f"      🌍 Standard Method: {dual_info['standard_confidence']:.3f}")
                print(f"      📏 Difference: {dual_info['confidence_difference']:.3f}")
                
                # Show processing times
                print(f"\n   ⏱️ Processing Times:")
                print(f"      🇮🇳 Indian: {dual_info['indian_processing_time']:.2f}s")
                print(f"      🌍 Standard: {dual_info['standard_processing_time']:.2f}s")
                print(f"      📊 Total: {dual_info['total_processing_time']:.2f}s")
                
                # Show extracted data
                extracted = result['extracted_data']
                print(f"\n   📋 Extracted Data:")
                if extracted['first_name']:
                    print(f"      Name: {extracted['first_name']} {extracted['last_name']}")
                if extracted['date_of_birth']:
                    print(f"      DOB: {extracted['date_of_birth']}")
                if extracted['ssn']:
                    print(f"      ID: {extracted['ssn']}")
                if extracted['current_address']:
                    print(f"      Address: {extracted['current_address']}")
                
            else:
                print(f"❌ Dual processing failed: {response.status_code}")
                print(f"   Error: {response.text}")
        
        # Clean up
        Path(test_file).unlink()
        
    except Exception as e:
        print(f"❌ Error testing US document: {e}")
    
    print("\n🎉 Dual processing test completed!")

if __name__ == "__main__":
    test_dual_processing()
