"""
Test Indian Document Integration with app_with_database.py
Comprehensive testing of the integrated system
"""

import requests
import json
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_test_pan_card():
    """Create a test PAN card image"""
    # Create a white background
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_medium = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw PAN card content
    y_pos = 50
    
    # Header
    draw.text((50, y_pos), "PERMANENT ACCOUNT NUMBER", fill='black', font=font_large)
    y_pos += 40
    
    # PAN Number
    draw.text((50, y_pos), "P.A.N. : ABCDE1234F", fill='black', font=font_medium)
    y_pos += 30
    
    # Income Tax Department
    draw.text((50, y_pos), "INCOME TAX DEPARTMENT", fill='black', font=font_medium)
    y_pos += 30
    
    # Government of India
    draw.text((50, y_pos), "GOVT. OF INDIA", fill='black', font=font_medium)
    y_pos += 50
    
    # Name
    draw.text((50, y_pos), "NAME: RAJESH KUMAR SHARMA", fill='black', font=font_medium)
    y_pos += 30
    
    # Father's Name
    draw.text((50, y_pos), "FATHER'S NAME: RAMESH KUMAR SHARMA", fill='black', font=font_medium)
    y_pos += 30
    
    # Date of Birth
    draw.text((50, y_pos), "DATE OF BIRTH: 15/01/1990", fill='black', font=font_medium)
    y_pos += 30
    
    # Signature
    draw.text((50, y_pos), "SIGNATURE: RAJESH KUMAR SHARMA", fill='black', font=font_medium)
    
    # Save the image
    img.save("test_pan_card.png")
    print("✅ Created test PAN card: test_pan_card.png")
    return "test_pan_card.png"

def create_test_aadhaar_card():
    """Create a test Aadhaar card image"""
    # Create a white background
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_medium = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw Aadhaar card content
    y_pos = 50
    
    # Header
    draw.text((50, y_pos), "AADHAAR", fill='black', font=font_large)
    y_pos += 40
    
    # UIDAI
    draw.text((50, y_pos), "UNIQUE IDENTIFICATION AUTHORITY OF INDIA", fill='black', font=font_medium)
    y_pos += 30
    
    # Government of India
    draw.text((50, y_pos), "GOVERNMENT OF INDIA", fill='black', font=font_medium)
    y_pos += 50
    
    # UID Number
    draw.text((50, y_pos), "UID: 1234 5678 9012", fill='black', font=font_medium)
    y_pos += 40
    
    # Name
    draw.text((50, y_pos), "NAME: PRIYA SHARMA", fill='black', font=font_medium)
    y_pos += 30
    
    # Father's Name
    draw.text((50, y_pos), "FATHER'S NAME: RAJESH KUMAR SHARMA", fill='black', font=font_medium)
    y_pos += 30
    
    # Mother's Name
    draw.text((50, y_pos), "MOTHER'S NAME: SUNITA SHARMA", fill='black', font=font_medium)
    y_pos += 30
    
    # Date of Birth
    draw.text((50, y_pos), "DATE OF BIRTH: 20/05/1995", fill='black', font=font_medium)
    y_pos += 30
    
    # Gender
    draw.text((50, y_pos), "GENDER: F", fill='black', font=font_medium)
    y_pos += 30
    
    # Address
    draw.text((50, y_pos), "ADDRESS: 123, SECTOR 15, NEW DELHI, 110015", fill='black', font=font_medium)
    y_pos += 30
    
    # PIN
    draw.text((50, y_pos), "PIN: 110015", fill='black', font=font_medium)
    
    # Save the image
    img.save("test_aadhaar_card.png")
    print("✅ Created test Aadhaar card: test_aadhaar_card.png")
    return "test_aadhaar_card.png"

def test_server_connection():
    """Test if the server is running"""
    try:
        response = requests.get("http://localhost:8001/")
        if response.status_code == 200:
            print("✅ Server is running on port 8001")
            return True
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running. Please start the server first:")
        print("   python app_with_database.py")
        return False
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return False

def test_indian_document_upload(file_path, document_type="auto"):
    """Test uploading an Indian document"""
    try:
        print(f"\n📄 Testing {document_type} document upload...")
        
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'image/png')}
            data = {'document_type': document_type}
            
            response = requests.post(
                "http://localhost:8001/process/indian",
                files=files,
                data=data
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful!")
            print(f"   Document ID: {result['id']}")
            print(f"   Document Type: {result['document_type']}")
            print(f"   Confidence: {result['confidence_score']:.2f}")
            print(f"   Processing Time: {result['processing_time']:.2f}s")
            
            # Print extracted fields
            extracted_data = result['extracted_data']
            print(f"   Extracted Fields:")
            for field, value in extracted_data.items():
                if value and field != 'financial_data':
                    print(f"     {field}: {value}")
            
            # Print Indian-specific fields
            if 'financial_data' in extracted_data and 'indian_fields' in extracted_data['financial_data']:
                indian_fields = extracted_data['financial_data']['indian_fields']
                print(f"   Indian Fields:")
                for field, value in indian_fields.items():
                    if value:
                        print(f"     {field}: {value}")
            
            return result
        else:
            print(f"❌ Upload failed with status code: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        return None

def test_regular_upload(file_path):
    """Test regular document upload (should auto-detect Indian documents)"""
    try:
        print(f"\n📄 Testing regular upload with auto-detection...")
        
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'image/png')}
            
            response = requests.post(
                "http://localhost:8001/upload",
                files=files
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Regular upload successful!")
            print(f"   Document ID: {result['id']}")
            print(f"   Document Type: {result['document_type']}")
            print(f"   Confidence: {result['confidence_score']:.2f}")
            print(f"   Processing Time: {result['processing_time']:.2f}s")
            
            # Check if Indian enhancement was used
            if 'indian_enhancement' in result.get('extracted_data', {}).get('financial_data', {}):
                print(f"   🇮🇳 Indian enhancement was applied!")
            
            return result
        else:
            print(f"❌ Regular upload failed with status code: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error during regular upload: {e}")
        return None

def test_indian_stats():
    """Test Indian document statistics endpoint"""
    try:
        print(f"\n📊 Testing Indian document statistics...")
        
        response = requests.get("http://localhost:8001/indian/stats")
        
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Statistics retrieved successfully!")
            print(f"   Total Indian Documents: {stats['total_indian_documents']}")
            print(f"   Document Types: {stats['document_types']}")
            print(f"   Average Confidence: {stats['average_confidence']:.2f}")
            print(f"   Recent Uploads: {stats['recent_uploads']}")
            return stats
        else:
            print(f"❌ Statistics request failed with status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")
        return None

def test_indian_documents_list():
    """Test getting list of Indian documents"""
    try:
        print(f"\n📋 Testing Indian documents list...")
        
        response = requests.get("http://localhost:8001/indian/documents")
        
        if response.status_code == 200:
            documents = response.json()
            print(f"✅ Retrieved {len(documents)} Indian documents!")
            
            for i, doc in enumerate(documents[:3]):  # Show first 3
                print(f"   Document {i+1}:")
                print(f"     ID: {doc['id']}")
                print(f"     Type: {doc.get('extracted_data', {}).get('financial_data', {}).get('document_type', 'unknown')}")
                print(f"     Confidence: {doc.get('confidence_score', 0):.2f}")
            
            return documents
        else:
            print(f"❌ Documents list request failed with status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error getting documents list: {e}")
        return None

def cleanup_test_files():
    """Clean up test files"""
    test_files = ["test_pan_card.png", "test_aadhaar_card.png"]
    for file in test_files:
        if Path(file).exists():
            Path(file).unlink()
            print(f"🗑️ Cleaned up {file}")

def main():
    """Main test function"""
    print("🇮🇳 Testing Indian Document Integration")
    print("=" * 50)
    
    # Test server connection
    if not test_server_connection():
        return
    
    # Create test documents
    pan_file = create_test_pan_card()
    aadhaar_file = create_test_aadhaar_card()
    
    try:
        # Test Indian document processing
        print("\n🧪 Testing Indian Document Processing...")
        pan_result = test_indian_document_upload(pan_file, "pan_card")
        aadhaar_result = test_indian_document_upload(aadhaar_file, "aadhaar_card")
        
        # Test auto-detection
        print("\n🔍 Testing Auto-Detection...")
        auto_pan_result = test_regular_upload(pan_file)
        auto_aadhaar_result = test_regular_upload(aadhaar_file)
        
        # Test statistics
        test_indian_stats()
        
        # Test documents list
        test_indian_documents_list()
        
        # Summary
        print("\n📊 Test Summary:")
        print("=" * 30)
        
        successful_tests = 0
        total_tests = 6
        
        if pan_result:
            successful_tests += 1
            print("✅ PAN Card Processing: PASSED")
        else:
            print("❌ PAN Card Processing: FAILED")
        
        if aadhaar_result:
            successful_tests += 1
            print("✅ Aadhaar Card Processing: PASSED")
        else:
            print("❌ Aadhaar Card Processing: FAILED")
        
        if auto_pan_result:
            successful_tests += 1
            print("✅ Auto-Detection (PAN): PASSED")
        else:
            print("❌ Auto-Detection (PAN): FAILED")
        
        if auto_aadhaar_result:
            successful_tests += 1
            print("✅ Auto-Detection (Aadhaar): PASSED")
        else:
            print("❌ Auto-Detection (Aadhaar): FAILED")
        
        if test_indian_stats():
            successful_tests += 1
            print("✅ Statistics Endpoint: PASSED")
        else:
            print("❌ Statistics Endpoint: FAILED")
        
        if test_indian_documents_list():
            successful_tests += 1
            print("✅ Documents List Endpoint: PASSED")
        else:
            print("❌ Documents List Endpoint: FAILED")
        
        print(f"\n🎯 Overall Result: {successful_tests}/{total_tests} tests passed")
        
        if successful_tests == total_tests:
            print("🎉 All tests passed! Indian document integration is working perfectly!")
        else:
            print("⚠️ Some tests failed. Check the output above for details.")
    
    finally:
        # Clean up test files
        cleanup_test_files()

if __name__ == "__main__":
    main()
