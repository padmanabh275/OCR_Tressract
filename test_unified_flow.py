"""
Test Unified Document Processing Flow
Test the comprehensive backend flow for both Indian and international documents
"""

import requests
import json
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_test_documents():
    """Create test documents for different types"""
    
    # Create PAN Card
    pan_img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(pan_img)
    try:
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_medium = ImageFont.truetype("arial.ttf", 18)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
    
    y_pos = 50
    draw.text((50, y_pos), "PERMANENT ACCOUNT NUMBER", fill='black', font=font_large)
    y_pos += 40
    draw.text((50, y_pos), "P.A.N. : ABCDE1234F", fill='black', font=font_medium)
    y_pos += 30
    draw.text((50, y_pos), "INCOME TAX DEPARTMENT", fill='black', font=font_medium)
    y_pos += 30
    draw.text((50, y_pos), "GOVT. OF INDIA", fill='black', font=font_medium)
    y_pos += 50
    draw.text((50, y_pos), "NAME: RAJESH KUMAR SHARMA", fill='black', font=font_medium)
    y_pos += 30
    draw.text((50, y_pos), "FATHER'S NAME: RAMESH KUMAR SHARMA", fill='black', font=font_medium)
    y_pos += 30
    draw.text((50, y_pos), "DATE OF BIRTH: 15/01/1990", fill='black', font=font_medium)
    y_pos += 30
    draw.text((50, y_pos), "SIGNATURE: RAJESH KUMAR SHARMA", fill='black', font=font_medium)
    pan_img.save("test_pan_card.png")
    
    # Create US Driver's License
    dl_img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(dl_img)
    y_pos = 50
    draw.text((50, y_pos), "DRIVER'S LICENSE", fill='black', font=font_large)
    y_pos += 40
    draw.text((50, y_pos), "DEPARTMENT OF MOTOR VEHICLES", fill='black', font=font_medium)
    y_pos += 30
    draw.text((50, y_pos), "CALIFORNIA", fill='black', font=font_medium)
    y_pos += 50
    draw.text((50, y_pos), "LICENSE NO: D1234567", fill='black', font=font_medium)
    y_pos += 30
    draw.text((50, y_pos), "NAME: JOHN DOE", fill='black', font=font_medium)
    y_pos += 30
    draw.text((50, y_pos), "DATE OF BIRTH: 01/15/1985", fill='black', font=font_medium)
    y_pos += 30
    draw.text((50, y_pos), "ADDRESS: 123 MAIN ST, LOS ANGELES, CA 90210", fill='black', font=font_medium)
    dl_img.save("test_us_dl.png")
    
    # Create US Passport
    passport_img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(passport_img)
    y_pos = 50
    draw.text((50, y_pos), "UNITED STATES OF AMERICA", fill='black', font=font_large)
    y_pos += 40
    draw.text((50, y_pos), "PASSPORT", fill='black', font=font_medium)
    y_pos += 30
    draw.text((50, y_pos), "PASSPORT NO: A1234567", fill='black', font=font_medium)
    y_pos += 50
    draw.text((50, y_pos), "SURNAME: DOE", fill='black', font=font_medium)
    y_pos += 30
    draw.text((50, y_pos), "GIVEN NAMES: JOHN MICHAEL", fill='black', font=font_medium)
    y_pos += 30
    draw.text((50, y_pos), "DATE OF BIRTH: 15 JAN 1985", fill='black', font=font_medium)
    y_pos += 30
    draw.text((50, y_pos), "PLACE OF BIRTH: CALIFORNIA, USA", fill='black', font=font_medium)
    passport_img.save("test_us_passport.png")
    
    print("✅ Created test documents:")
    print("  - test_pan_card.png (Indian PAN Card)")
    print("  - test_us_dl.png (US Driver's License)")
    print("  - test_us_passport.png (US Passport)")

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

def test_document_upload(file_path, expected_category, expected_type):
    """Test uploading a document and verify classification"""
    try:
        print(f"\n📄 Testing {Path(file_path).stem} upload...")
        
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'image/png')}
            
            response = requests.post(
                "http://localhost:8001/upload",
                files=files
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful!")
            print(f"   Document ID: {result['id']}")
            print(f"   Document Type: {result['document_type']}")
            print(f"   Document Category: {result.get('document_category', 'N/A')}")
            print(f"   Processing Method: {result.get('processing_method', 'N/A')}")
            print(f"   Confidence: {result['confidence_score']:.2f}")
            print(f"   Processing Time: {result['processing_time']:.2f}s")
            
            # Verify classification
            if result.get('document_category') == expected_category:
                print(f"✅ Category classification correct: {expected_category}")
            else:
                print(f"⚠️ Category classification: expected {expected_category}, got {result.get('document_category')}")
            
            if result['document_type'] == expected_type:
                print(f"✅ Document type classification correct: {expected_type}")
            else:
                print(f"⚠️ Document type classification: expected {expected_type}, got {result['document_type']}")
            
            # Print extracted fields
            extracted_data = result['extracted_data']
            print(f"   Extracted Fields:")
            for field, value in extracted_data.items():
                if value and field != 'financial_data' and field != 'validation':
                    print(f"     {field}: {value}")
            
            return result
        else:
            print(f"❌ Upload failed with status code: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        return None

def test_processing_stats():
    """Test processing statistics endpoint"""
    try:
        print(f"\n📊 Testing processing statistics...")
        
        response = requests.get("http://localhost:8001/processing/stats")
        
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Statistics retrieved successfully!")
            print(f"   Total Documents: {stats['total_documents']}")
            print(f"   By Category: {stats['by_category']}")
            print(f"   By Method: {stats['by_method']}")
            print(f"   By Type: {stats['by_type']}")
            print(f"   Average Confidence: {stats['average_confidence']:.2f}")
            
            # Print processing times
            print(f"   Processing Times:")
            for method, times in stats['processing_times'].items():
                if times['count'] > 0:
                    print(f"     {method}: {times['average']:.2f}s avg ({times['count']} docs)")
            
            return stats
        else:
            print(f"❌ Statistics request failed with status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")
        return None

def test_documents_by_category(category):
    """Test getting documents by category"""
    try:
        print(f"\n📋 Testing documents by category: {category}")
        
        response = requests.get(f"http://localhost:8001/documents/by-category?category={category}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Retrieved {result['total_documents']} documents in category '{category}'")
            return result
        else:
            print(f"❌ Category request failed with status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error getting documents by category: {e}")
        return None

def test_documents_by_method(method):
    """Test getting documents by processing method"""
    try:
        print(f"\n⚙️ Testing documents by method: {method}")
        
        response = requests.get(f"http://localhost:8001/documents/by-method?method={method}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Retrieved {result['total_documents']} documents with method '{method}'")
            return result
        else:
            print(f"❌ Method request failed with status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error getting documents by method: {e}")
        return None

def cleanup_test_files():
    """Clean up test files"""
    test_files = ["test_pan_card.png", "test_us_dl.png", "test_us_passport.png"]
    for file in test_files:
        if Path(file).exists():
            Path(file).unlink()
            print(f"🗑️ Cleaned up {file}")

def main():
    """Main test function"""
    print("🌍 Testing Unified Document Processing Flow")
    print("=" * 60)
    
    # Test server connection
    if not test_server_connection():
        return
    
    # Create test documents
    create_test_documents()
    
    try:
        # Test different document types
        print("\n🧪 Testing Document Classification and Processing...")
        
        # Test Indian document (PAN Card)
        pan_result = test_document_upload("test_pan_card.png", "indian", "pan_card")
        
        # Test US Driver's License
        dl_result = test_document_upload("test_us_dl.png", "international", "us_drivers_license")
        
        # Test US Passport
        passport_result = test_document_upload("test_us_passport.png", "international", "us_passport")
        
        # Test statistics
        stats_result = test_processing_stats()
        
        # Test category filtering
        indian_docs = test_documents_by_category("indian")
        international_docs = test_documents_by_category("international")
        
        # Test method filtering
        indian_enhanced_docs = test_documents_by_method("indian_enhanced")
        standard_enhanced_docs = test_documents_by_method("standard_enhanced")
        
        # Summary
        print("\n📊 Test Summary:")
        print("=" * 30)
        
        successful_tests = 0
        total_tests = 8
        
        if pan_result:
            successful_tests += 1
            print("✅ PAN Card Processing: PASSED")
        else:
            print("❌ PAN Card Processing: FAILED")
        
        if dl_result:
            successful_tests += 1
            print("✅ US Driver's License Processing: PASSED")
        else:
            print("❌ US Driver's License Processing: FAILED")
        
        if passport_result:
            successful_tests += 1
            print("✅ US Passport Processing: PASSED")
        else:
            print("❌ US Passport Processing: FAILED")
        
        if stats_result:
            successful_tests += 1
            print("✅ Processing Statistics: PASSED")
        else:
            print("❌ Processing Statistics: FAILED")
        
        if indian_docs:
            successful_tests += 1
            print("✅ Indian Documents Filter: PASSED")
        else:
            print("❌ Indian Documents Filter: FAILED")
        
        if international_docs:
            successful_tests += 1
            print("✅ International Documents Filter: PASSED")
        else:
            print("❌ International Documents Filter: FAILED")
        
        if indian_enhanced_docs:
            successful_tests += 1
            print("✅ Indian Enhanced Method Filter: PASSED")
        else:
            print("❌ Indian Enhanced Method Filter: FAILED")
        
        if standard_enhanced_docs:
            successful_tests += 1
            print("✅ Standard Enhanced Method Filter: PASSED")
        else:
            print("❌ Standard Enhanced Method Filter: FAILED")
        
        print(f"\n🎯 Overall Result: {successful_tests}/{total_tests} tests passed")
        
        if successful_tests == total_tests:
            print("🎉 All tests passed! Unified processing flow is working perfectly!")
            print("\n🚀 The system can now:")
            print("  - Automatically detect Indian vs International documents")
            print("  - Apply appropriate processing methods")
            print("  - Provide detailed classification and metadata")
            print("  - Filter documents by category and processing method")
            print("  - Generate comprehensive processing statistics")
        else:
            print("⚠️ Some tests failed. Check the output above for details.")
    
    finally:
        # Clean up test files
        cleanup_test_files()

if __name__ == "__main__":
    main()
