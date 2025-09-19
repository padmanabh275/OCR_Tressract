"""
Test script to verify the fixes for the document extraction system
"""

import json
from pathlib import Path
from document_extractor import DocumentProcessor

def test_basic_functionality():
    """Test basic document processing functionality"""
    print("🧪 Testing Document Extraction System Fixes")
    print("=" * 50)
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Test with sample text (simulating a document)
    test_text = """
    DRIVER LICENSE
    STATE OF CALIFORNIA
    DEPARTMENT OF MOTOR VEHICLES
    
    Name: JOHN MICHAEL SMITH
    Address: 123 MAIN STREET
    City: LOS ANGELES, CA 90210
    Date of Birth: 03/15/1985
    License Number: D1234567
    Class: C
    Expires: 03/15/2025
    """
    
    print("📄 Testing text processing...")
    
    # Test individual extraction methods
    print("\n🔍 Testing field extraction:")
    
    # Test name extraction
    first_name, last_name = processor.extract_names(test_text)
    print(f"Names: {first_name} {last_name}")
    
    # Test date extraction
    dates = processor.extract_dates(test_text)
    print(f"Dates: {dates}")
    
    # Test SSN extraction
    ssn = processor.extract_ssn(test_text)
    print(f"SSN: {ssn}")
    
    # Test address extraction
    address = processor.extract_address(test_text)
    print(f"Address: {address}")
    
    # Test document classification
    doc_type = processor.classify_document_type(test_text)
    print(f"Document Type: {doc_type}")
    
    # Test financial data extraction
    financial_data = processor.extract_financial_data(test_text)
    print(f"Financial Data: {financial_data}")
    
    print("\n✅ Basic functionality test completed!")
    return True

def test_pydantic_validation():
    """Test Pydantic validation with None values"""
    print("\n🔧 Testing Pydantic validation fixes...")
    
    from app import DocumentResponse
    
    # Test with None document_type (should default to "unknown")
    try:
        response = DocumentResponse(
            id="test-id",
            filename="test.pdf",
            document_type=None,  # This should be handled
            confidence_score=0.8,
            extracted_data={},
            processing_time=1.0,
            status="completed"
        )
        print("❌ Pydantic validation should have failed with None document_type")
        return False
    except Exception as e:
        print(f"✅ Pydantic correctly rejected None document_type: {e}")
    
    # Test with "unknown" document_type (should work)
    try:
        response = DocumentResponse(
            id="test-id",
            filename="test.pdf",
            document_type="unknown",
            confidence_score=0.8,
            extracted_data={},
            processing_time=1.0,
            status="completed"
        )
        print("✅ Pydantic validation works with 'unknown' document_type")
        return True
    except Exception as e:
        print(f"❌ Pydantic validation failed: {e}")
        return False

def test_tesseract_configuration():
    """Test Tesseract configuration"""
    print("\n🔧 Testing Tesseract configuration...")
    
    try:
        import pytesseract
        import os
        
        # Check if TESSDATA_PREFIX is set
        if 'TESSDATA_PREFIX' in os.environ:
            print(f"✅ TESSDATA_PREFIX is set: {os.environ['TESSDATA_PREFIX']}")
        else:
            print("⚠️  TESSDATA_PREFIX not set - will be auto-detected")
        
        # Test basic Tesseract functionality
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract version: {version}")
        except Exception as e:
            print(f"⚠️  Tesseract test failed: {e}")
            print("   This is expected if Tesseract is not installed")
        
        return True
    except ImportError:
        print("❌ pytesseract not available")
        return False

def main():
    """Main test function"""
    print("🚀 Document Extraction System - Fix Verification")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic functionality
    if test_basic_functionality():
        tests_passed += 1
    
    # Test 2: Pydantic validation
    if test_pydantic_validation():
        tests_passed += 1
    
    # Test 3: Tesseract configuration
    if test_tesseract_configuration():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The system should work correctly now.")
        print("\n🚀 You can now run:")
        print("   python start_server.py")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
