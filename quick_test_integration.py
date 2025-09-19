"""
Quick test to verify Indian document integration is working
"""

def test_imports():
    """Test that all modules can be imported"""
    try:
        print("🧪 Testing imports...")
        
        # Test Indian document enhancer
        from indian_document_enhancer import IndianDocumentEnhancer
        print("✅ IndianDocumentEnhancer imported")
        
        # Test advanced accuracy
        from advanced_indian_accuracy import AdvancedIndianAccuracy
        print("✅ AdvancedIndianAccuracy imported")
        
        # Test document extractor
        from document_extractor import DocumentProcessor
        print("✅ DocumentProcessor imported")
        
        # Test app with database
        from app_with_database import app
        print("✅ FastAPI app imported")
        
        print("\n🎉 All imports successful! Integration is working.")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_indian_enhancer():
    """Test Indian document enhancer functionality"""
    try:
        print("\n🧪 Testing Indian document enhancer...")
        
        from indian_document_enhancer import IndianDocumentEnhancer
        
        enhancer = IndianDocumentEnhancer()
        
        # Test with sample PAN card text
        sample_text = """
        PERMANENT ACCOUNT NUMBER
        P.A.N. : ABCDE1234F
        INCOME TAX DEPARTMENT
        GOVT. OF INDIA
        NAME: RAJESH KUMAR SHARMA
        FATHER'S NAME: RAMESH KUMAR SHARMA
        DATE OF BIRTH: 15/01/1990
        SIGNATURE: RAJESH KUMAR SHARMA
        """
        
        # Test classification
        doc_type = enhancer.classify_indian_document_type(sample_text)
        print(f"✅ Document type classified as: {doc_type}")
        
        # Test field extraction
        fields = enhancer.extract_indian_fields(sample_text, 'pan_card')
        print(f"✅ Fields extracted: {list(fields.keys())}")
        
        # Test validation
        validation = enhancer.validate_indian_data(fields, 'pan_card')
        print(f"✅ Validation results: {list(validation.keys())}")
        
        print("✅ Indian document enhancer working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Indian enhancer error: {e}")
        return False

def test_document_processor():
    """Test document processor with Indian integration"""
    try:
        print("\n🧪 Testing document processor...")
        
        from document_extractor import DocumentProcessor
        
        processor = DocumentProcessor()
        print("✅ DocumentProcessor initialized")
        
        # Test if Indian enhancer is available
        if hasattr(processor, 'indian_enhancer'):
            print("✅ Indian enhancer integrated")
        else:
            print("❌ Indian enhancer not found")
            return False
        
        print("✅ Document processor working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Document processor error: {e}")
        return False

def main():
    """Main test function"""
    print("🇮🇳 Quick Integration Test")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 3
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test Indian enhancer
    if test_indian_enhancer():
        tests_passed += 1
    
    # Test document processor
    if test_document_processor():
        tests_passed += 1
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Integration is working perfectly!")
        print("\n🚀 You can now start the server:")
        print("   python app_with_database.py")
        print("\n🧪 And run the full test suite:")
        print("   python test_indian_integration.py")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
