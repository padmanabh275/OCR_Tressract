"""
Test script for the Document Information Extraction System
Demonstrates the system's capabilities with sample documents
"""

import json
from pathlib import Path
from document_extractor import DocumentProcessor, create_sample_documents

def test_document_extraction():
    """Test the document extraction system with sample documents"""
    print("Testing Document Information Extraction System")
    print("=" * 50)
    
    # Create sample documents
    create_sample_documents()
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Test with sample documents
    sample_dir = Path("sample_documents")
    test_results = []
    
    for file_path in sample_dir.glob("*.txt"):
        print(f"\nTesting with: {file_path.name}")
        print("-" * 30)
        
        # Extract information
        result = processor.extract_information(str(file_path))
        
        # Display results
        print(f"✓ Document Type: {result.document_type}")
        print(f"✓ Confidence Score: {result.confidence_score:.2f}")
        print(f"✓ First Name: {result.first_name or 'Not found'}")
        print(f"✓ Last Name: {result.last_name or 'Not found'}")
        print(f"✓ Date of Birth: {result.date_of_birth or 'Not found'}")
        print(f"✓ Marriage Date: {result.marriage_date or 'Not found'}")
        print(f"✓ SSN: {result.ssn or 'Not found'}")
        print(f"✓ Address: {result.current_address or 'Not found'}")
        
        if result.financial_data:
            print(f"✓ Financial Data: {json.dumps(result.financial_data, indent=2)}")
        
        test_results.append({
            'file': file_path.name,
            'document_type': result.document_type,
            'confidence': result.confidence_score,
            'fields_extracted': sum([
                1 if result.first_name else 0,
                1 if result.last_name else 0,
                1 if result.date_of_birth else 0,
                1 if result.ssn else 0,
                1 if result.current_address else 0
            ])
        })
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    total_docs = len(test_results)
    avg_confidence = sum(r['confidence'] for r in test_results) / total_docs
    avg_fields = sum(r['fields_extracted'] for r in test_results) / total_docs
    
    print(f"Total Documents Processed: {total_docs}")
    print(f"Average Confidence Score: {avg_confidence:.2f}")
    print(f"Average Fields Extracted: {avg_fields:.1f}")
    
    print("\nDetailed Results:")
    for result in test_results:
        print(f"  {result['file']}: {result['fields_extracted']} fields, {result['confidence']:.2f} confidence")
    
    return test_results

def test_specific_patterns():
    """Test specific extraction patterns"""
    print("\n" + "=" * 50)
    print("PATTERN TESTING")
    print("=" * 50)
    
    processor = DocumentProcessor()
    
    # Test SSN extraction
    test_texts = [
        "SSN: 123-45-6789",
        "Social Security Number: 123456789",
        "SS# 123 45 6789"
    ]
    
    print("SSN Extraction Tests:")
    for text in test_texts:
        ssn = processor.extract_ssn(text)
        print(f"  '{text}' -> {ssn}")
    
    # Test date extraction
    test_dates = [
        "Born on 03/15/1985",
        "Date of Birth: March 15, 1985",
        "Birth: 1985-03-15"
    ]
    
    print("\nDate Extraction Tests:")
    for text in test_dates:
        dates = processor.extract_dates(text)
        print(f"  '{text}' -> {dates}")
    
    # Test name extraction
    test_names = [
        "Name: John Michael Smith",
        "Employee: Jane Doe",
        "Applicant: Robert J. Wilson"
    ]
    
    print("\nName Extraction Tests:")
    for text in test_names:
        first, last = processor.extract_names(text)
        print(f"  '{text}' -> First: {first}, Last: {last}")

if __name__ == "__main__":
    # Run main test
    results = test_document_extraction()
    
    # Run pattern tests
    test_specific_patterns()
    
    print("\n" + "=" * 50)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 50)
