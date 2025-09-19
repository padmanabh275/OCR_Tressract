"""
Debug Confidence Scores
Debug why confidence scores are showing 0.000
"""

import cv2
import numpy as np
from unified_document_processor import UnifiedDocumentProcessor

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

def debug_confidence():
    """Debug confidence calculation"""
    print("🔍 Debugging Confidence Calculation...")
    
    # Create test image
    test_image = create_test_pan_image()
    test_file = "debug_pan_card.png"
    cv2.imwrite(test_file, test_image)
    
    # Initialize processor
    processor = UnifiedDocumentProcessor()
    
    print("\n🇮🇳 Testing Indian Method...")
    try:
        # Test Indian method directly
        indian_result = processor._process_with_indian_method(test_image, "")
        print(f"   Indian confidence: {indian_result.confidence_score}")
        print(f"   Indian success: {indian_result.success}")
        print(f"   Indian document type: {indian_result.document_type}")
        print(f"   Indian fields: {indian_result.extracted_data.financial_data}")
    except Exception as e:
        print(f"   ❌ Indian method error: {e}")
    
    print("\n🌍 Testing Standard Method...")
    try:
        # Test standard method directly
        standard_result = processor._process_with_standard_method(test_image, "")
        print(f"   Standard confidence: {standard_result.confidence_score}")
        print(f"   Standard success: {standard_result.success}")
        print(f"   Standard document type: {standard_result.document_type}")
        print(f"   Standard fields: {standard_result.extracted_data.financial_data}")
    except Exception as e:
        print(f"   ❌ Standard method error: {e}")
    
    print("\n🔄 Testing Dual Processing...")
    try:
        # Test dual processing
        dual_result = processor.process_document_dual(test_file)
        print(f"   Best confidence: {dual_result.best_result.confidence_score}")
        print(f"   Best method: {dual_result.best_result.processing_method}")
        print(f"   Indian confidence: {dual_result.confidence_comparison['indian']}")
        print(f"   Standard confidence: {dual_result.confidence_comparison['standard']}")
    except Exception as e:
        print(f"   ❌ Dual processing error: {e}")
    
    # Clean up
    import os
    if os.path.exists(test_file):
        os.remove(test_file)

if __name__ == "__main__":
    debug_confidence()
