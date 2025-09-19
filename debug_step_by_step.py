"""
Debug Step by Step
Debug each step of the processing to find where it's failing
"""

import cv2
import numpy as np
import pytesseract
from indian_document_enhancer import IndianDocumentEnhancer
from document_extractor import DocumentProcessor

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

def debug_step_by_step():
    """Debug each step of processing"""
    print("🔍 Debugging Step by Step...")
    
    # Create test image
    test_image = create_test_pan_image()
    test_file = "debug_pan_card.png"
    cv2.imwrite(test_file, test_image)
    
    print("\n📄 Step 1: OCR Text Extraction...")
    try:
        text = pytesseract.image_to_string(test_image, config='--psm 6')
        print(f"   Text extracted: {len(text)} characters")
        print(f"   First 200 chars: {text[:200]}")
    except Exception as e:
        print(f"   ❌ OCR error: {e}")
        return
    
    print("\n🇮🇳 Step 2: Indian Document Enhancer...")
    try:
        enhancer = IndianDocumentEnhancer()
        indian_result = enhancer.enhance_indian_document(test_image, None)
        print(f"   Indian result type: {type(indian_result)}")
        print(f"   Indian confidence: {indian_result.confidence_score}")
        print(f"   Indian document type: {indian_result.document_type}")
        print(f"   Indian fields: {indian_result.extracted_fields}")
        print(f"   Indian validation: {indian_result.validation_results}")
    except Exception as e:
        print(f"   ❌ Indian enhancer error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🌍 Step 3: Standard Document Processor...")
    try:
        processor = DocumentProcessor()
        # Save image to temporary file for standard processor
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, test_image)
            temp_path = tmp_file.name
        
        try:
            standard_result = processor.extract_information(temp_path)
            print(f"   Standard result type: {type(standard_result)}")
            if standard_result:
                print(f"   Standard confidence: {standard_result.confidence_score}")
                print(f"   Standard document type: {standard_result.document_type}")
                print(f"   Standard first name: {standard_result.first_name}")
                print(f"   Standard last name: {standard_result.last_name}")
            else:
                print("   Standard result is None")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    except Exception as e:
        print(f"   ❌ Standard processor error: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    import os
    if os.path.exists(test_file):
        os.remove(test_file)

if __name__ == "__main__":
    debug_step_by_step()
