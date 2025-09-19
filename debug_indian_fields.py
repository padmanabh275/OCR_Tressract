"""
Debug Indian Fields Extraction
Debug the Indian fields extraction to see what's being returned
"""

import cv2
import numpy as np
import pytesseract
from indian_document_enhancer import IndianDocumentEnhancer

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

def debug_indian_fields():
    """Debug Indian fields extraction"""
    print("🔍 Debugging Indian Fields Extraction...")
    
    # Create test image
    test_image = create_test_pan_image()
    
    # Extract text
    text = pytesseract.image_to_string(test_image, config='--psm 6')
    print(f"📄 Extracted text: {text[:100]}...")
    
    # Initialize enhancer
    enhancer = IndianDocumentEnhancer()
    
    # Test field extraction directly
    print("\n🇮🇳 Testing field extraction...")
    try:
        extracted_fields = enhancer.extract_indian_fields(text, 'pan_card')
        print(f"   Extracted fields: {extracted_fields}")
        print(f"   Field types: {[(k, type(v)) for k, v in extracted_fields.items()]}")
        
        # Test validation
        print("\n🔍 Testing validation...")
        validation_results = enhancer.validate_indian_data(extracted_fields, 'pan_card')
        print(f"   Validation results: {validation_results}")
        print(f"   Validation types: {[(k, type(v)) for k, v in validation_results.items()]}")
        
        # Test confidence calculation
        print("\n📊 Testing confidence calculation...")
        confidence = enhancer.calculate_indian_confidence(extracted_fields, validation_results)
        print(f"   Confidence: {confidence}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_indian_fields()
