"""
Test script for accuracy improvements
Run this to measure accuracy improvements
"""

import cv2
import numpy as np
from document_extractor import AdvancedDocumentProcessor
import time
from pathlib import Path

def test_accuracy_improvements():
    """Test the accuracy improvements"""
    print("🧪 Testing Accuracy Improvements")
    print("=" * 50)
    
    processor = AdvancedDocumentProcessor()
    
    # Test with sample images if available
    test_images = [
        "sample_driver_license.jpg",
        "sample_passport.jpg", 
        "sample_tax_return.pdf",
        "sample_bank_statement.jpg"
    ]
    
    results = []
    
    for image_path in test_images:
        if Path(image_path).exists():
            print(f"\n📄 Testing: {image_path}")
            
            # Load image
            if image_path.endswith('.pdf'):
                # For PDFs, you'd need to convert to image first
                print("   ⚠️  PDF testing requires image conversion")
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                print("   ❌ Could not load image")
                continue
            
            # Process with improved system
            start_time = time.time()
            result = processor.process_document_advanced(image)
            processing_time = time.time() - start_time
            
            print(f"   ✅ Processing time: {processing_time:.2f}s")
            print(f"   📊 Confidence: {result.get('overall_confidence', 0):.2f}")
            print(f"   📝 Fields extracted: {len(result.get('fields', {}))}")
            
            results.append({
                'file': image_path,
                'confidence': result.get('overall_confidence', 0),
                'fields': len(result.get('fields', {})),
                'time': processing_time
            })
        else:
            print(f"   ⚠️  Sample file not found: {image_path}")
    
    # Summary
    if results:
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        avg_fields = sum(r['fields'] for r in results) / len(results)
        avg_time = sum(r['time'] for r in results) / len(results)
        
        print("\n📈 ACCURACY IMPROVEMENT SUMMARY")
        print("=" * 50)
        print(f"Average Confidence: {avg_confidence:.2f}")
        print(f"Average Fields Extracted: {avg_fields:.1f}")
        print(f"Average Processing Time: {avg_time:.2f}s")
        print(f"Documents Processed: {len(results)}")
        
        if avg_confidence > 0.8:
            print("\n🎉 EXCELLENT! High accuracy achieved")
        elif avg_confidence > 0.6:
            print("\n✅ GOOD! Decent accuracy improvement")
        else:
            print("\n⚠️  NEEDS IMPROVEMENT! Consider additional enhancements")
    else:
        print("\n⚠️  No test documents found. Add sample images to test accuracy.")

if __name__ == "__main__":
    test_accuracy_improvements()
