"""
Document Information Extraction System - Demo Script
Demonstrates extraction of structured information from various document types
"""

import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Import our custom modules
from unified_document_processor import UnifiedDocumentProcessor
from indian_document_enhancer import IndianDocumentEnhancer
from enhanced_document_classifier import EnhancedDocumentClassifier
from dynamic_field_extractor import DynamicFieldExtractor

@dataclass
class ExtractionResult:
    """Result of document information extraction"""
    document_type: str
    confidence_score: float
    extracted_fields: Dict[str, Any]
    processing_method: str
    processing_time: float
    validation_results: Dict[str, Any]
    raw_text: str

class DocumentExtractionDemo:
    """Demo class for document information extraction"""
    
    def __init__(self):
        """Initialize the demo system"""
        print("🚀 Initializing Document Extraction System...")
        
        # Initialize processors
        self.unified_processor = UnifiedDocumentProcessor()
        self.indian_enhancer = IndianDocumentEnhancer()
        self.classifier = EnhancedDocumentClassifier()
        self.dynamic_extractor = DynamicFieldExtractor()
        
        print("✅ System initialized successfully!")
    
    def extract_from_image(self, image_path: str) -> ExtractionResult:
        """Extract information from an image file"""
        print(f"\n📄 Processing image: {Path(image_path).name}")
        
        start_time = datetime.now()
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Use unified processor for dual processing
            result = self.unified_processor.process_document_dual(image_path)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Extract the best result
            best_result = result.best_result
            
            return ExtractionResult(
                document_type=best_result.document_type,
                confidence_score=best_result.confidence_score,
                extracted_fields=self._extract_structured_fields(best_result.extracted_data),
                processing_method=best_result.processing_method,
                processing_time=processing_time,
                validation_results=best_result.validation_results,
                raw_text=best_result.extracted_data.raw_text or ""
            )
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return ExtractionResult(
                document_type="unknown",
                confidence_score=0.0,
                extracted_fields={},
                processing_method="error",
                processing_time=(datetime.now() - start_time).total_seconds(),
                validation_results={"valid": False, "errors": [str(e)]},
                raw_text=""
            )
    
    def extract_from_pdf(self, pdf_path: str) -> ExtractionResult:
        """Extract information from a PDF file"""
        print(f"\n📄 Processing PDF: {Path(pdf_path).name}")
        
        start_time = datetime.now()
        
        try:
            # Convert PDF to image
            doc = fitz.open(pdf_path)
            page = doc[0]  # First page
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            doc.close()
            
            if image is None:
                raise ValueError(f"Could not convert PDF to image: {pdf_path}")
            
            # Use unified processor
            result = self.unified_processor.process_document_dual(pdf_path)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Extract the best result
            best_result = result.best_result
            
            return ExtractionResult(
                document_type=best_result.document_type,
                confidence_score=best_result.confidence_score,
                extracted_fields=self._extract_structured_fields(best_result.extracted_data),
                processing_method=best_result.processing_method,
                processing_time=processing_time,
                validation_results=best_result.validation_results,
                raw_text=best_result.extracted_data.raw_text or ""
            )
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")
            return ExtractionResult(
                document_type="unknown",
                confidence_score=0.0,
                extracted_fields={},
                processing_method="error",
                processing_time=(datetime.now() - start_time).total_seconds(),
                validation_results={"valid": False, "errors": [str(e)]},
                raw_text=""
            )
    
    def _extract_structured_fields(self, extracted_data) -> Dict[str, Any]:
        """Extract structured fields from extracted data"""
        fields = {
            "personal_info": {
                "first_name": extracted_data.first_name,
                "last_name": extracted_data.last_name,
                "full_name": f"{extracted_data.first_name or ''} {extracted_data.last_name or ''}".strip(),
                "date_of_birth": extracted_data.date_of_birth,
                "marriage_date": extracted_data.marriage_date,
                "birth_city": extracted_data.birth_city,
                "current_address": extracted_data.current_address
            },
            "identification": {
                "ssn": extracted_data.ssn,
                "document_type": extracted_data.document_type
            },
            "financial_data": extracted_data.financial_data or {},
            "metadata": {
                "confidence_score": extracted_data.confidence_score,
                "extraction_method": extracted_data.extraction_method,
                "raw_text_length": len(extracted_data.raw_text or "")
            }
        }
        
        return fields
    
    def batch_process(self, file_paths: List[str]) -> List[ExtractionResult]:
        """Process multiple documents in batch"""
        print(f"\n🔄 Batch processing {len(file_paths)} documents...")
        
        results = []
        for i, file_path in enumerate(file_paths, 1):
            print(f"\n[{i}/{len(file_paths)}] Processing: {Path(file_path).name}")
            
            if file_path.lower().endswith('.pdf'):
                result = self.extract_from_pdf(file_path)
            else:
                result = self.extract_from_image(file_path)
            
            results.append(result)
            
            # Print summary
            print(f"   📊 Type: {result.document_type}")
            print(f"   📈 Confidence: {result.confidence_score:.3f}")
            print(f"   ⏱️ Time: {result.processing_time:.2f}s")
            print(f"   🎯 Method: {result.processing_method}")
        
        return results
    
    def generate_report(self, results: List[ExtractionResult], output_file: str = "extraction_report.json"):
        """Generate a comprehensive extraction report"""
        print(f"\n📊 Generating extraction report: {output_file}")
        
        # Calculate statistics
        total_documents = len(results)
        successful_extractions = len([r for r in results if r.confidence_score > 0])
        average_confidence = sum(r.confidence_score for r in results) / total_documents if total_documents > 0 else 0
        average_processing_time = sum(r.processing_time for r in results) / total_documents if total_documents > 0 else 0
        
        # Group by document type
        document_types = {}
        for result in results:
            doc_type = result.document_type
            if doc_type not in document_types:
                document_types[doc_type] = []
            document_types[doc_type].append(result)
        
        # Create report
        report = {
            "summary": {
                "total_documents": total_documents,
                "successful_extractions": successful_extractions,
                "success_rate": f"{(successful_extractions/total_documents)*100:.1f}%" if total_documents > 0 else "0%",
                "average_confidence": round(average_confidence, 3),
                "average_processing_time": round(average_processing_time, 2),
                "generated_at": datetime.now().isoformat()
            },
            "document_types": {
                doc_type: {
                    "count": len(docs),
                    "average_confidence": round(sum(d.confidence_score for d in docs) / len(docs), 3),
                    "average_processing_time": round(sum(d.processing_time for d in docs) / len(docs), 2)
                }
                for doc_type, docs in document_types.items()
            },
            "results": [asdict(result) for result in results]
        }
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Report saved to: {output_file}")
        
        # Print summary
        print(f"\n📈 Extraction Summary:")
        print(f"   Total Documents: {total_documents}")
        print(f"   Successful: {successful_extractions} ({(successful_extractions/total_documents)*100:.1f}%)")
        print(f"   Average Confidence: {average_confidence:.3f}")
        print(f"   Average Time: {average_processing_time:.2f}s")
        
        return report

def create_sample_documents():
    """Create sample documents for testing"""
    print("\n🎨 Creating sample documents for testing...")
    
    # Create sample PAN card
    pan_image = np.ones((300, 500, 3), dtype=np.uint8) * 255
    cv2.putText(pan_image, "INCOME TAX DEPARTMENT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(pan_image, "GOVT. OF INDIA", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(pan_image, "PERMANENT ACCOUNT NUMBER", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    cv2.putText(pan_image, "P.A.N. : ABCDE1234F", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    cv2.putText(pan_image, "NAME: RAJESH KUMAR SHARMA", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(pan_image, "FATHER'S NAME: RAMESH KUMAR SHARMA", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(pan_image, "DATE OF BIRTH: 15/01/1990", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(pan_image, "SIGNATURE: RAJESH KUMAR SHARMA", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.imwrite("sample_pan_card.png", pan_image)
    
    # Create sample US document
    us_image = np.ones((300, 500, 3), dtype=np.uint8) * 255
    cv2.putText(us_image, "UNITED STATES OF AMERICA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(us_image, "DRIVER LICENSE", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(us_image, "NAME: JOHN DOE", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(us_image, "DATE OF BIRTH: 01/15/1990", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(us_image, "ADDRESS: 123 MAIN ST", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(us_image, "CITY: NEW YORK, NY 10001", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(us_image, "LICENSE NO: D123456789", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.putText(us_image, "EXPIRES: 01/15/2025", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.imwrite("sample_us_document.png", us_image)
    
    print("✅ Sample documents created: sample_pan_card.png, sample_us_document.png")

def main():
    """Main demo function"""
    print("🎯 Document Information Extraction System - Demo")
    print("=" * 60)
    
    # Create sample documents
    create_sample_documents()
    
    # Initialize demo system
    demo = DocumentExtractionDemo()
    
    # Process sample documents
    sample_files = ["sample_pan_card.png", "sample_us_document.png"]
    
    # Add any existing documents from uploads folder
    uploads_dir = Path("uploads")
    if uploads_dir.exists():
        existing_files = list(uploads_dir.glob("*.png")) + list(uploads_dir.glob("*.pdf"))
        sample_files.extend([str(f) for f in existing_files[:3]])  # Add up to 3 existing files
    
    print(f"\n📁 Processing {len(sample_files)} sample documents...")
    
    # Process documents
    results = demo.batch_process(sample_files)
    
    # Generate report
    report = demo.generate_report(results)
    
    # Print detailed results
    print(f"\n📋 Detailed Results:")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {Path(sample_files[i-1]).name}")
        print(f"   Type: {result.document_type}")
        print(f"   Confidence: {result.confidence_score:.3f}")
        print(f"   Method: {result.processing_method}")
        print(f"   Time: {result.processing_time:.2f}s")
        
        if result.extracted_fields.get("personal_info", {}).get("full_name"):
            print(f"   Name: {result.extracted_fields['personal_info']['full_name']}")
        if result.extracted_fields.get("personal_info", {}).get("date_of_birth"):
            print(f"   DOB: {result.extracted_fields['personal_info']['date_of_birth']}")
        if result.extracted_fields.get("identification", {}).get("ssn"):
            print(f"   ID: {result.extracted_fields['identification']['ssn']}")
    
    print(f"\n🎉 Demo completed! Check 'extraction_report.json' for detailed results.")
    print(f"📊 Success Rate: {report['summary']['success_rate']}")
    print(f"📈 Average Confidence: {report['summary']['average_confidence']}")

if __name__ == "__main__":
    main()
