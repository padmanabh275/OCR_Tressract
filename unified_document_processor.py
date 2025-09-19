"""
Unified Document Processing Flow
Intelligent document type detection and processing for both Indian and international documents
"""

import cv2
import numpy as np
import pytesseract
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

from indian_document_enhancer import IndianDocumentEnhancer
from advanced_indian_accuracy import AdvancedIndianAccuracy
from document_extractor import DocumentProcessor, ExtractedData
from enhanced_document_classifier import EnhancedDocumentClassifier
from dynamic_field_extractor import DynamicFieldExtractor

@dataclass
class DocumentClassificationResult:
    """Result of document classification"""
    document_type: str
    document_category: str  # 'indian', 'international', 'unknown'
    confidence_score: float
    processing_method: str
    detected_patterns: List[str]
    region_indicators: List[str]

@dataclass
class ProcessingResult:
    """Result of document processing"""
    success: bool
    document_type: str
    document_category: str
    extracted_data: ExtractedData
    processing_method: str
    confidence_score: float
    processing_time: float
    validation_results: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class DualProcessingResult:
    """Result of dual processing (Indian + Standard)"""
    indian_result: Optional[ProcessingResult]
    standard_result: Optional[ProcessingResult]
    best_result: ProcessingResult
    confidence_comparison: Dict[str, float]
    processing_summary: Dict[str, Any]

class UnifiedDocumentProcessor:
    """Unified processor for both Indian and international documents"""
    
    def __init__(self):
        self.indian_enhancer = IndianDocumentEnhancer()
        self.advanced_accuracy = AdvancedIndianAccuracy()
        self.standard_processor = DocumentProcessor()
        self.classifier = EnhancedDocumentClassifier()
        self.dynamic_extractor = DynamicFieldExtractor()
        
        # Setup document type patterns
        self.setup_document_patterns()
        
    def setup_document_patterns(self):
        """Setup patterns for different document types and regions"""
        
        self.document_patterns = {
            # Indian Documents
            'indian': {
                'pan_card': {
                    'keywords': ['permanent account number', 'pan', 'income tax', 'govt of india'],
                    'patterns': [r'[A-Z]{5}[0-9]{4}[A-Z]{1}', r'P\.A\.N\.?\s*:?', r'INCOME\s+TAX'],
                    'weight': 1.0
                },
                'aadhaar_card': {
                    'keywords': ['aadhaar', 'uid', 'unique identification', 'government of india'],
                    'patterns': [r'\d{4}\s?\d{4}\s?\d{4}', r'AADHAAR', r'UNIQUE\s+IDENTIFICATION'],
                    'weight': 1.0
                },
                'driving_license': {
                    'keywords': ['driving licence', 'transport authority', 'rto'],
                    'patterns': [r'DRIVING\s+LICEN[CS]E', r'TRANSPORT\s+AUTHORITY', r'RTO'],
                    'weight': 0.9
                },
                'voter_id': {
                    'keywords': ['electoral photo identity card', 'epic', 'election commission'],
                    'patterns': [r'[A-Z]{3}[0-9]{7}', r'ELECTORAL\s+PHOTO', r'ELECTION\s+COMMISSION'],
                    'weight': 0.9
                },
                'passport': {
                    'keywords': ['passport', 'ministry of external affairs', 'government of india'],
                    'patterns': [r'[A-Z]{1}[0-9]{7}', r'PASSPORT', r'MINISTRY\s+OF\s+EXTERNAL'],
                    'weight': 0.9
                }
            },
            
            # International Documents
            'international': {
                'us_drivers_license': {
                    'keywords': ['drivers license', 'department of motor vehicles', 'dmv'],
                    'patterns': [r'DRIVER\'?S?\s+LICEN[CS]E', r'DEPARTMENT\s+OF\s+MOTOR', r'DMV'],
                    'weight': 0.9
                },
                'us_passport': {
                    'keywords': ['passport', 'department of state', 'united states'],
                    'patterns': [r'PASSPORT', r'DEPARTMENT\s+OF\s+STATE', r'UNITED\s+STATES'],
                    'weight': 0.9
                },
                'us_ssn_card': {
                    'keywords': ['social security', 'social security number', 'ssn'],
                    'patterns': [r'\d{3}-\d{2}-\d{4}', r'SOCIAL\s+SECURITY', r'SSN'],
                    'weight': 0.9
                },
                'birth_certificate': {
                    'keywords': ['birth certificate', 'certificate of birth', 'vital records'],
                    'patterns': [r'BIRTH\s+CERTIFICATE', r'CERTIFICATE\s+OF\s+BIRTH', r'VITAL\s+RECORDS'],
                    'weight': 0.8
                },
                'marriage_certificate': {
                    'keywords': ['marriage certificate', 'certificate of marriage', 'marriage license'],
                    'patterns': [r'MARRIAGE\s+CERTIFICATE', r'CERTIFICATE\s+OF\s+MARRIAGE', r'MARRIAGE\s+LICENSE'],
                    'weight': 0.8
                },
                'bank_statement': {
                    'keywords': ['bank statement', 'account statement', 'financial statement'],
                    'patterns': [r'BANK\s+STATEMENT', r'ACCOUNT\s+STATEMENT', r'FINANCIAL\s+STATEMENT'],
                    'weight': 0.8
                },
                'w2_form': {
                    'keywords': ['w-2', 'wage and tax statement', 'internal revenue service'],
                    'patterns': [r'W-?2', r'WAGE\s+AND\s+TAX\s+STATEMENT', r'INTERNAL\s+REVENUE\s+SERVICE'],
                    'weight': 0.9
                },
                'tax_return': {
                    'keywords': ['tax return', 'form 1040', 'internal revenue service'],
                    'patterns': [r'TAX\s+RETURN', r'FORM\s+1040', r'INTERNAL\s+REVENUE\s+SERVICE'],
                    'weight': 0.8
                }
            }
        }
        
        # Regional indicators
        self.regional_indicators = {
            'indian': [
                'govt of india', 'government of india', 'income tax department',
                'election commission', 'transport authority', 'rto',
                'ministry of external affairs', 'unique identification authority',
                'permanent account number', 'aadhaar', 'uid'
            ],
            'us': [
                'united states', 'department of state', 'department of motor vehicles',
                'social security administration', 'internal revenue service',
                'dmv', 'ssn', 'w-2', 'form 1040'
            ],
            'uk': [
                'united kingdom', 'hm passport office', 'dvla',
                'national insurance', 'hm revenue and customs'
            ],
            'canada': [
                'canada', 'service canada', 'revenue canada',
                'ministry of transport', 'citizenship and immigration'
            ]
        }
    
    def classify_document(self, text: str, image: np.ndarray = None) -> DocumentClassificationResult:
        """Classify document type and region"""
        
        if not text:
            return DocumentClassificationResult(
                document_type='unknown',
                document_category='unknown',
                confidence_score=0.0,
                processing_method='standard',
                detected_patterns=[],
                region_indicators=[]
            )
        
        text_upper = text.upper()
        scores = {}
        detected_patterns = []
        region_indicators = []
        
        # Check each region and document type
        for region, documents in self.document_patterns.items():
            for doc_type, config in documents.items():
                score = 0
                doc_patterns = []
                doc_indicators = []
                
                # Keyword matching
                for keyword in config['keywords']:
                    if keyword.upper() in text_upper:
                        score += 3.0
                        doc_indicators.append(keyword)
                
                # Pattern matching
                for pattern in config['patterns']:
                    if re.search(pattern, text_upper):
                        score += 2.0
                        doc_patterns.append(pattern)
                
                # Regional indicator matching
                if region in self.regional_indicators:
                    for indicator in self.regional_indicators[region]:
                        if indicator.upper() in text_upper:
                            score += 1.5
                            doc_indicators.append(indicator)
                
                # Apply weight
                final_score = score * config['weight']
                
                if final_score > 0:
                    scores[f"{region}_{doc_type}"] = {
                        'score': final_score,
                        'region': region,
                        'doc_type': doc_type,
                        'patterns': doc_patterns,
                        'indicators': doc_indicators
                    }
        
        # Find best match
        if scores:
            best_match = max(scores.items(), key=lambda x: x[1]['score'])
            best_key, best_data = best_match
            
            # Determine processing method
            if best_data['region'] == 'indian':
                processing_method = 'indian_enhanced'
            else:
                processing_method = 'standard_enhanced'
            
            return DocumentClassificationResult(
                document_type=best_data['doc_type'],
                document_category=best_data['region'],
                confidence_score=min(best_data['score'] / 10.0, 1.0),  # Normalize to 0-1
                processing_method=processing_method,
                detected_patterns=best_data['patterns'],
                region_indicators=best_data['indicators']
            )
        
        # Fallback to standard classification
        try:
            standard_result = self.classifier.classify_document_type(text, image)
            return DocumentClassificationResult(
                document_type=standard_result.document_type,
                document_category='international',
                confidence_score=standard_result.confidence,
                processing_method='standard',
                detected_patterns=[],
                region_indicators=[]
            )
        except:
            return DocumentClassificationResult(
                document_type='unknown',
                document_category='unknown',
                confidence_score=0.0,
                processing_method='standard',
                detected_patterns=[],
                region_indicators=[]
            )
    
    def process_document(self, file_path: str) -> ProcessingResult:
        """Process document with unified flow"""
        
        start_time = datetime.now()
        
        try:
            # Load image
            image = self.load_image(file_path)
            if image is None:
                return self.create_error_result("Could not load image", start_time)
            
            # Extract text for classification
            text = self.extract_text_basic(image)
            
            # Classify document
            classification = self.classify_document(text, image)
            
            print(f"🔍 Document classified as: {classification.document_type}")
            print(f"🌍 Category: {classification.document_category}")
            print(f"⚙️ Processing method: {classification.processing_method}")
            print(f"📊 Confidence: {classification.confidence_score:.2f}")
            
            # Process based on classification
            if classification.processing_method == 'indian_enhanced':
                result = self.process_indian_document(image, text, classification)
            elif classification.processing_method == 'standard_enhanced':
                result = self.process_international_document(image, text, classification)
            else:
                result = self.process_standard_document(file_path, classification)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create processing result
            return ProcessingResult(
                success=True,
                document_type=classification.document_type,
                document_category=classification.document_category,
                extracted_data=result,
                processing_method=classification.processing_method,
                confidence_score=classification.confidence_score,
                processing_time=processing_time,
                validation_results=self.validate_extracted_data(result, classification),
                metadata={
                    'detected_patterns': classification.detected_patterns,
                    'region_indicators': classification.region_indicators,
                    'text_length': len(text),
                    'image_shape': image.shape if image is not None else None
                }
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return self.create_error_result(f"Processing error: {str(e)}", start_time)
    
    def process_indian_document(self, image: np.ndarray, text: str, classification: DocumentClassificationResult) -> ExtractedData:
        """Process Indian documents with enhanced accuracy"""
        
        print(f"🇮🇳 Processing Indian document: {classification.document_type}")
        
        # Use Indian document enhancer
        indian_result = self.indian_enhancer.enhance_indian_document(image, classification.document_type)
        
        # Convert to ExtractedData format
        extracted = ExtractedData()
        extracted.raw_text = text
        extracted.confidence_score = indian_result.confidence_score
        extracted.extraction_method = "indian_enhanced"
        extracted.document_type = indian_result.document_type
        
        # Map Indian fields to common fields
        indian_fields = indian_result.extracted_fields
        extracted.first_name = indian_fields.get('name', '').split()[0] if indian_fields.get('name') else None
        extracted.last_name = ' '.join(indian_fields.get('name', '').split()[1:]) if indian_fields.get('name') and len(indian_fields.get('name', '').split()) > 1 else None
        extracted.date_of_birth = indian_fields.get('date_of_birth')
        extracted.current_address = indian_fields.get('address')
        extracted.ssn = (indian_fields.get('pan') or 
                       indian_fields.get('aadhaar') or 
                       indian_fields.get('license_no') or 
                       indian_fields.get('epic_no') or 
                       indian_fields.get('passport_no'))
        
        # Store all Indian fields in financial_data
        extracted.financial_data = {
            'indian_fields': indian_fields,
            'validation_results': indian_result.validation_results,
            'enhanced_features': indian_result.enhanced_features,
            'document_type': indian_result.document_type,
            'region': 'indian'
        }
        
        return extracted
    
    def process_international_document(self, image: np.ndarray, text: str, classification: DocumentClassificationResult) -> ExtractedData:
        """Process international documents with enhanced accuracy"""
        
        print(f"🌍 Processing international document: {classification.document_type}")
        
        # Use advanced accuracy for international documents
        if classification.document_type in ['us_drivers_license', 'us_passport', 'us_ssn_card']:
            advanced_result = self.advanced_accuracy.process_indian_document_advanced(image, classification.document_type)
            
            # Convert to ExtractedData format
            extracted = ExtractedData()
            extracted.raw_text = text
            extracted.confidence_score = advanced_result.get('confidence_score', 0.0)
            extracted.extraction_method = "international_enhanced"
            extracted.document_type = classification.document_type
            
            # Map fields
            fields = advanced_result.get('extracted_fields', {})
            extracted.first_name = fields.get('name', '').split()[0] if fields.get('name') else None
            extracted.last_name = ' '.join(fields.get('name', '').split()[1:]) if fields.get('name') and len(fields.get('name', '').split()) > 1 else None
            extracted.date_of_birth = fields.get('date_of_birth')
            extracted.current_address = fields.get('address')
            extracted.ssn = fields.get('ssn') or fields.get('license_no') or fields.get('passport_no')
            
            # Store international fields
            extracted.financial_data = {
                'international_fields': fields,
                'document_type': classification.document_type,
                'region': classification.document_category,
                'enhanced_features': advanced_result.get('enhanced_features', {})
            }
            
            return extracted
        else:
            # Use standard processing for other international documents
            return self.process_standard_document_with_image(image, text, classification)
    
    def process_standard_document(self, file_path: str, classification: DocumentClassificationResult) -> ExtractedData:
        """Process document with standard method"""
        
        print(f"📄 Processing with standard method: {classification.document_type}")
        
        result = self.standard_processor.extract_information(file_path)
        if result is None:
            result = ExtractedData()
            result.document_type = classification.document_type
            result.confidence_score = 0.0
            result.raw_text = "No text extracted"
        
        result.extraction_method = "standard"
        return result
    
    def process_standard_document_with_image(self, image: np.ndarray, text: str, classification: DocumentClassificationResult) -> ExtractedData:
        """Process document with standard method using image"""
        
        print(f"📄 Processing with standard method (image): {classification.document_type}")
        
        # Use dynamic field extractor
        dynamic_fields = self.dynamic_extractor.extract_dynamic_fields(text, image)
        
        # Convert to ExtractedData format
        extracted = ExtractedData()
        extracted.raw_text = text
        extracted.confidence_score = dynamic_fields.overall_confidence
        extracted.extraction_method = "standard_enhanced"
        extracted.document_type = classification.document_type
        
        # Map common fields
        common_fields = dynamic_fields.common_fields
        extracted.first_name = common_fields.get('first_name') or common_fields.get('name')
        extracted.last_name = common_fields.get('last_name')
        extracted.date_of_birth = common_fields.get('date_of_birth')
        extracted.current_address = common_fields.get('address')
        extracted.ssn = common_fields.get('ssn')
        
        # Store all fields
        extracted.financial_data = {
            'dynamic_fields': {field.field_name: field.field_value for field in dynamic_fields.all_fields},
            'field_types': {field.field_name: field.field_type for field in dynamic_fields.all_fields},
            'field_confidences': {field.field_name: field.confidence for field in dynamic_fields.all_fields},
            'document_type': classification.document_type,
            'region': classification.document_category
        }
        
        return extracted
    
    def load_image(self, file_path: str) -> Optional[np.ndarray]:
        """Load image from file path"""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.pdf':
                import fitz
                doc = fitz.open(str(file_path))
                page = doc[0]
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                doc.close()
                return image
            else:
                image = cv2.imread(str(file_path))
                return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def extract_text_basic(self, image: np.ndarray) -> str:
        """Extract text using basic OCR"""
        try:
            text = pytesseract.image_to_string(image, config='--psm 6')
            return text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def validate_extracted_data(self, data: ExtractedData, classification: DocumentClassificationResult) -> Dict[str, Any]:
        """Validate extracted data based on document type"""
        
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'field_scores': {},
            'document_type_score': classification.confidence_score,
            'region_validation': True
        }
        
        # Validate based on document type
        if classification.document_category == 'indian':
            validation.update(self.validate_indian_document(data, classification))
        elif classification.document_category == 'international':
            validation.update(self.validate_international_document(data, classification))
        else:
            validation.update(self.validate_standard_document(data, classification))
        
        return validation
    
    def validate_indian_document(self, data: ExtractedData, classification: DocumentClassificationResult) -> Dict[str, Any]:
        """Validate Indian document data"""
        validation = {'warnings': [], 'errors': [], 'field_scores': {}}
        
        if classification.document_type == 'pan_card':
            if data.ssn and not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', data.ssn.upper()):
                validation['warnings'].append("PAN format may be incorrect")
                validation['field_scores']['ssn'] = 0.5
            else:
                validation['field_scores']['ssn'] = 1.0
        
        elif classification.document_type == 'aadhaar_card':
            if data.ssn and not re.match(r'^\d{12}$', re.sub(r'\s', '', data.ssn)):
                validation['warnings'].append("Aadhaar format may be incorrect")
                validation['field_scores']['ssn'] = 0.5
            else:
                validation['field_scores']['ssn'] = 1.0
        
        return validation
    
    def validate_international_document(self, data: ExtractedData, classification: DocumentClassificationResult) -> Dict[str, Any]:
        """Validate international document data"""
        validation = {'warnings': [], 'errors': [], 'field_scores': {}}
        
        if classification.document_type == 'us_ssn_card':
            if data.ssn and not re.match(r'^\d{3}-\d{2}-\d{4}$', data.ssn):
                validation['warnings'].append("SSN format may be incorrect")
                validation['field_scores']['ssn'] = 0.5
            else:
                validation['field_scores']['ssn'] = 1.0
        
        return validation
    
    def validate_standard_document(self, data: ExtractedData, classification: DocumentClassificationResult) -> Dict[str, Any]:
        """Validate standard document data"""
        validation = {'warnings': [], 'errors': [], 'field_scores': {}}
        
        # Basic validation
        if data.first_name and len(data.first_name) < 2:
            validation['warnings'].append("First name seems too short")
            validation['field_scores']['first_name'] = 0.5
        else:
            validation['field_scores']['first_name'] = 1.0
        
        return validation
    
    def create_error_result(self, error_message: str, start_time: datetime) -> ProcessingResult:
        """Create error result"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        error_data = ExtractedData()
        error_data.document_type = 'unknown'
        error_data.confidence_score = 0.0
        error_data.raw_text = error_message
        error_data.extraction_method = 'error'
        
        return ProcessingResult(
            success=False,
            document_type='unknown',
            document_category='unknown',
            extracted_data=error_data,
            processing_method='error',
            confidence_score=0.0,
            processing_time=processing_time,
            validation_results={'is_valid': False, 'errors': [error_message]},
            metadata={'error': error_message}
        )
    
    def process_document_dual(self, file_path: str) -> DualProcessingResult:
        """Process document with both Indian and standard methods, return the best result"""
        import time
        from datetime import datetime
        
        print(f"🔄 Starting dual processing for: {Path(file_path).name}")
        
        # Initialize results
        indian_result = None
        standard_result = None
        processing_summary = {
            'total_processing_time': 0.0,
            'indian_processing_time': 0.0,
            'standard_processing_time': 0.0,
            'confidence_difference': 0.0,
            'best_method': 'unknown'
        }
        
        start_time = time.time()
        
        try:
            # Load image
            file_path = Path(file_path)
            if file_path.suffix.lower() == '.pdf':
                import fitz
                doc = fitz.open(str(file_path))
                page = doc[0]
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                doc.close()
            else:
                image = cv2.imread(str(file_path))
            
            if image is None:
                raise ValueError(f"Could not load image: {file_path}")
            
            # Step 1: Quick classification to determine if it's likely Indian
            print("🔍 Step 1: Quick document classification...")
            text = pytesseract.image_to_string(image, config='--psm 6')
            is_likely_indian = self._is_likely_indian_document(text)
            print(f"   Indian document indicators: {is_likely_indian}")
            
            # Step 2: Process with Indian method (if likely Indian or always try)
            print("🇮🇳 Step 2: Processing with Indian enhanced method...")
            indian_start = time.time()
            try:
                indian_result = self._process_with_indian_method(image, text)
                indian_processing_time = time.time() - indian_start
                processing_summary['indian_processing_time'] = indian_processing_time
                print(f"   ✅ Indian processing completed: {indian_result.confidence_score:.3f} confidence")
            except Exception as e:
                print(f"   ❌ Indian processing failed: {e}")
                indian_result = None
            
            # Step 3: Process with standard method
            print("🌍 Step 3: Processing with standard method...")
            standard_start = time.time()
            try:
                standard_result = self._process_with_standard_method(image, text)
                standard_processing_time = time.time() - standard_start
                processing_summary['standard_processing_time'] = standard_processing_time
                print(f"   ✅ Standard processing completed: {standard_result.confidence_score:.3f} confidence")
            except Exception as e:
                print(f"   ❌ Standard processing failed: {e}")
                standard_result = None
            
            # Step 4: Compare results and select the best one
            print("📊 Step 4: Comparing results and selecting best method...")
            best_result, confidence_comparison = self._compare_and_select_best(
                indian_result, standard_result
            )
            
            processing_summary['confidence_difference'] = abs(
                confidence_comparison.get('indian', 0) - confidence_comparison.get('standard', 0)
            )
            processing_summary['best_method'] = best_result.processing_method
            processing_summary['total_processing_time'] = time.time() - start_time
            
            print(f"   🏆 Best method: {best_result.processing_method}")
            print(f"   📈 Confidence comparison: {confidence_comparison}")
            print(f"   ⏱️ Total processing time: {processing_summary['total_processing_time']:.2f}s")
            
            return DualProcessingResult(
                indian_result=indian_result,
                standard_result=standard_result,
                best_result=best_result,
                confidence_comparison=confidence_comparison,
                processing_summary=processing_summary
            )
            
        except Exception as e:
            error_message = f"Dual processing error: {str(e)}"
            print(f"❌ {error_message}")
            
            # Return error result
            error_result = ProcessingResult(
                success=False,
                document_type='unknown',
                document_category='unknown',
                extracted_data=ExtractedData(),
                processing_method='error',
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                validation_results={'valid': False, 'errors': [error_message]},
                metadata={'error': error_message}
            )
            
            return DualProcessingResult(
                indian_result=None,
                standard_result=None,
                best_result=error_result,
                confidence_comparison={'indian': 0.0, 'standard': 0.0},
                processing_summary={'error': error_message}
            )
    
    def _is_likely_indian_document(self, text: str) -> bool:
        """Quick check if document is likely Indian"""
        if not text:
            return False
        
        text_upper = text.upper()
        
        # Indian document indicators
        indian_indicators = [
            'GOVT OF INDIA', 'INCOME TAX', 'AADHAAR', 'UNIQUE IDENTIFICATION',
            'DRIVING LICENCE', 'TRANSPORT AUTHORITY', 'RTO', 'ELECTION COMMISSION',
            'ELECTORAL PHOTO', 'PASSPORT', 'MINISTRY OF EXTERNAL AFFAIRS',
            'PERMANENT ACCOUNT NUMBER', 'PAN', 'UID', 'EPIC'
        ]
        
        # Indian patterns
        indian_patterns = [
            r'[A-Z]{5}[0-9]{4}[A-Z]{1}',  # PAN format
            r'\d{4}\s?\d{4}\s?\d{4}',     # Aadhaar format
            r'[A-Z]{3}[0-9]{7}',          # EPIC format
            r'[A-Z]{1}[0-9]{7}'           # Passport format
        ]
        
        # Check indicators
        indicator_count = sum(1 for indicator in indian_indicators if indicator in text_upper)
        
        # Check patterns
        pattern_count = sum(1 for pattern in indian_patterns if re.search(pattern, text_upper))
        
        # Consider it Indian if we have indicators or patterns
        is_indian = indicator_count >= 2 or pattern_count >= 1
        
        print(f"   📋 Indian indicators found: {indicator_count}")
        print(f"   🔍 Indian patterns found: {pattern_count}")
        print(f"   🇮🇳 Likely Indian document: {is_indian}")
        
        return is_indian
    
    def _process_with_indian_method(self, image: np.ndarray, text: str) -> ProcessingResult:
        """Process document using Indian enhanced method"""
        import time
        from datetime import datetime
        
        start_time = time.time()
        
        try:
            # Use Indian document enhancer
            indian_result = self.indian_enhancer.enhance_indian_document(image, None)
            
            # Convert to ExtractedData format
            extracted = ExtractedData()
            extracted.raw_text = text
            extracted.confidence_score = indian_result.confidence_score
            extracted.extraction_method = "indian_enhanced"
            extracted.document_type = indian_result.document_type
            
            # Map Indian fields to common fields
            indian_fields = indian_result.extracted_fields
            extracted.first_name = indian_fields.get('name', '').split()[0] if indian_fields.get('name') else None
            extracted.last_name = ' '.join(indian_fields.get('name', '').split()[1:]) if indian_fields.get('name') and len(indian_fields.get('name', '').split()) > 1 else None
            extracted.date_of_birth = indian_fields.get('date_of_birth')
            extracted.current_address = indian_fields.get('address')
            extracted.ssn = (indian_fields.get('pan') or 
                           indian_fields.get('aadhaar') or 
                           indian_fields.get('license_no') or 
                           indian_fields.get('epic_no') or 
                           indian_fields.get('passport_no'))
            
            # Store all Indian fields in financial_data
            extracted.financial_data = {
                'indian_fields': indian_fields,
                'validation_results': indian_result.validation_results,
                'enhanced_features': indian_result.enhanced_features,
                'document_type': indian_result.document_type
            }
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                document_type=indian_result.document_type,
                document_category='indian',
                extracted_data=extracted,
                processing_method='indian_enhanced',
                confidence_score=indian_result.confidence_score,
                processing_time=processing_time,
                validation_results=indian_result.validation_results,
                metadata={
                    'indian_fields': indian_fields,
                    'enhanced_features': indian_result.enhanced_features,
                    'processing_method': 'indian_enhanced'
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                success=False,
                document_type='unknown',
                document_category='indian',
                extracted_data=ExtractedData(),
                processing_method='indian_enhanced',
                confidence_score=0.0,
                processing_time=processing_time,
                validation_results={'valid': False, 'errors': [str(e)]},
                metadata={'error': str(e)}
            )
    
    def _process_with_standard_method(self, image: np.ndarray, text: str) -> ProcessingResult:
        """Process document using standard method"""
        import time
        from datetime import datetime
        
        start_time = time.time()
        
        try:
            # Use standard document processor - we need to save image temporarily
            import tempfile
            import os
            
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, image)
                temp_path = tmp_file.name
            
            try:
                result = self.standard_processor.extract_information(temp_path)
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
            if result is None:
                result = ExtractedData()
            
            # Enhanced document type classification
            classification_result = self.classifier.classify_document_type(text, image)
            
            # Dynamic field extraction
            dynamic_fields = self.dynamic_extractor.extract_dynamic_fields(text, image)
            
            # Convert to ExtractedData format
            extracted = ExtractedData()
            extracted.raw_text = text
            extracted.confidence_score = result.confidence_score or 0.0
            extracted.extraction_method = "standard_enhanced"
            extracted.document_type = classification_result.document_type
            
            # Extract common fields from dynamic extraction
            common_fields = dynamic_fields.common_fields
            custom_fields = dynamic_fields.custom_fields
            
            # Map common fields to ExtractedData attributes
            extracted.first_name = common_fields.get('first_name') or common_fields.get('name')
            extracted.last_name = common_fields.get('last_name')
            extracted.date_of_birth = common_fields.get('date_of_birth')
            extracted.marriage_date = common_fields.get('marriage_date')
            extracted.ssn = common_fields.get('ssn')
            extracted.current_address = common_fields.get('address')
            
            # Store custom fields in financial_data
            extracted.financial_data = {
                'custom_fields': custom_fields,
                'all_dynamic_fields': {field.field_name: field.field_value for field in dynamic_fields.all_fields},
                'field_types': {field.field_name: field.field_type for field in dynamic_fields.all_fields},
                'field_confidences': {field.field_name: field.confidence for field in dynamic_fields.all_fields}
            }
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                document_type=classification_result.document_type,
                document_category='international',
                extracted_data=extracted,
                processing_method='standard_enhanced',
                confidence_score=extracted.confidence_score,
                processing_time=processing_time,
                validation_results={'valid': True, 'warnings': []},
                metadata={
                    'classification_confidence': classification_result.confidence,
                    'dynamic_fields_count': len(dynamic_fields.all_fields),
                    'processing_method': 'standard_enhanced'
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                success=False,
                document_type='unknown',
                document_category='international',
                extracted_data=ExtractedData(),
                processing_method='standard_enhanced',
                confidence_score=0.0,
                processing_time=processing_time,
                validation_results={'valid': False, 'errors': [str(e)]},
                metadata={'error': str(e)}
            )
    
    def _compare_and_select_best(self, indian_result: Optional[ProcessingResult], 
                                standard_result: Optional[ProcessingResult]) -> Tuple[ProcessingResult, Dict[str, float]]:
        """Compare results and select the best one based on confidence"""
        
        confidence_comparison = {
            'indian': indian_result.confidence_score if indian_result else 0.0,
            'standard': standard_result.confidence_score if standard_result else 0.0
        }
        
        # If only one method succeeded, return that one
        if indian_result and not standard_result:
            print(f"   🇮🇳 Only Indian method succeeded: {indian_result.confidence_score:.3f}")
            return indian_result, confidence_comparison
        
        if standard_result and not indian_result:
            print(f"   🌍 Only standard method succeeded: {standard_result.confidence_score:.3f}")
            return standard_result, confidence_comparison
        
        # If both failed, return the one with higher confidence (even if 0)
        if not indian_result and not standard_result:
            print("   ❌ Both methods failed")
            return ProcessingResult(
                success=False,
                document_type='unknown',
                document_category='unknown',
                extracted_data=ExtractedData(),
                processing_method='error',
                confidence_score=0.0,
                processing_time=0.0,
                validation_results={'valid': False, 'errors': ['Both processing methods failed']},
                metadata={'error': 'Both processing methods failed'}
            ), confidence_comparison
        
        # Compare confidence scores
        if indian_result.confidence_score >= standard_result.confidence_score:
            print(f"   🏆 Indian method selected: {indian_result.confidence_score:.3f} vs {standard_result.confidence_score:.3f}")
            return indian_result, confidence_comparison
        else:
            print(f"   🏆 Standard method selected: {standard_result.confidence_score:.3f} vs {indian_result.confidence_score:.3f}")
            return standard_result, confidence_comparison

# Usage example
def test_unified_processor():
    """Test the unified document processor"""
    processor = UnifiedDocumentProcessor()
    
    # Test with sample text
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
    
    classification = processor.classify_document(sample_text)
    print(f"Classification: {classification.document_type}")
    print(f"Category: {classification.document_category}")
    print(f"Confidence: {classification.confidence_score:.2f}")
    print(f"Method: {classification.processing_method}")

if __name__ == "__main__":
    test_unified_processor()
