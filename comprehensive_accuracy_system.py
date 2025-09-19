"""
Comprehensive Accuracy System
Integrates all accuracy improvement techniques for maximum performance
"""

import cv2
import numpy as np
import pytesseract
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path
import time
from datetime import datetime

from advanced_accuracy_enhancements import AdvancedAccuracyEnhancements, MultiModelEnsemble, AdaptivePreprocessing
from ml_accuracy_boosters import TextCorrectionML, DocumentQualityAssessment, ConfidenceBoosting

@dataclass
class ComprehensiveAccuracyResult:
    """Result of comprehensive accuracy enhancement"""
    enhanced_image: np.ndarray
    corrected_text: str
    extracted_fields: Dict[str, str]
    confidence_score: float
    accuracy_boost: float
    techniques_applied: List[str]
    processing_time: float
    quality_assessment: Dict[str, Any]
    ml_enhancements: Dict[str, Any]

class ComprehensiveAccuracySystem:
    """Comprehensive accuracy enhancement system"""
    
    def __init__(self):
        self.setup_accuracy_system()
    
    def setup_accuracy_system(self):
        """Setup the comprehensive accuracy system"""
        self.enhancement_system = AdvancedAccuracyEnhancements()
        self.ensemble_ocr = MultiModelEnsemble()
        self.adaptive_preprocessing = AdaptivePreprocessing()
        self.text_corrector = TextCorrectionML()
        self.quality_assessor = DocumentQualityAssessment()
        self.confidence_booster = ConfidenceBoosting()
        
        # Setup accuracy configurations
        self.accuracy_configs = {
            'maximum_accuracy': {
                'use_ensemble_ocr': True,
                'use_ml_correction': True,
                'use_confidence_boosting': True,
                'use_adaptive_preprocessing': True,
                'use_quality_assessment': True,
                'max_processing_time': 10.0
            },
            'balanced_accuracy': {
                'use_ensemble_ocr': True,
                'use_ml_correction': True,
                'use_confidence_boosting': True,
                'use_adaptive_preprocessing': False,
                'use_quality_assessment': True,
                'max_processing_time': 5.0
            },
            'fast_accuracy': {
                'use_ensemble_ocr': False,
                'use_ml_correction': True,
                'use_confidence_boosting': False,
                'use_adaptive_preprocessing': False,
                'use_quality_assessment': False,
                'max_processing_time': 2.0
            }
        }
    
    def enhance_document_accuracy(self, image: np.ndarray, document_type: str = 'unknown', 
                                accuracy_mode: str = 'balanced_accuracy') -> ComprehensiveAccuracyResult:
        """Apply comprehensive accuracy enhancements"""
        
        start_time = time.time()
        techniques_applied = []
        accuracy_boost = 0.0
        
        # Get configuration
        config = self.accuracy_configs.get(accuracy_mode, self.accuracy_configs['balanced_accuracy'])
        
        # Step 1: Quality Assessment
        quality_assessment = {}
        if config['use_quality_assessment']:
            quality_assessment = self.quality_assessor.assess_document_quality(image)
            techniques_applied.append('quality_assessment')
        
        # Step 2: Adaptive Preprocessing
        enhanced_image = image.copy()
        if config['use_adaptive_preprocessing']:
            enhanced_image = self.adaptive_preprocessing.apply_adaptive_preprocessing(image)
            techniques_applied.append('adaptive_preprocessing')
            accuracy_boost += 0.05
        
        # Step 3: Advanced Enhancement
        enhancement_result = self.enhancement_system.enhance_document_accuracy(enhanced_image, document_type)
        enhanced_image = enhancement_result.enhanced_image
        accuracy_boost += enhancement_result.confidence_boost
        techniques_applied.extend(enhancement_result.enhancement_applied)
        
        # Step 4: Ensemble OCR
        if config['use_ensemble_ocr']:
            ensemble_result = self.ensemble_ocr.ensemble_extract(enhanced_image)
            extracted_text = ensemble_result['best_text']
            techniques_applied.append('ensemble_ocr')
            accuracy_boost += 0.08
        else:
            # Single OCR
            extracted_text = pytesseract.image_to_string(enhanced_image, config='--psm 6')
        
        # Step 5: ML Text Correction
        corrected_text = extracted_text
        ml_enhancements = {}
        if config['use_ml_correction']:
            correction_result = self.text_corrector.correct_text_ml(extracted_text, document_type)
            corrected_text = correction_result.enhanced_text
            accuracy_boost += correction_result.accuracy_boost
            techniques_applied.extend(correction_result.ml_techniques_applied)
            ml_enhancements['text_correction'] = {
                'original_text': extracted_text,
                'corrected_text': corrected_text,
                'techniques_applied': correction_result.ml_techniques_applied,
                'accuracy_boost': correction_result.accuracy_boost
            }
        
        # Step 6: Field Extraction
        extracted_fields = self.extract_fields_comprehensive(corrected_text, document_type)
        
        # Step 7: Confidence Boosting
        confidence_score = 0.0
        if config['use_confidence_boosting']:
            boost_result = self.confidence_booster.boost_confidence(corrected_text, document_type, extracted_fields)
            confidence_score = boost_result['boosted_confidence']
            accuracy_boost += boost_result['total_boost']
            techniques_applied.append('confidence_boosting')
            ml_enhancements['confidence_boosting'] = boost_result
        else:
            confidence_score = self.calculate_basic_confidence(corrected_text, document_type)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Check if processing time is within limits
        if processing_time > config['max_processing_time']:
            print(f"⚠️ Processing time exceeded limit: {processing_time:.2f}s > {config['max_processing_time']}s")
        
        return ComprehensiveAccuracyResult(
            enhanced_image=enhanced_image,
            corrected_text=corrected_text,
            extracted_fields=extracted_fields,
            confidence_score=min(confidence_score, 1.0),
            accuracy_boost=min(accuracy_boost, 0.5),  # Cap at 50% boost
            techniques_applied=techniques_applied,
            processing_time=processing_time,
            quality_assessment=quality_assessment,
            ml_enhancements=ml_enhancements
        )
    
    def extract_fields_comprehensive(self, text: str, document_type: str) -> Dict[str, str]:
        """Extract fields using comprehensive techniques"""
        fields = {}
        
        if document_type == 'pan_card':
            fields = self.extract_pan_fields(text)
        elif document_type == 'aadhaar_card':
            fields = self.extract_aadhaar_fields(text)
        elif document_type == 'driving_license':
            fields = self.extract_driving_license_fields(text)
        elif document_type == 'voter_id':
            fields = self.extract_voter_id_fields(text)
        elif document_type == 'passport':
            fields = self.extract_passport_fields(text)
        else:
            fields = self.extract_generic_fields(text)
        
        return fields
    
    def extract_pan_fields(self, text: str) -> Dict[str, str]:
        """Extract PAN card fields"""
        fields = {}
        
        # PAN number
        pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', text)
        if pan_match:
            fields['pan'] = pan_match.group()
        
        # Name
        name_match = re.search(r'NAME\s*:?\s*([A-Z\s]+?)(?:\n|FATHER|MOTHER|DATE|SIGNATURE)', text, re.IGNORECASE)
        if name_match:
            fields['name'] = name_match.group(1).strip()
        
        # Father's name
        father_match = re.search(r'FATHER\'?S?\s+NAME\s*:?\s*([A-Z\s]+?)(?:\n|DATE|SIGNATURE)', text, re.IGNORECASE)
        if father_match:
            fields['father_name'] = father_match.group(1).strip()
        
        # Date of birth
        dob_match = re.search(r'DATE\s+OF\s+BIRTH\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})', text, re.IGNORECASE)
        if dob_match:
            fields['date_of_birth'] = dob_match.group(1).strip()
        
        # Signature
        signature_match = re.search(r'SIGNATURE\s*:?\s*([A-Z\s]+?)(?:\n|$)', text, re.IGNORECASE)
        if signature_match:
            fields['signature'] = signature_match.group(1).strip()
        
        return fields
    
    def extract_aadhaar_fields(self, text: str) -> Dict[str, str]:
        """Extract Aadhaar card fields"""
        fields = {}
        
        # Aadhaar number
        aadhaar_match = re.search(r'(\d{4}\s?\d{4}\s?\d{4})', text)
        if aadhaar_match:
            fields['aadhaar'] = aadhaar_match.group(1).strip()
        
        # Name
        name_match = re.search(r'NAME\s*:?\s*([A-Z\s]+?)(?:\n|FATHER|MOTHER|DATE|GENDER)', text, re.IGNORECASE)
        if name_match:
            fields['name'] = name_match.group(1).strip()
        
        # Father's name
        father_match = re.search(r'FATHER\'?S?\s+NAME\s*:?\s*([A-Z\s]+?)(?:\n|MOTHER|DATE|GENDER)', text, re.IGNORECASE)
        if father_match:
            fields['father_name'] = father_match.group(1).strip()
        
        # Mother's name
        mother_match = re.search(r'MOTHER\'?S?\s+NAME\s*:?\s*([A-Z\s]+?)(?:\n|DATE|GENDER)', text, re.IGNORECASE)
        if mother_match:
            fields['mother_name'] = mother_match.group(1).strip()
        
        # Date of birth
        dob_match = re.search(r'DATE\s+OF\s+BIRTH\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})', text, re.IGNORECASE)
        if dob_match:
            fields['date_of_birth'] = dob_match.group(1).strip()
        
        # Gender
        gender_match = re.search(r'GENDER\s*:?\s*([MF])', text, re.IGNORECASE)
        if gender_match:
            fields['gender'] = gender_match.group(1).strip()
        
        # Address
        address_match = re.search(r'ADDRESS\s*:?\s*([A-Za-z0-9\s,.-]+?)(?:\n|PIN)', text, re.IGNORECASE)
        if address_match:
            fields['address'] = address_match.group(1).strip()
        
        # PIN code
        pin_match = re.search(r'PIN\s*:?\s*(\d{6})', text, re.IGNORECASE)
        if pin_match:
            fields['pin_code'] = pin_match.group(1).strip()
        
        return fields
    
    def extract_driving_license_fields(self, text: str) -> Dict[str, str]:
        """Extract driving license fields"""
        fields = {}
        
        # License number
        license_match = re.search(r'LICEN[CS]E\s+NO\.?\s*:?\s*([A-Z0-9]+)', text, re.IGNORECASE)
        if license_match:
            fields['license_no'] = license_match.group(1).strip()
        
        # Name
        name_match = re.search(r'NAME\s*:?\s*([A-Z\s]+?)(?:\n|FATHER|MOTHER|DATE|VALID)', text, re.IGNORECASE)
        if name_match:
            fields['name'] = name_match.group(1).strip()
        
        # Father's name
        father_match = re.search(r'FATHER\'?S?\s+NAME\s*:?\s*([A-Z\s]+?)(?:\n|DATE|VALID)', text, re.IGNORECASE)
        if father_match:
            fields['father_name'] = father_match.group(1).strip()
        
        # Date of birth
        dob_match = re.search(r'DATE\s+OF\s+BIRTH\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})', text, re.IGNORECASE)
        if dob_match:
            fields['date_of_birth'] = dob_match.group(1).strip()
        
        # Valid from
        valid_from_match = re.search(r'VALID\s+FROM\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})', text, re.IGNORECASE)
        if valid_from_match:
            fields['valid_from'] = valid_from_match.group(1).strip()
        
        # Valid upto
        valid_upto_match = re.search(r'VALID\s+UPTO\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})', text, re.IGNORECASE)
        if valid_upto_match:
            fields['valid_upto'] = valid_upto_match.group(1).strip()
        
        # Address
        address_match = re.search(r'ADDRESS\s*:?\s*([A-Za-z0-9\s,.-]+?)(?:\n|BLOOD)', text, re.IGNORECASE)
        if address_match:
            fields['address'] = address_match.group(1).strip()
        
        # Blood group
        blood_match = re.search(r'BLOOD\s+GROUP\s*:?\s*([A-Z]+[+-]?)', text, re.IGNORECASE)
        if blood_match:
            fields['blood_group'] = blood_match.group(1).strip()
        
        return fields
    
    def extract_voter_id_fields(self, text: str) -> Dict[str, str]:
        """Extract voter ID fields"""
        fields = {}
        
        # EPIC number
        epic_match = re.search(r'([A-Z]{3}[0-9]{7})', text)
        if epic_match:
            fields['epic_no'] = epic_match.group(1).strip()
        
        # Elector's name
        name_match = re.search(r'ELECTOR\'?S?\s+NAME\s*:?\s*([A-Z\s]+?)(?:\n|FATHER|HUSBAND|DATE)', text, re.IGNORECASE)
        if name_match:
            fields['name'] = name_match.group(1).strip()
        
        # Father's name
        father_match = re.search(r'FATHER\'?S?\s+NAME\s*:?\s*([A-Z\s]+?)(?:\n|HUSBAND|DATE)', text, re.IGNORECASE)
        if father_match:
            fields['father_name'] = father_match.group(1).strip()
        
        # Husband's name
        husband_match = re.search(r'HUSBAND\'?S?\s+NAME\s*:?\s*([A-Z\s]+?)(?:\n|DATE)', text, re.IGNORECASE)
        if husband_match:
            fields['husband_name'] = husband_match.group(1).strip()
        
        # Date of birth
        dob_match = re.search(r'DATE\s+OF\s+BIRTH\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})', text, re.IGNORECASE)
        if dob_match:
            fields['date_of_birth'] = dob_match.group(1).strip()
        
        # Gender
        gender_match = re.search(r'GENDER\s*:?\s*([MF])', text, re.IGNORECASE)
        if gender_match:
            fields['gender'] = gender_match.group(1).strip()
        
        # Age
        age_match = re.search(r'AGE\s*:?\s*(\d{2})', text, re.IGNORECASE)
        if age_match:
            fields['age'] = age_match.group(1).strip()
        
        # Address
        address_match = re.search(r'ADDRESS\s*:?\s*([A-Za-z0-9\s,.-]+?)(?:\n|CONSTITUENCY)', text, re.IGNORECASE)
        if address_match:
            fields['address'] = address_match.group(1).strip()
        
        # Constituency
        constituency_match = re.search(r'CONSTITUENCY\s*:?\s*([A-Z\s]+?)(?:\n|PART)', text, re.IGNORECASE)
        if constituency_match:
            fields['constituency'] = constituency_match.group(1).strip()
        
        return fields
    
    def extract_passport_fields(self, text: str) -> Dict[str, str]:
        """Extract passport fields"""
        fields = {}
        
        # Passport number
        passport_match = re.search(r'PASSPORT\s+NO\.?\s*:?\s*([A-Z]{1}[0-9]{7})', text, re.IGNORECASE)
        if passport_match:
            fields['passport_no'] = passport_match.group(1).strip()
        
        # File number
        file_match = re.search(r'FILE\s+NO\.?\s*:?\s*([A-Z0-9]+)', text, re.IGNORECASE)
        if file_match:
            fields['file_no'] = file_match.group(1).strip()
        
        # Name
        name_match = re.search(r'NAME\s*:?\s*([A-Z\s]+?)(?:\n|FATHER|MOTHER|DATE)', text, re.IGNORECASE)
        if name_match:
            fields['name'] = name_match.group(1).strip()
        
        # Father's name
        father_match = re.search(r'FATHER\'?S?\s+NAME\s*:?\s*([A-Z\s]+?)(?:\n|MOTHER|DATE)', text, re.IGNORECASE)
        if father_match:
            fields['father_name'] = father_match.group(1).strip()
        
        # Mother's name
        mother_match = re.search(r'MOTHER\'?S?\s+NAME\s*:?\s*([A-Z\s]+?)(?:\n|DATE)', text, re.IGNORECASE)
        if mother_match:
            fields['mother_name'] = mother_match.group(1).strip()
        
        # Date of birth
        dob_match = re.search(r'DATE\s+OF\s+BIRTH\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})', text, re.IGNORECASE)
        if dob_match:
            fields['date_of_birth'] = dob_match.group(1).strip()
        
        # Place of birth
        pob_match = re.search(r'PLACE\s+OF\s+BIRTH\s*:?\s*([A-Z\s]+?)(?:\n|PLACE\s+OF\s+ISSUE)', text, re.IGNORECASE)
        if pob_match:
            fields['place_of_birth'] = pob_match.group(1).strip()
        
        # Place of issue
        poi_match = re.search(r'PLACE\s+OF\s+ISSUE\s*:?\s*([A-Z\s]+?)(?:\n|NATIONALITY)', text, re.IGNORECASE)
        if poi_match:
            fields['place_of_issue'] = poi_match.group(1).strip()
        
        # Nationality
        nationality_match = re.search(r'NATIONALITY\s*:?\s*([A-Z\s]+?)(?:\n|DATE\s+OF\s+ISSUE)', text, re.IGNORECASE)
        if nationality_match:
            fields['nationality'] = nationality_match.group(1).strip()
        
        # Date of issue
        doi_match = re.search(r'DATE\s+OF\s+ISSUE\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})', text, re.IGNORECASE)
        if doi_match:
            fields['date_of_issue'] = doi_match.group(1).strip()
        
        # Date of expire
        doe_match = re.search(r'DATE\s+OF\s+EXPIRE\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})', text, re.IGNORECASE)
        if doe_match:
            fields['date_of_expire'] = doe_match.group(1).strip()
        
        return fields
    
    def extract_generic_fields(self, text: str) -> Dict[str, str]:
        """Extract generic fields from any document"""
        fields = {}
        
        # Name patterns
        name_patterns = [
            r'NAME\s*:?\s*([A-Z\s]+?)(?:\n|FATHER|MOTHER|DATE|SIGNATURE)',
            r'Name\s*:?\s*([A-Z\s]+?)(?:\n|Father|Mother|Date|Signature)',
            r'1\.\s*NAME\s*:?\s*([A-Z\s]+)',
            r'Name\s+of\s+the\s+Cardholder\s*:?\s*([A-Z\s]+)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                fields['name'] = match.group(1).strip()
                break
        
        # Date patterns
        date_patterns = [
            r'DATE\s+OF\s+BIRTH\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
            r'Date\s+of\s+Birth\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
            r'DOB\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
            r'Birth\s+Date\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields['date_of_birth'] = match.group(1).strip()
                break
        
        # Address patterns
        address_patterns = [
            r'ADDRESS\s*:?\s*([A-Za-z0-9\s,.-]+?)(?:\n|PIN|PHONE)',
            r'Address\s*:?\s*([A-Za-z0-9\s,.-]+?)(?:\n|Pin|Phone)',
            r'Residential\s+Address\s*:?\s*([A-Za-z0-9\s,.-]+?)(?:\n|Pin|Phone)'
        ]
        
        for pattern in address_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                fields['address'] = match.group(1).strip()
                break
        
        return fields
    
    def calculate_basic_confidence(self, text: str, document_type: str) -> float:
        """Calculate basic confidence score"""
        if not text:
            return 0.0
        
        confidence = 0.0
        
        # Length factor
        confidence += min(len(text) / 100, 1.0) * 0.3
        
        # Character diversity
        unique_chars = len(set(text.upper()))
        confidence += min(unique_chars / 26, 1.0) * 0.2
        
        # Word count
        word_count = len(text.split())
        confidence += min(word_count / 20, 1.0) * 0.2
        
        # Document-specific patterns
        confidence += self.check_document_patterns(text, document_type) * 0.3
        
        return min(confidence, 1.0)
    
    def check_document_patterns(self, text: str, document_type: str) -> float:
        """Check document-specific patterns"""
        patterns = {
            'pan_card': [r'[A-Z]{5}[0-9]{4}[A-Z]{1}', r'INCOME TAX', r'GOVT OF INDIA'],
            'aadhaar_card': [r'\d{4}\s?\d{4}\s?\d{4}', r'AADHAAR', r'UNIQUE IDENTIFICATION'],
            'driving_license': [r'DRIVING LICENCE', r'TRANSPORT AUTHORITY', r'RTO'],
            'voter_id': [r'ELECTORAL PHOTO', r'ELECTION COMMISSION', r'EPIC'],
            'passport': [r'PASSPORT', r'MINISTRY OF EXTERNAL', r'PASSPORT NO']
        }
        
        doc_patterns = patterns.get(document_type, [])
        matches = sum(1 for pattern in doc_patterns if re.search(pattern, text.upper()))
        
        return matches / len(doc_patterns) if doc_patterns else 0.0

# Usage example
def test_comprehensive_accuracy_system():
    """Test comprehensive accuracy system"""
    system = ComprehensiveAccuracySystem()
    
    print("🧪 Testing Comprehensive Accuracy System...")
    
    # Test with sample image
    sample_image = np.ones((100, 200), dtype=np.uint8) * 255
    cv2.putText(sample_image, "PERMANENT ACCOUNT NUMBER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    cv2.putText(sample_image, "P.A.N. : ABCDE1234F", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    cv2.putText(sample_image, "NAME: RAJESH KUMAR SHARMA", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    
    # Test different accuracy modes
    for mode in ['fast_accuracy', 'balanced_accuracy', 'maximum_accuracy']:
        print(f"\n📊 Testing {mode} mode...")
        result = system.enhance_document_accuracy(sample_image, 'pan_card', mode)
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Accuracy boost: {result.accuracy_boost:.2f}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Techniques applied: {len(result.techniques_applied)}")

if __name__ == "__main__":
    test_comprehensive_accuracy_system()
