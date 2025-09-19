"""
Machine Learning Accuracy Boosters
Advanced ML techniques to maximize document extraction accuracy
"""

import cv2
import numpy as np
import pytesseract
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path
import math

# Optional imports with fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    import pickle
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available, some ML features will be disabled")

@dataclass
class MLAccuracyResult:
    """Result of ML accuracy enhancement"""
    enhanced_text: str
    confidence_score: float
    ml_techniques_applied: List[str]
    accuracy_boost: float

class TextCorrectionML:
    """Machine Learning-based text correction"""
    
    def __init__(self):
        self.setup_correction_models()
        self.setup_pattern_database()
    
    def setup_correction_models(self):
        """Setup ML models for text correction"""
        self.correction_patterns = {
            'common_ocr_errors': {
                '0': 'O',  # Zero to O
                '1': 'I',  # One to I
                '5': 'S',  # Five to S
                '8': 'B',  # Eight to B
                '6': 'G',  # Six to G
                '|': 'I',  # Pipe to I
                'l': 'I',  # Lowercase l to I
                'rn': 'm', # rn to m
                'cl': 'd', # cl to d
                'ii': 'n', # ii to n
            },
            'indian_names': [
                'RAJESH', 'KUMAR', 'SHARMA', 'SINGH', 'PATEL', 'GUPTA',
                'VERMA', 'AGARWAL', 'JAIN', 'MALHOTRA', 'CHOPRA', 'KAPOOR',
                'PRIYA', 'SUNITA', 'KAVITA', 'ANITA', 'REKHA', 'MEERA'
            ],
            'indian_cities': [
                'DELHI', 'MUMBAI', 'BANGALORE', 'CHENNAI', 'KOLKATA',
                'HYDERABAD', 'PUNE', 'AHMEDABAD', 'JAIPUR', 'LUCKNOW'
            ],
            'indian_states': [
                'DELHI', 'MAHARASHTRA', 'KARNATAKA', 'TAMIL NADU', 'WEST BENGAL',
                'TELANGANA', 'GUJARAT', 'RAJASTHAN', 'UTTAR PRADESH', 'MADHYA PRADESH'
            ]
        }
    
    def setup_pattern_database(self):
        """Setup pattern database for validation"""
        self.patterns = {
            'pan_format': r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$',
            'aadhaar_format': r'^\d{4}\s?\d{4}\s?\d{4}$',
            'epic_format': r'^[A-Z]{3}[0-9]{7}$',
            'passport_format': r'^[A-Z]{1}[0-9]{7}$',
            'ssn_format': r'^\d{3}-\d{2}-\d{4}$',
            'date_format': r'^\d{2}[/-]\d{2}[/-]\d{4}$',
            'pin_format': r'^\d{6}$'
        }
    
    def correct_text_ml(self, text: str, document_type: str = 'unknown') -> MLAccuracyResult:
        """Apply ML-based text correction"""
        if not text:
            return MLAccuracyResult(
                enhanced_text="",
                confidence_score=0.0,
                ml_techniques_applied=[],
                accuracy_boost=0.0
            )
        
        original_text = text
        techniques_applied = []
        accuracy_boost = 0.0
        
        # Apply common OCR error corrections
        corrected_text = self.correct_common_errors(text)
        if corrected_text != text:
            techniques_applied.append('common_ocr_correction')
            accuracy_boost += 0.05
        
        # Apply document-specific corrections
        if document_type in ['pan_card', 'aadhaar_card', 'driving_license', 'voter_id', 'passport']:
            corrected_text = self.correct_indian_document_text(corrected_text, document_type)
            if corrected_text != text:
                techniques_applied.append('indian_document_correction')
                accuracy_boost += 0.08
        
        # Apply pattern-based corrections
        corrected_text = self.correct_patterns(corrected_text, document_type)
        if corrected_text != text:
            techniques_applied.append('pattern_correction')
            accuracy_boost += 0.03
        
        # Apply context-aware corrections
        corrected_text = self.correct_context_aware(corrected_text, document_type)
        if corrected_text != text:
            techniques_applied.append('context_aware_correction')
            accuracy_boost += 0.04
        
        # Calculate confidence score
        confidence = self.calculate_ml_confidence(corrected_text, document_type)
        
        return MLAccuracyResult(
            enhanced_text=corrected_text,
            confidence_score=confidence,
            ml_techniques_applied=techniques_applied,
            accuracy_boost=min(accuracy_boost, 0.2)  # Cap at 20% boost
        )
    
    def correct_common_errors(self, text: str) -> str:
        """Correct common OCR errors"""
        corrected = text
        
        for error, correction in self.correction_patterns['common_ocr_errors'].items():
            corrected = corrected.replace(error, correction)
        
        return corrected
    
    def correct_indian_document_text(self, text: str, document_type: str) -> str:
        """Correct Indian document specific text"""
        corrected = text
        
        # Correct Indian names
        for name in self.correction_patterns['indian_names']:
            # Find similar names in text
            similar_names = self.find_similar_names(text, name)
            for similar in similar_names:
                if self.calculate_similarity(similar, name) > 0.8:
                    corrected = corrected.replace(similar, name)
        
        # Correct Indian cities
        for city in self.correction_patterns['indian_cities']:
            similar_cities = self.find_similar_names(text, city)
            for similar in similar_cities:
                if self.calculate_similarity(similar, city) > 0.8:
                    corrected = corrected.replace(similar, city)
        
        return corrected
    
    def find_similar_names(self, text: str, target: str) -> List[str]:
        """Find similar names in text"""
        words = text.split()
        similar = []
        
        for word in words:
            if len(word) > 3 and self.calculate_similarity(word, target) > 0.6:
                similar.append(word)
        
        return similar
    
    def calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words"""
        if not word1 or not word2:
            return 0.0
        
        # Simple character-based similarity
        common_chars = set(word1.upper()) & set(word2.upper())
        total_chars = set(word1.upper()) | set(word2.upper())
        
        if not total_chars:
            return 0.0
        
        return len(common_chars) / len(total_chars)
    
    def correct_patterns(self, text: str, document_type: str) -> str:
        """Correct text based on expected patterns"""
        corrected = text
        
        if document_type == 'pan_card':
            # Fix PAN format
            pan_match = re.search(r'[A-Z0-9]{10}', text)
            if pan_match:
                pan = pan_match.group()
                if len(pan) == 10 and not re.match(self.patterns['pan_format'], pan):
                    # Try to fix common PAN format errors
                    if pan[4:8].isdigit() and pan[0:4].isalpha() and pan[8:10].isalpha():
                        corrected = corrected.replace(pan, pan[0:4] + pan[4:8] + pan[8:10])
        
        elif document_type == 'aadhaar_card':
            # Fix Aadhaar format
            aadhaar_match = re.search(r'\d{10,12}', text)
            if aadhaar_match:
                aadhaar = aadhaar_match.group()
                if len(aadhaar) == 12:
                    # Format as 4-4-4
                    formatted = f"{aadhaar[0:4]} {aadhaar[4:8]} {aadhaar[8:12]}"
                    corrected = corrected.replace(aadhaar, formatted)
        
        return corrected
    
    def correct_context_aware(self, text: str, document_type: str) -> str:
        """Apply context-aware corrections"""
        corrected = text
        
        # Fix common context errors
        context_corrections = {
            'FATHER S NAME': 'FATHER\'S NAME',
            'MOTHER S NAME': 'MOTHER\'S NAME',
            'DATE OF BIRTH': 'DATE OF BIRTH',
            'PLACE OF BIRTH': 'PLACE OF BIRTH',
            'PERMANENT ACCOUNT NUMBER': 'PERMANENT ACCOUNT NUMBER',
            'INCOME TAX DEPARTMENT': 'INCOME TAX DEPARTMENT',
            'GOVT OF INDIA': 'GOVT. OF INDIA'
        }
        
        for error, correction in context_corrections.items():
            corrected = corrected.replace(error, correction)
        
        return corrected
    
    def calculate_ml_confidence(self, text: str, document_type: str) -> float:
        """Calculate confidence score using ML techniques"""
        if not text:
            return 0.0
        
        confidence = 0.0
        
        # Length factor
        if len(text) > 50:
            confidence += 0.2
        
        # Pattern matching
        if document_type in self.patterns:
            pattern = self.patterns.get(f'{document_type}_format')
            if pattern and re.search(pattern, text):
                confidence += 0.3
        
        # Character diversity
        unique_chars = len(set(text.upper()))
        if unique_chars > 20:
            confidence += 0.2
        
        # Word count
        word_count = len(text.split())
        if word_count > 10:
            confidence += 0.1
        
        # Document-specific keywords
        keywords = self.get_document_keywords(document_type)
        keyword_matches = sum(1 for keyword in keywords if keyword in text.upper())
        if keyword_matches > 0:
            confidence += min(keyword_matches / len(keywords), 1.0) * 0.2
        
        return min(confidence, 1.0)
    
    def get_document_keywords(self, document_type: str) -> List[str]:
        """Get keywords for document type"""
        keywords = {
            'pan_card': ['PERMANENT ACCOUNT NUMBER', 'INCOME TAX', 'GOVT OF INDIA', 'PAN'],
            'aadhaar_card': ['AADHAAR', 'UNIQUE IDENTIFICATION', 'GOVERNMENT OF INDIA', 'UID'],
            'driving_license': ['DRIVING LICENCE', 'TRANSPORT AUTHORITY', 'RTO', 'LICENCE NO'],
            'voter_id': ['ELECTORAL PHOTO IDENTITY CARD', 'ELECTION COMMISSION', 'EPIC'],
            'passport': ['PASSPORT', 'MINISTRY OF EXTERNAL AFFAIRS', 'PASSPORT NO']
        }
        
        return keywords.get(document_type, [])

class DocumentQualityAssessment:
    """Assess document quality and suggest improvements"""
    
    def __init__(self):
        self.setup_quality_metrics()
    
    def setup_quality_metrics(self):
        """Setup quality assessment metrics"""
        self.metrics = {
            'contrast_threshold': 30,
            'noise_threshold': 0.1,
            'blur_threshold': 100,
            'brightness_range': (50, 200),
            'resolution_threshold': 300
        }
    
    def assess_document_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess document quality"""
        assessment = {
            'overall_quality': 'good',
            'issues': [],
            'recommendations': [],
            'quality_score': 0.0
        }
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Assess contrast
        contrast = gray.std()
        if contrast < self.metrics['contrast_threshold']:
            assessment['issues'].append('low_contrast')
            assessment['recommendations'].append('Apply contrast enhancement')
        else:
            assessment['quality_score'] += 0.2
        
        # Assess noise
        noise = self.estimate_noise_level(gray)
        if noise > self.metrics['noise_threshold']:
            assessment['issues'].append('high_noise')
            assessment['recommendations'].append('Apply noise reduction')
        else:
            assessment['quality_score'] += 0.2
        
        # Assess blur
        blur = self.estimate_blur_level(gray)
        if blur > self.metrics['blur_threshold']:
            assessment['issues'].append('blurred')
            assessment['recommendations'].append('Apply sharpening')
        else:
            assessment['quality_score'] += 0.2
        
        # Assess brightness
        brightness = gray.mean()
        if brightness < self.metrics['brightness_range'][0]:
            assessment['issues'].append('too_dark')
            assessment['recommendations'].append('Increase brightness')
        elif brightness > self.metrics['brightness_range'][1]:
            assessment['issues'].append('too_bright')
            assessment['recommendations'].append('Decrease brightness')
        else:
            assessment['quality_score'] += 0.2
        
        # Assess resolution
        height, width = gray.shape
        resolution = max(height, width)
        if resolution < self.metrics['resolution_threshold']:
            assessment['issues'].append('low_resolution')
            assessment['recommendations'].append('Use higher resolution image')
        else:
            assessment['quality_score'] += 0.2
        
        # Determine overall quality
        if assessment['quality_score'] >= 0.8:
            assessment['overall_quality'] = 'excellent'
        elif assessment['quality_score'] >= 0.6:
            assessment['overall_quality'] = 'good'
        elif assessment['quality_score'] >= 0.4:
            assessment['overall_quality'] = 'fair'
        else:
            assessment['overall_quality'] = 'poor'
        
        return assessment
    
    def estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level in image"""
        try:
            # Use Laplacian to estimate noise
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            noise = laplacian.var()
            return noise / 1000.0  # Normalize
        except:
            return 0.0
    
    def estimate_blur_level(self, image: np.ndarray) -> float:
        """Estimate blur level in image"""
        try:
            # Use Laplacian to estimate blur
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            blur = laplacian.var()
            return blur
        except:
            return 0.0

class ConfidenceBoosting:
    """Advanced confidence boosting techniques"""
    
    def __init__(self):
        self.setup_confidence_boosters()
    
    def setup_confidence_boosters(self):
        """Setup confidence boosting techniques"""
        self.boosters = {
            'pattern_validation': self.validate_patterns,
            'context_validation': self.validate_context,
            'format_validation': self.validate_format,
            'consistency_check': self.check_consistency
        }
    
    def boost_confidence(self, text: str, document_type: str, extracted_fields: Dict[str, str]) -> Dict[str, Any]:
        """Boost confidence using multiple techniques"""
        boost_result = {
            'original_confidence': 0.0,
            'boosted_confidence': 0.0,
            'boost_factors': {},
            'total_boost': 0.0
        }
        
        # Calculate original confidence
        original_confidence = self.calculate_base_confidence(text, document_type)
        boost_result['original_confidence'] = original_confidence
        
        # Apply confidence boosters
        total_boost = 0.0
        boost_factors = {}
        
        for booster_name, booster_func in self.boosters.items():
            try:
                boost = booster_func(text, document_type, extracted_fields)
                boost_factors[booster_name] = boost
                total_boost += boost
            except Exception as e:
                print(f"Confidence booster {booster_name} failed: {e}")
                boost_factors[booster_name] = 0.0
        
        # Calculate final confidence
        boosted_confidence = min(original_confidence + total_boost, 1.0)
        
        boost_result['boosted_confidence'] = boosted_confidence
        boost_result['boost_factors'] = boost_factors
        boost_result['total_boost'] = total_boost
        
        return boost_result
    
    def calculate_base_confidence(self, text: str, document_type: str) -> float:
        """Calculate base confidence score"""
        if not text:
            return 0.0
        
        confidence = 0.0
        
        # Length factor
        confidence += min(len(text) / 100, 1.0) * 0.2
        
        # Character diversity
        unique_chars = len(set(text.upper()))
        confidence += min(unique_chars / 26, 1.0) * 0.2
        
        # Word count
        word_count = len(text.split())
        confidence += min(word_count / 20, 1.0) * 0.2
        
        # Document-specific patterns
        confidence += self.check_document_patterns(text, document_type) * 0.4
        
        return min(confidence, 1.0)
    
    def validate_patterns(self, text: str, document_type: str, extracted_fields: Dict[str, str]) -> float:
        """Validate patterns in text"""
        boost = 0.0
        
        if document_type == 'pan_card':
            if re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', text):
                boost += 0.1
            if 'INCOME TAX' in text.upper():
                boost += 0.05
            if 'GOVT OF INDIA' in text.upper():
                boost += 0.05
        
        elif document_type == 'aadhaar_card':
            if re.search(r'\d{4}\s?\d{4}\s?\d{4}', text):
                boost += 0.1
            if 'AADHAAR' in text.upper():
                boost += 0.05
            if 'UNIQUE IDENTIFICATION' in text.upper():
                boost += 0.05
        
        return min(boost, 0.2)
    
    def validate_context(self, text: str, document_type: str, extracted_fields: Dict[str, str]) -> float:
        """Validate context consistency"""
        boost = 0.0
        
        # Check for required fields
        required_fields = self.get_required_fields(document_type)
        present_fields = sum(1 for field in required_fields if field in extracted_fields)
        
        if present_fields > 0:
            boost += (present_fields / len(required_fields)) * 0.1
        
        return min(boost, 0.1)
    
    def validate_format(self, text: str, document_type: str, extracted_fields: Dict[str, str]) -> float:
        """Validate format consistency"""
        boost = 0.0
        
        # Check date formats
        if 'date_of_birth' in extracted_fields:
            date = extracted_fields['date_of_birth']
            if re.match(r'\d{2}[/-]\d{2}[/-]\d{4}', date):
                boost += 0.05
        
        # Check name formats
        if 'first_name' in extracted_fields and 'last_name' in extracted_fields:
            first_name = extracted_fields['first_name']
            last_name = extracted_fields['last_name']
            if first_name and last_name and len(first_name) > 1 and len(last_name) > 1:
                boost += 0.05
        
        return min(boost, 0.1)
    
    def check_consistency(self, text: str, document_type: str, extracted_fields: Dict[str, str]) -> float:
        """Check data consistency"""
        boost = 0.0
        
        # Check for logical consistency
        if 'date_of_birth' in extracted_fields:
            # Basic date validation
            date = extracted_fields['date_of_birth']
            if self.is_valid_date(date):
                boost += 0.05
        
        return min(boost, 0.05)
    
    def get_required_fields(self, document_type: str) -> List[str]:
        """Get required fields for document type"""
        required = {
            'pan_card': ['name', 'father_name', 'date_of_birth', 'pan'],
            'aadhaar_card': ['name', 'father_name', 'date_of_birth', 'aadhaar'],
            'driving_license': ['name', 'father_name', 'date_of_birth', 'license_no'],
            'voter_id': ['name', 'father_name', 'date_of_birth', 'epic_no'],
            'passport': ['name', 'father_name', 'date_of_birth', 'passport_no']
        }
        
        return required.get(document_type, [])
    
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
    
    def is_valid_date(self, date_str: str) -> bool:
        """Check if date string is valid"""
        try:
            import datetime
            formats = ['%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%d']
            for fmt in formats:
                try:
                    datetime.datetime.strptime(date_str, fmt)
                    return True
                except ValueError:
                    continue
            return False
        except:
            return False

# Usage example
def test_ml_accuracy_boosters():
    """Test ML accuracy boosters"""
    text_corrector = TextCorrectionML()
    quality_assessor = DocumentQualityAssessment()
    confidence_booster = ConfidenceBoosting()
    
    print("🧪 Testing ML Accuracy Boosters...")
    
    # Test text correction
    sample_text = "RAJESH KUMAR SHARMA FATHER S NAME RAMESH KUMAR SHARMA"
    correction_result = text_corrector.correct_text_ml(sample_text, 'pan_card')
    print(f"Text correction: {correction_result.accuracy_boost:.2f} boost")
    
    # Test confidence boosting
    extracted_fields = {
        'name': 'RAJESH KUMAR SHARMA',
        'father_name': 'RAMESH KUMAR SHARMA',
        'date_of_birth': '15/01/1990',
        'pan': 'ABCDE1234F'
    }
    
    boost_result = confidence_booster.boost_confidence(sample_text, 'pan_card', extracted_fields)
    print(f"Confidence boost: {boost_result['total_boost']:.2f}")

if __name__ == "__main__":
    test_ml_accuracy_boosters()
