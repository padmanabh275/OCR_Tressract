"""
Enhanced Document Type Classification System
Advanced techniques for accurate document type recognition
"""

import re
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import pytesseract

@dataclass
class DocumentTypeResult:
    """Result of document type classification"""
    document_type: str
    confidence: float
    matched_patterns: List[str]
    visual_features: Dict[str, float]
    text_features: Dict[str, float]

class EnhancedDocumentClassifier:
    """Advanced document type classifier with multiple recognition methods"""
    
    def __init__(self):
        self.setup_classification_patterns()
        self.setup_visual_features()
        
    def setup_classification_patterns(self):
        """Setup comprehensive document type patterns"""
        
        # Enhanced keyword patterns with weights and context
        self.document_patterns = {
            'driver_license': {
                'keywords': {
                    'primary': ['driver', 'license', 'driving', 'dmv', 'department of motor vehicles'],
                    'secondary': ['dl', 'license number', 'class', 'endorsements', 'restrictions'],
                    'state_specific': ['california', 'texas', 'florida', 'new york', 'illinois']
                },
                'patterns': [
                    r'DRIVER.*LICENSE',
                    r'DRIVING.*LICENSE',
                    r'CLASS\s+[A-Z]',
                    r'ENDORSEMENTS?',
                    r'RESTRICTIONS?',
                    r'EXPIRES?\s+\d{2}/\d{2}/\d{4}',
                    r'DOB\s*:\s*\d{2}/\d{2}/\d{4}',
                    r'SEX\s*:\s*[MF]',
                    r'HEIGHT\s*:\s*\d+\'?\d*"',
                    r'WEIGHT\s*:\s*\d+',
                    r'HAIR\s*:\s*[A-Za-z]+',
                    r'EYES\s*:\s*[A-Za-z]+'
                ],
                'visual_indicators': ['rectangular_format', 'photo_present', 'barcode_present'],
                'weight': 1.0
            },
            
            'passport': {
                'keywords': {
                    'primary': ['passport', 'passport number', 'issuing country', 'nationality'],
                    'secondary': ['passport no', 'passport #', 'country code', 'mrz'],
                    'specific': ['united states', 'usa', 'canada', 'mexico', 'united kingdom']
                },
                'patterns': [
                    r'PASSPORT',
                    r'PASSPORT\s+NO\.?\s*:?\s*[A-Z0-9]+',
                    r'ISSUING\s+COUNTRY',
                    r'NATIONALITY',
                    r'COUNTRY\s+CODE',
                    r'MRZ\s*:?\s*[A-Z0-9<]+',
                    r'P<[A-Z]{3}[A-Z0-9<]+',
                    r'[A-Z]{2}\d{7}',
                    r'UNITED\s+STATES\s+OF\s+AMERICA'
                ],
                'visual_indicators': ['booklet_format', 'coat_of_arms', 'gold_embossing'],
                'weight': 1.0
            },
            
            'birth_certificate': {
                'keywords': {
                    'primary': ['birth', 'certificate', 'born', 'birthplace', 'birth certificate'],
                    'secondary': ['date of birth', 'place of birth', 'mother', 'father', 'parents'],
                    'official': ['vital records', 'registrar', 'county', 'state', 'certified']
                },
                'patterns': [
                    r'BIRTH\s+CERTIFICATE',
                    r'CERTIFICATE\s+OF\s+BIRTH',
                    r'DATE\s+OF\s+BIRTH',
                    r'PLACE\s+OF\s+BIRTH',
                    r'MOTHER.*NAME',
                    r'FATHER.*NAME',
                    r'PARENTS.*NAME',
                    r'BORN\s+ON',
                    r'BIRTHPLACE',
                    r'VITAL\s+RECORDS',
                    r'REGISTRAR',
                    r'CERTIFIED\s+COPY'
                ],
                'visual_indicators': ['official_seal', 'government_header', 'formal_layout'],
                'weight': 0.9
            },
            
            'ssn_document': {
                'keywords': {
                    'primary': ['social security', 'ssn', 'ss#', 'social security number'],
                    'secondary': ['ssa', 'social security administration', 'card'],
                    'specific': ['social security card', 'ss-5', 'replacement card']
                },
                'patterns': [
                    r'SOCIAL\s+SECURITY',
                    r'SSN\s*:?\s*\d{3}-?\d{2}-?\d{4}',
                    r'SS#\s*:?\s*\d{3}-?\d{2}-?\d{4}',
                    r'SOCIAL\s+SECURITY\s+NUMBER',
                    r'SSA\s*:?\s*SOCIAL\s+SECURITY\s+ADMINISTRATION',
                    r'REPLACEMENT\s+CARD',
                    r'SS-5',
                    r'NOT\s+VALID\s+FOR\s+EMPLOYMENT'
                ],
                'visual_indicators': ['card_format', 'blue_background', 'ssa_logo'],
                'weight': 1.0
            },
            
            'utility_bill': {
                'keywords': {
                    'primary': ['utility', 'electric', 'gas', 'water', 'power', 'energy'],
                    'secondary': ['bill', 'statement', 'account', 'service', 'usage'],
                    'companies': ['coned', 'pge', 'southern california edison', 'duke energy']
                },
                'patterns': [
                    r'UTILITY\s+BILL',
                    r'ELECTRIC\s+BILL',
                    r'GAS\s+BILL',
                    r'WATER\s+BILL',
                    r'POWER\s+BILL',
                    r'ENERGY\s+BILL',
                    r'ACCOUNT\s+NUMBER',
                    r'SERVICE\s+ADDRESS',
                    r'BILLING\s+DATE',
                    r'DUE\s+DATE',
                    r'AMOUNT\s+DUE',
                    r'KWH\s+USED',
                    r'THERMS\s+USED'
                ],
                'visual_indicators': ['table_format', 'amount_due', 'usage_data'],
                'weight': 0.8
            },
            
            'rental_agreement': {
                'keywords': {
                    'primary': ['rental', 'lease', 'tenant', 'landlord', 'rent'],
                    'secondary': ['agreement', 'contract', 'property', 'apartment', 'house'],
                    'legal': ['terms', 'conditions', 'deposit', 'security deposit', 'monthly rent']
                },
                'patterns': [
                    r'RENTAL\s+AGREEMENT',
                    r'LEASE\s+AGREEMENT',
                    r'TENANT\s+AGREEMENT',
                    r'LANDLORD.*TENANT',
                    r'RENT\s+AGREEMENT',
                    r'PROPERTY\s+ADDRESS',
                    r'MONTHLY\s+RENT',
                    r'SECURITY\s+DEPOSIT',
                    r'LEASE\s+TERM',
                    r'RENTAL\s+PERIOD',
                    r'UTILITIES\s+INCLUDED'
                ],
                'visual_indicators': ['legal_document', 'signature_lines', 'terms_section'],
                'weight': 0.8
            },
            
            'tax_return': {
                'keywords': {
                    'primary': ['tax return', 'form 1040', 'irs', 'federal tax', 'income tax'],
                    'secondary': ['adjusted gross income', 'taxable income', 'refund', 'amount owed'],
                    'forms': ['w-2', '1099', 'schedule a', 'schedule b', 'schedule c']
                },
                'patterns': [
                    r'FORM\s+1040',
                    r'TAX\s+RETURN',
                    r'FEDERAL\s+INCOME\s+TAX',
                    r'INTERNAL\s+REVENUE\s+SERVICE',
                    r'ADJUSTED\s+GROSS\s+INCOME',
                    r'TAXABLE\s+INCOME',
                    r'REFUND\s+AMOUNT',
                    r'AMOUNT\s+OWED',
                    r'SCHEDULE\s+[A-Z]',
                    r'W-2\s+WAGE\s+AND\s+TAX\s+STATEMENT',
                    r'1099'
                ],
                'visual_indicators': ['form_layout', 'irs_logo', 'tax_tables'],
                'weight': 0.9
            },
            
            'w2_form': {
                'keywords': {
                    'primary': ['w-2', 'w2', 'wage and tax statement', 'employer'],
                    'secondary': ['wages', 'tips', 'compensation', 'federal income tax'],
                    'specific': ['box 1', 'box 2', 'box 3', 'box 4', 'box 5']
                },
                'patterns': [
                    r'W-2\s+WAGE\s+AND\s+TAX\s+STATEMENT',
                    r'FORM\s+W-2',
                    r'EMPLOYER.*IDENTIFICATION',
                    r'EMPLOYEE.*SOCIAL\s+SECURITY',
                    r'WAGES.*TIPS.*COMPENSATION',
                    r'FEDERAL\s+INCOME\s+TAX\s+WITHHELD',
                    r'BOX\s+[1-9]',
                    r'STATE\s+INCOME\s+TAX',
                    r'LOCAL\s+INCOME\s+TAX'
                ],
                'visual_indicators': ['form_layout', 'box_structure', 'employer_info'],
                'weight': 0.9
            },
            
            'bank_statement': {
                'keywords': {
                    'primary': ['bank', 'account', 'statement', 'checking', 'savings'],
                    'secondary': ['balance', 'transactions', 'deposits', 'withdrawals'],
                    'banks': ['chase', 'bank of america', 'wells fargo', 'citibank', 'us bank']
                },
                'patterns': [
                    r'BANK\s+STATEMENT',
                    r'ACCOUNT\s+STATEMENT',
                    r'CHECKING\s+ACCOUNT',
                    r'SAVINGS\s+ACCOUNT',
                    r'ACCOUNT\s+NUMBER',
                    r'STATEMENT\s+PERIOD',
                    r'CURRENT\s+BALANCE',
                    r'PREVIOUS\s+BALANCE',
                    r'TRANSACTIONS?',
                    r'DEPOSITS?',
                    r'WITHDRAWALS?',
                    r'DEBIT\s+CARD',
                    r'CREDIT\s+CARD'
                ],
                'visual_indicators': ['table_format', 'bank_logo', 'account_numbers'],
                'weight': 0.8
            }
        }
    
    def setup_visual_features(self):
        """Setup visual feature detection for document classification"""
        self.visual_detectors = {
            'rectangular_format': self.detect_rectangular_format,
            'photo_present': self.detect_photo,
            'barcode_present': self.detect_barcode,
            'official_seal': self.detect_official_seal,
            'government_header': self.detect_government_header,
            'table_format': self.detect_table_format,
            'amount_due': self.detect_amount_due,
            'legal_document': self.detect_legal_document,
            'signature_lines': self.detect_signature_lines,
            'form_layout': self.detect_form_layout,
            'irs_logo': self.detect_irs_logo,
            'box_structure': self.detect_box_structure,
            'bank_logo': self.detect_bank_logo,
            'account_numbers': self.detect_account_numbers
        }
    
    def classify_document_type(self, text: str, image: np.ndarray = None) -> DocumentTypeResult:
        """Classify document type using text and visual analysis"""
        
        # Text-based classification
        text_features = self.analyze_text_features(text)
        
        # Visual-based classification (if image provided)
        visual_features = {}
        if image is not None:
            visual_features = self.analyze_visual_features(image)
        
        # Calculate scores for each document type
        type_scores = {}
        matched_patterns = {}
        
        for doc_type, config in self.document_patterns.items():
            score = 0.0
            patterns_found = []
            
            # Text pattern matching
            text_score = self.calculate_text_score(text, config)
            score += text_score * 0.7  # 70% weight for text
            
            # Visual feature matching
            if image is not None:
                visual_score = self.calculate_visual_score(visual_features, config)
                score += visual_score * 0.3  # 30% weight for visual
            
            # Apply document type weight
            score *= config['weight']
            
            type_scores[doc_type] = score
            matched_patterns[doc_type] = patterns_found
        
        # Find best match
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            best_score = type_scores[best_type]
            
            # Normalize confidence (0-1)
            confidence = min(best_score / 10.0, 1.0)  # Adjust divisor based on testing
            
            return DocumentTypeResult(
                document_type=best_type,
                confidence=confidence,
                matched_patterns=matched_patterns.get(best_type, []),
                visual_features=visual_features,
                text_features=text_features
            )
        
        return DocumentTypeResult(
            document_type='unknown',
            confidence=0.0,
            matched_patterns=[],
            visual_features=visual_features,
            text_features=text_features
        )
    
    def calculate_text_score(self, text: str, config: Dict) -> float:
        """Calculate text-based score for document type"""
        score = 0.0
        text_lower = text.lower()
        
        # Primary keywords (highest weight)
        for keyword in config['keywords']['primary']:
            if keyword in text_lower:
                score += 3.0
        
        # Secondary keywords (medium weight)
        for keyword in config['keywords']['secondary']:
            if keyword in text_lower:
                score += 2.0
        
        # Specific keywords (context-dependent weight)
        for keyword in config['keywords'].get('specific', []):
            if keyword in text_lower:
                score += 1.5
        
        # Pattern matching (regex)
        for pattern in config['patterns']:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                score += 2.0
        
        return score
    
    def calculate_visual_score(self, visual_features: Dict, config: Dict) -> float:
        """Calculate visual-based score for document type"""
        score = 0.0
        
        for indicator in config.get('visual_indicators', []):
            if visual_features.get(indicator, False):
                score += 1.0
        
        return score
    
    def analyze_text_features(self, text: str) -> Dict[str, float]:
        """Analyze text features for classification"""
        features = {}
        
        # Text length
        features['text_length'] = len(text)
        
        # Word count
        words = text.split()
        features['word_count'] = len(words)
        
        # Average word length
        if words:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
        
        # Number of numbers
        numbers = re.findall(r'\d+', text)
        features['number_count'] = len(numbers)
        
        # Number of dates
        dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
        features['date_count'] = len(dates)
        
        # Number of monetary amounts
        money = re.findall(r'\$[\d,]+\.?\d*', text)
        features['money_count'] = len(money)
        
        # Uppercase ratio
        if text:
            features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
        
        return features
    
    def analyze_visual_features(self, image: np.ndarray) -> Dict[str, bool]:
        """Analyze visual features for classification"""
        features = {}
        
        for feature_name, detector in self.visual_detectors.items():
            try:
                features[feature_name] = detector(image)
            except Exception as e:
                features[feature_name] = False
        
        return features
    
    # Visual feature detectors
    def detect_rectangular_format(self, image: np.ndarray) -> bool:
        """Detect if document has rectangular format (like driver's license)"""
        h, w = image.shape[:2]
        aspect_ratio = w / h
        return 1.5 < aspect_ratio < 3.0  # Typical license aspect ratio
    
    def detect_photo(self, image: np.ndarray) -> bool:
        """Detect if document contains a photo"""
        # Simple edge detection to find rectangular photo areas
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular contours that could be photos
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area for photo
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.7 < aspect_ratio < 1.3:  # Square-ish aspect ratio
                    return True
        return False
    
    def detect_barcode(self, image: np.ndarray) -> bool:
        """Detect if document contains barcode"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Look for horizontal lines (barcode pattern)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Count horizontal lines
        lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        return lines is not None and len(lines) > 5
    
    def detect_official_seal(self, image: np.ndarray) -> bool:
        """Detect if document has official seal"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Look for circular patterns (seals)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=100)
        return circles is not None
    
    def detect_government_header(self, image: np.ndarray) -> bool:
        """Detect government document header"""
        # Look for text patterns that indicate government documents
        text = pytesseract.image_to_string(image, config='--psm 6')
        government_terms = ['government', 'state', 'county', 'federal', 'department', 'bureau']
        return any(term in text.lower() for term in government_terms)
    
    def detect_table_format(self, image: np.ndarray) -> bool:
        """Detect if document has table format"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Count lines
        h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        return (h_lines is not None and len(h_lines) > 3) and (v_lines is not None and len(v_lines) > 3)
    
    def detect_amount_due(self, image: np.ndarray) -> bool:
        """Detect monetary amounts in document"""
        text = pytesseract.image_to_string(image, config='--psm 6')
        money_patterns = [r'\$[\d,]+\.?\d*', r'amount due', r'total due', r'balance due']
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in money_patterns)
    
    def detect_legal_document(self, image: np.ndarray) -> bool:
        """Detect legal document characteristics"""
        text = pytesseract.image_to_string(image, config='--psm 6')
        legal_terms = ['agreement', 'contract', 'terms', 'conditions', 'hereby', 'whereas']
        return sum(1 for term in legal_terms if term in text.lower()) >= 3
    
    def detect_signature_lines(self, image: np.ndarray) -> bool:
        """Detect signature lines"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Look for horizontal lines that could be signature lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        line_count = cv2.HoughLinesP(lines, 1, np.pi/180, threshold=30, minLineLength=50, maxLineGap=10)
        return line_count is not None and len(line_count) >= 2
    
    def detect_form_layout(self, image: np.ndarray) -> bool:
        """Detect form-like layout"""
        return self.detect_table_format(image)  # Forms often have table-like structure
    
    def detect_irs_logo(self, image: np.ndarray) -> bool:
        """Detect IRS logo or branding"""
        text = pytesseract.image_to_string(image, config='--psm 6')
        irs_terms = ['internal revenue service', 'irs', 'form 1040', 'federal income tax']
        return any(term in text.lower() for term in irs_terms)
    
    def detect_box_structure(self, image: np.ndarray) -> bool:
        """Detect box-like structure (like W-2 forms)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Look for rectangular boxes
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        box_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area for box
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 2.0:  # Reasonable box aspect ratio
                    box_count += 1
        
        return box_count >= 3
    
    def detect_bank_logo(self, image: np.ndarray) -> bool:
        """Detect bank logos or branding"""
        text = pytesseract.image_to_string(image, config='--psm 6')
        bank_terms = ['bank', 'chase', 'wells fargo', 'bank of america', 'citibank', 'account']
        return any(term in text.lower() for term in bank_terms)
    
    def detect_account_numbers(self, image: np.ndarray) -> bool:
        """Detect account numbers"""
        text = pytesseract.image_to_string(image, config='--psm 6')
        account_patterns = [r'account\s+number', r'account\s+#', r'\d{10,}', r'routing\s+number']
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in account_patterns)

# Usage example
def test_document_classifier():
    """Test the enhanced document classifier"""
    classifier = EnhancedDocumentClassifier()
    
    # Test with sample text
    sample_text = """
    DRIVER LICENSE
    STATE OF CALIFORNIA
    DEPARTMENT OF MOTOR VEHICLES
    CLASS C
    EXPIRES 12/25/2025
    DOB: 01/15/1990
    SEX: M
    HEIGHT: 6'0"
    """
    
    result = classifier.classify_document_type(sample_text)
    print(f"Document Type: {result.document_type}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Matched Patterns: {result.matched_patterns}")

if __name__ == "__main__":
    test_document_classifier()
