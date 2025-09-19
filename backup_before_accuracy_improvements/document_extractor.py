"""
Advanced Accuracy Improvements for Document Extraction System
Multiple strategies to significantly boost extraction accuracy
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass

@dataclass
class AccuracyConfig:
    """Configuration for accuracy improvements"""
    use_advanced_preprocessing: bool = True
    use_multiple_ocr_engines: bool = True
    use_ml_classification: bool = True
    use_ensemble_extraction: bool = True
    confidence_threshold: float = 0.7

class AdvancedDocumentProcessor:
    """Advanced processor with multiple accuracy improvement techniques"""
    
    def __init__(self):
        self.setup_advanced_ocr()
        self.setup_ml_models()
        
    def setup_advanced_ocr(self):
        """Setup advanced OCR configurations"""
        # Multiple Tesseract configurations for different document types
        self.ocr_configs = {
            'standard': '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/:.,() ',
            'single_text_block': '--psm 6',
            'single_column': '--psm 4',
            'single_word': '--psm 8',
            'single_line': '--psm 7',
            'raw_line': '--psm 13',
            'sparse_text': '--psm 11',
            'orientation_script': '--psm 0',
            'numbers_only': '--psm 6 -c tessedit_char_whitelist=0123456789-',
            'letters_only': '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ',
        }
        
        # Language-specific configurations
        self.language_configs = {
            'english': 'eng',
            'spanish': 'spa',
            'french': 'fra',
            'german': 'deu'
        }
    
    def setup_ml_models(self):
        """Setup machine learning models for better accuracy"""
        # This would integrate with actual ML models
        self.document_classifier = None  # Placeholder for ML model
        self.field_extractor = None      # Placeholder for NER model
        
    def advanced_image_preprocessing(self, image: np.ndarray) -> List[np.ndarray]:
        """Advanced image preprocessing with 15+ techniques"""
        processed_images = []
        
        # 1. Original image
        processed_images.append(('original', image))
        
        # 2. Grayscale conversion
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_images.append(('grayscale', gray))
        
        # 3. Noise reduction techniques
        # Gaussian blur
        gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
        processed_images.append(('gaussian_blur', gaussian))
        
        # Bilateral filter for edge-preserving smoothing
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        processed_images.append(('bilateral', bilateral))
        
        # 4. Contrast enhancement
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(gray)
        processed_images.append(('clahe', clahe_img))
        
        # Histogram equalization
        hist_eq = cv2.equalizeHist(gray)
        processed_images.append(('histogram_eq', hist_eq))
        
        # 5. Thresholding techniques
        # Otsu's thresholding
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(('otsu', otsu))
        
        # Adaptive thresholding
        adaptive_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_images.append(('adaptive_mean', adaptive_mean))
        
        adaptive_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_images.append(('adaptive_gaussian', adaptive_gaussian))
        
        # 6. Morphological operations
        kernel = np.ones((2,2), np.uint8)
        opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
        processed_images.append(('morphology_open', opening))
        
        closing = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        processed_images.append(('morphology_close', closing))
        
        # 7. Edge detection and enhancement
        edges = cv2.Canny(gray, 50, 150)
        processed_images.append(('canny_edges', edges))
        
        # 8. PIL-based enhancements
        pil_image = Image.fromarray(gray)
        
        # Contrast enhancement
        enhancer = ImageEnhance.Contrast(pil_image)
        contrast_img = enhancer.enhance(2.0)
        processed_images.append(('contrast_enhanced', np.array(contrast_img)))
        
        # Sharpening
        sharp_img = pil_image.filter(ImageFilter.SHARPEN)
        processed_images.append(('sharpened', np.array(sharp_img)))
        
        # Brightness adjustment
        brightness_enhancer = ImageEnhance.Brightness(pil_image)
        bright_img = brightness_enhancer.enhance(1.2)
        processed_images.append(('brightness_enhanced', np.array(bright_img)))
        
        # 9. Deskewing (rotation correction)
        deskewed = self.deskew_image(gray)
        processed_images.append(('deskewed', deskewed))
        
        # 10. Scale normalization
        scaled = self.normalize_scale(gray)
        processed_images.append(('scaled', scaled))
        
        return processed_images
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew/rotation for better OCR"""
        try:
            # Find contours
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return image
            
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            
            # Correct angle
            if angle < -45:
                angle += 90
            
            # Rotate image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
        except:
            return image
    
    def normalize_scale(self, image: np.ndarray, target_height: int = 1000) -> np.ndarray:
        """Normalize image scale for consistent OCR"""
        h, w = image.shape[:2]
        if h < target_height:
            scale_factor = target_height / h
            new_w = int(w * scale_factor)
            return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
        return image
    
    def extract_text_ensemble(self, image: np.ndarray) -> Dict[str, str]:
        """Extract text using ensemble of methods"""
        results = {}
        
        # Get all preprocessed images
        processed_images = self.advanced_image_preprocessing(image)
        
        # Try each OCR configuration on each preprocessed image
        for img_name, processed_img in processed_images:
            for config_name, config in self.ocr_configs.items():
                try:
                    text = pytesseract.image_to_string(processed_img, config=config)
                    if text.strip():
                        results[f"{img_name}_{config_name}"] = text.strip()
                except Exception as e:
                    continue
        
        return results
    
    def score_text_quality(self, text: str) -> float:
        """Score text quality based on multiple factors"""
        if not text.strip():
            return 0.0
        
        score = 0.0
        
        # Length factor (longer text generally better)
        score += min(len(text) / 100, 1.0) * 0.2
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        score += min(unique_chars / 26, 1.0) * 0.2
        
        # Word count
        word_count = len(text.split())
        score += min(word_count / 20, 1.0) * 0.2
        
        # Pattern matching bonus
        patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Names
            r'\b\d{3}-?\d{2}-?\d{4}\b',        # SSN
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates
            r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd)\b'  # Addresses
        ]
        
        pattern_matches = sum(1 for pattern in patterns if re.search(pattern, text))
        score += (pattern_matches / len(patterns)) * 0.4
        
        return min(score, 1.0)
    
    def select_best_text(self, text_results: Dict[str, str]) -> Tuple[str, float]:
        """Select the best text result based on quality scoring"""
        if not text_results:
            return "", 0.0
        
        best_text = ""
        best_score = 0.0
        
        for method, text in text_results.items():
            score = self.score_text_quality(text)
            if score > best_score:
                best_score = score
                best_text = text
        
        return best_text, best_score
    
    def advanced_field_extraction(self, text: str) -> Dict[str, any]:
        """Advanced field extraction with multiple validation layers"""
        extracted = {}
        
        # Enhanced regex patterns with better accuracy
        patterns = {
            'ssn': [
                r'\b\d{3}-?\d{2}-?\d{4}\b',
                r'\b\d{9}\b',
                r'SSN[:\s]*(\d{3}-?\d{2}-?\d{4})',
                r'Social Security[:\s]*(\d{3}-?\d{2}-?\d{4})'
            ],
            'first_name': [
                r'(?:First Name|Given Name|First)[:\s]*([A-Z][a-z]+)',
                r'^([A-Z][a-z]+)\s+[A-Z][a-z]+',  # First word in "First Last"
                r'Name[:\s]*([A-Z][a-z]+)'
            ],
            'last_name': [
                r'(?:Last Name|Surname|Family Name|Last)[:\s]*([A-Z][a-z]+)',
                r'[A-Z][a-z]+\s+([A-Z][a-z]+)$',  # Last word in "First Last"
                r'Name[:\s]*[A-Z][a-z]+\s+([A-Z][a-z]+)'
            ],
            'date_of_birth': [
                r'(?:DOB|Date of Birth|Born|Birth Date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}'
            ],
            'address': [
                r'(?:Address|Residence|Home Address)[:\s]*([A-Za-z0-9\s,.-]+)',
                r'\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)',
                r'[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)[A-Za-z0-9\s,.-]*'
            ],
            'phone': [
                r'(?:Phone|Tel|Mobile|Cell)[:\s]*(\d{3}[-.]?\d{3}[-.]?\d{4})',
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                r'\(\d{3}\)\s*\d{3}[-.]?\d{4}'
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ]
        }
        
        # Extract each field with validation
        for field_name, field_patterns in patterns.items():
            best_match = None
            best_confidence = 0.0
            
            for pattern in field_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    
                    if match and match.strip():
                        # Validate the match
                        confidence = self.validate_field(field_name, match.strip())
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_match = match.strip()
            
            if best_match and best_confidence > 0.5:
                extracted[field_name] = {
                    'value': best_match,
                    'confidence': best_confidence
                }
        
        return extracted
    
    def validate_field(self, field_name: str, value: str) -> float:
        """Validate extracted field values and return confidence score"""
        if not value or not value.strip():
            return 0.0
        
        value = value.strip()
        
        if field_name == 'ssn':
            # SSN validation
            digits = re.sub(r'\D', '', value)
            if len(digits) == 9:
                # Check for invalid SSNs
                if digits.startswith('000') or digits.startswith('666') or digits[3:5] == '00' or digits[5:] == '0000':
                    return 0.3
                return 0.9
            return 0.1
        
        elif field_name in ['first_name', 'last_name']:
            # Name validation
            if len(value) < 2 or len(value) > 20:
                return 0.2
            if not re.match(r'^[A-Za-z\s-]+$', value):
                return 0.3
            if value.isupper() or value.islower():
                return 0.7  # Mixed case is better
            return 0.9
        
        elif field_name == 'date_of_birth':
            # Date validation
            try:
                from dateutil import parser
                parsed_date = parser.parse(value, fuzzy=True)
                # Check if date is reasonable (not in future, not too old)
                from datetime import datetime
                now = datetime.now()
                if parsed_date > now:
                    return 0.1
                if (now - parsed_date).days > 36500:  # More than 100 years
                    return 0.1
                return 0.8
            except:
                return 0.2
        
        elif field_name == 'address':
            # Address validation
            if len(value) < 10:
                return 0.2
            # Check for common address indicators
            address_indicators = ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'drive', 'dr', 'lane', 'ln']
            if any(indicator in value.lower() for indicator in address_indicators):
                return 0.8
            return 0.5
        
        elif field_name == 'phone':
            # Phone validation
            digits = re.sub(r'\D', '', value)
            if len(digits) == 10:
                return 0.9
            elif len(digits) == 11 and digits.startswith('1'):
                return 0.8
            return 0.3
        
        elif field_name == 'email':
            # Email validation
            if re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$', value):
                return 0.9
            return 0.2
        
        return 0.5  # Default confidence
    
    def extract_with_ml_enhancement(self, text: str, image: np.ndarray) -> Dict[str, any]:
        """Extract fields using ML enhancement (placeholder for actual ML models)"""
        # This would integrate with actual ML models
        # For now, use the advanced regex approach
        return self.advanced_field_extraction(text)
    
    def process_document_advanced(self, image: np.ndarray) -> Dict[str, any]:
        """Main processing function with all accuracy improvements"""
        # Step 1: Extract text using ensemble method
        text_results = self.extract_text_ensemble(image)
        
        # Step 2: Select best text
        best_text, text_confidence = self.select_best_text(text_results)
        
        if not best_text:
            return {
                'success': False,
                'error': 'No text extracted',
                'confidence': 0.0
            }
        
        # Step 3: Extract fields with advanced methods
        extracted_fields = self.advanced_field_extraction(best_text)
        
        # Step 4: Calculate overall confidence
        field_confidences = [field['confidence'] for field in extracted_fields.values() if isinstance(field, dict)]
        overall_confidence = (text_confidence + sum(field_confidences) / max(len(field_confidences), 1)) / 2
        
        return {
            'success': True,
            'text': best_text,
            'text_confidence': text_confidence,
            'fields': extracted_fields,
            'overall_confidence': overall_confidence,
            'extraction_methods_used': len(text_results)
        }

# Usage example
def demonstrate_accuracy_improvements():
    """Demonstrate the accuracy improvements"""
    processor = AdvancedAccuracyProcessor()
    
    # Load a sample image
    # image = cv2.imread('sample_document.jpg')
    # result = processor.process_document_advanced(image)
    # print(f"Extraction confidence: {result['overall_confidence']:.2f}")
    # print(f"Fields extracted: {len(result['fields'])}")
    
    print("Advanced accuracy processor initialized with:")
    print(f"- {len(processor.ocr_configs)} OCR configurations")
    print(f"- 15+ image preprocessing techniques")
    print(f"- Advanced field validation")
    print(f"- Ensemble text extraction")

if __name__ == "__main__":
    demonstrate_accuracy_improvements()
