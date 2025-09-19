"""
Improved AI/ML Document Information Extraction System
Enhanced with better OCR accuracy, multiple extraction methods, and confidence scoring
"""

import re
import cv2
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF
from dateutil import parser
import json

# Optional imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (OSError, ImportError):
    nlp = None
    SPACY_AVAILABLE = False

@dataclass
class ExtractedData:
    """Data class to store extracted information with confidence scores"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    marriage_date: Optional[str] = None
    birth_city: Optional[str] = None
    ssn: Optional[str] = None
    current_address: Optional[str] = None
    financial_data: Optional[Dict[str, Any]] = None
    document_type: Optional[str] = None
    confidence_score: float = 0.0
    extraction_method: Optional[str] = None
    raw_text: Optional[str] = None

class ImprovedDocumentProcessor:
    """Enhanced document processor with improved accuracy"""
    
    def __init__(self):
        self.setup_tesseract()
        
        # Enhanced regex patterns with better accuracy
        self.patterns = {
            'ssn': [
                r'\b\d{3}-?\d{2}-?\d{4}\b',  # Standard SSN format
                r'\b\d{9}\b',  # 9 consecutive digits
                r'SSN[:\s]*(\d{3}-?\d{2}-?\d{4})',  # SSN with label
            ],
            'date_of_birth': [
                r'(?:DOB|Date of Birth|Born)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(?:Birth|Born)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
            ],
            'name': [
                r'(?:Name|Full Name)[:\s]*([A-Za-z\s]+)',
                r'(?:First Name|Given Name)[:\s]*([A-Za-z]+)',
                r'(?:Last Name|Surname|Family Name)[:\s]*([A-Za-z]+)',
                r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)$',  # First Last format
            ],
            'address': [
                r'(?:Address|Residence)[:\s]*([A-Za-z0-9\s,.-]+)',
                r'\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)',
            ],
            'phone': [
                r'(?:Phone|Tel|Mobile)[:\s]*(\d{3}[-.]?\d{3}[-.]?\d{4})',
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            ]
        }
        
        # Document type keywords with weights
        self.doc_keywords = {
            'driver_license': {
                'keywords': ['driver', 'license', 'dmv', 'department of motor vehicles', 'dl', 'driving'],
                'weight': 1.0
            },
            'passport': {
                'keywords': ['passport', 'passport number', 'issuing country', 'passport no'],
                'weight': 1.0
            },
            'birth_certificate': {
                'keywords': ['birth', 'certificate', 'born', 'birthplace', 'birth certificate'],
                'weight': 1.0
            },
            'ssn_document': {
                'keywords': ['social security', 'ssn', 'ss#', 'social security number'],
                'weight': 1.0
            },
            'utility_bill': {
                'keywords': ['utility', 'electric', 'gas', 'water', 'bill', 'statement', 'power', 'energy'],
                'weight': 0.8
            },
            'rental_agreement': {
                'keywords': ['rental', 'lease', 'tenant', 'landlord', 'rent', 'agreement'],
                'weight': 0.8
            },
            'tax_return': {
                'keywords': ['tax return', 'form 1040', 'irs', 'federal tax', 'income tax'],
                'weight': 0.9
            },
            'w2_form': {
                'keywords': ['w-2', 'w2', 'wage and tax statement', 'employer'],
                'weight': 0.9
            },
            'bank_statement': {
                'keywords': ['bank', 'account', 'statement', 'checking', 'savings', 'financial'],
                'weight': 0.8
            }
        }

    def setup_tesseract(self):
        """Setup Tesseract with optimal configuration"""
        import os
        
        # Set Tesseract data path if not set
        if 'TESSDATA_PREFIX' not in os.environ:
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tessdata',
                r'C:\Program Files (x86)\Tesseract-OCR\tessdata',
                r'C:\Users\{}\AppData\Local\Tesseract-OCR\tessdata'.format(os.getenv('USERNAME', '')),
                r'C:\tesseract\tessdata',
                r'C:\Tesseract-OCR\tessdata',
                '/usr/share/tesseract-ocr/4.00/tessdata',
                '/usr/share/tesseract-ocr/5/tessdata',
                '/usr/local/share/tessdata'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    os.environ['TESSDATA_PREFIX'] = path
                    break

        # Set Tesseract command if not set
        if 'TESSERACT_CMD' not in os.environ:
            try:
                import shutil
                tesseract_cmd = shutil.which('tesseract')
                if tesseract_cmd:
                    os.environ['TESSERACT_CMD'] = tesseract_cmd
            except:
                pass

    def preprocess_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply multiple preprocessing techniques for better OCR accuracy"""
        processed_images = []
        
        # Original image
        processed_images.append(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_images.append(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        processed_images.append(blurred)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_images.append(thresh)
        
        # Apply morphological operations
        kernel = np.ones((1, 1), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed_images.append(morph)
        
        # Enhance contrast
        pil_image = Image.fromarray(gray)
        enhancer = ImageEnhance.Contrast(pil_image)
        contrast_img = enhancer.enhance(2.0)
        processed_images.append(np.array(contrast_img))
        
        # Apply sharpening
        sharp_img = pil_image.filter(ImageFilter.SHARPEN)
        processed_images.append(np.array(sharp_img))
        
        return processed_images

    def extract_text_multiple_methods(self, image: np.ndarray) -> Dict[str, str]:
        """Extract text using multiple OCR methods and configurations"""
        results = {}
        
        # Preprocess image
        processed_images = self.preprocess_image(image)
        
        # Different Tesseract configurations
        configs = [
            '--psm 6',  # Uniform block of text
            '--psm 3',  # Fully automatic page segmentation
            '--psm 4',  # Assume a single column of text
            '--psm 8',  # Treat the image as a single word
            '--psm 13', # Raw line. Treat the image as a single text line
        ]
        
        for i, processed_img in enumerate(processed_images):
            for j, config in enumerate(configs):
                try:
                    text = pytesseract.image_to_string(processed_img, config=config)
                    if text.strip():
                        results[f'method_{i}_{j}'] = text.strip()
                except Exception as e:
                    continue
        
        return results

    def extract_text_ocr(self, image: np.ndarray) -> str:
        """Extract text using OCR with multiple methods and return best result"""
        try:
            # Try multiple methods
            methods_results = self.extract_text_multiple_methods(image)
            
            if not methods_results:
                return ""
            
            # Score each method based on text length and content quality
            scored_results = []
            for method, text in methods_results.items():
                score = len(text)  # Basic scoring by length
                
                # Bonus for containing expected patterns
                for pattern_type, patterns in self.patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, text, re.IGNORECASE):
                            score += 10
                
                scored_results.append((score, text, method))
            
            # Return the best result
            if scored_results:
                scored_results.sort(key=lambda x: x[0], reverse=True)
                return scored_results[0][1]
            
            return ""
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    def extract_text_pdf(self, file_path: str) -> str:
        """Extract text from PDF with better handling"""
        try:
            doc = fitz.open(file_path)
            text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                text += page_text + "\n"
                
                # Also try to extract text from images in PDF
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_array = np.frombuffer(img_data, np.uint8)
                            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            
                            if img_cv is not None:
                                ocr_text = self.extract_text_ocr(img_cv)
                                if ocr_text:
                                    text += ocr_text + "\n"
                        pix = None
                    except Exception as e:
                        continue
            
            doc.close()
            return text.strip()
            
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""

    def classify_document_type(self, text: str) -> Tuple[str, float]:
        """Classify document type with confidence score"""
        text_lower = text.lower()
        
        scores = {}
        for doc_type, config in self.doc_keywords.items():
            score = 0
            for keyword in config['keywords']:
                if keyword in text_lower:
                    score += config['weight']
            scores[doc_type] = score
        
        if scores and max(scores.values()) > 0:
            best_type = max(scores, key=scores.get)
            confidence = min(scores[best_type] / 3.0, 1.0)  # Normalize to 0-1
            return best_type, confidence
        
        return 'unknown', 0.0

    def extract_with_regex(self, text: str, pattern_type: str) -> List[str]:
        """Extract data using regex patterns with validation"""
        if pattern_type not in self.patterns:
            return []
        
        results = []
        for pattern in self.patterns[pattern_type]:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]  # Get first group
                if match and match.strip():
                    results.append(match.strip())
        
        return list(set(results))  # Remove duplicates

    def validate_and_clean_data(self, data: str, data_type: str) -> Optional[str]:
        """Validate and clean extracted data"""
        if not data or not data.strip():
            return None
        
        data = data.strip()
        
        if data_type == 'ssn':
            # Remove non-digits and format
            digits = re.sub(r'\D', '', data)
            if len(digits) == 9:
                return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
        
        elif data_type in ['date_of_birth', 'marriage_date']:
            try:
                # Try to parse and reformat date
                parsed_date = parser.parse(data, fuzzy=True)
                return parsed_date.strftime('%Y-%m-%d')
            except:
                return data
        
        elif data_type in ['first_name', 'last_name']:
            # Clean name data
            cleaned = re.sub(r'[^A-Za-z\s-]', '', data)
            return cleaned.strip()
        
        return data

    def extract_information(self, file_path: str) -> Optional[ExtractedData]:
        """Extract information from document with improved accuracy"""
        try:
            file_path = Path(file_path)
            
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                text = self.extract_text_pdf(str(file_path))
            else:
                # Load and process image
                image = cv2.imread(str(file_path))
                if image is None:
                    return None
                text = self.extract_text_ocr(image)
            
            if not text.strip():
                return None
            
            # Classify document type
            doc_type, doc_confidence = self.classify_document_type(text)
            
            # Extract structured data
            extracted = ExtractedData()
            extracted.raw_text = text
            extracted.document_type = doc_type
            extracted.confidence_score = doc_confidence
            extracted.extraction_method = "improved_ocr"
            
            # Extract names
            name_matches = self.extract_with_regex(text, 'name')
            if name_matches:
                # Try to split first and last name
                for name_match in name_matches:
                    name_parts = name_match.split()
                    if len(name_parts) >= 2:
                        extracted.first_name = self.validate_and_clean_data(name_parts[0], 'first_name')
                        extracted.last_name = self.validate_and_clean_data(' '.join(name_parts[1:]), 'last_name')
                        break
                    elif len(name_parts) == 1 and not extracted.first_name:
                        extracted.first_name = self.validate_and_clean_data(name_parts[0], 'first_name')
            
            # Extract SSN
            ssn_matches = self.extract_with_regex(text, 'ssn')
            if ssn_matches:
                extracted.ssn = self.validate_and_clean_data(ssn_matches[0], 'ssn')
            
            # Extract dates
            dob_matches = self.extract_with_regex(text, 'date_of_birth')
            if dob_matches:
                extracted.date_of_birth = self.validate_and_clean_data(dob_matches[0], 'date_of_birth')
            
            # Extract address
            address_matches = self.extract_with_regex(text, 'address')
            if address_matches:
                extracted.current_address = address_matches[0]
            
            # Extract phone and email for additional context
            phone_matches = self.extract_with_regex(text, 'phone')
            email_matches = self.extract_with_regex(text, 'email')
            
            # Calculate overall confidence
            confidence_factors = [doc_confidence]
            if extracted.first_name: confidence_factors.append(0.8)
            if extracted.last_name: confidence_factors.append(0.8)
            if extracted.ssn: confidence_factors.append(0.9)
            if extracted.date_of_birth: confidence_factors.append(0.7)
            if extracted.current_address: confidence_factors.append(0.6)
            
            extracted.confidence_score = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
            
            return extracted
            
        except Exception as e:
            print(f"Error in extract_information: {e}")
            return None

    def validate_extracted_data(self, data: ExtractedData) -> Dict[str, Any]:
        """Validate extracted data and return validation results"""
        if not data:
            return {"valid": False, "errors": ["No data extracted"]}
        
        validation = {"valid": True, "errors": [], "warnings": []}
        
        # Validate SSN format
        if data.ssn:
            ssn_digits = re.sub(r'\D', '', data.ssn)
            if len(ssn_digits) != 9:
                validation["warnings"].append("SSN format may be incorrect")
        
        # Validate date format
        if data.date_of_birth:
            try:
                datetime.strptime(data.date_of_birth, '%Y-%m-%d')
            except:
                validation["warnings"].append("Date of birth format may be incorrect")
        
        # Check for required fields based on document type
        if data.document_type in ['driver_license', 'passport']:
            if not data.first_name or not data.last_name:
                validation["warnings"].append("Name information may be incomplete")
        
        if data.confidence_score < 0.3:
            validation["warnings"].append("Low confidence in extraction results")
        
        return validation
