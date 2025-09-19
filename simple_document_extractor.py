py"""
Simplified AI/ML Document Information Extraction System
Works without spaCy to avoid dependency conflicts
"""

import re
import cv2
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from dateutil import parser
import json

# Optional imports - will work without these
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Pandas not available - some features may be limited")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available - using basic text processing")

@dataclass
class ExtractedData:
    """Data class to store extracted information"""
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

class SimpleDocumentProcessor:
    """Simplified document processor without spaCy dependency"""
    
    def __init__(self):
        self.ssn_pattern = re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b')
        self.date_patterns = [
            re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            re.compile(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'),
            re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE)
        ]
        self.name_patterns = [
            re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'),
            re.compile(r'\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b'),
            re.compile(r'Name:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', re.IGNORECASE),
            re.compile(r'Employee:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', re.IGNORECASE)
        ]
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for better OCR results"""
        if image_path.lower().endswith('.pdf'):
            return self._process_pdf(image_path)
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _process_pdf(self, pdf_path: str) -> np.ndarray:
        """Extract text from PDF and create image representation"""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # Create a simple text image for processing
        img = np.ones((800, 1200), dtype=np.uint8) * 255
        cv2.putText(img, text[:1000], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        return img
    
    def extract_text_ocr(self, image: np.ndarray) -> str:
        """Extract text using OCR"""
        try:
            text = pytesseract.image_to_string(image, config='--psm 6')
            return text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def extract_text_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF directly"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"PDF processing error: {e}")
            return ""
    
    def extract_dates(self, text: str) -> List[str]:
        """Extract dates from text"""
        dates = []
        for pattern in self.date_patterns:
            matches = pattern.findall(text)
            dates.extend(matches)
        
        # Validate and format dates
        valid_dates = []
        for date_str in dates:
            try:
                parsed_date = parser.parse(date_str, fuzzy=True)
                valid_dates.append(parsed_date.strftime("%Y-%m-%d"))
            except:
                continue
        
        return valid_dates
    
    def extract_names(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract first and last names from text using regex patterns"""
        for pattern in self.name_patterns:
            matches = pattern.findall(text)
            if matches:
                full_name = matches[0].strip()
                name_parts = full_name.split()
                if len(name_parts) >= 2:
                    return name_parts[0], " ".join(name_parts[1:])
        
        return None, None
    
    def extract_ssn(self, text: str) -> Optional[str]:
        """Extract SSN from text"""
        matches = self.ssn_pattern.findall(text)
        if matches:
            ssn = matches[0].replace("-", "")
            if len(ssn) == 9 and ssn.isdigit():
                return f"{ssn[:3]}-{ssn[3:5]}-{ssn[5:]}"
        return None
    
    def extract_address(self, text: str) -> Optional[str]:
        """Extract address from text using regex patterns"""
        # Look for address patterns
        address_patterns = [
            r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Place|Pl)',
            r'Address:\s*([^\n]+)',
            r'Residence:\s*([^\n]+)'
        ]
        
        for pattern in address_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        return None
    
    def extract_financial_data(self, text: str) -> Dict[str, Any]:
        """Extract financial information from text"""
        financial_data = {}
        
        # Look for monetary amounts
        money_pattern = re.compile(r'\$[\d,]+\.?\d*')
        amounts = money_pattern.findall(text)
        if amounts:
            financial_data['amounts_found'] = amounts
        
        # Look for tax year information
        tax_year_pattern = re.compile(r'(?:tax\s+year|year)\s*:?\s*(\d{4})', re.IGNORECASE)
        tax_years = tax_year_pattern.findall(text)
        if tax_years:
            financial_data['tax_years'] = tax_years
        
        # Look for income-related keywords
        income_keywords = ['wages', 'salary', 'income', 'earnings', 'gross', 'net', 'taxable']
        found_keywords = []
        for keyword in income_keywords:
            if keyword.lower() in text.lower():
                found_keywords.append(keyword)
        
        if found_keywords:
            financial_data['income_keywords'] = found_keywords
        
        return financial_data
    
    def classify_document_type(self, text: str) -> str:
        """Classify the type of document based on content"""
        text_lower = text.lower()
        
        # Keywords for different document types
        doc_keywords = {
            'driver_license': ['driver', 'license', 'dmv', 'department of motor vehicles'],
            'passport': ['passport', 'passport number', 'issuing country'],
            'birth_certificate': ['birth', 'certificate', 'born', 'birthplace'],
            'ssn_document': ['social security', 'ssn', 'ss#'],
            'utility_bill': ['utility', 'electric', 'gas', 'water', 'bill', 'statement'],
            'rental_agreement': ['rental', 'lease', 'tenant', 'landlord'],
            'tax_return': ['tax return', 'form 1040', 'irs', 'federal tax'],
            'w2_form': ['w-2', 'w2', 'wage and tax statement'],
            'bank_statement': ['bank', 'account', 'statement', 'checking', 'savings']
        }
        
        scores = {}
        for doc_type, keywords in doc_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[doc_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        return 'unknown'
    
    def calculate_confidence_score(self, extracted_data: ExtractedData) -> float:
        """Calculate confidence score based on extracted fields"""
        score = 0.0
        total_fields = 8
        
        if extracted_data.first_name:
            score += 1.0
        if extracted_data.last_name:
            score += 1.0
        if extracted_data.date_of_birth:
            score += 1.0
        if extracted_data.ssn:
            score += 1.0
        if extracted_data.current_address:
            score += 1.0
        if extracted_data.document_type != 'unknown':
            score += 1.0
        if extracted_data.marriage_date:
            score += 0.5
        if extracted_data.birth_city:
            score += 0.5
        if extracted_data.financial_data:
            score += 1.0
        
        return min(score / total_fields, 1.0)
    
    def extract_information(self, file_path: str) -> ExtractedData:
        """Main method to extract information from a document"""
        print(f"Processing document: {file_path}")
        
        result = ExtractedData()
        
        try:
            # Extract text based on file type
            if file_path.lower().endswith('.pdf'):
                text = self.extract_text_pdf(file_path)
            else:
                # Process image
                image = self.preprocess_image(file_path)
                text = self.extract_text_ocr(image)
            
            if not text.strip():
                print("No text extracted from document")
                return result
            
            print(f"Extracted text length: {len(text)} characters")
            
            # Extract various fields
            result.first_name, result.last_name = self.extract_names(text)
            
            # Extract dates and determine which is DOB and marriage date
            dates = self.extract_dates(text)
            if dates:
                dates.sort()
                if len(dates) >= 1:
                    result.date_of_birth = dates[0]
                if len(dates) >= 2:
                    result.marriage_date = dates[1]
            
            result.ssn = self.extract_ssn(text)
            result.current_address = self.extract_address(text)
            result.financial_data = self.extract_financial_data(text)
            result.document_type = self.classify_document_type(text)
            
            # Calculate confidence score
            result.confidence_score = self.calculate_confidence_score(result)
            
            print(f"Document type: {result.document_type}")
            print(f"Confidence score: {result.confidence_score:.2f}")
            
        except Exception as e:
            print(f"Error processing document: {e}")
        
        return result

def create_sample_documents():
    """Create sample documents for testing"""
    sample_dir = Path("sample_documents")
    sample_dir.mkdir(exist_ok=True)
    
    # Sample driver's license text
    driver_license_text = """
    DRIVER LICENSE
    STATE OF CALIFORNIA
    DEPARTMENT OF MOTOR VEHICLES
    
    Name: JOHN MICHAEL SMITH
    Address: 123 MAIN STREET
    City: LOS ANGELES, CA 90210
    Date of Birth: 03/15/1985
    License Number: D1234567
    Class: C
    Expires: 03/15/2025
    """
    
    # Sample W-2 form text
    w2_text = """
    FORM W-2
    WAGE AND TAX STATEMENT
    
    Employee: JANE DOE
    Social Security Number: 123-45-6789
    Address: 456 OAK AVENUE, SPRINGFIELD, IL 62701
    
    Wages, tips, other compensation: $75,000.00
    Federal income tax withheld: $12,500.00
    Social security wages: $75,000.00
    Social security tax withheld: $4,650.00
    Medicare wages and tips: $75,000.00
    Medicare tax withheld: $1,087.50
    
    Tax Year: 2023
    """
    
    # Sample birth certificate text
    birth_cert_text = """
    CERTIFICATE OF BIRTH
    STATE OF TEXAS
    
    Child's Name: ROBERT JAMES WILSON
    Date of Birth: 07/22/1990
    Place of Birth: HOUSTON, TEXAS
    Father's Name: MICHAEL WILSON
    Mother's Name: SARAH WILSON
    Marriage Date: 05/15/1988
    """
    
    # Save sample documents as text files
    with open(sample_dir / "driver_license.txt", "w") as f:
        f.write(driver_license_text)
    
    with open(sample_dir / "w2_form.txt", "w") as f:
        f.write(w2_text)
    
    with open(sample_dir / "birth_certificate.txt", "w") as f:
        f.write(birth_cert_text)
    
    print("Sample documents created in 'sample_documents' directory")

def main():
    """Main function to demonstrate the document extraction system"""
    print("Simplified AI/ML Document Information Extraction System")
    print("=" * 60)
    
    # Create sample documents
    create_sample_documents()
    
    # Initialize processor
    processor = SimpleDocumentProcessor()
    
    # Process sample documents
    sample_dir = Path("sample_documents")
    results = []
    
    for file_path in sample_dir.glob("*.txt"):
        print(f"\nProcessing: {file_path.name}")
        result = processor.extract_information(str(file_path))
        results.append(result)
        
        # Display results
        print(f"First Name: {result.first_name}")
        print(f"Last Name: {result.last_name}")
        print(f"Date of Birth: {result.date_of_birth}")
        print(f"Marriage Date: {result.marriage_date}")
        print(f"SSN: {result.ssn}")
        print(f"Current Address: {result.current_address}")
        print(f"Document Type: {result.document_type}")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        if result.financial_data:
            print(f"Financial Data: {result.financial_data}")
        print("-" * 30)
    
    # Save results to JSON
    results_data = []
    for i, result in enumerate(results):
        result_dict = {
            "document_id": i + 1,
            "first_name": result.first_name,
            "last_name": result.last_name,
            "date_of_birth": result.date_of_birth,
            "marriage_date": result.marriage_date,
            "birth_city": result.birth_city,
            "ssn": result.ssn,
            "current_address": result.current_address,
            "financial_data": result.financial_data,
            "document_type": result.document_type,
            "confidence_score": result.confidence_score
        }
        results_data.append(result_dict)
    
    with open("extraction_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to 'extraction_results.json'")
    print(f"Processed {len(results)} documents")

if __name__ == "__main__":
    main()
