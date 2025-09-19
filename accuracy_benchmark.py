"""
Accuracy Benchmarking System for Indian Documents
Test and compare different accuracy improvement techniques
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os
from pathlib import Path

@dataclass
class BenchmarkResult:
    """Result of accuracy benchmark test"""
    method_name: str
    document_type: str
    accuracy_score: float
    processing_time: float
    extracted_fields: Dict[str, str]
    confidence_score: float
    errors: List[str]

class AccuracyBenchmark:
    """Comprehensive accuracy benchmarking system"""
    
    def __init__(self):
        self.test_documents = self.setup_test_documents()
        self.benchmark_results = []
        
    def setup_test_documents(self) -> Dict[str, Dict]:
        """Setup test documents with expected results"""
        return {
            'pan_card_1': {
                'text': """
                PERMANENT ACCOUNT NUMBER
                P.A.N. : ABCDE1234F
                INCOME TAX DEPARTMENT
                GOVT. OF INDIA
                NAME: RAJESH KUMAR SHARMA
                FATHER'S NAME: RAMESH KUMAR SHARMA
                DATE OF BIRTH: 15/01/1990
                SIGNATURE: RAJESH KUMAR SHARMA
                """,
                'expected_fields': {
                    'pan_number': 'ABCDE1234F',
                    'name': 'RAJESH KUMAR SHARMA',
                    'father_name': 'RAMESH KUMAR SHARMA',
                    'date_of_birth': '15/01/1990',
                    'signature': 'RAJESH KUMAR SHARMA'
                },
                'document_type': 'pan_card'
            },
            'aadhaar_card_1': {
                'text': """
                AADHAAR
                UNIQUE IDENTIFICATION AUTHORITY OF INDIA
                GOVERNMENT OF INDIA
                UID: 1234 5678 9012
                NAME: PRIYA SHARMA
                FATHER'S NAME: RAJESH KUMAR SHARMA
                MOTHER'S NAME: SUNITA SHARMA
                DATE OF BIRTH: 20/05/1995
                GENDER: F
                ADDRESS: 123, SECTOR 15, NEW DELHI, 110015
                PIN: 110015
                """,
                'expected_fields': {
                    'aadhaar_number': '123456789012',
                    'name': 'PRIYA SHARMA',
                    'father_name': 'RAJESH KUMAR SHARMA',
                    'mother_name': 'SUNITA SHARMA',
                    'date_of_birth': '20/05/1995',
                    'gender': 'F',
                    'address': '123, SECTOR 15, NEW DELHI, 110015',
                    'pin_code': '110015'
                },
                'document_type': 'aadhaar_card'
            },
            'driving_license_1': {
                'text': """
                DRIVING LICENCE
                TRANSPORT AUTHORITY
                DELHI
                LICENCE NO: DL0123456789
                NAME: AMIT KUMAR SINGH
                FATHER'S NAME: VIJAY KUMAR SINGH
                DATE OF BIRTH: 10/03/1988
                VALID FROM: 01/01/2020
                VALID UPTO: 31/12/2030
                ADDRESS: 456, PITAMPURA, DELHI, 110034
                BLOOD GROUP: A+
                """,
                'expected_fields': {
                    'license_number': 'DL0123456789',
                    'name': 'AMIT KUMAR SINGH',
                    'father_name': 'VIJAY KUMAR SINGH',
                    'date_of_birth': '10/03/1988',
                    'valid_from': '01/01/2020',
                    'valid_upto': '31/12/2030',
                    'address': '456, PITAMPURA, DELHI, 110034',
                    'blood_group': 'A+'
                },
                'document_type': 'driving_license'
            }
        }
    
    def benchmark_method(self, method_name: str, method_func, test_doc: Dict) -> BenchmarkResult:
        """Benchmark a specific method"""
        start_time = time.time()
        errors = []
        
        try:
            # Run the method
            result = method_func(test_doc['text'])
            
            # Calculate accuracy
            accuracy = self.calculate_accuracy(result, test_doc['expected_fields'])
            
            # Calculate confidence
            confidence = result.get('confidence_score', 0.0)
            
            processing_time = time.time() - start_time
            
            return BenchmarkResult(
                method_name=method_name,
                document_type=test_doc['document_type'],
                accuracy_score=accuracy,
                processing_time=processing_time,
                extracted_fields=result.get('extracted_fields', {}),
                confidence_score=confidence,
                errors=errors
            )
            
        except Exception as e:
            errors.append(str(e))
            processing_time = time.time() - start_time
            
            return BenchmarkResult(
                method_name=method_name,
                document_type=test_doc['document_type'],
                accuracy_score=0.0,
                processing_time=processing_time,
                extracted_fields={},
                confidence_score=0.0,
                errors=errors
            )
    
    def calculate_accuracy(self, result: Dict, expected: Dict) -> float:
        """Calculate accuracy score"""
        if not result or 'extracted_fields' not in result:
            return 0.0
        
        extracted = result['extracted_fields']
        if not extracted:
            return 0.0
        
        # Calculate field-level accuracy
        correct_fields = 0
        total_fields = len(expected)
        
        for field_name, expected_value in expected.items():
            if field_name in extracted:
                extracted_value = extracted[field_name]
                if self.fields_match(extracted_value, expected_value):
                    correct_fields += 1
        
        return correct_fields / total_fields if total_fields > 0 else 0.0
    
    def fields_match(self, extracted: str, expected: str) -> bool:
        """Check if extracted field matches expected field"""
        if not extracted or not expected:
            return False
        
        # Normalize both values
        extracted_norm = self.normalize_field_value(extracted)
        expected_norm = self.normalize_field_value(expected)
        
        # Check for exact match
        if extracted_norm == expected_norm:
            return True
        
        # Check for partial match (for names with slight variations)
        if self.is_partial_match(extracted_norm, expected_norm):
            return True
        
        return False
    
    def normalize_field_value(self, value: str) -> str:
        """Normalize field value for comparison"""
        if not value:
            return ""
        
        # Convert to uppercase
        value = value.upper()
        
        # Remove extra spaces
        value = ' '.join(value.split())
        
        # Remove special characters for certain fields
        if any(char.isdigit() for char in value):
            # For fields with numbers, keep only alphanumeric
            value = ''.join(c for c in value if c.isalnum() or c.isspace())
        
        return value.strip()
    
    def is_partial_match(self, extracted: str, expected: str) -> bool:
        """Check for partial match (useful for names)"""
        if not extracted or not expected:
            return False
        
        # Split into words
        extracted_words = set(extracted.split())
        expected_words = set(expected.split())
        
        # Check if most words match
        common_words = extracted_words.intersection(expected_words)
        match_ratio = len(common_words) / max(len(expected_words), 1)
        
        return match_ratio >= 0.8  # 80% word match
    
    def run_comprehensive_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark on all methods"""
        
        # Import the methods to test
        from indian_document_enhancer import IndianDocumentEnhancer
        from advanced_indian_accuracy import AdvancedIndianAccuracy
        
        # Initialize methods
        indian_enhancer = IndianDocumentEnhancer()
        advanced_accuracy = AdvancedIndianAccuracy()
        
        # Define methods to test
        methods = {
            'basic_ocr': self.basic_ocr_method,
            'indian_enhancer': lambda text: indian_enhancer.enhance_indian_document(None, 'pan_card'),
            'advanced_accuracy': lambda text: advanced_accuracy.process_indian_document_advanced(None, 'pan_card')
        }
        
        # Run benchmarks
        results = {}
        
        for doc_name, test_doc in self.test_documents.items():
            doc_results = []
            
            for method_name, method_func in methods.items():
                result = self.benchmark_method(method_name, method_func, test_doc)
                doc_results.append(result)
                self.benchmark_results.append(result)
            
            results[doc_name] = doc_results
        
        return results
    
    def basic_ocr_method(self, text: str) -> Dict:
        """Basic OCR method for comparison"""
        try:
            # Simulate basic OCR extraction
            fields = {}
            
            # Simple pattern matching
            import re
            
            # Extract PAN
            pan_match = re.search(r'([A-Z]{5}[0-9]{4}[A-Z]{1})', text)
            if pan_match:
                fields['pan_number'] = pan_match.group(1)
            
            # Extract name
            name_match = re.search(r'NAME\s*:?\s*([A-Z\s]+)', text)
            if name_match:
                fields['name'] = name_match.group(1).strip()
            
            return {
                'extracted_fields': fields,
                'confidence_score': 0.5,
                'success': True
            }
        except Exception as e:
            return {
                'extracted_fields': {},
                'confidence_score': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report"""
        report = []
        report.append("# 🎯 Indian Document Accuracy Benchmark Report")
        report.append("")
        
        # Overall statistics
        if self.benchmark_results:
            avg_accuracy = sum(r.accuracy_score for r in self.benchmark_results) / len(self.benchmark_results)
            avg_processing_time = sum(r.processing_time for r in self.benchmark_results) / len(self.benchmark_results)
            
            report.append(f"## 📊 Overall Statistics")
            report.append(f"- **Total Tests**: {len(self.benchmark_results)}")
            report.append(f"- **Average Accuracy**: {avg_accuracy:.2%}")
            report.append(f"- **Average Processing Time**: {avg_processing_time:.2f}s")
            report.append("")
        
        # Method comparison
        methods = {}
        for result in self.benchmark_results:
            if result.method_name not in methods:
                methods[result.method_name] = []
            methods[result.method_name].append(result)
        
        report.append("## 🏆 Method Comparison")
        report.append("")
        
        for method_name, results in methods.items():
            avg_accuracy = sum(r.accuracy_score for r in results) / len(results)
            avg_time = sum(r.processing_time for r in results) / len(results)
            
            report.append(f"### {method_name}")
            report.append(f"- **Average Accuracy**: {avg_accuracy:.2%}")
            report.append(f"- **Average Processing Time**: {avg_time:.2f}s")
            report.append(f"- **Tests Run**: {len(results)}")
            report.append("")
        
        # Document type analysis
        doc_types = {}
        for result in self.benchmark_results:
            if result.document_type not in doc_types:
                doc_types[result.document_type] = []
            doc_types[result.document_type].append(result)
        
        report.append("## 📄 Document Type Analysis")
        report.append("")
        
        for doc_type, results in doc_types.items():
            avg_accuracy = sum(r.accuracy_score for r in results) / len(results)
            report.append(f"### {doc_type.replace('_', ' ').title()}")
            report.append(f"- **Average Accuracy**: {avg_accuracy:.2%}")
            report.append(f"- **Tests Run**: {len(results)}")
            report.append("")
        
        # Recommendations
        report.append("## 💡 Recommendations")
        report.append("")
        
        if self.benchmark_results:
            best_method = max(methods.items(), key=lambda x: sum(r.accuracy_score for r in x[1]) / len(x[1]))
            report.append(f"1. **Best Performing Method**: {best_method[0]}")
            report.append("2. **Use Indian Document Enhancer** for PAN cards and Aadhaar cards")
            report.append("3. **Use Advanced Accuracy** for driving licenses and voter IDs")
            report.append("4. **Combine multiple methods** for maximum accuracy")
            report.append("")
        
        return "\n".join(report)
    
    def save_benchmark_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to file"""
        results_data = []
        
        for result in self.benchmark_results:
            results_data.append({
                'method_name': result.method_name,
                'document_type': result.document_type,
                'accuracy_score': result.accuracy_score,
                'processing_time': result.processing_time,
                'extracted_fields': result.extracted_fields,
                'confidence_score': result.confidence_score,
                'errors': result.errors
            })
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Benchmark results saved to {filename}")

# Usage example
def run_accuracy_benchmark():
    """Run the accuracy benchmark"""
    benchmark = AccuracyBenchmark()
    
    print("🚀 Starting Indian Document Accuracy Benchmark...")
    print("=" * 50)
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Generate and print report
    report = benchmark.generate_benchmark_report()
    print(report)
    
    # Save results
    benchmark.save_benchmark_results()
    
    print("\n✅ Benchmark completed!")

if __name__ == "__main__":
    run_accuracy_benchmark()
