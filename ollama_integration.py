"""
Ollama Integration for Enhanced Document Processing
Integrates local Ollama models for advanced text understanding and field extraction
"""

import requests
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OllamaResponse:
    """Response from Ollama API"""
    model: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

class OllamaDocumentProcessor:
    """Enhanced document processor using Ollama models"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:latest"):
        """
        Initialize Ollama document processor
        
        Args:
            base_url: Ollama API base URL
            model: Model name to use (llama3.2:latest, smollm2:135m, etc.)
        """
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/api"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama API"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if self.model in model_names:
                    logger.info(f"✅ Connected to Ollama. Using model: {self.model}")
                    return True
                else:
                    logger.warning(f"⚠️ Model {self.model} not found. Available: {model_names}")
                    return False
            else:
                logger.error(f"❌ Failed to connect to Ollama: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Error connecting to Ollama: {e}")
            return False
    
    def _call_ollama(self, prompt: str, system_prompt: str = None) -> OllamaResponse:
        """Call Ollama API with a prompt"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent extraction
                    "top_p": 0.9,
                    "max_tokens": 2000
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return OllamaResponse(
                    model=data.get('model', self.model),
                    response=data.get('response', ''),
                    done=data.get('done', False),
                    context=data.get('context'),
                    total_duration=data.get('total_duration'),
                    load_duration=data.get('load_duration'),
                    prompt_eval_count=data.get('prompt_eval_count'),
                    prompt_eval_duration=data.get('prompt_eval_duration'),
                    eval_count=data.get('eval_count'),
                    eval_duration=data.get('eval_duration')
                )
            else:
                logger.error(f"❌ Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error calling Ollama: {e}")
            return None
    
    def extract_fields_with_llm(self, text: str, document_type: str = "unknown") -> Dict[str, Any]:
        """
        Extract structured fields using Ollama LLM
        
        Args:
            text: Extracted text from document
            document_type: Type of document (passport, license, etc.)
            
        Returns:
            Dictionary of extracted fields with confidence scores
        """
        system_prompt = f"""You are an expert document information extraction system. 
        Extract structured information from the given document text.
        
        Document Type: {document_type}
        
        Extract the following fields if present:
        - first_name: Legal first name
        - last_name: Legal last name
        - full_name: Complete name
        - date_of_birth: Date of birth (format: YYYY-MM-DD)
        - marriage_date: Marriage date (format: YYYY-MM-DD)
        - birth_city: City of birth
        - ssn: Social Security Number
        - current_address: Complete current address
        - document_number: Document ID number
        - issue_date: Document issue date
        - expiry_date: Document expiry date
        - nationality: Nationality
        - gender: Gender
        - financial_data: Any financial information (last 3 years)
        
        Return ONLY a valid JSON object with the extracted fields.
        Use null for missing fields.
        Include confidence scores (0.0-1.0) for each field.
        """
        
        prompt = f"""Extract information from this document text:

TEXT:
{text}

Return the extracted information as a JSON object with this structure:
{{
    "personal_info": {{
        "first_name": "value",
        "last_name": "value", 
        "full_name": "value",
        "date_of_birth": "YYYY-MM-DD",
        "marriage_date": "YYYY-MM-DD",
        "birth_city": "value",
        "gender": "value",
        "nationality": "value"
    }},
    "identification": {{
        "ssn": "value",
        "document_number": "value",
        "document_type": "{document_type}"
    }},
    "address": {{
        "current_address": "complete address"
    }},
    "dates": {{
        "issue_date": "YYYY-MM-DD",
        "expiry_date": "YYYY-MM-DD"
    }},
    "financial_data": {{
        "amount": "value",
        "period": "value",
        "description": "value"
    }},
    "confidence_scores": {{
        "overall": 0.95,
        "personal_info": 0.90,
        "identification": 0.85,
        "address": 0.80,
        "financial_data": 0.75
    }}
}}"""
        
        logger.info(f"🤖 Calling Ollama model: {self.model}")
        response = self._call_ollama(prompt, system_prompt)
        
        if not response or not response.done:
            logger.error("❌ Failed to get response from Ollama")
            return self._create_fallback_response()
        
        try:
            # Extract JSON from response
            response_text = response.response.strip()
            
            # Find JSON object in response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                extracted_data = json.loads(json_str)
                logger.info("✅ Successfully extracted fields using Ollama")
                return extracted_data
            else:
                logger.warning("⚠️ No JSON found in Ollama response")
                return self._create_fallback_response()
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            logger.error(f"Raw response: {response.response}")
            return self._create_fallback_response()
    
    def classify_document_type_with_llm(self, text: str) -> Dict[str, Any]:
        """
        Classify document type using Ollama LLM
        
        Args:
            text: Extracted text from document
            
        Returns:
            Document classification with confidence
        """
        system_prompt = """You are an expert document classifier. 
        Classify the document type based on the text content.
        
        Possible types:
        - passport: International passport
        - driver_license: Driver's license
        - birth_certificate: Birth certificate
        - ssn_card: Social Security card
        - utility_bill: Utility bill
        - bank_statement: Bank statement
        - tax_return: Tax return
        - w2_form: W-2 form
        - pan_card: Indian PAN card
        - aadhaar_card: Indian Aadhaar card
        - indian_driving_license: Indian driving license
        - unknown: Cannot determine type
        
        Return ONLY a JSON object with classification results.
        """
        
        prompt = f"""Classify this document based on the text content:

TEXT:
{text[:1000]}...

Return classification as JSON:
{{
    "document_type": "type",
    "confidence": 0.95,
    "reasoning": "explanation",
    "country": "country_code",
    "is_indian_document": true/false
}}"""
        
        logger.info(f"🔍 Classifying document type using Ollama")
        response = self._call_ollama(prompt, system_prompt)
        
        if not response or not response.done:
            return {"document_type": "unknown", "confidence": 0.0}
        
        try:
            response_text = response.response.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                classification = json.loads(json_str)
                logger.info(f"✅ Document classified as: {classification.get('document_type', 'unknown')}")
                return classification
            else:
                return {"document_type": "unknown", "confidence": 0.0}
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ Classification JSON decode error: {e}")
            return {"document_type": "unknown", "confidence": 0.0}
    
    def enhance_text_with_llm(self, text: str) -> str:
        """
        Enhance and clean text using Ollama LLM
        
        Args:
            text: Raw extracted text
            
        Returns:
            Enhanced and cleaned text
        """
        system_prompt = """You are a text enhancement specialist.
        Clean and enhance OCR-extracted text while preserving all important information.
        
        Tasks:
        1. Fix common OCR errors
        2. Correct spacing and formatting
        3. Standardize date formats
        4. Fix name formatting
        5. Clean up addresses
        6. Preserve all numbers and codes exactly
        
        Return the enhanced text without any additional formatting or explanations.
        """
        
        prompt = f"""Enhance this OCR-extracted text:

ORIGINAL TEXT:
{text}

Return the enhanced text:"""
        
        logger.info("✨ Enhancing text with Ollama")
        response = self._call_ollama(prompt, system_prompt)
        
        if response and response.done:
            enhanced_text = response.response.strip()
            logger.info("✅ Text enhanced successfully")
            return enhanced_text
        else:
            logger.warning("⚠️ Text enhancement failed, returning original")
            return text
    
    def validate_extracted_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted data using Ollama LLM
        
        Args:
            extracted_data: Previously extracted data
            
        Returns:
            Validation results with corrections
        """
        system_prompt = """You are a data validation expert.
        Validate and correct extracted document information.
        
        Check for:
        1. Date format consistency (YYYY-MM-DD)
        2. Name formatting (proper case)
        3. Address completeness
        4. SSN format validation
        5. Logical consistency
        6. Missing required fields
        
        Return validation results as JSON.
        """
        
        prompt = f"""Validate this extracted document data:

EXTRACTED DATA:
{json.dumps(extracted_data, indent=2)}

Return validation as JSON:
{{
    "is_valid": true/false,
    "confidence": 0.95,
    "corrections": {{
        "field_name": "corrected_value"
    }},
    "missing_fields": ["field1", "field2"],
    "validation_notes": "explanation"
}}"""
        
        logger.info("🔍 Validating extracted data with Ollama")
        response = self._call_ollama(prompt, system_prompt)
        
        if response and response.done:
            try:
                response_text = response.response.strip()
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    validation = json.loads(json_str)
                    logger.info("✅ Data validation completed")
                    return validation
            except json.JSONDecodeError as e:
                logger.error(f"❌ Validation JSON decode error: {e}")
        
        return {"is_valid": False, "confidence": 0.0, "corrections": {}, "missing_fields": []}
    
    def _create_fallback_response(self) -> Dict[str, Any]:
        """Create fallback response when Ollama fails"""
        return {
            "personal_info": {
                "first_name": None,
                "last_name": None,
                "full_name": None,
                "date_of_birth": None,
                "marriage_date": None,
                "birth_city": None,
                "gender": None,
                "nationality": None
            },
            "identification": {
                "ssn": None,
                "document_number": None,
                "document_type": "unknown"
            },
            "address": {
                "current_address": None
            },
            "dates": {
                "issue_date": None,
                "expiry_date": None
            },
            "financial_data": {
                "amount": None,
                "period": None,
                "description": None
            },
            "confidence_scores": {
                "overall": 0.0,
                "personal_info": 0.0,
                "identification": 0.0,
                "address": 0.0,
                "financial_data": 0.0
            }
        }

class OllamaEnhancedProcessor:
    """Enhanced processor that combines traditional OCR with Ollama LLM"""
    
    def __init__(self, model: str = "llama3.2:latest"):
        """Initialize enhanced processor"""
        self.ollama_processor = OllamaDocumentProcessor(model=model)
        self.is_available = self.ollama_processor._test_connection()
        
        if self.is_available:
            logger.info("🚀 Ollama integration enabled")
        else:
            logger.warning("⚠️ Ollama not available, falling back to traditional methods")
    
    def process_document_with_llm(self, text: str, document_type: str = "unknown") -> Dict[str, Any]:
        """
        Process document using both traditional methods and Ollama LLM
        
        Args:
            text: Extracted text from document
            document_type: Type of document
            
        Returns:
            Enhanced extraction results
        """
        if not self.is_available:
            logger.warning("⚠️ Ollama not available, using fallback")
            return self.ollama_processor._create_fallback_response()
        
        try:
            # Step 1: Enhance text with LLM
            enhanced_text = self.ollama_processor.enhance_text_with_llm(text)
            
            # Step 2: Classify document type
            classification = self.ollama_processor.classify_document_type_with_llm(enhanced_text)
            
            # Step 3: Extract fields with LLM
            extracted_data = self.ollama_processor.extract_fields_with_llm(
                enhanced_text, 
                classification.get('document_type', document_type)
            )
            
            # Step 4: Validate extracted data
            validation = self.ollama_processor.validate_extracted_data(extracted_data)
            
            # Combine results
            result = {
                "extraction_method": "ollama_llm",
                "model_used": self.ollama_processor.model,
                "text_enhancement": {
                    "original_length": len(text),
                    "enhanced_length": len(enhanced_text),
                    "improvement_ratio": len(enhanced_text) / len(text) if text else 1.0
                },
                "classification": classification,
                "extracted_data": extracted_data,
                "validation": validation,
                "confidence_score": validation.get('confidence', 0.0)
            }
            
            logger.info(f"✅ Document processed with Ollama LLM (confidence: {result['confidence_score']:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in Ollama processing: {e}")
            return self.ollama_processor._create_fallback_response()

# Test function
def test_ollama_integration():
    """Test Ollama integration"""
    print("🧪 Testing Ollama Integration...")
    
    # Test with sample text
    sample_text = """
    INCOME TAX DEPARTMENT
    GOVT. OF INDIA
    PERMANENT ACCOUNT NUMBER
    P.A.N. : ABCDE1234F
    NAME: RAJESH KUMAR SHARMA
    FATHER'S NAME: RAMESH KUMAR SHARMA
    DATE OF BIRTH: 15/01/1990
    SIGNATURE: RAJESH KUMAR SHARMA
    """
    
    processor = OllamaEnhancedProcessor()
    
    if processor.is_available:
        result = processor.process_document_with_llm(sample_text, "pan_card")
        print(f"✅ Test completed. Confidence: {result['confidence_score']:.3f}")
        print(f"📄 Document Type: {result['classification'].get('document_type', 'unknown')}")
        print(f"👤 Name: {result['extracted_data']['personal_info'].get('full_name', 'N/A')}")
    else:
        print("❌ Ollama not available for testing")

if __name__ == "__main__":
    test_ollama_integration()
