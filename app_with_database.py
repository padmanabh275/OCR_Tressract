"""
AI Document Extraction System with Database Integration
Enhanced version with SQLite database for storing extracted fields
"""

import os
import json
import uuid
import asyncio
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil
import numpy as np
import pytesseract

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from document_extractor import DocumentProcessor, ExtractedData
from improved_document_extractor import ImprovedDocumentProcessor
from database_setup import DocumentDatabase
from indian_document_enhancer import IndianDocumentEnhancer
from advanced_indian_accuracy import AdvancedIndianAccuracy
from unified_document_processor import UnifiedDocumentProcessor
from comprehensive_accuracy_system import ComprehensiveAccuracySystem

# Initialize FastAPI app
app = FastAPI(
    title="AI Document Extraction System with Database",
    description="Advanced AI/ML system with database storage for extracted information",
    version="2.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
processor = DocumentProcessor()
db = DocumentDatabase()
upload_dir = Path("uploads")
results_dir = Path("results")

# Initialize document processors
indian_enhancer = IndianDocumentEnhancer()
advanced_accuracy = AdvancedIndianAccuracy()
unified_processor = UnifiedDocumentProcessor()
comprehensive_accuracy = ComprehensiveAccuracySystem()

# Create directories
upload_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

# Pydantic models
class DocumentResponse(BaseModel):
    id: str
    filename: str
    document_type: str
    confidence_score: float
    extracted_data: Dict[str, Any]
    processing_time: float
    status: str
    upload_timestamp: str

class DocumentSearchRequest(BaseModel):
    field_name: str
    field_value: str

class DocumentSearchResponse(BaseModel):
    documents: List[Dict[str, Any]]
    total_results: int

class DatabaseStatsResponse(BaseModel):
    total_documents: int
    document_types: Dict[str, int]
    average_confidence: float
    recent_uploads: int

class EnhancedAccuracyRequest(BaseModel):
    accuracy_mode: str = "balanced_accuracy"  # fast_accuracy, balanced_accuracy, maximum_accuracy
    document_type: Optional[str] = None

class EnhancedAccuracyResponse(BaseModel):
    id: str
    filename: str
    document_type: str
    confidence_score: float
    accuracy_boost: float
    techniques_applied: List[str]
    processing_time: float
    quality_assessment: Dict[str, Any]
    ml_enhancements: Dict[str, Any]
    extracted_data: Dict[str, Any]
    status: str
    upload_timestamp: str

# Enhanced document processor with database integration
class AdvancedDocumentProcessor(DocumentProcessor):
    def __init__(self):
        super().__init__()
        self.db = DocumentDatabase()
    
    def extract_with_confidence(self, text: str, field: str) -> tuple:
        """Extract field with confidence score"""
        if field == "names":
            first, last = self.extract_names(text)
            confidence = 0.9 if first and last else 0.3
            return (first, last), confidence
        elif field == "ssn":
            ssn = self.extract_ssn(text)
            confidence = 0.95 if ssn else 0.0
            return ssn, confidence
        elif field == "dates":
            dates = self.extract_dates(text)
            confidence = 0.8 if dates else 0.0
            return dates, confidence
        elif field == "address":
            address = self.extract_address(text)
            confidence = 0.7 if address else 0.0
            return address, confidence
        return None, 0.0
    
    def extract_financial_data_advanced(self, text: str) -> Dict[str, Any]:
        """Advanced financial data extraction"""
        financial_data = super().extract_financial_data(text)
        
        patterns = {
            'wages': r'wages[:\s]*\$?([\d,]+\.?\d*)',
            'tax_withheld': r'tax\s+withheld[:\s]*\$?([\d,]+\.?\d*)',
            'gross_income': r'gross[:\s]*\$?([\d,]+\.?\d*)',
            'net_income': r'net[:\s]*\$?([\d,]+\.?\d*)',
            'account_balance': r'balance[:\s]*\$?([\d,]+\.?\d*)'
        }
        
        for field, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                financial_data[field] = matches[0]
        
        return financial_data
    
    def validate_extracted_data(self, data: ExtractedData) -> Dict[str, Any]:
        """Validate and score extracted data"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'field_scores': {}
        }
        
        # Validate SSN format
        if data.ssn:
            if not re.match(r'^\d{3}-\d{2}-\d{4}$', data.ssn):
                validation_results['warnings'].append("SSN format may be incorrect")
                validation_results['field_scores']['ssn'] = 0.5
            else:
                validation_results['field_scores']['ssn'] = 1.0
        
        # Validate date formats
        if data.date_of_birth:
            try:
                datetime.strptime(data.date_of_birth, '%Y-%m-%d')
                validation_results['field_scores']['date_of_birth'] = 1.0
            except:
                validation_results['warnings'].append("Date of birth format may be incorrect")
                validation_results['field_scores']['date_of_birth'] = 0.5
        
        # Validate names
        if data.first_name and data.last_name:
            if len(data.first_name) < 2 or len(data.last_name) < 2:
                validation_results['warnings'].append("Name fields seem too short")
                validation_results['field_scores']['names'] = 0.7
            else:
                validation_results['field_scores']['names'] = 1.0
        
        return validation_results

# Initialize improved processor
advanced_processor = ImprovedDocumentProcessor()

# Enhanced processor with Indian document accuracy
class IndianDocumentProcessor(ImprovedDocumentProcessor):
    """Enhanced processor with Indian document accuracy improvements"""
    
    def __init__(self):
        super().__init__()
        self.indian_enhancer = IndianDocumentEnhancer()
        self.advanced_accuracy = AdvancedIndianAccuracy()
    
    def extract_information(self, file_path: str) -> Optional[ExtractedData]:
        """Extract information with Indian document enhancements"""
        try:
            import cv2
            from pathlib import Path
            
            file_path = Path(file_path)
            
            # Load image
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
                print(f"Could not load image: {file_path}")
                return None
            
            # First try Indian document enhancement
            indian_document_types = ['pan_card', 'aadhaar_card', 'driving_license', 'voter_id', 'passport']
            
            # Quick classification to check if it's an Indian document
            text = self.extract_text_basic(image)
            is_indian_document = self.is_indian_document(text)
            
            if is_indian_document:
                print(f"🇮🇳 Indian document detected, using enhanced processing...")
                
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
                
                print(f"✅ Indian document processed: {indian_result.document_type}")
                print(f"✅ Confidence: {indian_result.confidence_score:.2f}")
                print(f"✅ Fields extracted: {list(indian_fields.keys())}")
                
                return extracted
            
            # Fall back to standard processing for non-Indian documents
            print("📄 Using standard processing for non-Indian document...")
            return super().extract_information(str(file_path))
            
        except Exception as e:
            print(f"Error in Indian document processing: {e}")
            # Fall back to standard processing
            return super().extract_information(str(file_path))
    
    def extract_text_basic(self, image):
        """Extract text using basic OCR"""
        try:
            import pytesseract
            text = pytesseract.image_to_string(image, config='--psm 6')
            return text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def is_indian_document(self, text: str) -> bool:
        """Check if document is likely an Indian document"""
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
        
        # Check for Indian patterns
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
        return indicator_count >= 2 or pattern_count >= 1

# Initialize the unified processor
enhanced_processor = unified_processor

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main frontend"""
    response = FileResponse("frontend/session_frontend.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/database", response_class=HTMLResponse)
async def database_viewer():
    """Serve the database viewer frontend"""
    response = FileResponse("frontend/database_viewer.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process a single document"""
    try:
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = upload_dir / f"{doc_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Process document with unified processor
        start_time = datetime.now()
        processing_result = enhanced_processor.process_document(str(file_path))
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Extract result from processing result
        result = processing_result.extracted_data
        
        # Use validation from processing result
        validation = processing_result.validation_results
        
        # Create response data
        response_data = {
            "id": doc_id,
            "filename": file.filename,
            "original_filename": file.filename,
            "file_path": str(file_path),
            "file_size": file_size,
            "document_type": result.document_type or "unknown",
            "document_category": processing_result.document_category,
            "confidence_score": result.confidence_score,
            "processing_method": processing_result.processing_method,
            "extracted_data": {
                "first_name": result.first_name,
                "last_name": result.last_name,
                "date_of_birth": result.date_of_birth,
                "marriage_date": result.marriage_date,
                "birth_city": result.birth_city,
                "ssn": result.ssn,
                "current_address": result.current_address,
                "financial_data": result.financial_data,
                "validation": validation
            },
            "processing_time": processing_time,
            "status": "completed",
            "metadata": processing_result.metadata
        }
        
        # Save to database
        db.save_document(response_data)
        
        # Create response
        response = DocumentResponse(
            id=doc_id,
            filename=file.filename,
            document_type=result.document_type or "unknown",
            confidence_score=result.confidence_score,
            extracted_data=response_data["extracted_data"],
            processing_time=processing_time,
            status="completed",
            upload_timestamp=datetime.now().isoformat()
        )
        
        # Clean up uploaded file
        background_tasks.add_task(cleanup_file, file_path)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/batch")
async def upload_batch_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload and process multiple documents in a single request"""
    batch_id = str(uuid.uuid4())
    results = []
    processed = 0
    failed = 0
    
    for file in files:
        try:
            # Process each file
            doc_id = str(uuid.uuid4())
            file_path = upload_dir / f"{doc_id}_{file.filename}"
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Get file size
            file_size = file_path.stat().st_size
            
            start_time = datetime.now()
            processing_result = enhanced_processor.process_document(str(file_path))
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Extract result from processing result
            result = processing_result.extracted_data
            
            # Handle case where extraction returns None
            if result is None:
                # Create a default ExtractedData object
                from document_extractor import ExtractedData
                result = ExtractedData(
                    document_type='unknown',
                    confidence_score=0.0,
                    raw_text='No text extracted'
                )
            
            validation = processing_result.validation_results
            
            # Create response data
            response_data = {
                "id": doc_id,
                "filename": file.filename,
                "original_filename": file.filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "document_type": result.document_type or "unknown",
                "confidence_score": result.confidence_score or 0.0,
                "extracted_data": {
                    "first_name": result.first_name,
                    "last_name": result.last_name,
                    "date_of_birth": result.date_of_birth,
                    "marriage_date": result.marriage_date,
                    "birth_city": result.birth_city,
                    "ssn": result.ssn,
                    "current_address": result.current_address,
                    "financial_data": result.financial_data,
                    "validation": validation
                },
                "processing_time": processing_time,
                "status": "completed"
            }
            
            # Save to database
            db.save_document(response_data)
            
            response = DocumentResponse(
                id=doc_id,
                filename=file.filename,
                document_type=result.document_type or "unknown",
                confidence_score=result.confidence_score,
                extracted_data=response_data["extracted_data"],
                processing_time=processing_time,
                status="completed",
                upload_timestamp=datetime.now().isoformat()
            )
            
            results.append(response)
            processed += 1
            
            # Clean up
            background_tasks.add_task(cleanup_file, file_path)
            
        except Exception as e:
            failed += 1
            print(f"Error processing {file.filename}: {e}")
    
    # Return batch results
    return {
        "batch_id": batch_id,
        "total_documents": len(files),
        "processed_documents": processed,
        "failed_documents": failed,
        "status": "completed",
        "results": results
    }

@app.get("/results")
async def get_all_results():
    """Get all processed results (for compatibility)"""
    try:
        documents = db.get_all_documents(100)
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_all_documents(limit: int = Query(100, ge=1, le=1000)):
    """Get all documents from database"""
    try:
        documents = db.get_all_documents(limit)
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}", response_model=Dict[str, Any])
async def get_document(document_id: str):
    """Get specific document with all extracted fields"""
    try:
        document = db.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document by ID"""
    try:
        success = db.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": "Document deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=DocumentSearchResponse)
async def search_documents(request: DocumentSearchRequest):
    """Search documents by field value"""
    try:
        results = db.search_documents(request.field_name, request.field_value)
        return DocumentSearchResponse(
            documents=results,
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=DatabaseStatsResponse)
async def get_database_stats():
    """Get database statistics"""
    try:
        stats = db.get_statistics()
        return DatabaseStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export/{document_id}")
async def export_document(document_id: str, format: str = "json"):
    """Export document in various formats"""
    try:
        document = db.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if format == "json":
            return document
        elif format == "csv":
            # Convert to CSV format
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write headers
            writer.writerow(["Field", "Value", "Confidence"])
            
            # Write extracted fields
            for field, value in document.get('extracted_fields', {}).items():
                writer.writerow([field, value, document.get('confidence_score', 0)])
            
            return {"content": output.getvalue(), "content_type": "text/csv"}
        
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/indian")
async def process_indian_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_type: str = Query("auto", description="Document type: pan_card, aadhaar_card, driving_license, voter_id, passport, or auto")
):
    """Process Indian documents with enhanced accuracy"""
    try:
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = upload_dir / f"{doc_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Process with Indian document enhancer
        start_time = datetime.now()
        
        # Load image
        import cv2
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
            raise HTTPException(status_code=400, detail="Could not load image")
        
        # Process with Indian enhancer
        if document_type == "auto":
            indian_result = indian_enhancer.enhance_indian_document(image, None)
        else:
            indian_result = indian_enhancer.enhance_indian_document(image, document_type)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Convert to ExtractedData format
        extracted = ExtractedData()
        extracted.raw_text = indian_result.raw_text if hasattr(indian_result, 'raw_text') else ""
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
        
        # Validate results
        validation = advanced_processor.validate_extracted_data(extracted)
        
        # Create response data
        response_data = {
            "id": doc_id,
            "filename": file.filename,
            "original_filename": file.filename,
            "file_path": str(file_path),
            "file_size": file_size,
            "document_type": indian_result.document_type,
            "confidence_score": indian_result.confidence_score,
            "extracted_data": {
                "first_name": extracted.first_name,
                "last_name": extracted.last_name,
                "date_of_birth": extracted.date_of_birth,
                "marriage_date": extracted.marriage_date,
                "birth_city": extracted.birth_city,
                "ssn": extracted.ssn,
                "current_address": extracted.current_address,
                "financial_data": extracted.financial_data,
                "validation": validation
            },
            "processing_time": processing_time,
            "status": "completed",
            "indian_enhancement": True
        }
        
        # Save to database
        db.save_document(response_data)
        
        # Create response
        response = DocumentResponse(
            id=doc_id,
            filename=file.filename,
            document_type=indian_result.document_type,
            confidence_score=indian_result.confidence_score,
            extracted_data=response_data["extracted_data"],
            processing_time=processing_time,
            status="completed",
            upload_timestamp=datetime.now().isoformat()
        )
        
        # Clean up uploaded file
        background_tasks.add_task(cleanup_file, file_path)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indian/stats")
async def get_indian_document_stats():
    """Get statistics for Indian documents"""
    try:
        documents = db.get_all_documents(1000)
        
        indian_docs = [doc for doc in documents if doc.get('extracted_data', {}).get('financial_data', {}).get('document_type') in 
                      ['pan_card', 'aadhaar_card', 'driving_license', 'voter_id', 'passport']]
        
        stats = {
            "total_indian_documents": len(indian_docs),
            "document_types": {},
            "average_confidence": 0.0,
            "recent_uploads": len([doc for doc in indian_docs if doc.get('upload_timestamp')])
        }
        
        if indian_docs:
            # Count by document type
            for doc in indian_docs:
                doc_type = doc.get('extracted_data', {}).get('financial_data', {}).get('document_type', 'unknown')
                stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1
            
            # Average confidence
            confidences = [doc.get('confidence_score', 0) for doc in indian_docs]
            stats["average_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indian/documents")
async def get_indian_documents(limit: int = Query(100, ge=1, le=1000)):
    """Get all Indian documents from database"""
    try:
        documents = db.get_all_documents(limit)
        
        indian_docs = [doc for doc in documents if doc.get('extracted_data', {}).get('financial_data', {}).get('document_type') in 
                      ['pan_card', 'aadhaar_card', 'driving_license', 'voter_id', 'passport']]
        
        return indian_docs
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/by-category")
async def get_documents_by_category(category: str = Query(..., description="Document category: indian, international, unknown")):
    """Get documents by category"""
    try:
        documents = db.get_all_documents(1000)
        
        filtered_docs = [doc for doc in documents if doc.get('document_category') == category]
        
        return {
            "category": category,
            "total_documents": len(filtered_docs),
            "documents": filtered_docs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/by-method")
async def get_documents_by_method(method: str = Query(..., description="Processing method: indian_enhanced, standard_enhanced, standard")):
    """Get documents by processing method"""
    try:
        documents = db.get_all_documents(1000)
        
        filtered_docs = [doc for doc in documents if doc.get('processing_method') == method]
        
        return {
            "method": method,
            "total_documents": len(filtered_docs),
            "documents": filtered_docs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/processing/stats")
async def get_processing_stats():
    """Get comprehensive processing statistics"""
    try:
        documents = db.get_all_documents(1000)
        
        stats = {
            "total_documents": len(documents),
            "by_category": {},
            "by_method": {},
            "by_type": {},
            "average_confidence": 0.0,
            "processing_times": {
                "indian_enhanced": [],
                "standard_enhanced": [],
                "standard": []
            }
        }
        
        if documents:
            # Count by category
            for doc in documents:
                category = doc.get('document_category', 'unknown')
                method = doc.get('processing_method', 'unknown')
                doc_type = doc.get('document_type', 'unknown')
                
                stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
                stats['by_method'][method] = stats['by_method'].get(method, 0) + 1
                stats['by_type'][doc_type] = stats['by_type'].get(doc_type, 0) + 1
                
                # Collect processing times
                if method in stats['processing_times']:
                    stats['processing_times'][method].append(doc.get('processing_time', 0))
            
            # Calculate average confidence
            confidences = [doc.get('confidence_score', 0) for doc in documents]
            stats['average_confidence'] = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Calculate average processing times
            for method, times in stats['processing_times'].items():
                if times:
                    stats['processing_times'][method] = {
                        'average': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times),
                        'count': len(times)
                    }
                else:
                    stats['processing_times'][method] = {'average': 0, 'min': 0, 'max': 0, 'count': 0}
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/enhanced", response_model=EnhancedAccuracyResponse)
async def upload_enhanced_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    accuracy_mode: str = Query("balanced_accuracy", description="Accuracy mode: fast_accuracy, balanced_accuracy, maximum_accuracy"),
    document_type: str = Query("auto", description="Document type: auto, pan_card, aadhaar_card, driving_license, voter_id, passport, or any other type")
):
    """Upload and process document with comprehensive accuracy enhancements"""
    try:
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = upload_dir / f"{doc_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Load image
        import cv2
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
            raise HTTPException(status_code=400, detail="Could not load image")
        
        # Process with comprehensive accuracy system
        start_time = datetime.now()
        
        # Determine document type if auto
        if document_type == "auto":
            # Quick classification
            text = pytesseract.image_to_string(image, config='--psm 6')
            if any(indicator in text.upper() for indicator in ['GOVT OF INDIA', 'INCOME TAX', 'AADHAAR', 'DRIVING LICENCE']):
                document_type = "indian_document"
            else:
                document_type = "international_document"
        
        # Apply comprehensive accuracy enhancements
        accuracy_result = comprehensive_accuracy.enhance_document_accuracy(
            image, document_type, accuracy_mode
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create response data
        response_data = {
            "id": doc_id,
            "filename": file.filename,
            "original_filename": file.filename,
            "file_path": str(file_path),
            "file_size": file_size,
            "document_type": document_type,
            "confidence_score": accuracy_result.confidence_score,
            "accuracy_boost": accuracy_result.accuracy_boost,
            "techniques_applied": accuracy_result.techniques_applied,
            "processing_time": processing_time,
            "quality_assessment": accuracy_result.quality_assessment,
            "ml_enhancements": accuracy_result.ml_enhancements,
            "extracted_data": {
                "first_name": accuracy_result.extracted_fields.get('name', '').split()[0] if accuracy_result.extracted_fields.get('name') else None,
                "last_name": ' '.join(accuracy_result.extracted_fields.get('name', '').split()[1:]) if accuracy_result.extracted_fields.get('name') and len(accuracy_result.extracted_fields.get('name', '').split()) > 1 else None,
                "date_of_birth": accuracy_result.extracted_fields.get('date_of_birth'),
                "marriage_date": accuracy_result.extracted_fields.get('marriage_date'),
                "birth_city": accuracy_result.extracted_fields.get('birth_city'),
                "ssn": (accuracy_result.extracted_fields.get('pan') or 
                       accuracy_result.extracted_fields.get('aadhaar') or 
                       accuracy_result.extracted_fields.get('license_no') or 
                       accuracy_result.extracted_fields.get('epic_no') or 
                       accuracy_result.extracted_fields.get('passport_no')),
                "current_address": accuracy_result.extracted_fields.get('address'),
                "financial_data": {
                    'enhanced_fields': accuracy_result.extracted_fields,
                    'quality_assessment': accuracy_result.quality_assessment,
                    'ml_enhancements': accuracy_result.ml_enhancements,
                    'techniques_applied': accuracy_result.techniques_applied
                }
            },
            "status": "completed",
            "accuracy_mode": accuracy_mode
        }
        
        # Save to database
        db.save_document(response_data)
        
        # Create response
        response = EnhancedAccuracyResponse(
            id=doc_id,
            filename=file.filename,
            document_type=document_type,
            confidence_score=accuracy_result.confidence_score,
            accuracy_boost=accuracy_result.accuracy_boost,
            techniques_applied=accuracy_result.techniques_applied,
            processing_time=processing_time,
            quality_assessment=accuracy_result.quality_assessment,
            ml_enhancements=accuracy_result.ml_enhancements,
            extracted_data=response_data["extracted_data"],
            status="completed",
            upload_timestamp=datetime.now().isoformat()
        )
        
        # Clean up uploaded file
        background_tasks.add_task(cleanup_file, file_path)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/enhanced/batch")
async def upload_enhanced_batch_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    accuracy_mode: str = Query("balanced_accuracy", description="Accuracy mode: fast_accuracy, balanced_accuracy, maximum_accuracy")
):
    """Upload and process multiple documents with comprehensive accuracy enhancements"""
    batch_id = str(uuid.uuid4())
    results = []
    processed = 0
    failed = 0
    
    for file in files:
        try:
            # Process each file
            doc_id = str(uuid.uuid4())
            file_path = upload_dir / f"{doc_id}_{file.filename}"
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Get file size
            file_size = file_path.stat().st_size
            
            # Load image
            import cv2
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
                failed += 1
                continue
            
            # Quick classification
            text = pytesseract.image_to_string(image, config='--psm 6')
            if any(indicator in text.upper() for indicator in ['GOVT OF INDIA', 'INCOME TAX', 'AADHAAR', 'DRIVING LICENCE']):
                document_type = "indian_document"
            else:
                document_type = "international_document"
            
            start_time = datetime.now()
            accuracy_result = comprehensive_accuracy.enhance_document_accuracy(
                image, document_type, accuracy_mode
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create response data
            response_data = {
                "id": doc_id,
                "filename": file.filename,
                "original_filename": file.filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "document_type": document_type,
                "confidence_score": accuracy_result.confidence_score,
                "accuracy_boost": accuracy_result.accuracy_boost,
                "techniques_applied": accuracy_result.techniques_applied,
                "processing_time": processing_time,
                "quality_assessment": accuracy_result.quality_assessment,
                "ml_enhancements": accuracy_result.ml_enhancements,
                "extracted_data": {
                    "first_name": accuracy_result.extracted_fields.get('name', '').split()[0] if accuracy_result.extracted_fields.get('name') else None,
                    "last_name": ' '.join(accuracy_result.extracted_fields.get('name', '').split()[1:]) if accuracy_result.extracted_fields.get('name') and len(accuracy_result.extracted_fields.get('name', '').split()) > 1 else None,
                    "date_of_birth": accuracy_result.extracted_fields.get('date_of_birth'),
                    "marriage_date": accuracy_result.extracted_fields.get('marriage_date'),
                    "birth_city": accuracy_result.extracted_fields.get('birth_city'),
                    "ssn": (accuracy_result.extracted_fields.get('pan') or 
                           accuracy_result.extracted_fields.get('aadhaar') or 
                           accuracy_result.extracted_fields.get('license_no') or 
                           accuracy_result.extracted_fields.get('epic_no') or 
                           accuracy_result.extracted_fields.get('passport_no')),
                    "current_address": accuracy_result.extracted_fields.get('address'),
                    "financial_data": {
                        'enhanced_fields': accuracy_result.extracted_fields,
                        'quality_assessment': accuracy_result.quality_assessment,
                        'ml_enhancements': accuracy_result.ml_enhancements,
                        'techniques_applied': accuracy_result.techniques_applied
                    }
                },
                "status": "completed",
                "accuracy_mode": accuracy_mode
            }
            
            # Save to database
            db.save_document(response_data)
            
            response = EnhancedAccuracyResponse(
                id=doc_id,
                filename=file.filename,
                document_type=document_type,
                confidence_score=accuracy_result.confidence_score,
                accuracy_boost=accuracy_result.accuracy_boost,
                techniques_applied=accuracy_result.techniques_applied,
                processing_time=processing_time,
                quality_assessment=accuracy_result.quality_assessment,
                ml_enhancements=accuracy_result.ml_enhancements,
                extracted_data=response_data["extracted_data"],
                status="completed",
                upload_timestamp=datetime.now().isoformat()
            )
            
            results.append(response)
            processed += 1
            
            # Clean up
            background_tasks.add_task(cleanup_file, file_path)
            
        except Exception as e:
            failed += 1
            print(f"Error processing {file.filename}: {e}")
    
    # Return batch results
    return {
        "batch_id": batch_id,
        "total_documents": len(files),
        "processed_documents": processed,
        "failed_documents": failed,
        "status": "completed",
        "accuracy_mode": accuracy_mode,
        "results": results
    }

@app.get("/accuracy/modes")
async def get_accuracy_modes():
    """Get available accuracy modes and their descriptions"""
    return {
        "accuracy_modes": {
            "fast_accuracy": {
                "description": "Fast processing with essential enhancements",
                "processing_time": "1-3 seconds",
                "accuracy_boost": "10-20%",
                "use_cases": "High-volume processing, real-time applications"
            },
            "balanced_accuracy": {
                "description": "Balanced processing with core enhancements",
                "processing_time": "4-6 seconds",
                "accuracy_boost": "20-30%",
                "use_cases": "General document processing, production use"
            },
            "maximum_accuracy": {
                "description": "Maximum accuracy with all enhancements",
                "processing_time": "8-12 seconds",
                "accuracy_boost": "30-50%",
                "use_cases": "Critical documents, high-value processing"
            }
        }
    }

@app.get("/accuracy/stats")
async def get_accuracy_stats():
    """Get accuracy enhancement statistics"""
    try:
        documents = db.get_all_documents(1000)
        
        # Filter enhanced documents
        enhanced_docs = [doc for doc in documents if doc.get('accuracy_mode')]
        
        stats = {
            "total_enhanced_documents": len(enhanced_docs),
            "accuracy_modes_used": {},
            "average_accuracy_boost": 0.0,
            "techniques_applied": {},
            "quality_assessments": {
                "excellent": 0,
                "good": 0,
                "fair": 0,
                "poor": 0
            }
        }
        
        if enhanced_docs:
            # Count by accuracy mode
            for doc in enhanced_docs:
                mode = doc.get('accuracy_mode', 'unknown')
                stats["accuracy_modes_used"][mode] = stats["accuracy_modes_used"].get(mode, 0) + 1
            
            # Calculate average accuracy boost
            boosts = [doc.get('accuracy_boost', 0) for doc in enhanced_docs if doc.get('accuracy_boost')]
            if boosts:
                stats["average_accuracy_boost"] = sum(boosts) / len(boosts)
            
            # Count techniques applied
            for doc in enhanced_docs:
                techniques = doc.get('techniques_applied', [])
                for technique in techniques:
                    stats["techniques_applied"][technique] = stats["techniques_applied"].get(technique, 0) + 1
            
            # Count quality assessments
            for doc in enhanced_docs:
                quality = doc.get('quality_assessment', {}).get('overall_quality', 'unknown')
                if quality in stats["quality_assessments"]:
                    stats["quality_assessments"][quality] += 1
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/dual")
async def upload_dual_processing_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process document with dual processing (Indian + Standard) and return the best result"""
    try:
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = upload_dir / f"{doc_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Process document with dual processing
        start_time = datetime.now()
        dual_result = unified_processor.process_document_dual(str(file_path))
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Extract the best result
        best_result = dual_result.best_result
        
        # Create response data
        response_data = {
            "id": doc_id,
            "filename": file.filename,
            "original_filename": file.filename,
            "file_path": str(file_path),
            "file_size": file_size,
            "document_type": best_result.document_type or "unknown",
            "document_category": best_result.document_category or "unknown",
            "confidence_score": best_result.confidence_score or 0.0,
            "processing_method": best_result.processing_method or "unknown",
            "extracted_data": {
                "first_name": best_result.extracted_data.first_name,
                "last_name": best_result.extracted_data.last_name,
                "date_of_birth": best_result.extracted_data.date_of_birth,
                "marriage_date": best_result.extracted_data.marriage_date,
                "birth_city": best_result.extracted_data.birth_city,
                "ssn": best_result.extracted_data.ssn,
                "current_address": best_result.extracted_data.current_address,
                "financial_data": best_result.extracted_data.financial_data,
                "validation": best_result.validation_results
            },
            "processing_time": processing_time,
            "status": "completed",
            "dual_processing": {
                "indian_confidence": dual_result.confidence_comparison.get('indian', 0.0),
                "standard_confidence": dual_result.confidence_comparison.get('standard', 0.0),
                "confidence_difference": dual_result.processing_summary.get('confidence_difference', 0.0),
                "indian_processing_time": dual_result.processing_summary.get('indian_processing_time', 0.0),
                "standard_processing_time": dual_result.processing_summary.get('standard_processing_time', 0.0),
                "best_method": dual_result.processing_summary.get('best_method', 'unknown'),
                "total_processing_time": dual_result.processing_summary.get('total_processing_time', 0.0)
            },
            "metadata": best_result.metadata
        }
        
        # Save to database
        db.save_document(response_data)
        
        # Create response
        response = DocumentResponse(
            id=doc_id,
            filename=file.filename,
            document_type=best_result.document_type,
            confidence_score=best_result.confidence_score,
            extracted_data=response_data["extracted_data"],
            processing_time=processing_time,
            status="completed",
            upload_timestamp=datetime.now().isoformat()
        )
        
        # Clean up uploaded file
        background_tasks.add_task(cleanup_file, file_path)
        
        # Add dual processing details to response
        response_dict = response.dict()
        response_dict["dual_processing"] = response_data["dual_processing"]
        
        return response_dict
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def cleanup_file(file_path: Path):
    """Clean up uploaded file after processing"""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Error cleaning up file {file_path}: {e}")

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
