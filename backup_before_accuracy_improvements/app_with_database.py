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

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from document_extractor import DocumentProcessor, ExtractedData
from improved_document_extractor import ImprovedDocumentProcessor
from database_setup import DocumentDatabase

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
        
        # Process document
        start_time = datetime.now()
        result = advanced_processor.extract_information(str(file_path))
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Validate results
        validation = advanced_processor.validate_extracted_data(result)
        
        # Create response data
        response_data = {
            "id": doc_id,
            "filename": file.filename,
            "original_filename": file.filename,
            "file_path": str(file_path),
            "file_size": file_size,
            "document_type": result.document_type or "unknown",
            "confidence_score": result.confidence_score,
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
            result = advanced_processor.extract_information(str(file_path))
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Handle case where extraction returns None
            if result is None:
                result = type('ExtractionResult', (), {
                    'document_type': 'unknown',
                    'confidence_score': 0.0,
                    'first_name': None,
                    'last_name': None,
                    'date_of_birth': None,
                    'marriage_date': None,
                    'birth_city': None,
                    'ssn': None,
                    'current_address': None,
                    'financial_data': None
                })()
            
            validation = advanced_processor.validate_extracted_data(result)
            
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

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete document from database"""
    try:
        # This would require implementing delete functionality in the database class
        # For now, just return success
        return {"message": "Document deletion not implemented yet"}
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
