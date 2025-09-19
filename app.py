"""
Advanced AI/ML Document Information Extraction System
FastAPI Backend with comprehensive features and user-friendly interface
"""

import os
import json
import uuid
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from document_extractor import DocumentProcessor, ExtractedData

# Initialize FastAPI app
app = FastAPI(
    title="AI Document Extraction System",
    description="Advanced AI/ML system for extracting structured information from documents",
    version="2.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
processor = DocumentProcessor()
upload_dir = Path("uploads")
results_dir = Path("results")
processing_status = {}

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

class BatchProcessResponse(BaseModel):
    batch_id: str
    total_documents: int
    processed_documents: int
    failed_documents: int
    status: str
    results: List[DocumentResponse]

class ProcessingStatus(BaseModel):
    document_id: str
    status: str
    progress: int
    message: str

# Enhanced document processor with advanced features
class AdvancedDocumentProcessor(DocumentProcessor):
    def __init__(self):
        super().__init__()
        self.processing_history = []
        self.confidence_threshold = 0.7
    
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
        
        # Extract specific financial fields
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

# Initialize advanced processor
advanced_processor = AdvancedDocumentProcessor()

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main frontend"""
    return FileResponse("frontend/index.html")

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
        
        # Process document
        start_time = datetime.now()
        result = advanced_processor.extract_information(str(file_path))
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Validate results
        validation = advanced_processor.validate_extracted_data(result)
        
        # Create response
        response = DocumentResponse(
            id=doc_id,
            filename=file.filename,
            document_type=result.document_type or "unknown",
            confidence_score=result.confidence_score,
            extracted_data={
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
            processing_time=processing_time,
            status="completed"
        )
        
        # Save results
        results_file = results_dir / f"{doc_id}_results.json"
        with open(results_file, "w") as f:
            json.dump(response.dict(), f, indent=2)
        
        # Clean up uploaded file
        background_tasks.add_task(cleanup_file, file_path)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/batch", response_model=BatchProcessResponse)
async def upload_batch_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload and process multiple documents"""
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
            
            start_time = datetime.now()
            result = advanced_processor.extract_information(str(file_path))
            processing_time = (datetime.now() - start_time).total_seconds()
            
            validation = advanced_processor.validate_extracted_data(result)
            
            response = DocumentResponse(
                id=doc_id,
                filename=file.filename,
                document_type=result.document_type or "unknown",
                confidence_score=result.confidence_score,
                extracted_data={
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
                processing_time=processing_time,
                status="completed"
            )
            
            results.append(response)
            processed += 1
            
            # Save individual result
            results_file = results_dir / f"{doc_id}_results.json"
            with open(results_file, "w") as f:
                json.dump(response.dict(), f, indent=2)
            
            # Clean up
            background_tasks.add_task(cleanup_file, file_path)
            
        except Exception as e:
            failed += 1
            print(f"Error processing {file.filename}: {e}")
    
    # Save batch results
    batch_results = BatchProcessResponse(
        batch_id=batch_id,
        total_documents=len(files),
        processed_documents=processed,
        failed_documents=failed,
        status="completed",
        results=results
    )
    
    batch_file = results_dir / f"{batch_id}_batch.json"
    with open(batch_file, "w") as f:
        json.dump(batch_results.dict(), f, indent=2)
    
    return batch_results

@app.get("/results/{document_id}")
async def get_document_results(document_id: str):
    """Get results for a specific document"""
    results_file = results_dir / f"{document_id}_results.json"
    if not results_file.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    with open(results_file, "r") as f:
        return json.load(f)

@app.get("/results")
async def get_all_results():
    """Get all processed results"""
    results = []
    for file_path in results_dir.glob("*_results.json"):
        with open(file_path, "r") as f:
            results.append(json.load(f))
    return results

@app.get("/export/{document_id}")
async def export_document_results(document_id: str, format: str = "json"):
    """Export document results in various formats"""
    results_file = results_dir / f"{document_id}_results.json"
    if not results_file.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    with open(results_file, "r") as f:
        data = json.load(f)
    
    if format == "json":
        return data
    elif format == "csv":
        # Convert to CSV format
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(["Field", "Value", "Confidence"])
        
        # Write data
        extracted = data["extracted_data"]
        writer.writerow(["First Name", extracted.get("first_name", ""), data["confidence_score"]])
        writer.writerow(["Last Name", extracted.get("last_name", ""), data["confidence_score"]])
        writer.writerow(["Date of Birth", extracted.get("date_of_birth", ""), data["confidence_score"]])
        writer.writerow(["SSN", extracted.get("ssn", ""), data["confidence_score"]])
        writer.writerow(["Address", extracted.get("current_address", ""), data["confidence_score"]])
        
        return {"content": output.getvalue(), "content_type": "text/csv"}
    
    return data

@app.get("/stats")
async def get_processing_stats():
    """Get processing statistics"""
    total_docs = len(list(results_dir.glob("*_results.json")))
    total_batches = len(list(results_dir.glob("*_batch.json")))
    
    # Calculate average confidence
    confidences = []
    for file_path in results_dir.glob("*_results.json"):
        with open(file_path, "r") as f:
            data = json.load(f)
            confidences.append(data["confidence_score"])
    
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        "total_documents": total_docs,
        "total_batches": total_batches,
        "average_confidence": round(avg_confidence, 2),
        "document_types": get_document_type_distribution()
    }

def get_document_type_distribution():
    """Get distribution of document types"""
    types = {}
    for file_path in results_dir.glob("*_results.json"):
        with open(file_path, "r") as f:
            data = json.load(f)
            doc_type = data["document_type"]
            types[doc_type] = types.get(doc_type, 0) + 1
    return types

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
