# API Documentation
## Document Information Extraction System

---

## 🌐 **Base URL**
```
http://localhost:8001
```

---

## 📋 **Authentication**
Currently, the API does not require authentication. For production deployment, consider implementing API keys or OAuth2.

---

## 📊 **Response Format**
All API responses follow a consistent JSON format:

```json
{
  "id": "uuid",
  "filename": "document.pdf",
  "document_type": "passport",
  "confidence_score": 0.95,
  "processing_time": 5.2,
  "status": "completed",
  "extracted_data": { ... },
  "metadata": { ... }
}
```

---

## 🔄 **Document Processing Endpoints**

### **1. Basic Document Processing**
```http
POST /upload
```

**Description**: Process a single document using standard OCR and field extraction.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**: File upload

**Response**: `DocumentResponse`

**Example**:
```bash
curl -X POST "http://localhost:8001/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

---

### **2. Dual Processing (Recommended)**
```http
POST /upload/dual
```

**Description**: Process document using both Indian and standard methods, returning the best result based on confidence.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**: File upload

**Response**: `DocumentResponse` with dual processing details

**Example**:
```bash
curl -X POST "http://localhost:8001/upload/dual" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**Response includes**:
```json
{
  "dual_processing": {
    "indian_confidence": 0.85,
    "standard_confidence": 0.92,
    "confidence_difference": 0.07,
    "best_method": "standard",
    "total_processing_time": 8.5
  }
}
```

---

### **3. Ollama LLM Enhanced Processing**
```http
POST /upload/ollama
```

**Description**: Process document using Ollama LLM for enhanced text understanding and field extraction.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**: File upload

**Response**: `OllamaProcessingResponse`

**Example**:
```bash
curl -X POST "http://localhost:8001/upload/ollama" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**Response includes**:
```json
{
  "extraction_method": "ollama_llm",
  "model_used": "llama3.2:latest",
  "text_enhancement": {
    "original_length": 500,
    "enhanced_length": 520,
    "improvement_ratio": 1.04
  },
  "classification": {
    "document_type": "passport",
    "confidence": 0.95,
    "country": "US",
    "is_indian_document": false
  },
  "validation": {
    "is_valid": true,
    "confidence": 0.95,
    "corrections": {},
    "missing_fields": []
  }
}
```

---

### **4. Enhanced Accuracy Processing**
```http
POST /upload/enhanced
```

**Description**: Process document with comprehensive accuracy enhancements.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**: File upload
- **Query Parameters**:
  - `accuracy_mode`: `fast` | `balanced` | `maximum`
  - `document_type`: Document type hint

**Response**: `EnhancedAccuracyResponse`

**Example**:
```bash
curl -X POST "http://localhost:8001/upload/enhanced?accuracy_mode=maximum&document_type=passport" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

---

### **5. Batch Processing**
```http
POST /upload/batch
```

**Description**: Process multiple documents in batch.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**: Multiple file uploads

**Response**: Array of `DocumentResponse`

**Example**:
```bash
curl -X POST "http://localhost:8001/upload/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.png" \
  -F "files=@doc3.jpg"
```

---

## 🤖 **Ollama Integration Endpoints**

### **1. Check Ollama Status**
```http
GET /ollama/status
```

**Description**: Check if Ollama service is running and get current model information.

**Response**:
```json
{
  "status": "available",
  "model": "llama3.2:latest",
  "base_url": "http://localhost:11434",
  "message": "Ollama service is running and ready for processing"
}
```

**Example**:
```bash
curl -X GET "http://localhost:8001/ollama/status"
```

---

### **2. List Available Models**
```http
GET /ollama/models
```

**Description**: Get list of available Ollama models.

**Response**:
```json
{
  "status": "success",
  "models": [
    {
      "name": "llama3.2:latest",
      "size": "2019393189"
    },
    {
      "name": "smollm2:135m",
      "size": "270898672"
    }
  ],
  "current_model": "llama3.2:latest"
}
```

**Example**:
```bash
curl -X GET "http://localhost:8001/ollama/models"
```

---

## 📊 **Database Management Endpoints**

### **1. Get All Documents**
```http
GET /documents
```

**Description**: Retrieve all processed documents with optional filtering.

**Query Parameters**:
- `limit`: Number of documents to return (default: 100)
- `offset`: Number of documents to skip (default: 0)
- `document_type`: Filter by document type
- `min_confidence`: Minimum confidence score
- `max_confidence`: Maximum confidence score

**Response**: Array of document objects

**Example**:
```bash
curl -X GET "http://localhost:8001/documents?limit=10&min_confidence=0.8"
```

---

### **2. Get Specific Document**
```http
GET /documents/{document_id}
```

**Description**: Retrieve a specific document by ID.

**Path Parameters**:
- `document_id`: UUID of the document

**Response**: Document object

**Example**:
```bash
curl -X GET "http://localhost:8001/documents/123e4567-e89b-12d3-a456-426614174000"
```

---

### **3. Delete Document**
```http
DELETE /documents/{document_id}
```

**Description**: Delete a specific document from the database.

**Path Parameters**:
- `document_id`: UUID of the document

**Response**: Success message

**Example**:
```bash
curl -X DELETE "http://localhost:8001/documents/123e4567-e89b-12d3-a456-426614174000"
```

---

### **4. Database Statistics**
```http
GET /stats
```

**Description**: Get processing statistics and metrics.

**Response**:
```json
{
  "total_documents": 150,
  "high_confidence_documents": 120,
  "medium_confidence_documents": 25,
  "low_confidence_documents": 5,
  "average_confidence": 0.87,
  "processing_times": {
    "average": 4.2,
    "min": 1.1,
    "max": 12.5
  },
  "document_types": {
    "passport": 45,
    "driver_license": 30,
    "pan_card": 25,
    "other": 50
  }
}
```

**Example**:
```bash
curl -X GET "http://localhost:8001/stats"
```

---

## 🎯 **Accuracy and Quality Endpoints**

### **1. Get Accuracy Modes**
```http
GET /accuracy/modes
```

**Description**: Get available accuracy modes and their descriptions.

**Response**:
```json
{
  "modes": {
    "fast": {
      "description": "Quick processing with basic accuracy",
      "processing_time": "1-3 seconds",
      "accuracy": "85-90%"
    },
    "balanced": {
      "description": "Balanced processing with good accuracy",
      "processing_time": "4-6 seconds",
      "accuracy": "90-95%"
    },
    "maximum": {
      "description": "Maximum accuracy with advanced processing",
      "processing_time": "8-12 seconds",
      "accuracy": "95-98%"
    }
  }
}
```

---

### **2. Get Accuracy Statistics**
```http
GET /accuracy/stats
```

**Description**: Get accuracy-related statistics and metrics.

**Response**:
```json
{
  "overall_accuracy": 0.92,
  "accuracy_by_type": {
    "indian_documents": 0.96,
    "international_documents": 0.89,
    "ollama_enhanced": 0.94
  },
  "confidence_distribution": {
    "high": 80,
    "medium": 15,
    "low": 5
  },
  "processing_methods": {
    "indian_enhanced": 45,
    "standard_enhanced": 60,
    "ollama_llm": 30
  }
}
```

---

## 🌐 **Web Interface Endpoints**

### **1. Main Upload Interface**
```http
GET /
```

**Description**: Main upload interface for document processing.

**Response**: HTML page

---

### **2. Database Viewer**
```http
GET /database
```

**Description**: Database viewer interface with filtering and export capabilities.

**Response**: HTML page

---

### **3. API Documentation**
```http
GET /docs
```

**Description**: Interactive API documentation (Swagger UI).

**Response**: HTML page

---

## 📝 **Data Models**

### **DocumentResponse**
```json
{
  "id": "string (UUID)",
  "filename": "string",
  "document_type": "string",
  "confidence_score": "number (0.0-1.0)",
  "processing_time": "number (seconds)",
  "extracted_data": {
    "first_name": "string",
    "last_name": "string",
    "date_of_birth": "string (YYYY-MM-DD)",
    "marriage_date": "string (YYYY-MM-DD)",
    "birth_city": "string",
    "ssn": "string",
    "current_address": "string",
    "financial_data": "object",
    "validation": "object"
  },
  "status": "string",
  "upload_timestamp": "string (ISO 8601)"
}
```

### **OllamaProcessingResponse**
```json
{
  "id": "string (UUID)",
  "filename": "string",
  "document_type": "string",
  "confidence_score": "number (0.0-1.0)",
  "processing_time": "number (seconds)",
  "extraction_method": "string",
  "model_used": "string",
  "text_enhancement": {
    "original_length": "number",
    "enhanced_length": "number",
    "improvement_ratio": "number"
  },
  "classification": {
    "document_type": "string",
    "confidence": "number",
    "reasoning": "string",
    "country": "string",
    "is_indian_document": "boolean"
  },
  "extracted_data": "object",
  "validation": "object",
  "metadata": "object",
  "upload_timestamp": "string (ISO 8601)"
}
```

---

## ⚠️ **Error Handling**

### **Common Error Responses**

#### **400 Bad Request**
```json
{
  "detail": "Invalid file format. Supported formats: PDF, PNG, JPG, JPEG"
}
```

#### **413 Payload Too Large**
```json
{
  "detail": "File size exceeds maximum limit of 10MB"
}
```

#### **422 Unprocessable Entity**
```json
{
  "detail": "Unable to process document. Please ensure the document is clear and readable"
}
```

#### **500 Internal Server Error**
```json
{
  "detail": "An error occurred during processing. Please try again later"
}
```

#### **503 Service Unavailable**
```json
{
  "detail": "Ollama service not available. Please ensure Ollama is running"
}
```

---

## 🔧 **Rate Limiting**

Currently, there are no rate limits implemented. For production deployment, consider implementing:
- Request rate limiting per IP
- File size limits
- Processing time limits
- Concurrent request limits

---

## 📊 **Monitoring and Logging**

### **Health Check**
```http
GET /health
```

**Description**: Check system health and status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "2.1.0",
  "services": {
    "database": "connected",
    "ollama": "available",
    "ocr": "ready"
  }
}
```

---

## 🚀 **SDK Examples**

### **Python SDK**
```python
import requests

# Upload document
with open("document.pdf", "rb") as f:
    files = {"file": ("document.pdf", f, "application/pdf")}
    response = requests.post("http://localhost:8001/upload/dual", files=files)
    result = response.json()
    print(f"Confidence: {result['confidence_score']:.3f}")
```

### **JavaScript SDK**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8001/upload/dual', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log('Confidence:', data.confidence_score));
```

### **cURL Examples**
```bash
# Basic upload
curl -X POST "http://localhost:8001/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# Dual processing
curl -X POST "http://localhost:8001/upload/dual" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# Ollama processing
curl -X POST "http://localhost:8001/upload/ollama" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

---

## 📞 **Support**

For API support and questions:
- **Documentation**: Check this file and `/docs` endpoint
- **Issues**: Create an issue on GitHub
- **Email**: [Your contact information]

---

**Last Updated**: January 2024  
**API Version**: 2.1.0