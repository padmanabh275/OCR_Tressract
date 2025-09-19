# Advanced AI Document Extraction System - API Documentation

## Overview

The Advanced AI Document Extraction System provides a comprehensive REST API for extracting structured information from various document types. The system supports both single document processing and batch processing with real-time progress tracking.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production environments, consider implementing proper authentication mechanisms.

## API Endpoints

### 1. Upload Single Document

**Endpoint:** `POST /upload`

**Description:** Upload and process a single document

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** File upload

**Parameters:**
- `file` (required): The document file to process
  - Supported formats: PDF, JPG, JPEG, PNG, TIFF, TXT
  - Maximum file size: 50MB

**Response:**
```json
{
  "id": "uuid-string",
  "filename": "document.pdf",
  "document_type": "driver_license",
  "confidence_score": 0.85,
  "extracted_data": {
    "first_name": "John",
    "last_name": "Smith",
    "date_of_birth": "1985-03-15",
    "marriage_date": null,
    "birth_city": null,
    "ssn": "123-45-6789",
    "current_address": "123 Main Street",
    "financial_data": {
      "amounts_found": ["$75,000.00"],
      "tax_years": ["2023"]
    },
    "validation": {
      "is_valid": true,
      "warnings": [],
      "errors": [],
      "field_scores": {
        "ssn": 1.0,
        "date_of_birth": 1.0,
        "names": 1.0
      }
    }
  },
  "processing_time": 2.34,
  "status": "completed"
}
```

### 2. Upload Multiple Documents (Batch Processing)

**Endpoint:** `POST /upload/batch`

**Description:** Upload and process multiple documents in a single request

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** Multiple file uploads

**Parameters:**
- `files` (required): Array of document files to process
  - Supported formats: PDF, JPG, JPEG, PNG, TIFF, TXT
  - Maximum files per batch: 50
  - Maximum total size: 500MB

**Response:**
```json
{
  "batch_id": "uuid-string",
  "total_documents": 5,
  "processed_documents": 4,
  "failed_documents": 1,
  "status": "completed",
  "results": [
    {
      "id": "uuid-string",
      "filename": "document1.pdf",
      "document_type": "driver_license",
      "confidence_score": 0.85,
      "extracted_data": { /* ... */ },
      "processing_time": 2.34,
      "status": "completed"
    }
    // ... more results
  ]
}
```

### 3. Get Document Results

**Endpoint:** `GET /results/{document_id}`

**Description:** Retrieve results for a specific document

**Parameters:**
- `document_id` (path): The unique identifier of the document

**Response:**
```json
{
  "id": "uuid-string",
  "filename": "document.pdf",
  "document_type": "driver_license",
  "confidence_score": 0.85,
  "extracted_data": { /* ... */ },
  "processing_time": 2.34,
  "status": "completed"
}
```

### 4. Get All Results

**Endpoint:** `GET /results`

**Description:** Retrieve all processed document results

**Response:**
```json
[
  {
    "id": "uuid-string",
    "filename": "document1.pdf",
    "document_type": "driver_license",
    "confidence_score": 0.85,
    "extracted_data": { /* ... */ },
    "processing_time": 2.34,
    "status": "completed"
  }
  // ... more results
]
```

### 5. Export Document Results

**Endpoint:** `GET /export/{document_id}`

**Description:** Export document results in various formats

**Parameters:**
- `document_id` (path): The unique identifier of the document
- `format` (query): Export format (`json` or `csv`)

**Response:**
- **JSON format:** Same as document results
- **CSV format:** CSV data with headers

### 6. Get Processing Statistics

**Endpoint:** `GET /stats`

**Description:** Get system processing statistics

**Response:**
```json
{
  "total_documents": 150,
  "total_batches": 25,
  "average_confidence": 0.82,
  "document_types": {
    "driver_license": 45,
    "w2_form": 30,
    "birth_certificate": 25,
    "utility_bill": 20,
    "bank_statement": 15,
    "tax_return": 15
  }
}
```

## Document Types Supported

The system can automatically classify and process the following document types:

1. **Personal Identification**
   - Driver's License
   - Passport
   - Birth Certificate

2. **Social Security Documents**
   - SSN-containing documents

3. **Proof of Address**
   - Utility Bills
   - Rental Agreements
   - Government-issued IDs

4. **Financial Information**
   - Tax Returns (Form 1040)
   - W-2 Forms
   - Bank Statements

## Extracted Fields

The system extracts the following structured information:

- **Personal Information**
  - First Name
  - Last Name
  - Date of Birth
  - Marriage Date
  - Birth City

- **Identification**
  - Social Security Number (SSN)
  - Current Address

- **Financial Data**
  - Monetary amounts
  - Tax years
  - Income-related keywords
  - Account balances

## Error Handling

The API uses standard HTTP status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Document or resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

Error responses include detailed error messages:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider implementing rate limiting to prevent abuse.

## File Size Limits

- **Single file:** 50MB maximum
- **Batch upload:** 500MB total maximum
- **Batch size:** 50 files maximum per batch

## Supported File Formats

- **PDF:** `.pdf`
- **Images:** `.jpg`, `.jpeg`, `.png`, `.tiff`
- **Text:** `.txt`

## Confidence Scoring

The system provides confidence scores (0.0 to 1.0) for extracted data:

- **0.8 - 1.0:** High confidence
- **0.5 - 0.8:** Medium confidence
- **0.0 - 0.5:** Low confidence

## Validation

The system includes built-in validation for:

- SSN format validation
- Date format validation
- Name field validation
- Address format validation

## Web Interface

The system includes a modern web interface accessible at:

- **Main Interface:** `http://localhost:8000`
- **API Documentation:** `http://localhost:8000/docs`
- **ReDoc Documentation:** `http://localhost:8000/redoc`

## Usage Examples

### cURL Examples

**Upload single document:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**Upload multiple documents:**
```bash
curl -X POST "http://localhost:8000/upload/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document1.pdf" \
  -F "files=@document2.jpg"
```

**Get document results:**
```bash
curl -X GET "http://localhost:8000/results/{document_id}"
```

**Export as CSV:**
```bash
curl -X GET "http://localhost:8000/export/{document_id}?format=csv"
```

### Python Examples

```python
import requests

# Upload single document
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload',
        files={'file': f}
    )
    result = response.json()

# Upload multiple documents
files = [
    ('files', open('doc1.pdf', 'rb')),
    ('files', open('doc2.jpg', 'rb'))
]
response = requests.post(
    'http://localhost:8000/upload/batch',
    files=files
)
batch_result = response.json()

# Get all results
response = requests.get('http://localhost:8000/results')
all_results = response.json()
```

## Performance Considerations

- **Processing time:** Typically 1-5 seconds per document
- **Concurrent processing:** Supports multiple simultaneous requests
- **Memory usage:** Optimized for large document processing
- **Storage:** Results are stored locally (consider cloud storage for production)

## Security Considerations

- **File validation:** Basic file type validation
- **Data privacy:** No permanent storage of sensitive data
- **CORS:** Configured for development (restrict in production)
- **Input sanitization:** Basic input validation

## Monitoring and Logging

The system provides:

- Processing statistics
- Error logging
- Performance metrics
- Document type distribution

## Future Enhancements

Planned features include:

- User authentication and authorization
- Cloud storage integration
- Advanced ML models
- Real-time processing status
- Webhook notifications
- API rate limiting
- Advanced analytics dashboard
