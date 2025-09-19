# Project Structure
## Document Information Extraction System

---

## 📁 **Directory Overview**

```
GlobalTech/
├── 📁 Core System Files
├── 📁 Advanced Processing Modules
├── 📁 LLM Integration
├── 📁 Frontend Interface
├── 📁 Testing & Demo Scripts
├── 📁 Documentation
├── 📁 Configuration & Setup
└── 📁 Generated Files
```

---

## 🔧 **Core System Files**

### **Main Application**
- **`app_with_database.py`** - Main FastAPI application with all endpoints
- **`document_extractor.py`** - Core document processing logic
- **`database_setup.py`** - Database initialization and management
- **`requirements_torch_env.txt`** - Python dependencies for torch_env

### **Purpose**
These files form the backbone of the system, handling:
- API endpoints and routing
- Basic document processing
- Database operations
- Dependency management

---

## 🚀 **Advanced Processing Modules**

### **Unified Processing**
- **`unified_document_processor.py`** - Orchestrates dual processing (Indian + Standard)
- **`dual_processing_processor.py`** - Implements dual processing logic

### **Indian Document Specialization**
- **`indian_document_enhancer.py`** - Specialized processing for Indian documents
- **`advanced_indian_accuracy.py`** - Advanced accuracy improvements for Indian docs

### **Document Classification & Field Extraction**
- **`enhanced_document_classifier.py`** - Multi-layer document type recognition
- **`dynamic_field_extractor.py`** - Adaptive field extraction beyond predefined patterns

### **Accuracy Enhancement**
- **`comprehensive_accuracy_system.py`** - Integrates all accuracy improvements
- **`advanced_accuracy_enhancements.py`** - Image preprocessing techniques
- **`ml_accuracy_boosters.py`** - Machine learning-based corrections

### **Purpose**
These modules provide:
- Specialized processing for different document types
- Advanced accuracy improvements
- Dynamic field detection
- Multi-method processing with confidence comparison

---

## 🤖 **LLM Integration**

### **Ollama Integration**
- **`ollama_integration.py`** - Local LLM integration with Ollama
- **`test_ollama_integration.py`** - Test script for Ollama functionality

### **Purpose**
Provides:
- Local LLM processing for enhanced accuracy
- Text enhancement and error correction
- Intelligent document classification
- Advanced field extraction with context understanding

---

## 🎨 **Frontend Interface**

### **Web Interface**
- **`frontend/database_viewer.html`** - React-based database management interface

### **Features**
- Document upload and processing
- Database viewing with filtering
- JSON export functionality
- Real-time processing status
- Confidence-based filtering

---

## 🧪 **Testing & Demo Scripts**

### **Test Scripts**
- **`test_accuracy.py`** - Test general accuracy improvements
- **`test_indian_integration.py`** - Test Indian document integration
- **`test_unified_flow.py`** - Test unified processing flow
- **`test_dual_processing.py`** - Test dual processing functionality
- **`test_enhanced_endpoints.py`** - Test enhanced accuracy endpoints
- **`test_ollama_integration.py`** - Test Ollama LLM integration

### **Demo Scripts**
- **`document_extraction_demo.py`** - Comprehensive demonstration script
- **`setup_torch_env_ollama.py`** - Automated setup script

### **Debug Scripts**
- **`debug_confidence.py`** - Debug confidence calculation
- **`debug_step_by_step.py`** - Step-by-step debugging
- **`debug_indian_fields.py`** - Debug Indian field extraction
- **`debug_confidence_ranges.py`** - Debug confidence ranges

### **Purpose**
These scripts provide:
- Comprehensive testing of all features
- Demonstration of system capabilities
- Debugging tools for troubleshooting
- Automated setup and configuration

---

## 📚 **Documentation**

### **Main Documentation**
- **`README.md`** - Main project documentation
- **`TECHNICAL_WRITEUP.md`** - Technical implementation details
- **`API_DOCUMENTATION.md`** - Complete API reference
- **`INSTALLATION_GUIDE.md`** - Detailed installation instructions
- **`PROJECT_STRUCTURE.md`** - This file

### **Feature Guides**
- **`OLLAMA_INTEGRATION_GUIDE.md`** - Ollama setup and usage guide
- **`ACCURACY_IMPROVEMENT_GUIDE.md`** - Accuracy enhancement techniques
- **`INDIAN_DOCUMENT_ACCURACY_GUIDE.md`** - Indian document processing guide
- **`DYNAMIC_FIELD_EXTRACTION_GUIDE.md`** - Dynamic field extraction guide

### **Implementation Summaries**
- **`UNIFIED_PROCESSING_FLOW_SUMMARY.md`** - Unified processing flow overview
- **`COMPREHENSIVE_ACCURACY_IMPROVEMENTS.md`** - Accuracy improvements summary
- **`INDIAN_INTEGRATION_SUMMARY.md`** - Indian integration summary
- **`ENHANCED_ACCURACY_INTEGRATION_SUMMARY.md`** - Enhanced accuracy integration
- **`DUAL_PROCESSING_IMPLEMENTATION.md`** - Dual processing implementation

### **Purpose**
Comprehensive documentation covering:
- Installation and setup
- API usage and reference
- Technical implementation details
- Feature-specific guides
- Troubleshooting and optimization

---

## ⚙️ **Configuration & Setup**

### **Setup Scripts**
- **`setup_torch_env_ollama.py`** - Automated setup for torch_env
- **`start_torch_env_system.bat`** - Windows startup script
- **`refresh_database.py`** - Database refresh utility

### **Configuration Files**
- **`requirements_torch_env.txt`** - Python dependencies
- **`requirements.txt`** - Alternative requirements file

### **Purpose**
These files handle:
- Automated setup and configuration
- Environment management
- Database maintenance
- System startup

---

## 📊 **Generated Files (Runtime)**

### **Database Files**
- **`document_extractions.db`** - SQLite database (created at runtime)
- **`document_extractions_backup.db`** - Database backup (created during refresh)

### **Upload Directories**
- **`uploads/`** - Temporary file storage (created at runtime)
- **`results/`** - Processing results storage (created at runtime)

### **Log Files**
- **`*.log`** - Application logs (created at runtime)
- **`debug_*.txt`** - Debug output files (created during debugging)

### **Purpose**
These files are created during system operation:
- Database storage for extracted data
- Temporary file handling
- Logging and debugging output

---

## 🔄 **File Dependencies**

### **Core Dependencies**
```
app_with_database.py
├── document_extractor.py
├── database_setup.py
├── unified_document_processor.py
├── ollama_integration.py
└── frontend/database_viewer.html
```

### **Processing Chain**
```
unified_document_processor.py
├── indian_document_enhancer.py
├── enhanced_document_classifier.py
├── dynamic_field_extractor.py
└── comprehensive_accuracy_system.py
    ├── advanced_accuracy_enhancements.py
    └── ml_accuracy_boosters.py
```

### **Ollama Integration**
```
ollama_integration.py
├── requests (HTTP client)
├── json (data serialization)
└── ollama service (external)
```

---

## 🚀 **Quick Start Files**

### **For New Users**
1. **`README.md`** - Start here for overview
2. **`INSTALLATION_GUIDE.md`** - Follow installation steps
3. **`setup_torch_env_ollama.py`** - Run automated setup
4. **`document_extraction_demo.py`** - Test the system

### **For Developers**
1. **`TECHNICAL_WRITEUP.md`** - Understand the architecture
2. **`API_DOCUMENTATION.md`** - API reference
3. **`test_*.py`** - Run test suite
4. **`PROJECT_STRUCTURE.md`** - This file

### **For Advanced Users**
1. **`OLLAMA_INTEGRATION_GUIDE.md`** - LLM integration
2. **`ACCURACY_IMPROVEMENT_GUIDE.md`** - Accuracy optimization
3. **`debug_*.py`** - Debugging tools
4. **`refresh_database.py`** - Database maintenance

---

## 📈 **File Size and Complexity**

### **Large Files (>500 lines)**
- **`app_with_database.py`** - 1,464 lines (Main application)
- **`unified_document_processor.py`** - 945 lines (Processing orchestrator)
- **`ollama_integration.py`** - 519 lines (LLM integration)
- **`document_extractor.py`** - 604 lines (Core processing)

### **Medium Files (200-500 lines)**
- **`indian_document_enhancer.py`** - ~400 lines
- **`comprehensive_accuracy_system.py`** - ~527 lines
- **`enhanced_document_classifier.py`** - ~300 lines
- **`dynamic_field_extractor.py`** - ~250 lines

### **Small Files (<200 lines)**
- **`database_setup.py`** - ~150 lines
- **`test_*.py`** - 100-200 lines each
- **`debug_*.py`** - 50-100 lines each

---

## 🔧 **Maintenance Files**

### **Regular Maintenance**
- **`refresh_database.py`** - Run when database schema changes
- **`setup_torch_env_ollama.py`** - Run when setting up new environment
- **`start_torch_env_system.bat`** - Use to start the system

### **Debugging**
- **`debug_*.py`** - Use when troubleshooting specific issues
- **`test_*.py`** - Run to verify system functionality

### **Updates**
- **`requirements_torch_env.txt`** - Update when adding new dependencies
- **`README.md`** - Update when adding new features
- **`API_DOCUMENTATION.md`** - Update when changing API

---

## 📊 **Performance Considerations**

### **CPU Intensive Files**
- **`ollama_integration.py`** - LLM processing
- **`comprehensive_accuracy_system.py`** - Image preprocessing
- **`indian_document_enhancer.py`** - Indian document processing

### **Memory Intensive Files**
- **`app_with_database.py`** - Main application with all processors
- **`unified_document_processor.py`** - Dual processing orchestration
- **`ollama_integration.py`** - LLM model loading

### **I/O Intensive Files**
- **`database_setup.py`** - Database operations
- **`document_extractor.py`** - File processing
- **`frontend/database_viewer.html`** - Web interface

---

## 🎯 **Key Files for Different Use Cases**

### **Production Deployment**
- **`app_with_database.py`** - Main application
- **`requirements_torch_env.txt`** - Dependencies
- **`start_torch_env_system.bat`** - Startup script
- **`database_setup.py`** - Database initialization

### **Development**
- **`test_*.py`** - Test suite
- **`debug_*.py`** - Debugging tools
- **`document_extraction_demo.py`** - Demo script

### **Documentation**
- **`README.md`** - Project overview
- **`API_DOCUMENTATION.md`** - API reference
- **`INSTALLATION_GUIDE.md`** - Setup instructions

### **Advanced Features**
- **`ollama_integration.py`** - LLM features
- **`OLLAMA_INTEGRATION_GUIDE.md`** - LLM setup
- **`comprehensive_accuracy_system.py`** - Advanced accuracy

---

This project structure provides a comprehensive, modular, and well-documented system for document information extraction with advanced AI capabilities.
