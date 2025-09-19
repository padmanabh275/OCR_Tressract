# Installation Guide
## Document Information Extraction System

---

## 🎯 **System Requirements**

### **Minimum Requirements**
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended for Ollama)
- **Storage**: 5GB free space
- **CPU**: Multi-core processor (4+ cores recommended)

### **Recommended Requirements**
- **OS**: Windows 11, macOS 12+, or Ubuntu 20.04+
- **Python**: 3.12
- **RAM**: 16GB+ (for optimal Ollama performance)
- **Storage**: 10GB+ free space
- **CPU**: 8+ cores
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster processing)

---

## 🚀 **Quick Installation**

### **Option 1: Automated Setup (Recommended)**
```bash
# Clone repository
git clone <repository-url>
cd GlobalTech

# Run automated setup
python setup_torch_env_ollama.py
```

### **Option 2: Manual Setup**
Follow the detailed steps below for manual installation.

---

## 📋 **Detailed Installation Steps**

### **Step 1: Install Prerequisites**

#### **1.1 Install Python and Conda**
```bash
# Download and install Miniconda from https://docs.conda.io/en/latest/miniconda.html
# Or use package managers:

# Windows (using winget)
winget install Anaconda.Miniconda3

# macOS (using Homebrew)
brew install miniconda

# Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

#### **1.2 Install Tesseract OCR**
```bash
# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Or using chocolatey:
choco install tesseract

# macOS
brew install tesseract

# Linux (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install tesseract-ocr

# Linux (CentOS/RHEL)
sudo yum install tesseract
```

#### **1.3 Install Ollama (Optional but Recommended)**
```bash
# Download and install from https://ollama.ai
# Or use package managers:

# Windows (using winget)
winget install Ollama.Ollama

# macOS (using Homebrew)
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

---

### **Step 2: Setup Conda Environment**

#### **2.1 Create Environment**
```bash
# Create torch_env environment
conda create -n torch_env python=3.12

# Activate environment
conda activate torch_env
```

#### **2.2 Install PyTorch**
```bash
# For CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# For GPU (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# For GPU (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### **2.3 Install Additional Dependencies**
```bash
# Install from requirements file
pip install -r requirements_torch_env.txt

# Or install individually
pip install opencv-python pytesseract pymupdf pillow numpy
pip install fastapi uvicorn python-multipart pydantic
pip install requests scikit-learn scipy imutils
```

---

### **Step 3: Setup Ollama (Optional)**

#### **3.1 Start Ollama Service**
```bash
# Start Ollama service
ollama serve

# In another terminal, pull models
ollama pull llama3.2:latest
ollama pull smollm2:135m
```

#### **3.2 Verify Ollama Installation**
```bash
# Check if Ollama is running
ollama list

# Test with a simple prompt
ollama run llama3.2:latest "Hello, world!"
```

---

### **Step 4: Initialize Database**

#### **4.1 Create Database**
```bash
# Activate environment
conda activate torch_env

# Initialize database
python database_setup.py
```

#### **4.2 Verify Database**
```bash
# Test database connection
python -c "from database_setup import DocumentDatabase; db = DocumentDatabase(); print('Database initialized successfully')"
```

---

### **Step 5: Test Installation**

#### **5.1 Run Test Suite**
```bash
# Test basic functionality
python test_accuracy.py

# Test Indian document integration
python test_indian_integration.py

# Test Ollama integration (if installed)
python test_ollama_integration.py
```

#### **5.2 Run Demo**
```bash
# Run comprehensive demo
python document_extraction_demo.py
```

---

### **Step 6: Start the System**

#### **6.1 Start Application**
```bash
# Option 1: Use startup script (Windows)
start_torch_env_system.bat

# Option 2: Manual start
conda activate torch_env
python app_with_database.py
```

#### **6.2 Access Web Interface**
- **Main Upload**: http://localhost:8001
- **Database Viewer**: http://localhost:8001/database
- **API Documentation**: http://localhost:8001/docs

---

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Optional: Set Tesseract path
export TESSERACT_CMD=/usr/bin/tesseract

# Optional: Set Ollama URL
export OLLAMA_BASE_URL=http://localhost:11434

# Optional: Set database path
export DATABASE_URL=sqlite:///document_extractions.db
```

### **Tesseract Configuration**
```python
# In your Python code
import pytesseract

# Set Tesseract path (if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set language data path
import os
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
```

### **Ollama Configuration**
```python
# In ollama_integration.py
processor = OllamaEnhancedProcessor(
    model="llama3.2:latest",  # or "smollm2:135m"
    base_url="http://localhost:11434"
)
```

---

## 🐳 **Docker Installation (Alternative)**

### **Dockerfile**
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements_torch_env.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_torch_env.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Start application
CMD ["python", "app_with_database.py"]
```

### **Docker Compose**
```yaml
version: '3.8'

services:
  document-extractor:
    build: .
    ports:
      - "8001:8001"
    volumes:
      - ./uploads:/app/uploads
      - ./results:/app/results
      - ./document_extractions.db:/app/document_extractions.db
    environment:
      - TESSERACT_CMD=/usr/bin/tesseract
      - OLLAMA_BASE_URL=http://ollama:11434

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0

volumes:
  ollama_data:
```

### **Build and Run**
```bash
# Build image
docker build -t document-extractor .

# Run with Docker Compose
docker-compose up -d

# Or run individually
docker run -p 8001:8001 document-extractor
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. Tesseract Not Found**
```bash
# Error: TesseractNotFoundError
# Solution: Install Tesseract and set path

# Windows
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# Linux/macOS
export TESSERACT_CMD=/usr/bin/tesseract
```

#### **2. Ollama Connection Failed**
```bash
# Error: Connection refused to Ollama
# Solution: Start Ollama service

ollama serve
```

#### **3. CUDA Out of Memory**
```bash
# Error: CUDA out of memory
# Solution: Use CPU-only PyTorch or smaller batch size

# Install CPU-only PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

#### **4. Database Locked**
```bash
# Error: Database is locked
# Solution: Close other instances or restart

# Kill all Python processes
taskkill /f /im python.exe  # Windows
pkill -f python            # Linux/macOS
```

#### **5. Port Already in Use**
```bash
# Error: Port 8001 already in use
# Solution: Use different port or kill existing process

# Kill process on port 8001
netstat -ano | findstr :8001  # Windows
lsof -ti:8001 | xargs kill    # Linux/macOS
```

### **Debug Commands**
```bash
# Check Python version
python --version

# Check installed packages
pip list

# Check conda environment
conda info --envs

# Check Tesseract installation
tesseract --version

# Check Ollama status
ollama list

# Check database
sqlite3 document_extractions.db ".tables"
```

---

## 📊 **Performance Optimization**

### **CPU Optimization**
```bash
# Set number of workers
export OMP_NUM_THREADS=4

# Set MKL threads
export MKL_NUM_THREADS=4
```

### **Memory Optimization**
```python
# In your code, limit batch size
batch_size = 1  # Process one document at a time

# Clear cache periodically
import gc
gc.collect()
```

### **GPU Optimization**
```python
# Use GPU if available
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.8)
```

---

## 🔄 **Updates and Maintenance**

### **Update Dependencies**
```bash
# Update pip packages
pip install --upgrade -r requirements_torch_env.txt

# Update conda packages
conda update --all
```

### **Update Ollama Models**
```bash
# Update models
ollama pull llama3.2:latest
ollama pull smollm2:135m
```

### **Database Maintenance**
```bash
# Refresh database
python refresh_database.py

# Backup database
cp document_extractions.db document_extractions_backup.db
```

---

## 📞 **Support**

### **Getting Help**
- **Documentation**: Check README.md and API_DOCUMENTATION.md
- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Email**: [Your contact information]

### **Log Files**
- **Application Logs**: Check console output
- **Error Logs**: Check stderr output
- **Database Logs**: Check SQLite logs

---

## ✅ **Verification Checklist**

After installation, verify the following:

- [ ] Python 3.8+ installed
- [ ] Conda environment created and activated
- [ ] PyTorch installed and working
- [ ] Tesseract OCR installed and accessible
- [ ] All Python dependencies installed
- [ ] Database initialized successfully
- [ ] Ollama service running (if using LLM features)
- [ ] Web interface accessible at http://localhost:8001
- [ ] Test suite passes
- [ ] Demo script runs successfully

---

**Installation completed successfully!** 🎉

For next steps, see the [README.md](README.md) and [API_DOCUMENTATION.md](API_DOCUMENTATION.md) files.
