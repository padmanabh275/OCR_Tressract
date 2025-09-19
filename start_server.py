"""
Startup script for the Advanced AI Document Extraction System
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    # Map package names to their import names
    package_imports = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'opencv-python': 'cv2',
        'pytesseract': 'pytesseract',
        'Pillow': 'PIL',
        'PyMuPDF': 'fitz',
        'python-dateutil': 'dateutil',
        'pydantic': 'pydantic'
    }
    
    missing_packages = []
    
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract OCR is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Tesseract OCR is not installed")
    print("📥 Download and install from: https://github.com/tesseract-ocr/tesseract")
    return False

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'results', 'frontend']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Created necessary directories")

def start_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting Advanced AI Document Extraction System...")
    print("=" * 60)
    print("📱 Web Interface: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔧 API ReDoc: http://localhost:8000/redoc")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 
            'app:app', 
            '--host', '0.0.0.0', 
            '--port', '8000', 
            '--reload'
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")

def main():
    """Main startup function"""
    print("🤖 Advanced AI Document Extraction System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check Tesseract
    if not check_tesseract():
        print("\n⚠️  Warning: Tesseract OCR is required for image processing")
        print("   The system will work with PDFs but not images without Tesseract")
    
    # Create directories
    create_directories()
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()
