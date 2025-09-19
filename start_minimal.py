"""
Minimal startup script that works around dependency conflicts
"""

import sys
import subprocess
from pathlib import Path

def check_basic_dependencies():
    """Check for basic Python functionality"""
    print("Checking basic dependencies...")
    
    # Check if we can import basic modules
    try:
        import json
        import re
        import datetime
        from pathlib import Path
        print("✅ Basic Python modules available")
        return True
    except ImportError as e:
        print(f"❌ Missing basic Python modules: {e}")
        return False

def install_minimal_packages():
    """Install only the essential packages that don't conflict"""
    print("Installing minimal compatible packages...")
    
    packages = [
        "python-dateutil",
        "pytesseract", 
        "PyMuPDF",
        "Pillow"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  {package} installation failed, but continuing...")

def create_directories():
    """Create necessary directories"""
    directories = ['sample_documents', 'results']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Created necessary directories")

def main():
    """Main startup function"""
    print("🤖 Minimal AI Document Extraction System")
    print("=" * 50)
    print("This version works around dependency conflicts")
    print("=" * 50)
    
    # Check basic dependencies
    if not check_basic_dependencies():
        print("❌ Basic Python modules not available")
        sys.exit(1)
    
    # Try to install minimal packages
    install_minimal_packages()
    
    # Create directories
    create_directories()
    
    print("\n🚀 Starting Minimal Document Extraction System...")
    print("=" * 50)
    print("📁 Processing sample documents...")
    print("=" * 50)
    
    try:
        # Import and run the minimal extractor
        from minimal_extractor import main as run_extractor
        run_extractor()
    except ImportError as e:
        print(f"❌ Error importing extractor: {e}")
        print("Try running: python minimal_extractor.py")
    except Exception as e:
        print(f"❌ Error running extractor: {e}")

if __name__ == "__main__":
    main()
