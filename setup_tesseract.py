"""
Tesseract OCR Setup Script
Helps install and configure Tesseract OCR for the document extraction system
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_tesseract_installation():
    """Check if Tesseract is installed and working"""
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Tesseract OCR is installed")
            print(f"Version: {result.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("❌ Tesseract OCR is not installed or not in PATH")
    return False

def find_tesseract_path():
    """Find Tesseract installation path"""
    system = platform.system().lower()
    
    if system == "windows":
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
        ]
    else:
        possible_paths = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract"
        ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found Tesseract at: {path}")
            return path
    
    print("❌ Tesseract not found in common locations")
    return None

def find_tessdata_path():
    """Find Tesseract data directory"""
    system = platform.system().lower()
    
    if system == "windows":
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tessdata",
            r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
            r"C:\Users\{}\AppData\Local\Tesseract-OCR\tessdata".format(os.getenv('USERNAME', '')),
        ]
    else:
        possible_paths = [
            "/usr/share/tesseract-ocr/4.00/tessdata",
            "/usr/share/tesseract-ocr/5/tessdata",
            "/usr/local/share/tessdata",
            "/opt/homebrew/share/tessdata"
        ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found tessdata at: {path}")
            return path
    
    print("❌ tessdata directory not found")
    return None

def install_tesseract_windows():
    """Instructions for installing Tesseract on Windows"""
    print("\n📥 Installing Tesseract OCR on Windows:")
    print("1. Download Tesseract from: https://github.com/tesseract-ocr/tesseract")
    print("2. Run the installer as Administrator")
    print("3. During installation, make sure to check 'Add to PATH'")
    print("4. Restart your command prompt/terminal")
    print("\nOr install via conda:")
    print("conda install -c conda-forge tesseract")

def install_tesseract_mac():
    """Instructions for installing Tesseract on macOS"""
    print("\n📥 Installing Tesseract OCR on macOS:")
    print("Using Homebrew:")
    print("brew install tesseract")
    print("\nOr using conda:")
    print("conda install -c conda-forge tesseract")

def install_tesseract_linux():
    """Instructions for installing Tesseract on Linux"""
    print("\n📥 Installing Tesseract OCR on Linux:")
    print("Ubuntu/Debian:")
    print("sudo apt update")
    print("sudo apt install tesseract-ocr")
    print("\nCentOS/RHEL:")
    print("sudo yum install tesseract")
    print("\nOr using conda:")
    print("conda install -c conda-forge tesseract")

def set_environment_variables():
    """Set Tesseract environment variables"""
    tesseract_path = find_tesseract_path()
    tessdata_path = find_tessdata_path()
    
    if tesseract_path and tessdata_path:
        print(f"\n🔧 Setting environment variables:")
        print(f"TESSERACT_CMD = {tesseract_path}")
        print(f"TESSDATA_PREFIX = {tessdata_path}")
        
        # Set for current session
        os.environ['TESSERACT_CMD'] = tesseract_path
        os.environ['TESSDATA_PREFIX'] = tessdata_path
        
        print("\n📝 To make these permanent, add to your environment variables:")
        print(f"TESSERACT_CMD={tesseract_path}")
        print(f"TESSDATA_PREFIX={tessdata_path}")
        
        return True
    else:
        print("❌ Cannot set environment variables - Tesseract not found")
        return False

def test_tesseract():
    """Test Tesseract OCR functionality"""
    try:
        import pytesseract
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        img = np.ones((100, 400), dtype=np.uint8) * 255
        # Add some text (this is a simple test)
        
        # Test OCR
        text = pytesseract.image_to_string(img, config='--psm 6')
        print("✅ Tesseract OCR test successful")
        return True
    except Exception as e:
        print(f"❌ Tesseract OCR test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🔧 Tesseract OCR Setup for Document Extraction System")
    print("=" * 60)
    
    # Check if Tesseract is installed
    if check_tesseract_installation():
        print("\n🔍 Checking configuration...")
        if set_environment_variables():
            print("\n🧪 Testing OCR functionality...")
            if test_tesseract():
                print("\n🎉 Tesseract OCR is properly configured!")
                print("You can now process images with the document extraction system.")
                return True
    
    # Tesseract not installed or not working
    print("\n📋 Installation Instructions:")
    system = platform.system().lower()
    
    if system == "windows":
        install_tesseract_windows()
    elif system == "darwin":  # macOS
        install_tesseract_mac()
    else:  # Linux
        install_tesseract_linux()
    
    print("\n⚠️  After installing Tesseract:")
    print("1. Restart your terminal/command prompt")
    print("2. Run this script again to verify installation")
    print("3. The document extraction system will work with PDFs and text files")
    print("   even without Tesseract, but images won't be processed")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
