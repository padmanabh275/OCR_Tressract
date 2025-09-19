"""
Find and configure Tesseract OCR installation
"""

import os
import subprocess
import sys
from pathlib import Path

def find_tesseract_executable():
    """Find Tesseract executable using multiple methods"""
    print("🔍 Searching for Tesseract executable...")
    
    # Method 1: Check if tesseract is in PATH
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Found Tesseract in PATH")
            return "tesseract"  # Command name
    except:
        pass
    
    # Method 2: Search common Windows locations
    if os.name == 'nt':  # Windows
        search_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
            r"C:\tesseract\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe"
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                print(f"✅ Found Tesseract at: {path}")
                return path
    
    # Method 3: Search in conda environment
    try:
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix:
            conda_tesseract = os.path.join(conda_prefix, 'Scripts', 'tesseract.exe')
            if os.path.exists(conda_tesseract):
                print(f"✅ Found Tesseract in conda: {conda_tesseract}")
                return conda_tesseract
    except:
        pass
    
    print("❌ Tesseract executable not found")
    return None

def find_tessdata_directory():
    """Find Tesseract data directory"""
    print("🔍 Searching for tessdata directory...")
    
    # Method 1: Check TESSDATA_PREFIX environment variable
    if 'TESSDATA_PREFIX' in os.environ:
        tessdata_path = os.environ['TESSDATA_PREFIX']
        if os.path.exists(tessdata_path):
            print(f"✅ Found tessdata via TESSDATA_PREFIX: {tessdata_path}")
            return tessdata_path
    
    # Method 2: Search relative to tesseract executable
    tesseract_path = find_tesseract_executable()
    if tesseract_path and tesseract_path != "tesseract":
        # Get directory containing tesseract.exe
        tesseract_dir = os.path.dirname(tesseract_path)
        tessdata_path = os.path.join(tesseract_dir, 'tessdata')
        if os.path.exists(tessdata_path):
            print(f"✅ Found tessdata relative to executable: {tessdata_path}")
            return tessdata_path
    
    # Method 3: Search common Windows locations
    if os.name == 'nt':  # Windows
        search_paths = [
            r"C:\Program Files\Tesseract-OCR\tessdata",
            r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
            r"C:\Users\{}\AppData\Local\Tesseract-OCR\tessdata".format(os.getenv('USERNAME', '')),
            r"C:\tesseract\tessdata",
            r"C:\Tesseract-OCR\tessdata"
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                print(f"✅ Found tessdata at: {path}")
                return path
    
    # Method 4: Search in conda environment
    try:
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix:
            conda_tessdata = os.path.join(conda_prefix, 'share', 'tessdata')
            if os.path.exists(conda_tessdata):
                print(f"✅ Found tessdata in conda: {conda_tessdata}")
                return conda_tessdata
    except:
        pass
    
    print("❌ tessdata directory not found")
    return None

def test_tesseract_with_paths():
    """Test Tesseract with found paths"""
    print("🧪 Testing Tesseract functionality...")
    
    tesseract_path = find_tesseract_executable()
    tessdata_path = find_tessdata_directory()
    
    if not tesseract_path or not tessdata_path:
        print("❌ Cannot test - missing paths")
        return False
    
    try:
        # Set environment variables
        os.environ['TESSERACT_CMD'] = tesseract_path
        os.environ['TESSDATA_PREFIX'] = tessdata_path
        
        # Test with pytesseract
        import pytesseract
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        img = np.ones((100, 400), dtype=np.uint8) * 255
        
        # Test OCR
        text = pytesseract.image_to_string(img, config='--psm 6')
        print("✅ Tesseract OCR test successful!")
        print(f"   Tesseract path: {tesseract_path}")
        print(f"   Tessdata path: {tessdata_path}")
        return True
        
    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")
        return False

def create_environment_script():
    """Create a script to set environment variables"""
    tesseract_path = find_tesseract_executable()
    tessdata_path = find_tessdata_directory()
    
    if tesseract_path and tessdata_path:
        script_content = f"""@echo off
echo Setting Tesseract environment variables...
set TESSERACT_CMD={tesseract_path}
set TESSDATA_PREFIX={tessdata_path}
echo Environment variables set!
echo TESSERACT_CMD={tesseract_path}
echo TESSDATA_PREFIX={tessdata_path}
"""
        
        with open("set_tesseract_env.bat", "w") as f:
            f.write(script_content)
        
        print("✅ Created set_tesseract_env.bat script")
        print("   Run this script before starting the server:")
        print("   set_tesseract_env.bat")
        print("   python start_server.py")

def main():
    """Main function"""
    print("🔍 Tesseract OCR Detection and Configuration")
    print("=" * 60)
    
    # Find paths
    tesseract_path = find_tesseract_executable()
    tessdata_path = find_tessdata_directory()
    
    if tesseract_path and tessdata_path:
        print("\n🎉 Tesseract OCR found and configured!")
        
        # Test functionality
        if test_tesseract_with_paths():
            print("\n✅ Tesseract is working correctly!")
            print("You can now process images with the document extraction system.")
            
            # Create environment script
            create_environment_script()
        else:
            print("\n⚠️  Tesseract found but not working properly")
    else:
        print("\n❌ Tesseract OCR not found")
        print("Please install Tesseract OCR first:")
        print("conda install -c conda-forge tesseract")

if __name__ == "__main__":
    main()
