"""
Quick fix for Tesseract tessdata path
"""

import os
import subprocess
import sys

def find_tessdata_in_conda():
    """Find tessdata in conda environment"""
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if not conda_prefix:
        print("❌ CONDA_PREFIX not found")
        return None
    
    print(f"🔍 Searching in conda environment: {conda_prefix}")
    
    # Common tessdata locations in conda
    possible_paths = [
        os.path.join(conda_prefix, 'share', 'tessdata'),
        os.path.join(conda_prefix, 'Library', 'share', 'tessdata'),
        os.path.join(conda_prefix, 'Library', 'bin', 'tessdata'),
        os.path.join(conda_prefix, 'tessdata'),
        os.path.join(conda_prefix, 'share', 'tesseract-ocr', 'tessdata'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found tessdata at: {path}")
            return path
    
    print("❌ tessdata not found in conda environment")
    return None

def download_tessdata():
    """Download tessdata if not found"""
    print("📥 Attempting to download tessdata...")
    
    try:
        # Try to download tessdata using conda
        result = subprocess.run([
            'conda', 'install', '-c', 'conda-forge', 'tesseract-data-eng'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Downloaded tessdata successfully")
            return True
        else:
            print(f"❌ Failed to download tessdata: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error downloading tessdata: {e}")
        return False

def set_environment_variables():
    """Set the correct environment variables"""
    tesseract_cmd = r"C:\Users\Padmanabh\.conda\envs\torch_env\Library\bin\tesseract.EXE"
    tessdata_path = find_tessdata_in_conda()
    
    if not tessdata_path:
        print("🔧 Trying to download tessdata...")
        if download_tessdata():
            tessdata_path = find_tessdata_in_conda()
    
    if tessdata_path:
        # Set environment variables
        os.environ['TESSERACT_CMD'] = tesseract_cmd
        os.environ['TESSDATA_PREFIX'] = tessdata_path
        
        print(f"✅ Set TESSERACT_CMD: {tesseract_cmd}")
        print(f"✅ Set TESSDATA_PREFIX: {tessdata_path}")
        
        # Test OCR
        try:
            import pytesseract
            from PIL import Image
            import numpy as np
            
            # Create test image
            img = np.ones((100, 400), dtype=np.uint8) * 255
            
            # Test OCR
            text = pytesseract.image_to_string(img, config='--psm 6')
            print("✅ Tesseract OCR test successful!")
            
            # Create updated batch file
            create_batch_file(tesseract_cmd, tessdata_path)
            return True
            
        except Exception as e:
            print(f"❌ OCR test failed: {e}")
            return False
    else:
        print("❌ Could not find or create tessdata directory")
        return False

def create_batch_file(tesseract_cmd, tessdata_path):
    """Create updated batch file with correct paths"""
    batch_content = f"""@echo off
echo Setting Tesseract environment variables...
set TESSERACT_CMD={tesseract_cmd}
set TESSDATA_PREFIX={tessdata_path}
echo Environment variables set!
echo TESSERACT_CMD={tesseract_cmd}
echo TESSDATA_PREFIX={tessdata_path}
"""
    
    with open("set_tesseract_env.bat", "w") as f:
        f.write(batch_content)
    
    print("✅ Updated set_tesseract_env.bat with correct paths")

def main():
    """Main function"""
    print("🔧 Tesseract OCR Quick Fix")
    print("=" * 40)
    
    if set_environment_variables():
        print("\n🎉 Tesseract OCR is now properly configured!")
        print("You can now process images with the document extraction system.")
        print("\nTo start the server:")
        print("1. Run: set_tesseract_env.bat")
        print("2. Run: python start_server.py")
    else:
        print("\n❌ Could not fix Tesseract configuration")
        print("The system will still work for PDFs and text files.")

if __name__ == "__main__":
    main()
