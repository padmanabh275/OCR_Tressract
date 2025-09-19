"""
Dependency installation script for the AI Document Extraction System
This script installs packages one by one to avoid conflicts
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a single package with error handling"""
    try:
        print(f"Installing {package}...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', package
        ], capture_output=True, text=True, check=True)
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}")
        print(f"Error: {e.stderr}")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is already installed"""
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def main():
    """Main installation function"""
    print("🤖 AI Document Extraction System - Dependency Installer")
    print("=" * 60)
    
    # List of packages to install
    packages = [
        ("opencv-python", "cv2"),
        ("Pillow", "PIL"),
        ("PyMuPDF", "fitz"),
        ("python-dateutil", "dateutil"),
        ("pytesseract", "pytesseract"),
        ("pydantic", "pydantic"),
        ("fastapi", "fastapi"),
        ("uvicorn[standard]", "uvicorn"),
        ("python-multipart", None),
        ("jinja2", "jinja2")
    ]
    
    print("Checking current installation status...")
    print("-" * 40)
    
    # Check which packages are already installed
    missing_packages = []
    for package, import_name in packages:
        if not check_package(package, import_name):
            missing_packages.append(package)
    
    if not missing_packages:
        print("\n🎉 All required packages are already installed!")
        return True
    
    print(f"\n📦 Installing {len(missing_packages)} missing packages...")
    print("-" * 40)
    
    # Install missing packages
    failed_packages = []
    for package in missing_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    print("\n" + "=" * 60)
    print("📊 Installation Summary")
    print("=" * 60)
    
    if failed_packages:
        print(f"❌ Failed to install {len(failed_packages)} packages:")
        for package in failed_packages:
            print(f"   - {package}")
        print("\n🔧 Try installing them manually:")
        for package in failed_packages:
            print(f"   pip install {package}")
        return False
    else:
        print("✅ All packages installed successfully!")
        print("\n🚀 You can now start the system with:")
        print("   python start_server.py")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
