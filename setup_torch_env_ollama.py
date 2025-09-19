"""
Setup script for Ollama integration with torch_env conda environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Failed")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def check_ollama_installation():
    """Check if Ollama is installed and running"""
    print("🔍 Checking Ollama installation...")
    
    # Check if ollama command exists
    result = subprocess.run("ollama --version", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Ollama is installed: {result.stdout.strip()}")
        return True
    else:
        print("❌ Ollama is not installed")
        print("Please install Ollama from https://ollama.ai")
        return False

def check_ollama_service():
    """Check if Ollama service is running"""
    print("🔍 Checking Ollama service...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            return True
        else:
            print("❌ Ollama service is not responding")
            return False
    except Exception as e:
        print(f"❌ Ollama service check failed: {e}")
        return False

def install_required_packages():
    """Install required packages in torch_env"""
    print("📦 Installing required packages in torch_env...")
    
    packages = [
        "requests",
        "opencv-python",
        "pytesseract",
        "pymupdf",
        "pillow",
        "numpy",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pydantic"
    ]
    
    success_count = 0
    for package in packages:
        if run_command(f"pip install {package}", f"Installing {package}"):
            success_count += 1
    
    print(f"📊 Installed {success_count}/{len(packages)} packages")
    return success_count == len(packages)

def setup_ollama_models():
    """Setup recommended Ollama models"""
    print("🤖 Setting up Ollama models...")
    
    models = [
        "smollm2:135m",  # Fast, lightweight model
        "llama3.2:latest"  # Balanced model
    ]
    
    success_count = 0
    for model in models:
        if run_command(f"ollama pull {model}", f"Pulling {model}"):
            success_count += 1
    
    print(f"📊 Pulled {success_count}/{len(models)} models")
    return success_count > 0

def create_torch_env_requirements():
    """Create requirements.txt for torch_env"""
    print("📝 Creating requirements.txt for torch_env...")
    
    requirements = """# Core AI/ML packages (already in torch_env)
torch>=2.5.0
torchvision>=0.20.0
torchaudio>=2.5.0
pytorch-lightning>=2.5.0

# Document processing
opencv-python>=4.8.0
pytesseract>=0.3.10
pymupdf>=1.23.0
pillow>=10.0.0
numpy>=1.24.0

# Web framework
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
pydantic>=2.5.0

# HTTP requests
requests>=2.31.0

# Optional: Advanced features
scikit-learn>=1.3.0
scipy>=1.11.0
imutils>=0.5.4

# Optional: NLP (if needed)
spacy>=3.7.0
"""
    
    try:
        with open("requirements_torch_env.txt", "w") as f:
            f.write(requirements)
        print("✅ Created requirements_torch_env.txt")
        return True
    except Exception as e:
        print(f"❌ Failed to create requirements file: {e}")
        return False

def test_ollama_integration():
    """Test Ollama integration"""
    print("🧪 Testing Ollama integration...")
    
    try:
        # Test basic Ollama connection
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama connection successful. Found {len(models)} models")
            
            # Test model availability
            model_names = [model['name'] for model in models]
            if 'smollm2:135m' in model_names:
                print("✅ smollm2:135m model available")
            if 'llama3.2:latest' in model_names:
                print("✅ llama3.2:latest model available")
            
            return True
        else:
            print(f"❌ Ollama connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama integration test failed: {e}")
        return False

def create_torch_env_startup_script():
    """Create startup script for torch_env"""
    print("📝 Creating torch_env startup script...")
    
    startup_script = """@echo off
echo Starting Document Extraction System with torch_env...

REM Activate torch_env
call conda activate torch_env

REM Check if Ollama is running
echo Checking Ollama service...
curl -s http://localhost:11434/api/tags > nul
if %errorlevel% neq 0 (
    echo Starting Ollama service...
    start "Ollama Service" ollama serve
    timeout /t 5 /nobreak > nul
)

REM Start the document extraction system
echo Starting Document Extraction System...
python app_with_database.py

pause
"""
    
    try:
        with open("start_torch_env_system.bat", "w") as f:
            f.write(startup_script)
        print("✅ Created start_torch_env_system.bat")
        return True
    except Exception as e:
        print(f"❌ Failed to create startup script: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Ollama Integration with torch_env")
    print("=" * 60)
    
    # Step 1: Check Ollama installation
    if not check_ollama_installation():
        print("\n❌ Please install Ollama first from https://ollama.ai")
        return False
    
    # Step 2: Check Ollama service
    if not check_ollama_service():
        print("\n⚠️ Ollama service not running. Starting it...")
        if not run_command("ollama serve", "Starting Ollama service"):
            print("❌ Failed to start Ollama service")
            return False
    
    # Step 3: Install required packages
    if not install_required_packages():
        print("\n❌ Failed to install required packages")
        return False
    
    # Step 4: Setup Ollama models
    if not setup_ollama_models():
        print("\n❌ Failed to setup Ollama models")
        return False
    
    # Step 5: Create requirements file
    create_torch_env_requirements()
    
    # Step 6: Create startup script
    create_torch_env_startup_script()
    
    # Step 7: Test integration
    if not test_ollama_integration():
        print("\n❌ Ollama integration test failed")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📚 Next steps:")
    print("1. Run: start_torch_env_system.bat")
    print("2. Or manually: conda activate torch_env && python app_with_database.py")
    print("3. Test Ollama: python test_ollama_integration.py")
    print("4. Access web interface: http://localhost:8001")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!")
        sys.exit(0)
