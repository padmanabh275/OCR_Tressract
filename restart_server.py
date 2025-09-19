"""
Restart the server with the latest fixes
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def kill_existing_server():
    """Kill any existing server processes"""
    try:
        # On Windows, find and kill uvicorn processes
        if os.name == 'nt':  # Windows
            subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                         capture_output=True, check=False)
        else:  # Unix-like systems
            subprocess.run(['pkill', '-f', 'uvicorn'], 
                         capture_output=True, check=False)
        print("🔄 Stopped existing server processes")
    except Exception as e:
        print(f"⚠️  Could not stop existing processes: {e}")

def start_server():
    """Start the server with the latest fixes"""
    print("🚀 Starting server with latest fixes...")
    
    try:
        # Start the server
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 
            'app:app', 
            '--host', '0.0.0.0', 
            '--port', '8000', 
            '--reload'
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def main():
    """Main restart function"""
    print("🔄 Restarting AI Document Extraction System")
    print("=" * 50)
    
    # Kill existing server
    kill_existing_server()
    
    # Wait a moment
    time.sleep(2)
    
    # Start new server
    start_server()

if __name__ == "__main__":
    main()
