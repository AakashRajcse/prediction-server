#!/usr/bin/env python3
"""
Fake Review Detection API - Server Startup Script
Run this from the server directory
"""
import subprocess
import sys
import os

def main():
    print("\n" + "="*70)
    print("FAKE REVIEW DETECTION API - SERVER STARTUP")
    print("="*70)

    print("\n[1] Checking dependencies...")
    try:
        import fastapi
        import sqlalchemy
        print("[OK] All dependencies installed\n")
    except ImportError:
        print("[ERROR] Missing dependencies!")
        print("Run: pip install -r requirements.txt")
        return

    print("[2] Starting API Server on http://localhost:8000")
    print("    API Documentation: http://localhost:8000/docs")
    print("    Alternative Docs: http://localhost:8000/redoc\n")
    print("Press CTRL+C to stop the server\n")
    print("="*70 + "\n")

    try:
        # Run uvicorn from current directory
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "src.api:app",
            "--reload",
            "--host", "127.0.0.1",
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n\n[INFO] Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
