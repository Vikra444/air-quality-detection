#!/usr/bin/env python3
"""
Status check script for AirGuard services.
"""

import requests
import time

def check_services():
    """Check if both API and web services are running."""
    print("🔍 Checking AirGuard Services...")
    print("=" * 40)
    
    # Check API server
    print("1. Checking API Server (Port 8001)...")
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API Server: RUNNING (Version: {data.get('version', 'Unknown')})")
        else:
            print(f"   ❌ API Server: ERROR (Status: {response.status_code})")
    except Exception as e:
        print(f"   ❌ API Server: NOT ACCESSIBLE ({str(e)})")
    
    # Check web dashboard
    print("2. Checking Web Dashboard (Port 8050)...")
    try:
        response = requests.get("http://localhost:8050/", timeout=5)
        if response.status_code == 200:
            print("   ✅ Web Dashboard: RUNNING")
        else:
            print(f"   ❌ Web Dashboard: ERROR (Status: {response.status_code})")
    except Exception as e:
        print(f"   ❌ Web Dashboard: NOT ACCESSIBLE ({str(e)})")
    
    print("\n" + "=" * 40)
    print("📋 Access Information:")
    print("   API Server:     http://localhost:8001")
    print("   Web Dashboard:  http://localhost:8050")
    print("   API Docs:       http://localhost:8001/docs")
    print("=" * 40)

if __name__ == "__main__":
    check_services()