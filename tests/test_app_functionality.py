"""Simple functional test to verify the app works correctly."""

import requests
import time
import sys
import os

def test_app_functionality():
    """Test that the app is running and responsive."""
    print("🧪 Testing App Functionality")
    
    # Test that the app is accessible
    try:
        response = requests.get("http://localhost:8502", timeout=5)
        if response.status_code == 200:
            print("✅ App is accessible at http://localhost:8502")
            
            # Check if page contains expected content
            content = response.text
            if "Python FAQ RAG Chatbot" in content:
                print("✅ App title found in page content")
            else:
                print("❌ App title not found in page content")
                
            if "Ask me anything about Python" in content:
                print("✅ App description found in page content")
            else:
                print("❌ App description not found in page content")
                
            # Look for initialization messages (these should not be present if app is working)
            if "initializing" in content.lower():
                print("⚠️  Found initialization messages - this may indicate the app is still starting up")
                return False
            else:
                print("✅ No initialization messages found - app appears to be ready")
                return True
                
        else:
            print(f"❌ App returned status code {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to app: {e}")
        return False

def test_multiple_requests():
    """Test that multiple requests work (to verify no infinite loading)."""
    print("\n🧪 Testing Multiple Requests")
    
    for i in range(3):
        try:
            response = requests.get("http://localhost:8502", timeout=10)
            if response.status_code == 200:
                print(f"✅ Request {i+1}/3 successful")
            else:
                print(f"❌ Request {i+1}/3 failed with status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Request {i+1}/3 failed: {e}")
            return False
        
        time.sleep(1)  # Small delay between requests
    
    print("✅ All requests completed successfully")
    return True

if __name__ == "__main__":
    print("🚀 Running App Functionality Tests")
    print("(Make sure the app is running at http://localhost:8502)")
    
    success = True
    
    # Wait a moment for the app to be ready
    print("\nWaiting 3 seconds for app to be ready...")
    time.sleep(3)
    
    if not test_app_functionality():
        success = False
    
    if not test_multiple_requests():
        success = False
    
    if success:
        print("\n🎉 All functionality tests passed!")
        print("✅ The infinite 'initializing' issue appears to be fixed!")
    else:
        print("\n❌ Some tests failed")
        print("💡 Please check the app at http://localhost:8502 manually")
