import requests
import time
import subprocess
import signal
import os
import sys

def test_api():
    # Start the server
    print("Starting FastAPI server...")
    server_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "backend.main:app",
        "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"
    ], cwd=os.getcwd())

    # Wait for server to start
    time.sleep(3)

    try:
        base_url = "http://localhost:8000"

        # Test root endpoint
        print("Testing root endpoint...")
        response = requests.get(f"{base_url}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")

        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")

        # Test analyze/frame endpoint (should return validation error)
        print("Testing analyze/frame endpoint...")
        response = requests.post(f"{base_url}/analyze/frame")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")

        print("All tests passed!")

    except Exception as e:
        print(f"Test failed: {e}")
        return False
    finally:
        # Stop the server
        server_process.terminate()
        server_process.wait()

    return True

if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1)