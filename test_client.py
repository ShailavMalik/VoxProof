"""
Test client for VoxProof API
Use this to test the voice detection endpoint with sample audio files.
"""

import base64
import sys
from pathlib import Path

import requests


def test_voice_detection(
    audio_path: str,
    language: str = "English",
    api_url: str = "http://localhost:8000",
    api_key: str = "voxproof-secret-key-2024"
):
    """
    Test the voice detection API with an audio file.
    
    Args:
        audio_path: Path to the MP3 audio file
        language: Language of the audio (Tamil, English, Hindi, Malayalam, Telugu)
        api_url: Base URL of the API
        api_key: API key for authentication
    """
    # Validate file exists
    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"❌ Error: File not found: {audio_path}")
        return None
    
    if not audio_file.suffix.lower() == ".mp3":
        print(f"⚠️ Warning: File may not be MP3 format: {audio_file.suffix}")
    
    # Read and encode audio
    print(f"📁 Reading audio file: {audio_path}")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    print(f"📦 Encoded {len(audio_bytes)} bytes to Base64 ({len(audio_base64)} chars)")
    
    # Prepare request
    endpoint = f"{api_url}/api/voice-detection"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    # Make request
    print(f"🚀 Sending request to {endpoint}...")
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        # Display results
        print("\n" + "=" * 50)
        print("📊 DETECTION RESULTS")
        print("=" * 50)
        print(f"Status:       {result['status']}")
        print(f"Language:     {result['language']}")
        print(f"Classification: {result['classification']}")
        print(f"Confidence:   {result['confidenceScore']:.2%}")
        print(f"Explanation:  {result['explanation']}")
        print("=" * 50 + "\n")
        
        return result
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Is the server running?")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if response.text:
            print(f"   Response: {response.text}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_health(api_url: str = "http://localhost:8000"):
    """Test the health endpoint."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        response.raise_for_status()
        print(f"✅ API is healthy: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def main():
    """Main entry point for CLI usage."""
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <audio_file.mp3> [language]")
        print("\nSupported languages: Tamil, English, Hindi, Malayalam, Telugu")
        print("\nExamples:")
        print("  python test_client.py sample.mp3")
        print("  python test_client.py sample.mp3 Hindi")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "English"
    
    # Check health first
    print("\n🔍 Checking API health...")
    if not test_health():
        print("❌ API is not available. Please start the server first:")
        print("   uvicorn app:app --reload")
        sys.exit(1)
    
    # Run detection
    print(f"\n🎤 Testing voice detection...")
    test_voice_detection(audio_path, language)


if __name__ == "__main__":
    main()
