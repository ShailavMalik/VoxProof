#!/usr/bin/env python
"""
VoxProof API Test Script
Test all 6 samples (3 human + 3 AI) against the deployed backend
"""
import requests
import json
import sys
from pathlib import Path
from datetime import datetime

API_KEY = "99d8f7fefa2c12ce971e4b320ee3af70"
API_URL = "https://voxproof-api.onrender.com/api/voice-detection"  # Update with your Render URL

HEADERS = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

def test_sample(sample_path, expected_type):
    """Test a single audio sample"""
    try:
        with open(sample_path, "r") as f:
            audio_base64 = f.read().strip()
        
        # Make API request
        response = requests.post(
            API_URL,
            json={"audio_base64": audio_base64},
            headers=HEADERS,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            classification = result.get("classification", "UNKNOWN")
            confidence = result.get("confidence", 0)
            is_correct = classification == expected_type
            
            # Status symbol
            status = "✅" if is_correct else "❌"
            
            return {
                "status": status,
                "classification": classification,
                "confidence": f"{confidence:.2%}",
                "expected": expected_type,
                "correct": is_correct,
                "explanation": result.get("explanation", "")[:80] + "..."
            }
        else:
            return {
                "status": "❌",
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    
    except FileNotFoundError:
        return {"status": "❌", "error": f"File not found: {sample_path}"}
    except requests.exceptions.Timeout:
        return {"status": "❌", "error": "Request timeout (>60s)"}
    except requests.exceptions.ConnectionError:
        return {"status": "❌", "error": "Cannot connect to API"}
    except Exception as e:
        return {"status": "❌", "error": str(e)}

def main():
    print("\n" + "="*90)
    print(" VOXPROOF API TEST SUITE ".center(90, "="))
    print("="*90)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"API URL: {API_URL}")
    print("="*90 + "\n")
    
    # Test configuration
    tests = [
        ("test_samples/human_sample_1.txt", "HUMAN", "Human Voice 1"),
        ("test_samples/human_sample_2.txt", "HUMAN", "Human Voice 2"),
        ("test_samples/human_sample_3.txt", "HUMAN", "Human Voice 3"),
        ("test_samples/ai_sample_1.txt", "AI", "AI-Generated 1"),
        ("test_samples/ai_sample_2.txt", "AI", "AI-Generated 2"),
        ("test_samples/ai_sample_3.txt", "AI", "AI-Generated 3"),
    ]
    
    results = []
    passed = 0
    failed = 0
    
    # Run tests
    print("📍 TESTING HUMAN SAMPLES:")
    print("-" * 90)
    for sample_path, expected, label in tests[:3]:
        print(f"Testing {label}...", end=" ", flush=True)
        result = test_sample(sample_path, expected)
        results.append((label, result))
        
        if "error" in result:
            print(f"{result['status']} ERROR: {result['error']}")
            failed += 1
        else:
            print(f"{result['status']} {result['classification']} (Confidence: {result['confidence']})")
            if result["correct"]:
                passed += 1
            else:
                failed += 1
    
    print("\n🤖 TESTING AI-GENERATED SAMPLES:")
    print("-" * 90)
    for sample_path, expected, label in tests[3:]:
        print(f"Testing {label}...", end=" ", flush=True)
        result = test_sample(sample_path, expected)
        results.append((label, result))
        
        if "error" in result:
            print(f"{result['status']} ERROR: {result['error']}")
            failed += 1
        else:
            print(f"{result['status']} {result['classification']} (Confidence: {result['confidence']})")
            if result["correct"]:
                passed += 1
            else:
                failed += 1
    
    # Summary
    print("\n" + "="*90)
    print("TEST SUMMARY".center(90))
    print("="*90)
    print(f"Total Tests: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed/len(results)*100):.1f}%")
    print("="*90 + "\n")
    
    # Detailed results table
    print("DETAILED RESULTS:")
    print("-" * 90)
    print(f"{'Sample':<20} {'Expected':<10} {'Got':<10} {'Confidence':<15} {'Status':<5}")
    print("-" * 90)
    
    for label, result in results:
        if "error" in result:
            print(f"{label:<20} {'N/A':<10} {'ERROR':<10} {'-':<15} {result['status']:<5}")
        else:
            expected = result["expected"]
            classification = result["classification"]
            confidence = result["confidence"]
            status = result["status"]
            print(f"{label:<20} {expected:<10} {classification:<10} {confidence:<15} {status}")
    
    print("-" * 90 + "\n")
    
    # Return exit code
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
