"""
Generate base64-encoded test samples from dataset for API testing
"""
import base64
import json
from pathlib import Path

def encode_file_to_base64(file_path):
    """Convert audio file to base64"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Sample files
human_samples = [
    "dataset/human/common_voice_en_111346.mp3",
    "dataset/human/common_voice_en_111487.mp3",
    "dataset/human/common_voice_en_140140.mp3",
]

ai_samples = [
    "dataset/ai/ai_0.mp3",
    "dataset/ai/ai_1.mp3",
    "dataset/ai/ai_2.mp3",
]

print("\n" + "="*80)
print("VOXPROOF API TEST SAMPLES")
print("="*80)

# Process human samples
print("\n📍 HUMAN VOICE SAMPLES (Expected: Human):n")
human_tests = []
for i, file_path in enumerate(human_samples, 1):
    if Path(file_path).exists():
        base64_audio = encode_file_to_base64(file_path)
        human_tests.append({
            "file": file_path,
            "base64_length": len(base64_audio),
            "base64": base64_audio[:100] + "..." if len(base64_audio) > 100 else base64_audio
        })
        print(f"[{i}] File: {file_path}")
        print(f"    Base64 Size: {len(base64_audio)} chars (~{len(base64_audio)//1024} KB)")
        print()

# Process AI samples  
print("\n🤖 AI-GENERATED VOICE SAMPLES (Expected: AI):n")
ai_tests = []
for i, file_path in enumerate(ai_samples, 1):
    if Path(file_path).exists():
        base64_audio = encode_file_to_base64(file_path)
        ai_tests.append({
            "file": file_path,
            "base64_length": len(base64_audio),
            "base64": base64_audio[:100] + "..." if len(base64_audio) > 100 else base64_audio
        })
        print(f"[{i}] File: {file_path}")
        print(f"    Base64 Size: {len(base64_audio)} chars (~{len(base64_audio)//1024} KB)")
        print()

# Save full base64 to files for easy copy-paste
print("\n" + "="*80)
print("SAVING FULL BASE64 TO FILES...")
print("="*80 + "n")

for idx, file_path in enumerate(human_samples, 1):
    if Path(file_path).exists():
        base64_audio = encode_file_to_base64(file_path)
        output_file = f"test_samples/human_sample_{idx}.txt"
        Path("test_samples").mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(base64_audio)
        print(f"✅ Saved: {output_file}")

for idx, file_path in enumerate(ai_samples, 1):
    if Path(file_path).exists():
        base64_audio = encode_file_to_base64(file_path)
        output_file = f"test_samples/ai_sample_{idx}.txt"
        Path("test_samples").mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(base64_audio)
        print(f"✅ Saved: {output_file}")

print("\n" + "="*80)
print("TEST CURL COMMANDS")
print("="*80)

print("\n📍 TEST HUMAN SAMPLE 1:")
print('-' * 80)
print("curl -X POST \"https://your-render-url.onrender.com/api/voice-detection\" \\")
print('  -H "x-api-key: 99d8f7fefa2c12ce971e4b320ee3af70" \\')
print('  -H "Content-Type: application/json" \\')
print('  -d @test_curl_human.json')
print()

print("🤖 TEST AI SAMPLE 1:")
print('-' * 80)
print("curl -X POST \"https://your-render-url.onrender.com/api/voice-detection\" \\")
print('  -H "x-api-key: 99d8f7fefa2c12ce971e4b320ee3af70" \\')
print('  -H "Content-Type: application/json" \\')
print('  -d @test_curl_ai.json')
print()

print("="*80)
print("✅ TEST SAMPLES READY! Check test_samples/ folder for base64 files")
print("="*80 + "n")
