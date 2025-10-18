#!/usr/bin/env python3
import requests
import json
import base64

# Test image upload API directly
def test_image_upload():
    url = "http://localhost:8000/api/chat_stream_with_image"
    
    # Create a simple test image (1x1 pixel PNG)
    # This is a valid PNG file in base64
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    # Prepare form data
    files = {
        'image': ('test.png', base64.b64decode(test_image_b64), 'image/png'),
        'session_id': (None, 'test_session'),
        'message': (None, 'What do you see in this image?')
    }
    
    print("🔍 Testing image upload API...")
    print(f"📡 URL: {url}")
    
    try:
        response = requests.post(url, files=files, stream=True, timeout=30)
        print(f"📊 Status Code: {response.status_code}")
        print(f"📋 Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("📝 Stream Content:")
            chunk_count = 0
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    chunk_count += 1
                    print(f"Chunk {chunk_count}: {repr(chunk[:200])}")  # First 200 chars
                    if chunk_count >= 5:  # Only show first 5 chunks
                        print("... (truncated)")
                        break
        else:
            print(f"❌ Error response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_image_upload()