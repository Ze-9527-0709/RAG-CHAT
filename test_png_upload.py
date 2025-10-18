#!/usr/bin/env python3
"""
Test script to simulate PNG upload to backend
"""
import requests
import base64

def test_png_upload():
    """Test PNG upload with LLaVA processing"""
    # Create a simple test PNG (1x1 pixel red image)
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAFicKFLvgAAAABJRU5ErkJggg=="
    )
    
    # Prepare form data
    form_data = {
        'session_id': 'test_png_session',
        'message': 'What do you see in this image?',
        'max_history': '8',
        'stream': 'true'
    }
    
    files = {
        'image': ('test.png', png_data, 'image/png')
    }
    
    print("🧪 Testing PNG upload to backend...")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/chat_stream_with_image", 
            data=form_data,
            files=files,
            timeout=60,
            stream=True
        )
        
        print(f"📡 HTTP status: {response.status_code}")
        
        if response.status_code == 200:
            print("📝 Streaming response:")
            content = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    print(f"  {decoded_line}")
                    content += decoded_line + "\n"
            
            if not content.strip():
                print("❌ No content received!")
            else:
                print("✅ Content received successfully!")
                
        else:
            print(f"❌ Request failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_png_upload()