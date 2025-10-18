import requests
import json

def test_stream_format():
    print("🔍 Testing detailed SSE stream format...")
    
    # Test image (1x1 pixel PNG)
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    url = "http://localhost:8000/api/chat_stream_with_image"
    
    # Use form data to match the expected format
    files = {
        'session_id': (None, 'test-debug'),
        'message': (None, 'Tell me about this image'),
        'image': ('test.png', f"data:image/png;base64,{test_image_b64}", 'image/png')
    }
    
    try:
        print(f"📡 URL: {url}")
        response = requests.post(url, files=files, stream=True)
        print(f"📊 Status Code: {response.status_code}")
        print(f"📋 Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("📝 Raw Stream Content:")
            chunk_count = 0
            accumulated_buffer = ""
            
            for chunk in response.iter_content(chunk_size=None, decode_unicode=False):
                if chunk:
                    chunk_count += 1
                    chunk_str = chunk.decode('utf-8', errors='ignore')
                    accumulated_buffer += chunk_str
                    
                    print(f"Chunk {chunk_count}: {repr(chunk_str)}")
                    
                    # Show buffer state
                    if '\n\n' in accumulated_buffer:
                        parts = accumulated_buffer.split('\n\n')
                        completed_events = parts[:-1]  # All but last part
                        accumulated_buffer = parts[-1]  # Keep remainder
                        
                        for i, event in enumerate(completed_events):
                            print(f"  ✅ Complete Event {i+1}: {repr(event)}")
                        
                        if accumulated_buffer:
                            print(f"  ⏳ Buffer remainder: {repr(accumulated_buffer)}")
                    
                    if chunk_count >= 15:  # Limit output
                        break
                        
            # Show any final buffer content
            if accumulated_buffer:
                print(f"📋 Final buffer: {repr(accumulated_buffer)}")
                
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_stream_format()