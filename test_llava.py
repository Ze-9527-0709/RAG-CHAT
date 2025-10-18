#!/usr/bin/env python3
"""
LLaVA Integration Test Script
"""
import requests
import base64
import json

def test_llava_integration():
    """Test whether LLaVA model can process images"""
    # 创建一个简单的测试图片（base64编码的简单图片）
    # 这是一个1x1像素的透明PNG图片
    test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    # 解码base64图片数据
    import base64
    image_data = base64.b64decode(test_image_base64)
    
    try:
        print("🧪 Testing LLaVA image processing functionality...")
        
        # 准备表单数据和文件
        form_data = {
            'session_id': 'test_session',
            'message': 'Please describe this image',
            'max_history': '8'
        }
        
        files = {
            'image': ('test.png', image_data, 'image/png')
        }
        
        # 发送请求到后端
        response = requests.post(
            "http://localhost:8000/api/chat_with_image", 
            data=form_data,
            files=files,
            timeout=30
        )
        
        print(f"📡 HTTP status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ LLaVA response successful!")
            print(f"📝 Response content: {result.get('response', 'No response')[:200]}...")
            return True
        else:
            print(f"❌ Request failed: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def test_ollama_direct():
    """Direct test of Ollama LLaVA model"""
    try:
        print("\n🔍 Direct testing Ollama LLaVA...")
        
        test_data = {
            "model": "llava:7b",
            "prompt": "Describe this image",
            "images": ["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="],
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=test_data,
            timeout=30
        )
        
        print(f"📡 Ollama HTTP status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Ollama LLaVA response successful!")
            print(f"📝 Response content: {result.get('response', 'No response')[:200]}...")
            return True
        else:
            print(f"❌ Ollama request failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Ollama test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting LLaVA integration test...")
    
    # 首先测试直接Ollama连接
    ollama_ok = test_ollama_direct()
    
    # 然后测试通过我们的后端
    backend_ok = test_llava_integration()
    
    print("\n📊 Test results:")
    print(f"   Ollama Direct: {'✅ Success' if ollama_ok else '❌ Failed'}")
    print(f"   Backend Integration: {'✅ Success' if backend_ok else '❌ Failed'}")
    
    if ollama_ok and backend_ok:
        print("\n🎉 LLaVA integration test completely successful!")
    elif ollama_ok:
        print("\n⚠️ Ollama works normally, but backend integration has issues")
    else:
        print("\n❌ LLaVA integration has issues, needs checking")