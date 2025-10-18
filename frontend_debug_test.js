// Debug test to trace frontend image upload
const fs = require('fs');
const FormData = require('form-data');
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));

async function testImageUpload() {
    console.log('🔍 Testing frontend image upload flow...');
    
    // Create a test image file (same as our blue square)
    const imageBuffer = Buffer.from([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
        0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00,
        0x0C, 0x49, 0x44, 0x41, 0x54, 0x08, 0x1D, 0x01, 0x01, 0x00, 0x00, 0xFF,
        0xFF, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0x73, 0x75, 0x01, 0x18, 0x00,
        0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82
    ]);
    
    // Test 1: Direct API call to backend (like frontend would do)
    console.log('\n📡 Testing direct API call via frontend proxy path...');
    
    const formData = new FormData();
    formData.append('session_id', 'test_frontend_session');
    formData.append('message', 'What do you see in this image?');
    formData.append('max_history', '8');
    formData.append('stream', 'true');
    formData.append('image', imageBuffer, {
        filename: 'test_frontend.png',
        contentType: 'image/png'
    });
    
    try {
        console.log('📤 Making request to frontend proxy...');
        const response = await fetch('http://localhost:5173/api/chat_stream_with_image', {
            method: 'POST',
            body: formData
        });
        
        console.log('📥 Response status:', response.status);
        console.log('📋 Response headers:', Object.fromEntries(response.headers.entries()));
        
        if (!response.ok) {
            const errorText = await response.text();
            console.log('❌ Error response:', errorText);
            return;
        }
        
        // Test streaming response reading
        console.log('\n🌊 Reading streaming response...');
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        let chunkCount = 0;
        
        while (true) {
            const { done, value } = await reader.read();
            chunkCount++;
            
            if (done) {
                console.log(`✅ Stream completed after ${chunkCount} chunks`);
                break;
            }
            
            const chunk = decoder.decode(value, { stream: true });
            buffer += chunk;
            
            // Process complete events
            const parts = buffer.split('\n\n');
            buffer = parts.pop() || '';
            
            for (const raw of parts) {
                if (!raw.trim()) continue;
                
                const lines = raw.split('\n');
                let event = null;
                let data = '';
                
                for (const line of lines) {
                    if (line.startsWith('event:')) {
                        event = line.slice(6).trim();
                    } else if (line.startsWith('data:')) {
                        data = line.slice(5);
                        if (data.startsWith(' ')) data = data.slice(1);
                    }
                }
                
                console.log(`📦 Event: ${event || 'data'}, Data: "${data}"`);
                
                if (event === 'done') {
                    console.log('🏁 Received done event');
                    break;
                }
            }
            
            if (chunkCount > 50) {
                console.log('⚠️ Stopping after 50 chunks for safety');
                break;
            }
        }
        
    } catch (error) {
        console.error('❌ Test failed:', error);
    }
}

testImageUpload();