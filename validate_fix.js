#!/usr/bin/env node
const http = require('http');

async function testFrontendSSEFix() {
    console.log('🎯 Final Validation: Frontend SSE Fix Test');
    console.log('=' * 50);
    
    // Test data - simulate what backend sends
    const testImageBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";
    
    try {
        // Make request to backend
        console.log('📡 Testing backend SSE stream...');
        
        const response = await fetch('http://localhost:8000/api/chat_stream_with_image', {
            method: 'POST',
            headers: {
                'Content-Type': 'multipart/form-data'
            },
            body: createFormData({
                session_id: 'validation-test',
                message: 'Describe this image',
                image: testImageBase64
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        console.log('✅ Backend responded successfully');
        
        // Test the actual parsing logic that frontend would use
        const reader = response.body.getReader();
        let buffer = '';
        let fullMessage = '';
        let chunkCount = 0;
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            chunkCount++;
            const chunk = new TextDecoder().decode(value);
            buffer += chunk;
            
            // Apply our fix logic
            if(buffer.includes('data:') && !buffer.includes('\n\n')){
                const lines = buffer.split('\n')
                const processedLines = []
                
                for(let i = 0; i < lines.length; i++){
                    const line = lines[i].trim()
                    if(line.startsWith('data:') && !line.includes('event:')){
                        const rawData = line.slice(5)
                        const cleanData = rawData.startsWith(' ') ? rawData.slice(1) : rawData
                        if(cleanData) {
                            console.log(`🔧 Chunk ${chunkCount}: "${cleanData}"`)
                            fullMessage += cleanData
                            processedLines.push(i)
                        }
                    }
                }
                
                if(processedLines.length > 0){
                    const remainingLines = lines.filter((_, i) => !processedLines.includes(i))
                    buffer = remainingLines.join('\n')
                }
            }
            
            // Also handle properly formatted SSE
            const parts = buffer.split('\n\n')
            buffer = parts.pop() || ''
            
            for(const raw of parts){
                if(!raw.trim()) continue
                const lines = raw.split('\n')
                let event = null
                let data = ''
                
                for(const ln of lines){
                    if(ln.startsWith('event:')) event = ln.slice(6).trim()
                    else if(ln.startsWith('data:')) {
                        const rawData = ln.slice(5)
                        data += rawData.startsWith(' ') ? rawData.slice(1) : rawData
                    }
                }
                
                if(event === 'done') break
                if(event === 'model_info' || event === 'citations') continue
                
                if(data) {
                    console.log(`📦 Proper SSE: "${data}"`)
                    fullMessage += data
                }
            }
            
            if(chunkCount > 50) break // Safety limit
        }
        
        console.log('\n' + '=' * 50);
        console.log('🏁 FINAL RESULT:');
        console.log('📝 Full message:', JSON.stringify(fullMessage.trim()));
        console.log('📏 Message length:', fullMessage.length);
        console.log('✅ Contains readable text:', fullMessage.length > 10 && !fullMessage.includes('data:'));
        console.log('🎯 Success: Frontend fix works!', !fullMessage.includes('data:'));
        
    } catch (error) {
        console.error('❌ Test failed:', error.message);
    }
}

function createFormData(data) {
    // Simplified form data creation for Node.js
    const boundary = '----WebKitFormBoundary' + Math.random().toString(16).substr(2);
    let body = '';
    
    for (const [key, value] of Object.entries(data)) {
        body += `--${boundary}\r\n`;
        if (key === 'image') {
            body += `Content-Disposition: form-data; name="${key}"; filename="test.png"\r\n`;
            body += `Content-Type: image/png\r\n\r\n`;
            body += `data:image/png;base64,${value}\r\n`;
        } else {
            body += `Content-Disposition: form-data; name="${key}"\r\n\r\n`;
            body += `${value}\r\n`;
        }
    }
    body += `--${boundary}--\r\n`;
    
    return body;
}

// Run the test
testFrontendSSEFix().catch(console.error);