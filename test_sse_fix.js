// Test the frontend SSE parsing fix
console.log('🧪 Testing SSE Parsing Fix...')

// Simulate the problematic SSE chunks
const malformedChunks = [
    'data:  The',
    'data:  image', 
    'data:  displays',
    'data:  a',
    'data:  screenshot',
    'data:  of',
    'data:  a',
    'data:  LinkedIn',
    'data:  profile'
]

// Simulate the frontend parsing logic
let buffer = ''
let result = ''
let botMessageAdded = false

function processChunk(chunk) {
    buffer += chunk + '\n'  // Add newline as decoder would
    
    console.log('📥 Processing chunk:', JSON.stringify(chunk))
    console.log('📋 Current buffer:', JSON.stringify(buffer))
    
    // Handle malformed SSE chunks that don't end with \n\n
    if(buffer.includes('data:') && !buffer.includes('\n\n')){
        const lines = buffer.split('\n')
        const processedLines = []
        
        for(let i = 0; i < lines.length; i++){
            const line = lines[i].trim()
            if(line.startsWith('data:') && !line.includes('event:')){
                const rawData = line.slice(5) // Remove "data:"
                const cleanData = rawData.startsWith(' ') ? rawData.slice(1) : rawData
                if(cleanData) {
                    console.log('🔧 Extracted clean data:', JSON.stringify(cleanData))
                    result += cleanData
                    processedLines.push(i)
                }
            }
        }
        
        // Remove processed lines from buffer
        if(processedLines.length > 0){
            const remainingLines = lines.filter((_, i) => !processedLines.includes(i))
            buffer = remainingLines.join('\n')
        }
    }
}

// Test the malformed chunks
console.log('\n🚀 Testing malformed SSE chunks...')
malformedChunks.forEach(chunk => processChunk(chunk))

console.log('\n✅ Final result:', JSON.stringify(result))
console.log('📏 Expected: "The image displays a screenshot of a LinkedIn profile"')
console.log('🎯 Matches expected:', result.trim() === 'The image displays a screenshot of a LinkedIn profile')

// Test with properly formatted SSE
console.log('\n🧪 Testing properly formatted SSE...')
buffer = ''
result = ''

const properSSE = 'event: response\ndata: This is a properly formatted SSE message\n\n'
buffer += properSSE

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
    
    console.log('📦 Proper SSE - Event:', event, 'Data:', JSON.stringify(data))
    result = data
}

console.log('✅ Proper SSE result:', JSON.stringify(result))
console.log('🎯 Working correctly:', result === 'This is a properly formatted SSE message')