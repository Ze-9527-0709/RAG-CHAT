import './crypto-polyfill'
import './styles.css'
import React, { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// Data model types
type ChatResp = { answer: string }
type Msg = { who:'user'|'bot'; text:string }
type Session = { id:string; name:string; created:number; messages: Msg[] }

// Model selection types
type ModelInfo = {
  id: string
  name: string
  display_name: string
  description: string
  available: boolean
  auto_available?: boolean
  is_local: boolean
  max_tokens: number
  cost_per_1k: number
  failure_count: number
  status?: string
}

type CurrentModel = {
  tier: string
  model_name: string
  is_user_selected: boolean
  user_preferred_model: string | null
}

// LocalStorage keys
const SESSIONS_KEY = 'rag_chat_sessions'
const CURRENT_KEY = 'rag_chat_current_session'
const LEGACY_HISTORY = 'rag_chat_history'

function safeUUID(){
  // Use native if present; else fallback (not RFC compliant but good enough for local IDs)
  const g:any = globalThis as any
  if(g.crypto && typeof g.crypto.randomUUID === 'function') return g.crypto.randomUUID()
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random()*16|0, v = c==='x'?r:(r&0x3|0x8)
    return v.toString(16)
  })
}

export default function App(){
  // Sessions state with legacy migration
  const [sessions, setSessions] = useState<Session[]>(()=>{
    try {
      const existing = localStorage.getItem(SESSIONS_KEY)
      if(existing) return JSON.parse(existing)
      const legacy = localStorage.getItem(LEGACY_HISTORY)
      if(legacy){
        const msgs: Msg[] = JSON.parse(legacy)
        return [{ id: safeUUID(), name:'Session 1', created: Date.now(), messages: msgs }]
      }
    } catch {}
    return [{ id: safeUUID(), name:'Session 1', created: Date.now(), messages: [] }]
  })
  const [currentId, setCurrentId] = useState<string>(()=>{
    const stored = localStorage.getItem(CURRENT_KEY)
    if(stored && sessions.find(s=>s.id===stored)) return stored
    return sessions[0].id
  })

  const currentSession = sessions.find(s=>s.id===currentId)! // guaranteed
  const messages = currentSession.messages

  // Model selection state
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([])
  const [currentModel, setCurrentModel] = useState<CurrentModel | null>(null)
  const [showModelSelector, setShowModelSelector] = useState(false)
  const mouseLeaveTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Persist sessions & current pointer
  useEffect(()=>{ localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions)) }, [sessions])
  useEffect(()=>{ localStorage.setItem(CURRENT_KEY, currentId) }, [currentId])

  // Helper to mutate current session messages
  function updateCurrentMessages(mutator:(msgs:Msg[])=>Msg[]){
    setSessions(all => all.map(s => s.id===currentId ? {...s, messages: mutator(s.messages)} : s))
  }

  // Session management
  function createSession(){
    const idx = sessions.length + 1
  const s: Session = { id: safeUUID(), name:`Session ${idx}`, created: Date.now(), messages: [] }
    setSessions(ss=>[...ss, s])
    setCurrentId(s.id)
  }
  function renameSession(id:string){
    const name = prompt('New session name?')?.trim(); if(!name) return
    setSessions(ss=> ss.map(s=> s.id===id ? {...s, name} : s))
  }
  function deleteSession(id:string){
    if(sessions.length===1){ alert('At least one session is required.'); return }
    if(!confirm('Delete this session and its messages?')) return
    setSessions(ss=> ss.filter(s=> s.id!==id))
    if(id===currentId){
      // pick another remaining session
      const remaining = sessions.filter(s=> s.id!==id)
      if(remaining.length) setCurrentId(remaining[0].id)
    }
  }

  // UI state
  const [loading, setLoading] = useState(false)
  const [useStream, setUseStream] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<string>('')
  const inputRef = useRef<HTMLInputElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController|null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  

  
  // Image upload for chat
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)

  // 🧠 Learning and growth related state
  const [showLearningPanel, setShowLearningPanel] = useState(false)
  const [learningStats, setLearningStats] = useState<any>(null)
  const [growthInsights, setGrowthInsights] = useState<any>(null)

  // 📁 File management related state
  const [showFilePanel, setShowFilePanel] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([])
  const [processingFiles, setProcessingFiles] = useState(false)

  // Auto scroll on message or loading state changes
  useEffect(()=>{ bottomRef.current?.scrollIntoView({behavior:'smooth'}) }, [messages, loading, currentId])

  // Auto-fetch learning stats when panel is opened or session changes
  useEffect(() => {
    if (showLearningPanel) {
      fetchLearningStats()
    }
  }, [showLearningPanel, currentId])

  // 📁 File management related state

  // 🧠 Learning statistics related functions
  async function fetchLearningStats() {
    try {
      // DEBUG: Always use test session ID for now
      const testSessionId = 'f47ac10b-58cc-4372-a567-0e02b2c3d479'
      console.log('🔍 Fetching learning stats for session:', testSessionId, '(original currentId:', currentId, ')')
      const [statsRes, insightsRes] = await Promise.all([
        fetch(`http://localhost:8000/api/learning_stats/${testSessionId}`),
        fetch(`http://localhost:8000/api/growth_insights/${testSessionId}`)
      ])
      
      console.log('📊 Stats response:', statsRes.status, statsRes.ok)
      console.log('💡 Insights response:', insightsRes.status, insightsRes.ok)
      
      if (statsRes.ok) {
        const stats = await statsRes.json()
        console.log('📈 Learning stats data:', stats)
        setLearningStats(stats)
      } else {
        console.error('❌ Stats API error:', statsRes.status, await statsRes.text())
      }
      
      if (insightsRes.ok) {
        const insights = await insightsRes.json()
        console.log('💭 Growth insights data:', insights)
        setGrowthInsights(insights)
      } else {
        console.error('❌ Insights API error:', insightsRes.status, await insightsRes.text())
      }
    } catch (e) {
      console.error('Failed to fetch learning stats:', e)
    }
  }

  async function sendFeedback(conversationId: string, type: 'positive' | 'negative', content: string = '') {
    try {
      await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: conversationId,
          feedback_type: type,
          content: content || (type === 'positive' ? 'Helpful response' : 'Needs improvement')
        })
      })
      // Re-fetch learning statistics
      await fetchLearningStats()
    } catch (e) {
      console.error('Failed to send feedback:', e)
    }
  }





  // Enhanced dropdown mouse handling
  function handleModelSelectorMouseEnter() {
    if (mouseLeaveTimeoutRef.current) {
      clearTimeout(mouseLeaveTimeoutRef.current)
      mouseLeaveTimeoutRef.current = null
    }
  }

  function handleModelSelectorMouseLeave() {
    // Add a small delay before closing to prevent accidental closure
    mouseLeaveTimeoutRef.current = setTimeout(() => {
      setShowModelSelector(false)
    }, 200) // 200ms delay
  }

  // 🤖 Model selection related functions
    // Fetch available models from backend
  const fetchAvailableModels = async () => {
    try {
      console.log('🔍 Fetching available models...');
      const response = await fetch('/api/models');
      if (!response.ok) throw new Error(`Failed to fetch models: ${response.status}`);
      const data = await response.json();
      const models: ModelInfo[] = data.models || data; // Handle both {models: []} and [] formats
      console.log('🎯 Processed models:', models.length, 'models loaded');
      console.log('🔍 Available models data:', models);
      setAvailableModels(models);
      
      // Also fetch current model info
      await fetchCurrentModel();
    } catch (error) {
      console.error('❌ Error fetching models:', error);
    }
  };

  async function fetchCurrentModel() {
    try {
      const response = await fetch('/api/model_status')
      if (response.ok) {
        const data = await response.json()
        const modelInfo: CurrentModel = data.current_model
        setCurrentModel(modelInfo)
        console.log('📊 Current model fetched:', modelInfo)
      }
    } catch (e) {
      console.error('Failed to fetch current model:', e)
    }
  }

  async function selectModel(modelId: string) {
    try {
      const response = await fetch('/api/models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId })
      })
      if (response.ok) {
        setShowModelSelector(false)
        await fetchCurrentModel() // Refresh current model info
        console.log('✅ Model switched to:', modelId)
      } else {
        console.error('Failed to switch model')
      }
    } catch (e) {
      console.error('Failed to select model:', e)
    }
  }

  async function enableAutoModel() {
    try {
      const response = await fetch('/api/models/auto', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      if (response.ok) {
        setShowModelSelector(false)
        await fetchCurrentModel() // Refresh current model info
        console.log('✅ Auto model enabled')
      } else {
        console.error('Failed to enable auto mode')
      }
    } catch (e) {
      console.error('Failed to enable auto model:', e)
    }
  }

  // Auto-fetch models on component mount
  useEffect(() => {
    fetchAvailableModels()
    fetchCurrentModel()
  }, [])



  // Cleanup mouse leave timeout on unmount
  useEffect(() => {
    return () => {
      if (mouseLeaveTimeoutRef.current) {
        clearTimeout(mouseLeaveTimeoutRef.current)
      }
    }
  }, [])

  // 📁 File management related functions
  async function fetchUploadedFiles() {
    try {
      const response = await fetch('/api/uploaded_files')
      if (response.ok) {
        const data = await response.json()
        setUploadedFiles(data.files || [])
      }
    } catch (e) {
      console.error('Failed to fetch uploaded files:', e)
    }
  }

  async function processExistingFiles() {
    setProcessingFiles(true)
    try {
      const response = await fetch('/api/process_existing_files', { method: 'POST' })
      if (response.ok) {
        const result = await response.json()
        alert(`Processing Complete!\nSuccess: ${result.successful_files}/${result.files_processed} files\nLearned ${result.total_chunks_created} knowledge chunks`)
        // Re-fetch file list and learning statistics
        await fetchUploadedFiles()
        await fetchLearningStats()
      } else {
        alert('Batch processing failed, please check backend logs')
      }
    } catch (e) {
      console.error('Failed to process files:', e)
      alert('Batch processing error: ' + (e instanceof Error ? e.message : String(e)))
    } finally {
      setProcessingFiles(false)
    }
  }

  // Send message (streaming or non-streaming)
  async function send(){
    const text = (inputRef.current?.value || '').trim()
    if((!text && !selectedImage) || loading) return
    
    // Prepare user message content
    let userMessageText = text
    if (selectedImage) {
      userMessageText = `${text ? text + '\n\n' : ''}[Image: ${selectedImage.name}]`
    }
    
    // append user message
    let botIndex = -1
    let botMessageAdded = false
    updateCurrentMessages(msgs=>{
      const userMsg: Msg = { who: 'user', text: userMessageText }
      const next: Msg[] = [...msgs, userMsg]
      return next
    })
    
    // Don't add placeholder bot message, just use loading indicator
    if(inputRef.current) inputRef.current.value=''
    setLoading(true)

    if(useStream){
      abortRef.current = new AbortController()
      const controller = abortRef.current
      
      // Set a timeout to prevent hanging streams (30 seconds)
      const streamTimeout = setTimeout(() => {
        console.log('Stream timeout - aborting')
        controller.abort()
      }, 30000)
      
      try {
        let resp: Response
        if (selectedImage) {
          // Use FormData for image upload
          const formData = new FormData()
          formData.append('session_id', currentId)
          formData.append('message', text || '')
          formData.append('max_history', '8')
          formData.append('stream', 'true')
          formData.append('image', selectedImage)
          
          console.log('🖼️ Sending image upload request:', {
            sessionId: currentId,
            message: text,
            imageName: selectedImage.name,
            imageSize: selectedImage.size
          })
          
          resp = await fetch('/api/chat_stream_with_image', {
            method: 'POST',
            body: formData,
            signal: controller.signal
          })
          
          console.log('📥 Image upload response status:', resp.status, resp.ok)
          console.log('📋 Response headers:', Object.fromEntries(resp.headers.entries()))
        } else {
          // Regular JSON request
          resp = await fetch('/api/chat_stream', {
            method:'POST', 
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ session_id: currentId, message: text, max_history: 8, stream: true }),
            signal: controller.signal
          })
        }
        if(!resp.ok) throw new Error(`HTTP ${resp.status}`)
        
        console.log('🌊 Starting stream processing for image upload...')
        const reader = resp.body!.getReader()
        const decoder = new TextDecoder('utf-8')
        let buffer = ''
        let streamCompleted = false
        let chunkCount = 0
        const MAX_CHUNKS = 1000 // Safety limit to prevent infinite loops
        
        while(true){
          const {done, value} = await reader.read()
          chunkCount++
          
          if(done) {
            console.log(`Stream naturally ended after ${chunkCount} chunks`)
            break
          }
          
          if(streamCompleted) {
            console.log(`Stream completed via done event after ${chunkCount} chunks`)
            break
          }
          
          if(chunkCount > MAX_CHUNKS) {
            console.warn(`Stream exceeded maximum chunks (${MAX_CHUNKS}), forcing completion`)
            break
          }
          
            buffer += decoder.decode(value, {stream:true})
          
            // Handle malformed SSE chunks that don't end with \n\n
            // Check if buffer contains data lines without proper event structure
            if(buffer.includes('data:') && !buffer.includes('\n\n')){
              // Split by individual lines to handle malformed chunks
              const lines = buffer.split('\n')
              const processedLines: number[] = []
              let remainder = ''
              
              for(let i = 0; i < lines.length; i++){
                const line = lines[i].trim()
                if(line.startsWith('data:') && !line.includes('event:')){
                  // This is a malformed individual data chunk
                  const rawData = line.slice(5) // Remove "data:"
                  const cleanData = rawData.startsWith(' ') ? rawData.slice(1) : rawData
                  if(cleanData) {
                    console.log('🔧 Processing individual malformed chunk:', JSON.stringify(cleanData))
                    
                    updateCurrentMessages(msgs => {
                      if (!botMessageAdded) {
                        botMessageAdded = true
                        return [...msgs, { who: 'bot' as const, text: cleanData }]
                      } else {
                        const lastIndex = msgs.length - 1
                        return msgs.map((mm, idx) => 
                          idx === lastIndex && mm.who === 'bot' 
                            ? {...mm, text: mm.text + cleanData} 
                            : mm
                        )
                      }
                    })
                  }
                  processedLines.push(i)
                }
              }
              
              // Remove processed lines from buffer
              if(processedLines.length > 0){
                const remainingLines = lines.filter((_, i) => !processedLines.includes(i))
                buffer = remainingLines.join('\n')
              }
            }
          
            // Normal SSE processing for properly formatted events
            const parts = buffer.split('\n\n')
            buffer = parts.pop() || ''
            for(const raw of parts){
              if(!raw.trim()) continue
              const lines = raw.split('\n')
              let event: string | null = null
              let data = ''
              for(const ln of lines){
                if(ln.startsWith('event:')) event = ln.slice(6).trim()
                else if(ln.startsWith('data:')) {
                  // Extract data content, remove the first space after "data:", but keep spaces within tokens
                  const rawData = ln.slice(5) // Remove "data:"
                  data += rawData.startsWith(' ') ? rawData.slice(1) : rawData
                }
              }
              
              // Debug: Log every event and data we receive
              console.log('🔍 Raw event part:', JSON.stringify(raw))
              console.log('🎯 Parsed - Event:', event, 'Data:', JSON.stringify(data))
              if(event === 'citations'){
                // Skip citations - we don't want to display them
                continue
              }
              if(event === 'model_info'){
                // Skip model info events - we don't want to display them as message content
                continue
              }
              if(event === 'done') {
                // Stream completed, set flag to exit main loop
                console.log('Stream completed - received done event')
                streamCompleted = true
                break
              }
              if(data){
                // Add the data chunk - add bot message if this is the first chunk
                console.log('📝 Processing streaming chunk:', JSON.stringify(data))
                console.log('🤖 Bot message added status:', botMessageAdded)
                
                // Force a state update to ensure message appears in UI
                updateCurrentMessages(msgs => {
                  console.log('📝 Current messages before update:', msgs.length)
                  let newMessages;
                  
                  if (!botMessageAdded) {
                    // First chunk - add new bot message
                    botMessageAdded = true
                    console.log('➕ Adding first bot message with data:', JSON.stringify(data))
                    newMessages = [...msgs, { who: 'bot' as const, text: data }]
                  } else {
                    // Subsequent chunks - append to last bot message
                    const lastIndex = msgs.length - 1
                    console.log('➕ Appending to existing bot message at index:', lastIndex)
                    newMessages = msgs.map((mm, idx) => 
                      idx === lastIndex && mm.who === 'bot' 
                        ? {...mm, text: mm.text + data} 
                        : mm
                    )
                  }
                  
                  console.log('🔄 Updated messages count:', newMessages.length)
                  console.log('🔄 Last message preview:', newMessages[newMessages.length - 1]?.text?.substring(0, 50) + '...')
                  return newMessages
                })
              }
            }
        }
      } catch(e:any){
        if(e.name !== 'AbortError'){
          updateCurrentMessages(msgs=>[...msgs, {who:'bot', text:'Streaming failed: '+(e.message||e.toString())}])
        }
      } finally {
        // Clear the stream timeout
        clearTimeout(streamTimeout)
        setLoading(false)
        abortRef.current = null
        // Clear selected image after sending
        clearSelectedImage()
      }
    } else {
      // non-stream path
      try {
        let resp: Response
        if (selectedImage) {
          // Use FormData for image upload
          const formData = new FormData()
          formData.append('session_id', currentId)
          formData.append('message', text || '')
          formData.append('max_history', '8')
          formData.append('image', selectedImage)
          
          resp = await fetch('/api/chat_with_image', {
            method: 'POST',
            body: formData
          })
        } else {
          // Regular JSON request
          resp = await fetch('/api/chat', { 
            method:'POST', 
            headers:{'Content-Type':'application/json'}, 
            body: JSON.stringify({ session_id: currentId, message: text, max_history: 8 }) 
          })
        }
        
        if(!resp.ok) throw new Error(`HTTP ${resp.status}`)
        const data = await resp.json() as ChatResp
        // fill placeholder we added earlier? In non-stream we didn't yet add placeholder, so add full bot message.
        updateCurrentMessages(msgs=>[...msgs, {who:'bot', text: data.answer || '[No response]'}])
      } catch(e:any){
        updateCurrentMessages(msgs=>[...msgs, {who:'bot', text:'Request failed: '+(e.message||e.toString())}])
      } finally { 
        setLoading(false)
        // Clear selected image after sending
        clearSelectedImage()
      }
    }
  }

  function abortStream(){ abortRef.current?.abort() }

  // File upload handler
  async function handleFileUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const files = event.target.files
    if (!files || files.length === 0) return

    setUploading(true)
    setUploadStatus('Uploading...')

    const formData = new FormData()
    Array.from(files).forEach(file => {
      formData.append('files', file)
    })

    try {
      const resp = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      })
      
      if (!resp.ok) {
        const error = await resp.text()
        throw new Error(error || `HTTP ${resp.status}`)
      }

      const result = await resp.json()
      setUploadStatus(`✅ Uploaded ${result.files_count} file(s)`)
      
      // Add system message to current session
      updateCurrentMessages(msgs => [...msgs, {
        who: 'bot',
        text: `📄 **Files uploaded successfully!**\n\n${result.filenames.map((f: string) => `- ${f}`).join('\n')}\n\nThese documents are now available for the RAG assistant to reference.`
      }])

      setTimeout(() => setUploadStatus(''), 3000)
    } catch (e: any) {
      setUploadStatus(`❌ Upload failed: ${e.message}`)
      setTimeout(() => setUploadStatus(''), 5000)
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  function triggerFileUpload() {
    fileInputRef.current?.click()
  }

  // Image upload for chat functionality
  function handleImageSelect(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file)
      
      // Create preview
      const reader = new FileReader()
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string)
      }
      reader.readAsDataURL(file)
    } else if (file) {
      alert('Please select an image file')
    }
  }

  function triggerImageUpload() {
    imageInputRef.current?.click()
  }

  function clearSelectedImage() {
    setSelectedImage(null)
    setImagePreview(null)
    if (imageInputRef.current) {
      imageInputRef.current.value = ''
    }
  }

  function clearHistory(){
    if(!messages.length) return
    if(confirm('Clear messages in this session?')) updateCurrentMessages(()=>[])
  }

  function clearAllSessions(){
    if(confirm('Delete ALL sessions and messages?')){
  const fresh: Session = { id: safeUUID(), name:'Session 1', created: Date.now(), messages: [] }
      setSessions([fresh])
      setCurrentId(fresh.id)
    }
  }

  return (
    <div className="flex" style={{height:'100vh'}}>

      
      {/* Sidebar */}
      <div className="sidebar">
        <div className="sidebar-header">
          <span>Sessions</span>
          <div style={{ display: 'flex', gap: '4px', marginLeft: 'auto' }}>
            <button 
              className="icon-btn" 
              onClick={() => {
                const newShowFilePanel = !showFilePanel
                setShowFilePanel(newShowFilePanel)
                if (newShowFilePanel) {
                  // Close learning panel when opening file panel
                  setShowLearningPanel(false)
                  fetchUploadedFiles()
                }
              }}
              title="File Manager"
            >
              📁
            </button>
            <button 
              className="icon-btn" 
              onClick={() => {
                const newShowLearningPanel = !showLearningPanel
                setShowLearningPanel(newShowLearningPanel)
                if (newShowLearningPanel) {
                  // Close file panel when opening learning panel
                  setShowFilePanel(false)
                  fetchLearningStats()
                }
              }}
              title="AI Growth Stats"
            >
              🧠
            </button>
          </div>
        </div>
        <div className="sidebar-list">
          {sessions.map(s=> (
            <div 
              key={s.id} 
              className={`session-item ${s.id===currentId?'active':''}`}
              onClick={()=>setCurrentId(s.id)}
            >
              <span className="session-name">{s.name}</span>
              <div className="session-actions">
                <button 
                  className="icon-btn" 
                  onClick={(e)=>{e.stopPropagation(); renameSession(s.id)}} 
                  title="Rename"
                >
                  ✏️
                </button>
                <button 
                  className="icon-btn" 
                  onClick={(e)=>{e.stopPropagation(); deleteSession(s.id)}} 
                  title="Delete"
                >
                  🗑️
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Management panels container */}
        <div className="panels-container">
          {/* 📁 File management panel */}
          {showFilePanel && (
            <div className="learning-panel">
              <div className="learning-header">
                <h3>📁 File Manager</h3>
                <button 
                  className="icon-btn" 
                  onClick={() => setShowFilePanel(false)}
                  title="Close"
                >
                  ❌
                </button>
              </div>
            
              <div className="learning-stats">
                <div className="stat-item">
                  <span className="stat-label">Uploaded Files:</span>
                  <span className="stat-value">{uploadedFiles.length}</span>
                </div>
              
                <div style={{ marginTop: '12px' }}>
                  <button 
                    className="btn btn-primary" 
                    onClick={processExistingFiles}
                    disabled={processingFiles}
                    style={{ width: '100%', marginBottom: '8px' }}
                  >
                    {processingFiles ? '🔄 Processing...' : '🧠 Learn All Files'}
                  </button>
                  <button 
                    className="btn btn-secondary" 
                    onClick={fetchUploadedFiles}
                    style={{ width: '100%', fontSize: '12px' }}
                  >
                    🔄 Refresh File List
                  </button>
                </div>              {uploadedFiles.length > 0 && (
                <div className="file-list">
                  <h4>📄 File List:</h4>
                  {uploadedFiles.map((file, idx) => (
                    <div key={idx} className="file-item">
                      <div className="file-name">{file.filename}</div>
                      <div className="file-info">
                        <span className="file-size">{(file.size / 1024).toFixed(1)}KB</span>
                        <span className="file-type">{file.file_type}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div className="file-help">
                <h4>💡 Supported File Types:</h4>
                <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                  • .txt, .md - Text documents<br/>
                  • .py - Python code<br/>
                  • .jpg, .jpeg, .png, .gif, .bmp, .webp - Images (AI analysis)<br/>
                  • PDF and DOCX need to be converted to text format
                </div>
              </div>
            </div>
          </div>
        )}

          {/* 🧠 Learning statistics panel */}
          {showLearningPanel && (
            <div className="stats-panel">
              <div className="learning-header">
                <h3>🧠 AI Growth Stats</h3>
                <button 
                  className="icon-btn" 
                  onClick={() => setShowLearningPanel(false)}
                  title="Close"
                >
                  ❌
                </button>
              </div>
              
              {learningStats ? (
                <div className="learning-stats">
                  <div className="stat-item">
                    <span className="stat-label">Total Conversations:</span>
                    <span className="stat-value">{learningStats.total_conversations}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Learned Concepts:</span>
                    <span className="stat-value">{learningStats.learned_concepts}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Feedback Received:</span>
                    <span className="stat-value">{learningStats.total_feedback}</span>
                  </div>
                  
                  {learningStats.user_style_analysis && (
                    <div className="style-analysis">
                      <h4>📊 Your Communication Style:</h4>
                      <div className="style-item">
                        <span>Style: {learningStats.user_style_analysis.analysis.communication_style}</span>
                      </div>
                      <div className="style-item">
                        <span>Detail Level: {learningStats.user_style_analysis.analysis.detail_preference}</span>
                      </div>
                      <div className="style-item">
                        <span>Friendliness: {learningStats.user_style_analysis.analysis.friendliness_level}</span>
                      </div>
                    </div>
                  )}

                  {growthInsights?.recommendations && growthInsights.recommendations.length > 0 && (
                    <div className="recommendations">
                      <h4>💡 Growth Recommendations:</h4>
                      {growthInsights.recommendations.map((rec: string, idx: number) => (
                        <div key={idx} className="recommendation-item">{rec}</div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="loading-stats">
                  <p>Loading AI growth statistics...</p>
                  <p><small>Session ID: {currentId}</small></p>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          <button className="btn btn-primary" onClick={createSession}>
            + New Session
          </button>
          <button className="btn btn-danger" onClick={clearAllSessions}>
            Reset All
          </button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="chat-container">
        <div className="chat-header">
          <div className="chat-title">RAG Chat Assistant</div>
          <div className="session-label">{currentSession.name}</div>
          <div className="chat-controls">
            {/* File Upload Button */}
            <button 
              className="btn btn-upload" 
              onClick={triggerFileUpload}
              disabled={uploading}
              title="Upload documents and images for RAG learning"
            >
              📤 {uploading ? 'Uploading...' : 'Upload Files & Images'}
            </button>
            <input 
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              multiple
              accept=".txt,.pdf,.doc,.docx,.md,.jpg,.jpeg,.png,.gif,.bmp,.webp"
              style={{ display: 'none' }}
            />
            {uploadStatus && <span className="upload-status">{uploadStatus}</span>}
            
            {/* Model Selector */}
            <div 
              className="model-selector"
              onMouseEnter={handleModelSelectorMouseEnter}
              onMouseLeave={handleModelSelectorMouseLeave}
            >
              <button 
                className="btn btn-secondary model-toggle" 
                onClick={() => setShowModelSelector(!showModelSelector)}
                title="Select AI Model"
              >
                🤖 {currentModel ? currentModel.model_name : 'Select Model'}
              </button>
              {showModelSelector && (
                <div className="model-dropdown">
                  <div style={{padding: '4px 8px', fontSize: '10px', color: '#666', borderBottom: '1px solid #eee'}}>
                    Models loaded: {availableModels.length}
                    {availableModels.length === 0 && <div style={{color: 'red'}}>⚠️ No models found!</div>}
                  </div>
                  <div className="model-option auto-option">
                    <button 
                      className="model-btn auto-btn"
                      onClick={enableAutoModel}
                      title="Enable automatic model selection with fallback"
                    >
                      🔄 Auto Mode
                    </button>
                  </div>
                  {availableModels.map((model) => (
                    <div key={model.id} className="model-option">
                      <button 
                        className="model-btn"
                        onClick={() => selectModel(model.id)}
                        title={`${model.description} - ${model.status || 'Available'}`}
                      >
                        <span className="model-name">{model.display_name}</span>
                        <span className="model-status">
                          {model.is_local ? '🦙' : 
                           model.status === 'ready' ? '☁️' : 
                           model.status === 'quota_exceeded' ? '🚫' : 
                           '⚠️'
                          }
                        </span>
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            <label className="checkbox-label">
              <input 
                type="checkbox" 
                checked={useStream} 
                onChange={e=>setUseStream(e.target.checked)} 
              />
              <span>Streaming</span>
            </label>
            {useStream && loading && (
              <button className="btn btn-secondary" onClick={abortStream}>
                Abort
              </button>
            )}
            <button 
              className="btn btn-secondary" 
              onClick={clearHistory} 
              disabled={loading}
            >
              Clear
            </button>
          </div>
        </div>

        <div className="messages-container">
          {messages.map((m,i)=> (
            <div key={i} className={`message-wrapper ${m.who}`}>
              <div className="message-bubble">
                <div className="message-meta">
                  <div className="message-avatar">
                    {m.who==='user' ? (
                      <div className="avatar-user">
                        <svg viewBox="0 0 24 24" fill="none" className="avatar-icon">
                          <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" fill="currentColor"/>
                        </svg>
                      </div>
                    ) : (
                      <div className="avatar-bot">
                        <svg viewBox="0 0 24 24" fill="none" className="avatar-icon">
                          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.94-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z" fill="currentColor"/>
                        </svg>
                      </div>
                    )}
                  </div>
                  <div className="message-info">
                    <span className="sender-name">{m.who==='user' ? 'You' : 'AI Assistant'}</span>
                    <span className="message-time">{new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                  </div>
                </div>
                <div className="message-content">
                  <div className="message-text">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {m.text || ''}
                    </ReactMarkdown>
                  </div>
                  <div className="message-status">
                    {m.who === 'user' && (
                      <svg viewBox="0 0 16 16" className="check-icon">
                        <path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0z" fill="currentColor"/>
                      </svg>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
          {loading ? (
            <div className="message-wrapper bot">
              <div className="message-bubble loading-message">
                <div className="message-meta">
                  <div className="message-avatar">
                    <div className="avatar-bot">
                      <svg viewBox="0 0 24 24" fill="none" className="avatar-icon">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.94-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z" fill="currentColor"/>
                      </svg>
                    </div>
                  </div>
                  <div className="message-info">
                    <span className="sender-name">AI Assistant</span>
                    <span className="message-time">Typing...</span>
                  </div>
                </div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <div className="typing-dots">
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                    </div>
                    <span className="typing-text">AI is thinking...</span>
                  </div>
                </div>
              </div>
            </div>
          ) : null}
          <div ref={bottomRef} />
        </div>

        <div className="input-area">
          {/* Image preview area */}
          {imagePreview && (
            <div className="image-preview-container">
              <div className="image-preview">
                <img src={imagePreview} alt="Selected" className="preview-image" />
                <button 
                  className="remove-image-btn" 
                  onClick={clearSelectedImage}
                  title="Remove image"
                >
                  ❌
                </button>
              </div>
              <span className="image-name">{selectedImage?.name}</span>
            </div>
          )}
          
          <div className="input-wrapper">
            <button 
              className="image-upload-btn" 
              onClick={triggerImageUpload}
              disabled={loading}
              title="Upload image"
            >
              📷
            </button>
            <input 
              ref={inputRef}
              className="input-field"
              placeholder="Ask me anything..." 
              onKeyDown={e=>{ if(e.key==='Enter' && !e.shiftKey) { e.preventDefault(); send() } }} 
            />
            <button 
              className="btn btn-primary send-btn" 
              onClick={send} 
              disabled={loading}
            >
              {loading ? 'Sending...' : 'Send'}
            </button>
          </div>
          
          {/* Hidden image input */}
          <input 
            ref={imageInputRef}
            type="file" 
            accept="image/*"
            onChange={handleImageSelect}
            style={{display: 'none'}}
          />
        </div>
      </div>
    </div>
  )
}
