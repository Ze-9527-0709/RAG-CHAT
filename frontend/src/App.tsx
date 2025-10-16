import './crypto-polyfill'
import './styles.css'
import React, { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// Data model types
type ChatResp = { answer: string }
type Msg = { who:'user'|'bot'; text:string }
type Session = { id:string; name:string; created:number; messages: Msg[] }

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
    updateCurrentMessages(msgs=>{
      const userMsg: Msg = { who: 'user', text: userMessageText }
      const next: Msg[] = [...msgs, userMsg]
      botIndex = next.length // placeholder index after we push bot placeholder
      return next
    })
    
    if(inputRef.current) inputRef.current.value=''
    setLoading(true)

    if(useStream){
      abortRef.current = new AbortController()
      const controller = abortRef.current
      // add placeholder bot message
      updateCurrentMessages(msgs=>[...msgs, {who:'bot', text:''}])
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
          
          resp = await fetch('/api/chat_stream_with_image', {
            method: 'POST',
            body: formData,
            signal: controller.signal
          })
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
        const reader = resp.body!.getReader()
        const decoder = new TextDecoder('utf-8')
        let buffer = ''
        while(true){
          const {done, value} = await reader.read()
            if(done) break
            buffer += decoder.decode(value, {stream:true})
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
                  data = rawData.startsWith(' ') ? rawData.slice(1) : rawData
                }
              }
              if(event === 'citations'){
                // Skip citations - we don't want to display them
                continue
              }
              if(event === 'model_info'){
                // Skip model info events - we don't want to display them as message content
                continue
              }
              if(event === 'done') continue
              if(data){
                // Add the data chunk directly - backend already includes proper spacing
                console.log('Streaming chunk:', JSON.stringify(data))
                updateCurrentMessages(msgs => msgs.map((mm, idx)=> idx===botIndex ? {...mm, text: mm.text + data} : mm))
              }
            }
        }
      } catch(e:any){
        if(e.name !== 'AbortError'){
          updateCurrentMessages(msgs=>[...msgs, {who:'bot', text:'Streaming failed: '+(e.message||e.toString())}])
        }
      } finally {
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
            <div key={i} className={`message ${m.who}`}>
              <div className="message-header">
                <div className="message-avatar">
                  {m.who==='user' ? '👤' : '🤖'}
                </div>
                <span>{m.who==='user' ? 'You' : 'Assistant'}</span>
              </div>
              <div className="message-content">
                <div className="markdown-body">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {m.text || ''}
                  </ReactMarkdown>
                </div>
              </div>
            </div>
          ))}
          {loading ? (
            <div className="loading-indicator">
              <div className="loading-dots">
                <div className="loading-dot"></div>
                <div className="loading-dot"></div>
                <div className="loading-dot"></div>
              </div>
              <span>Generating response...</span>
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
