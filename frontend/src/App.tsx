import './crypto-polyfill'
import './styles.css'
import './performance-styles.css'
import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// Data model types
type ChatResp = { answer: string }
type Msg = { 
  who:'user'|'bot'; 
  text:string; 
  id?: string;
}
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
  id: string
  name: string
  display_name: string
  description: string
  is_local: boolean
  max_tokens: number
  cost_per_1k: number
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

  // Performance-optimized current session and messages
  const currentSession = useMemo(() => 
    sessions.find(s => s.id === currentId) || sessions[0], [sessions, currentId]
  )
  const messages = useMemo(() => currentSession?.messages || [], [currentSession])

  // Memoized message component to prevent unnecessary re-renders
  const MessageComponent = useMemo(() => React.memo(({ message }: { message: Msg }) => (
    <div className={`message-wrapper ${message.who}`} key={`${message.who}-${message.text?.slice(0, 20)}`}>
      <div className={`message-bubble ${message.who === 'user' ? 'user-message' : 'bot-message'}`}>
        <div className="message-meta">
          <div className="message-avatar">
            <div className={`avatar-${message.who}`}>
              {message.who === 'user' ? (
                <svg viewBox="0 0 24 24" fill="none" className="avatar-icon">
                  <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" fill="currentColor"/>
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" fill="none" className="avatar-icon">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.94-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z" fill="currentColor"/>
                </svg>
              )}
            </div>
          </div>
          <div className="message-info">
            <span className="sender-name">{message.who === 'user' ? 'You' : 'AUV'}</span>
            <span className="message-time">{new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
          </div>
        </div>
        <div className="message-content">
          <div className="message-text">
            {/* Handle image messages specially to prevent download behavior */}
            {message.text && message.text.includes('[Image:') ? (
              <div className="image-message">
                {message.text.split('\n\n').map((part, idx) => 
                  part.startsWith('[Image:') ? (
                    <div key={idx} className="image-indicator">
                      📷 {part.replace(/^\[Image:\s*|\]$/g, '')}
                    </div>
                  ) : part ? (
                    <ReactMarkdown key={idx} remarkPlugins={[remarkGfm]}>
                      {part}
                    </ReactMarkdown>
                  ) : null
                )}
              </div>
            ) : (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.text || ''}
              </ReactMarkdown>
            )}
          </div>
          <div className="message-status">
            {message.who === 'user' && (
              <svg viewBox="0 0 16 16" className="check-icon">
                <path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0z" fill="currentColor"/>
              </svg>
            )}
          </div>
        </div>
      </div>
    </div>
  ), (prevProps, nextProps) => {
    // Only re-render if message text actually changed
    return prevProps.message.text === nextProps.message.text && 
           prevProps.message.who === nextProps.message.who
  }), [])

  // Model selection state
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([])
  const [currentModel, setCurrentModel] = useState<CurrentModel | null>(null)
  const [showModelSelector, setShowModelSelector] = useState(false)
  const [isModelSwitching, setIsModelSwitching] = useState(false)

  // Persist sessions & current pointer
  useEffect(()=>{ localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions)) }, [sessions])
  useEffect(()=>{ localStorage.setItem(CURRENT_KEY, currentId) }, [currentId])

  // Helper to mutate current session messages (simplified for reliability)
  function updateCurrentMessages(mutator:(msgs:Msg[])=>Msg[]){
    setSessions(all => {
      const targetIndex = all.findIndex(s => s.id === currentId)
      if (targetIndex === -1) return all
      
      const targetSession = all[targetIndex]
      const newMessages = mutator(targetSession.messages)
      
      const newAll = [...all]
      newAll[targetIndex] = {...targetSession, messages: newMessages}
      return newAll
    })
  }

  // Session management - optimized with useCallback
  const createSession = useCallback(() => {
    const idx = sessions.length + 1
    const s: Session = { id: safeUUID(), name: `Session ${idx}`, created: Date.now(), messages: [] }
    setSessions(ss => [...ss, s])
    setCurrentId(s.id)
  }, [sessions.length])
  
  const renameSession = useCallback((id: string) => {
    const name = prompt('New session name?')?.trim(); if (!name) return
    setSessions(ss => ss.map(s => s.id === id ? { ...s, name } : s))
  }, [])
  
  const deleteSession = useCallback((id: string) => {
    if (sessions.length === 1) { alert('At least one session is required.'); return }
    if (!confirm('Delete this session and its messages?')) return
    setSessions(ss => ss.filter(s => s.id !== id))
    if (id === currentId) {
      // pick another remaining session
      const remaining = sessions.filter(s => s.id !== id)
      if (remaining.length) setCurrentId(remaining[0].id)
    }
  }, [sessions, currentId])

  // UI state
  const [loading, setLoading] = useState(false)
  const [useStream, setUseStream] = useState(true)
  const [uploading, setUploading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController|null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const modelSelectorRef = useRef<HTMLDivElement>(null)
  

  
  // Image upload for chat
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)

  // Drag and drop state
  const [isDragOver, setIsDragOver] = useState(false)

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





  // Simplified dropdown handling - click only (no mouse hover auto-close)
  function handleDropdownToggle() {
    setShowModelSelector(prev => !prev)
  }

  // Helper function to check if a model is currently selected
  function isModelSelected(modelId: string) {
    return currentModel && (
      currentModel.name === modelId || 
      currentModel.id === modelId ||
      modelId === currentModel.name
    )
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
      // Close dropdown immediately when user makes a choice
      setShowModelSelector(false)
      setIsModelSwitching(true)
      
      console.log('🔄 Attempting to select model:', modelId)
      
      const response = await fetch('/api/models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId })
      })
      
      if (response.ok) {
        const data = await response.json()
        console.log('✅ Model selection response:', data)
        
        // Update current model from response instead of making another request
        if (data.current_model) {
          setCurrentModel(data.current_model)
          console.log('✅ Model switched to:', data.current_model.name)
          
          // Show a brief success message to user
          const modelDisplayName = data.current_model.name === 'gpt-4o' ? 'GPT-4o' :
                                  data.current_model.name === 'gpt-4o-mini' ? 'GPT-4o Mini' :
                                  data.current_model.name === 'gpt-3.5-turbo' ? 'GPT-3.5 Turbo' :
                                  (data.current_model.name && data.current_model.name.includes('llama')) ? 'Local Llama' :
                                  data.current_model.display_name || data.current_model.name || 'Unknown Model'
          
          // Brief success notification (you can replace this with a toast library)
          const notification = document.createElement('div')
          notification.style.cssText = `
            position: fixed; top: 20px; right: 20px; z-index: 9999;
            background: linear-gradient(135deg, #059669, #10b981);
            color: white; padding: 12px 20px; border-radius: 8px;
            font-weight: 600; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transform: translateX(100%); transition: transform 0.3s ease;
          `
          notification.textContent = `✅ Switched to ${modelDisplayName}`
          document.body.appendChild(notification)
          
          // Animate in
          setTimeout(() => notification.style.transform = 'translateX(0)', 100)
          
          // Remove after 3 seconds
          setTimeout(() => {
            notification.style.transform = 'translateX(100%)'
            setTimeout(() => document.body.removeChild(notification), 300)
          }, 3000)
          
        } else {
          // Fallback to fetch if not included in response
          await fetchCurrentModel()
        }
      } else {
        const errorText = await response.text()
        console.error('❌ Failed to switch model:', response.status, errorText)
        
        // Show user-friendly error message
        alert(`Failed to switch to model: ${response.status} - ${errorText}`)
        
        // Reopen selector on error for better UX
        setShowModelSelector(true)
      }
    } catch (e) {
      console.error('❌ Error selecting model:', e)
      
      // Show user-friendly error message
      alert(`Error selecting model: ${e instanceof Error ? e.message : 'Unknown error'}`)
      
      // Reopen selector on error for better UX
      setShowModelSelector(true)
    } finally {
      setIsModelSwitching(false)
    }
  }

  // Auto-fetch models on component mount
  useEffect(() => {
    fetchAvailableModels()
    fetchCurrentModel()
  }, [])

  // Click outside to close model selector
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (modelSelectorRef.current && !modelSelectorRef.current.contains(event.target as Node)) {
        setShowModelSelector(false)
      }
    }

    if (showModelSelector) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => {
        document.removeEventListener('mousedown', handleClickOutside)
      }
    }
  }, [showModelSelector])

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
    
    // Store image reference before clearing
    const imageToSend = selectedImage
    
    // Prepare user message content
    let userMessageText = text
    if (imageToSend) {
      userMessageText = `${text ? text + '\n\n' : ''}[Image: ${imageToSend.name}]`
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
    
    // Clear selected image immediately after sending
    clearSelectedImageFunction()
    
    setLoading(true)

    if(useStream){
      abortRef.current = new AbortController()
      const controller = abortRef.current
      
      // Set a timeout to prevent hanging streams (30 seconds)
      const streamTimeout = setTimeout(() => {
        console.log('Stream timeout - aborting')
        controller.abort()
      }, 30000)
      
      // Optimized batched streaming updates
      let updateBuffer = ''
      let rafId: number | null = null
      let botMessageIndex = -1
      
      const flushBatchedUpdate = () => {
        if (updateBuffer.length === 0) return
        
        const textToAdd = updateBuffer
        updateBuffer = ''
        
        updateCurrentMessages(msgs => {
          if (botMessageIndex === -1) {
            // First chunk - add new bot message
            botMessageIndex = msgs.length
            console.log('➕ Adding first bot message')
            return [...msgs, { who: 'bot' as const, text: textToAdd }]
          } else {
            // Subsequent chunks - append to existing bot message
            const newMessages = [...msgs]
            if (newMessages[botMessageIndex]) {
              newMessages[botMessageIndex] = {
                ...newMessages[botMessageIndex], 
                text: newMessages[botMessageIndex].text + textToAdd
              }
            }
            return newMessages
          }
        })
      }
      
      const addTextToBot = (text: string) => {
        updateBuffer += text
        
        // Cancel previous RAF and schedule new one
        if (rafId) cancelAnimationFrame(rafId)
        rafId = requestAnimationFrame(flushBatchedUpdate)
      }
      
      try {
        let resp: Response
        if (imageToSend) {
          // Use FormData for image upload
          const formData = new FormData()
          formData.append('session_id', currentId)
          formData.append('message', text || '')
          formData.append('max_history', '8')
          formData.append('stream', 'true')
          formData.append('image', imageToSend)
          
          console.log('🖼️ Sending image upload request:', {
            sessionId: currentId,
            message: text,
            imageName: imageToSend.name,
            imageSize: imageToSend.size
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
        
        console.log('🌊 Starting stream processing...')
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
            setLoading(false)
            break
          }
          
          if(streamCompleted) {
            console.log(`🎯 Stream completed via done event after ${chunkCount} chunks`)
            console.log('🎯 Setting loading to false in main loop due to streamCompleted')
            // Ensure loading is cleared immediately when stream is marked complete
            setLoading(false)
            break
          }
          
          if(chunkCount > MAX_CHUNKS) {
            console.warn(`Stream exceeded maximum chunks (${MAX_CHUNKS}), forcing completion`)
            setLoading(false)
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
                    
                    // Add text immediately without batching
                    addTextToBot(cleanData)
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
                // Stream completed
                console.log('🎯 DONE EVENT RECEIVED - Setting loading to false')
                console.log('🎯 Current loading state before setting:', loading)
                setLoading(false)
                streamCompleted = true
                console.log('🎯 Done event processed, streamCompleted set to true')
                break
              }
              if(data){
                // Add the data chunk with optimized batching
                console.log('📝 Processing streaming chunk:', JSON.stringify(data))
                console.log('🤖 Bot message added status:', botMessageAdded)
                
                // Add text with batching
                addTextToBot(data)
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
        console.log('🎯 Setting loading to false in finally block')
        setLoading(false)
        abortRef.current = null
      }
    } else {
      // non-stream path
      try {
        let resp: Response
        if (imageToSend) {
          // Use FormData for image upload
          const formData = new FormData()
          formData.append('session_id', currentId)
          formData.append('message', text || '')
          formData.append('max_history', '8')
          formData.append('image', imageToSend)
          
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
      }
    }
  }

  // Optimized callback functions to prevent unnecessary re-renders
  const clearHistory = useCallback(() => {
    updateCurrentMessages(() => [])
  }, [])

  const abortStream = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
      setLoading(false)
    }
  }, [])

  const triggerImageUpload = useCallback(() => {
    imageInputRef.current?.click()
  }, [])

  // File upload handler
  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files || files.length === 0) return

    const fileNames = Array.from(files).map(f => f.name)
    const uploadMessageId = Date.now().toString()
    
    // Add upload message to chat
    updateCurrentMessages(msgs => [...msgs, {
      who: 'bot',
      text: `📤 **Uploading ${files.length} file(s)...**\n\n${fileNames.map(name => `📄 ${name}`).join('\n')}`,
      id: uploadMessageId
    }])

    setUploading(true)

    const formData = new FormData()
    Array.from(files).forEach(file => {
      formData.append('files', file)
    })

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`)
      }
      
      const result = await response.json()
      
      // Update chat with success message
      updateCurrentMessages(msgs => msgs.map(msg => 
        msg.id === uploadMessageId 
          ? {
              ...msg,
              text: `✅ **Files uploaded successfully!**\n\n${result.filenames.map((f: string) => `📄 ${f}`).join('\n')}\n\nThese documents are now available for the RAG assistant to reference.`
            }
          : msg
      ))
    } catch (e: any) {
      // Update chat with error message
      updateCurrentMessages(msgs => msgs.map(msg => 
        msg.id === uploadMessageId 
          ? {
              ...msg,
              text: `❌ **Upload failed: ${e.message}**\n\n${fileNames.map(name => `📄 ${name}`).join('\n')}`
            }
          : msg
      ))
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }, [])

  const triggerFileUpload = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  // Enhanced image selection for chat - following Copilot best practices
  const handleImageSelect = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    await processSelectedImage(file)
  }, [])

  const clearSelectedImageFunction = useCallback(() => {
    setSelectedImage(null)
    setImagePreview(null)
    if (imageInputRef.current) {
      imageInputRef.current.value = ''
    }
    console.log('🗑️ Cleared selected image')
  }, [])

  // Enhanced drag and drop functionality
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    // Only hide drag overlay if leaving the main container
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragOver(false)
    }
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)

    const files = Array.from(e.dataTransfer.files)
    const imageFile = files.find(file => file.type.startsWith('image/'))
    
    if (imageFile) {
      await processSelectedImage(imageFile)
    } else if (files.length > 0) {
      alert('Please drop an image file (PNG, JPG, GIF, WebP)')
    }
  }

  // Helper function to process image files consistently
  async function processSelectedImage(file: File) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file (PNG, JPG, GIF, WebP)')
      return
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024 // 10MB
    if (file.size > maxSize) {
      alert('Image size must be less than 10MB')
      return
    }

    try {
      // Create image preview
      const reader = new FileReader()
      reader.onload = (e) => {
        const result = e.target?.result as string
        setImagePreview(result)
      }
      reader.readAsDataURL(file)

      // Set selected image for chat
      setSelectedImage(file)
      
      // Log for debugging
      console.log('📷 Image selected for chat:', {
        name: file.name,
        size: file.size,
        type: file.type
      })

    } catch (e: any) {
      console.error('Error processing image:', e)
      alert(`Error processing image: ${e.message}`)
    }
  }

  const clearAllSessions = useCallback(() => {
    if (confirm('Delete ALL sessions and messages?')) {
      const fresh: Session = { id: safeUUID(), name: 'Session 1', created: Date.now(), messages: [] }
      setSessions([fresh])
      setCurrentId(fresh.id)
    }
  }, [])

  return (
    <div 
      className="flex" 
      style={{height:'100vh'}}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      
      {/* Drag and Drop Overlay */}
      {isDragOver && (
        <div className="drag-overlay">
          <div className="drag-content">
            <div className="drag-icon">📷</div>
            <div className="drag-text">Drop image here to chat</div>
            <div className="drag-subtext">PNG, JPG, GIF, WebP supported</div>
          </div>
        </div>
      )}

      
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
          {/* Current Model Indicator */}
          {currentModel && (
            <div className="current-model-indicator">
              <span className="model-indicator-label">Using:</span>
              <span className="model-indicator-name">
                {currentModel.name === 'gpt-4o' ? '🧠 GPT-4o' :
                 currentModel.name === 'gpt-4o-mini' ? '⚡ GPT-4o Mini' :
                 currentModel.name === 'gpt-3.5-turbo' ? '🚀 GPT-3.5 Turbo' :
                 (currentModel.name && currentModel.name.includes('llama')) ? '🦙 Local Llama' :
                 `🤖 ${currentModel.display_name || currentModel.name || 'Unknown'}`}
              </span>
            </div>
          )}
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
              className="hidden-file-input"
            />

            
            {/* Model Selector */}
            <div className="model-selector" ref={modelSelectorRef}>
              <button 
                className="btn btn-secondary model-toggle" 
                onClick={handleDropdownToggle}
                title="Select AI Model"
                disabled={isModelSwitching}
              >
                {isModelSwitching ? (
                  <span>🔄 Switching...</span>
                ) : currentModel ? (
                  <span>
                    {currentModel.name === 'gpt-4o' ? '� GPT-4o' :
                     currentModel.name === 'gpt-4o-mini' ? '⚡ GPT-4o Mini' :
                     currentModel.name === 'gpt-3.5-turbo' ? '🚀 GPT-3.5 Turbo' :
                     (currentModel.name && currentModel.name.includes('llama')) ? '🦙 Local Llama' :
                     `🤖 ${currentModel.display_name || currentModel.name || 'Unknown'}`}
                  </span>
                ) : (
                  <span>🤖 Select Model</span>
                )}
              </button>
              {showModelSelector && (
                <div className="model-dropdown">
                  <div style={{padding: '4px 8px', fontSize: '10px', color: '#666', borderBottom: '1px solid #eee'}}>
                    Models loaded: {availableModels.length}
                    {availableModels.length === 0 && <div style={{color: 'red'}}>⚠️ No models found!</div>}
                  </div>
                  {availableModels
                    .filter(model => model.available || model.status === 'ready')
                    .map((model) => (
                    <div key={model.id} className={`model-option ${isModelSelected(model.id) ? 'selected' : ''}`}>
                      <button 
                        className={`model-btn ${isModelSelected(model.id) ? 'selected' : ''}`}
                        onClick={() => selectModel(model.id)}
                        title={`${model.description} - ${model.status || 'Available'}`}
                        disabled={isModelSwitching || (!model.available && model.status !== 'ready')}
                      >
                        <span className="model-name">
                          {isModelSelected(model.id) && <span className="selected-indicator">✅ </span>}
                          {model.display_name}
                        </span>
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
                  {availableModels.filter(model => model.available || model.status === 'ready').length === 0 && (
                    <div className="model-option">
                      <div style={{padding: '8px', color: '#666', fontSize: '12px'}}>
                        No available models. Check your API configuration.
                      </div>
                    </div>
                  )}
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
          {messages.map((message, i) => (
            <MessageComponent key={`${message.who}-${i}-${message.text?.slice(0, 20)}`} message={message} />
          ))}
          {loading ? (
            <div className="message-wrapper bot thinking">
              <div className="message-bubble thinking-bubble">
                <div className="message-meta">
                  <div className="message-avatar">
                    <div className="avatar-bot thinking-avatar">
                      <svg viewBox="0 0 24 24" fill="none" className="avatar-icon">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.94-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z" fill="currentColor"/>
                      </svg>
                      <div className="brain-waves">
                        <div className="wave wave-1"></div>
                        <div className="wave wave-2"></div>
                        <div className="wave wave-3"></div>
                      </div>
                    </div>
                  </div>
                  <div className="message-info">
                    <span className="sender-name">AUV</span>
                    <span className="message-time thinking-status">
                      <div className="pulse-dot"></div>
                      Thinking...
                    </span>
                  </div>
                </div>
                <div className="message-content thinking-content">
                  <div className="thinking-animation">
                    <div className="thinking-dots">
                      <span className="dot"></span>
                      <span className="dot"></span>
                      <span className="dot"></span>
                    </div>
                    <div className="thinking-text">Analyzing your request</div>
                  </div>
                </div>
              </div>
            </div>
          ) : null}
          <div ref={bottomRef} />
        </div>

        <div className="input-area">
          {/* Enhanced image preview area */}
          {imagePreview && selectedImage && (
            <div className="image-preview-container">
              <div className="image-preview">
                <img src={imagePreview} alt="Selected image" className="preview-image" />
                <button 
                  className="remove-image-btn" 
                  onClick={clearSelectedImageFunction}
                  title="Remove image"
                >
                  ❌
                </button>
              </div>
              <div className="image-info">
                <span className="image-name">{selectedImage.name}</span>
                <span className="image-size">
                  {(selectedImage.size / 1024 / 1024).toFixed(1)} MB
                </span>
                <span className="image-ready-badge">✅ Ready to send</span>
              </div>
            </div>
          )}
          
          <div className="input-wrapper">
            <div className="image-upload-group">
              <button 
                className="image-upload-btn primary" 
                onClick={triggerImageUpload}
                disabled={loading}
                title="Upload image"
              >
                📷
              </button>
            </div>
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
            className="hidden-image-input"
          />
        </div>
      </div>
    </div>
  )
}
