import os
import asyncio
from typing import Dict, List
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ---- 智能模型降级系统 ----
from model_fallback import ModelFallbackManager, ModelTier, MODEL_CONFIGS, ModelConfig, estimate_tokens

# ---- 环境变量 ----
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Initialize OpenAI client with graceful handling of missing API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
    print("✅ OpenAI client initialized successfully")
else:
    client = None
    print("⚠️ No OpenAI API key found - OpenAI models will be unavailable")

# Initialize model fallback manager
model_manager = ModelFallbackManager(client)

# ---- RAG 组件（Pinecone + LangChain）----
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

USE_RAG = True
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "default-index")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# 延迟初始化向量存储和嵌入模型，避免启动时连接失败
embeddings = None
vectorstore = None
retriever = None

def init_vectorstore():
    global retriever
    try:
        # Initialize Pinecone client
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        pinecone_index = pc.Index(INDEX_NAME)
        
        # Initialize embeddings
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vectorstore = PineconeVectorStore(
            index=pinecone_index, 
            embedding=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        print("✅ Vectorstore initialized successfully")
        return True
    except Exception as e:
        print(f"⚠️ Failed to initialize vectorstore: {e}")
        print("🔧 RAG functionality will be disabled, but chat will continue to work")
        return False

# ---- 记忆和学习系统 ----
from memory_system import ConversationMemory, AdaptivePersonality

# ---- FastAPI 与内存会话 ----
app = FastAPI(title="RAG Chat Backend", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.on_event("startup")
async def startup_event():
    print("🚀 Starting RAG Chat Backend...")
    init_vectorstore()
    print("✅ Backend startup complete")

SESSIONS: Dict[str, List[Dict[str, str]]] = {}

# 初始化记忆系统
memory_system = ConversationMemory("conversations.db")
personality_system = AdaptivePersonality(memory_system)

class ChatRequest(BaseModel):
    session_id: str
    message: str
    max_history: int | None = 8

class ChatResponse(BaseModel):
    answer: str

class StreamChatRequest(ChatRequest):
    stream: bool | None = True

def _sse_format(event: str | None, data: str):
    if event:
        return f"event: {event}\ndata: {data}\n\n"
    return f"data: {data}\n\n"

@app.get("/health")
def health():
    return {"status": "ok"}

def retrieve_context(query: str, k: int = 4):
    try:
        if not USE_RAG or retriever is None:
            print(f"🔍 RAG disabled or retriever not available, using no context")
            docs = []
        else:
            docs = retriever.invoke(query)
            print(f"🔍 Retrieved {len(docs)} documents for query: '{query}'")
    except Exception as e:
        print(f"⚠️ RAG retrieval failed (falling back to no context): {e}")
        docs = []
    
    cites, parts = [], []
    for i, d in enumerate(docs, 1):
        src  = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        preview = d.page_content[:240].replace("\n", " ")
        cites.append({"source": src, "preview": preview, "page": str(page) if page is not None else ""})
        parts.append(f"[{i}] {d.page_content}\n(source: {src}" + (f", page: {page})" if page is not None else ")"))
    
    context = "\n\n".join(parts)
    print(f"📄 Context length: {len(context)} chars")
    return context, cites

def build_messages(session_id: str, user_msg: str, system_prompt: str, max_history: int):
    history = SESSIONS.get(session_id, [])
    hist = history[-(max_history*2):] if (max_history and max_history>0) else history
    return [{"role":"system","content":system_prompt}] + hist + [{"role":"user","content":user_msg}]

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 1) 检索上下文
    context, _ = retrieve_context(req.message, k=4)  # 忽略citations
    
    if context and context.strip():
        # 有相关知识库内容，使用RAG模式
        instruction = (
            "You are a helpful AI assistant. Use the Retrieved Context to answer the question when relevant. "
            "If the context contains relevant information, prioritize it in your answer. "
            "If the context is not relevant or helpful, you can also use your general knowledge to provide a helpful response."
        )
        system_prompt = f"{instruction}\n\nRetrieved Context:\n{context}"
    else:
        # 没有相关知识库内容，使用普通对话模式
        instruction = (
            "You are a helpful AI assistant. Answer questions using your knowledge and capabilities. "
            "Be conversational, helpful, and informative."
        )
        system_prompt = instruction

    # 2) 构造消息
    messages = build_messages(
        session_id=req.session_id, user_msg=req.message,
        system_prompt=system_prompt, max_history=req.max_history or 8
    )
    
    # 3) 智能模型选择和执行聊天 (with automatic fallback)
    estimated_tokens = estimate_tokens(str(messages))
    selected_tier, selection_reason = await model_manager.get_optimal_model(estimated_tokens)
    
    answer = None
    model_info = None
    max_fallback_attempts = 4  # Try up to 4 different models (all OpenAI + local)
    
    for attempt in range(max_fallback_attempts):
        try:
            print(f"🔄 Attempt {attempt + 1}: Trying {selected_tier.value}")
            # 4) 执行聊天
            completion = await model_manager.execute_chat(messages, selected_tier, stream=False)
            print(f"✅ Got completion from {selected_tier.value}")
            
            if selected_tier in [ModelTier.GPT4, ModelTier.GPT4_MINI, ModelTier.GPT35]:
                answer = completion.choices[0].message.content
            else:
                # Handle local model response (Ollama)
                try:
                    if hasattr(completion, 'json'):
                        # Ollama response
                        response_data = completion.json()
                        answer = response_data.get('response', 'Local model response')
                        print(f"✅ Local model response: {answer[:100]}...")
                    elif hasattr(completion, 'text'):
                        answer = completion.text
                        print(f"✅ Local model text: {answer[:100]}...")
                    else:
                        answer = str(completion)
                        print(f"✅ Local model string: {answer[:100]}...")
                except Exception as parse_error:
                    print(f"⚠️ Error parsing local model response: {parse_error}")
                    answer = "Local model responded but couldn't parse the response."
            
            # Success! Break out of retry loop
            model_info = model_manager.get_current_model_info()
            break
            
        except Exception as e:
            print(f"Model {selected_tier.value} failed (attempt {attempt + 1}): {str(e)}")
            
            # Check if this is a quota/rate limit error OR connection error
            error_str = str(e)
            is_recoverable_error = (
                "429" in error_str or 
                "insufficient_quota" in error_str or 
                "rate_limit" in error_str or
                "Connection error" in error_str or
                "timeout" in error_str.lower()
            )
            
            if is_recoverable_error:
                # Try to get next available model
                print(f"🔄 Recoverable error detected, attempting fallback...")
                try:
                    next_tier, next_reason = await model_manager.get_optimal_model(estimated_tokens)
                    if next_tier != selected_tier:
                        print(f"🔄 Switching from {selected_tier.value} to {next_tier.value}")
                        selected_tier = next_tier
                        selection_reason = next_reason
                        continue  # Try again with new model
                except Exception as fallback_error:
                    print(f"Fallback selection failed: {fallback_error}")
            
            # If this is the last attempt or not a recoverable error, fail
            if attempt == max_fallback_attempts - 1:
                answer = f"⚠️ All models temporarily unavailable. Error: {str(e)}"
                model_info = {"model_name": "fallback", "tier": "error"}
                break

    # 🧠 集成记忆和个性化系统 (only if we got a successful response)
    if answer and not answer.startswith("⚠️"):
        try:
            user_style = personality_system.analyze_user_style(req.session_id)
            relevant_memories = memory_system.get_relevant_memory(req.message, req.session_id, limit=3)
            
            # 如果有相关记忆，增强回答
            if relevant_memories:
                memory_context = "\n\n🧠 **Based on our previous conversations:**\n"
                for mem in relevant_memories[:2]:  # 只显示最相关的2个
                    memory_context += f"- {mem['user_message'][:50]}... → {mem['assistant_response'][:100]}...\n"
                answer = memory_context + answer
            
            # 根据用户风格调整回应
            answer = personality_system.adapt_response_style(answer, user_style)
            
            # Add model info with learning stats
            learning_stats = memory_system.get_learning_stats()
            answer += f"\n\n*[Used: {model_info['model_name']} - {selection_reason}]*"
            if learning_stats['total_conversations'] > 0:
                answer += f"\n*[💡 Learned from {learning_stats['total_conversations']} conversations, mastered {learning_stats['learned_concepts']} concepts]*"
        except Exception as memory_error:
            print(f"Memory/personality error (non-critical): {memory_error}")
    
    # Set default answer if nothing was set
    if not answer:
        answer = "⚠️ Unable to generate response. Please try again."

    # 🎯 存储到记忆系统
    try:
        memory_system.store_conversation(
            session_id=req.session_id,
            user_message=req.message,
            assistant_response=answer,
            context_used=context if context else None
        )
    except Exception as memory_error:
        print(f"Memory storage error: {memory_error}")

    # 5) 存会话 (保持原有功能)
    SESSIONS.setdefault(req.session_id, []).extend([
        {"role":"user","content":req.message},
        {"role":"assistant","content":answer}
    ])
    return ChatResponse(answer=answer)

@app.post("/api/chat_stream")
async def chat_stream(req: StreamChatRequest):
    # Build context
    context, _ = retrieve_context(req.message, k=4)  # 忽略citations
    
    if context and context.strip():
        # 有相关知识库内容，使用RAG模式
        instruction = (
            "You are a helpful AI assistant. Use the Retrieved Context to answer the question when relevant. "
            "If the context contains relevant information, prioritize it in your answer. "
            "If the context is not relevant or helpful, you can also use your general knowledge to provide a helpful response."
        )
        system_prompt = f"{instruction}\n\nRetrieved Context:\n{context}"
    else:
        # 没有相关知识库内容，使用普通对话模式
        instruction = (
            "You are a helpful AI assistant. Answer questions using your knowledge and capabilities. "
            "Be conversational, helpful, and informative."
        )
        system_prompt = instruction
    messages = build_messages(req.session_id, req.message, system_prompt, req.max_history or 8)

    async def token_stream():
        accumulated = []
        import json
        
        # 智能模型选择
        estimated_tokens = estimate_tokens(str(messages))
        selected_tier, selection_reason = await model_manager.get_optimal_model(estimated_tokens)
        
        # Send model info
        model_info = model_manager.get_current_model_info()
        yield _sse_format("model_info", json.dumps({
            "model": model_info['model_name'],
            "tier": model_info['tier'],
            "reason": selection_reason
        }))
        
        # Try multiple models with automatic fallback
        max_retries = 3
        current_tier = selected_tier
        success = False
        
        for retry_count in range(max_retries):
            try:
                # Execute streaming chat
                stream = await model_manager.execute_chat(messages, current_tier, stream=True)
                
                if current_tier in [ModelTier.GPT4, ModelTier.GPT4_MINI, ModelTier.GPT35]:
                    # Handle OpenAI streaming
                    for chunk in stream:
                        choice = chunk.choices[0]
                        delta = getattr(choice, 'delta', None) or getattr(choice, 'message', None) or None
                        if delta and delta.content:
                            text_part = delta.content
                            accumulated.append(text_part)
                            yield _sse_format(None, text_part)
                else:
                    # Handle local model streaming (Ollama format)
                    import json as json_lib
                    for line in stream.iter_lines():
                        if line:
                            try:
                                chunk_data = json_lib.loads(line)
                                if 'response' in chunk_data:
                                    text_part = chunk_data['response']
                                    accumulated.append(text_part)
                                    yield _sse_format(None, text_part)
                            except:
                                continue
                
                success = True
                break  # Success, exit retry loop
                            
            except Exception as e:
                error_msg = f"⚠️ Model {model_info.get('model_name', 'unknown')} failed: {str(e)}"
                print(f"Model failure attempt {retry_count + 1}: {error_msg}")
                
                # Check if this is a rate limit error and try next model
                if "rate_limit_exceeded" in str(e) or "429" in str(e) or "insufficient_quota" in str(e):
                    # Try to get next available model
                    try:
                        next_tier, next_reason = await model_manager.get_optimal_model(estimated_tokens)
                        if next_tier != current_tier and retry_count < max_retries - 1:
                            print(f"🔄 Rate limit hit, switching from {current_tier.value} to {next_tier.value}")
                            current_tier = next_tier
                            
                            # Update model info for user
                            model_info = model_manager.get_current_model_info()
                            yield _sse_format("model_info", json.dumps({
                                "model": model_info['model_name'],
                                "tier": next_tier.value,
                                "reason": f"Fallback: {next_reason}"
                            }))
                            continue  # Try again with new model
                    except:
                        pass
                
                # If this is the last retry AND it's not a quota/rate limit error, show error
                # Don't show quota errors to user - they should be handled silently by fallback
                if retry_count == max_retries - 1 and not ("429" in str(e) or "insufficient_quota" in str(e) or "rate_limit_exceeded" in str(e)):
                    accumulated.append(error_msg)
                    yield _sse_format(None, error_msg)
                    break
                elif retry_count == max_retries - 1:
                    # For quota errors on final retry, just break without showing error
                    print(f"All models exhausted due to quota limits")
                    break
        
        full_text = ''.join(accumulated)
        
        # 🧠 从记忆中获取相关上下文并学习用户风格
        user_style = personality_system.analyze_user_style(req.session_id)
        relevant_memories = memory_system.get_relevant_memory(req.message, req.session_id, limit=3)
        
        # 根据用户风格调整回应
        adapted_response = personality_system.adapt_response_style(full_text, user_style)
        
        # Add model signature with learning info
        learning_stats = memory_system.get_learning_stats()
        signature = f"\n\n*[Generated by: {model_info.get('model_name', 'unknown')} - {selection_reason}]*"
        if learning_stats['total_conversations'] > 0:
            signature += f"\n*[💡 Learned from {learning_stats['total_conversations']} conversations, mastered {learning_stats['learned_concepts']} concepts]*"
        
        adapted_response += signature
        yield _sse_format(None, signature)
        
        # 🎯 存储对话到记忆系统 (异步学习)
        try:
            conversation_id = memory_system.store_conversation(
                session_id=req.session_id,
                user_message=req.message,
                assistant_response=adapted_response,
                context_used=context if context else None,
                topics=None,  # 可以通过NLP提取
                sentiment=0.5  # 可以通过情感分析得出
            )
        except Exception as memory_error:
            print(f"Memory storage error: {memory_error}")
        
        # persist session (保持原有功能)
        SESSIONS.setdefault(req.session_id, []).extend([
            {"role":"user","content":req.message},
            {"role":"assistant","content":adapted_response}
        ])
        yield _sse_format("done", "true")

    return StreamingResponse(token_stream(), media_type="text/event-stream")

@app.get("/api/model_status")
async def get_model_status():
    """Get current model status and availability"""
    model_info = model_manager.get_current_model_info()
    
    # Check availability of all models
    model_availability = {}
    for tier in ModelTier:
        try:
            can_use = await model_manager._can_use_model(tier, 1000)  # Test with 1k tokens
            model_availability[tier.value] = {
                "available": can_use,
                "config": {
                    "name": MODEL_CONFIGS[tier].name,
                    "max_tokens": MODEL_CONFIGS[tier].max_tokens,
                    "cost_per_1k": MODEL_CONFIGS[tier].cost_per_1k_tokens,
                    "requires_api": MODEL_CONFIGS[tier].requires_api
                }
            }
        except:
            model_availability[tier.value] = {"available": False, "config": None}
    
    return {
        "current_model": model_info,
        "model_availability": model_availability,
        "quota_cache": model_manager.quota_cache,
        "failure_counts": {k.value: v for k, v in model_manager.failure_counts.items()}
    }

def _get_model_display_name(tier: ModelTier, config: ModelConfig) -> str:
    """Get user-friendly display name for model"""
    display_names = {
        ModelTier.GPT4: "GPT-4o (Best Quality)",
        ModelTier.GPT4_MINI: "GPT-4o Mini (Balanced)",
        ModelTier.GPT35: "GPT-3.5 Turbo (Fast)",
        ModelTier.LOCAL: "Llama 3.1 8B (Local)"
    }
    return display_names.get(tier, config.name)

def _get_model_description(tier: ModelTier, config: ModelConfig) -> str:
    """Get description for model"""
    descriptions = {
        ModelTier.GPT4: "Highest quality responses, best for complex tasks",
        ModelTier.GPT4_MINI: "Great balance of quality and speed, cost-effective",
        ModelTier.GPT35: "Fast responses, good for simple questions",
        ModelTier.LOCAL: "Private local model, no API costs, works offline"
    }
    return descriptions.get(tier, "AI language model")

@app.get("/api/models")
async def get_available_models():
    """Get list of all models for user selection (always shows all models)"""
    models = []
    for tier, config in MODEL_CONFIGS.items():
        try:
            # Check if model can be used for automatic selection
            is_auto_available = await model_manager._can_use_model(tier, 1000)
        except:
            is_auto_available = False
        
        # All models are always selectable manually, regardless of auto availability
        models.append({
            "id": tier.value,
            "name": config.name,
            "display_name": _get_model_display_name(tier, config),
            "description": _get_model_description(tier, config),
            "available": True,  # Always selectable for manual selection
            "auto_available": is_auto_available,  # Available for automatic selection
            "is_local": not config.requires_api,
            "max_tokens": config.max_tokens,
            "cost_per_1k": config.cost_per_1k_tokens,
            "failure_count": model_manager.failure_counts.get(tier, 0),
            "status": "ready" if is_auto_available else ("quota_exceeded" if model_manager.failure_counts.get(tier, 0) > 0 else "unavailable")
        })
    return {"models": models}

@app.post("/api/models/select")
async def select_model(request: dict):
    """Allow user to manually select a preferred model (overrides availability checks)"""
    model_id = request.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    
    # Find the model tier
    selected_tier = None
    for tier in ModelTier:
        if tier.value == model_id:
            selected_tier = tier
            break
    
    if not selected_tier:
        raise HTTPException(status_code=400, detail=f"Invalid model_id: {model_id}")
    
    # Allow manual selection regardless of current availability status
    # Reset failure count for manually selected model to give it a fresh chance
    model_manager.failure_counts[selected_tier] = 0
    
    # Set the preferred model
    model_manager.user_preferred_model = selected_tier
    model_manager.current_tier = selected_tier
    model_manager.is_user_selected = True
    
    config = MODEL_CONFIGS[selected_tier]
    status_message = "Successfully switched to"
    
    # Check if model might have issues and warn user
    if not config.requires_api:
        status_message = "Successfully switched to"
    elif model_manager.client is None:
        status_message = "Switched to (Note: No OpenAI API key configured)"
    else:
        status_message = "Successfully switched to"
    
    return {
        "success": True,
        "message": f"{status_message} {_get_model_display_name(selected_tier, config)}",
        "current_model": model_manager.get_current_model_info()
    }

@app.post("/api/models/auto")
async def enable_auto_model():
    """Re-enable automatic model selection"""
    model_manager.user_preferred_model = None
    estimated_tokens = 1000
    selected_tier, reason = await model_manager.get_optimal_model(estimated_tokens)
    
    return {
        "success": True,
        "message": "Automatic model selection enabled",
        "current_model": model_manager.get_current_model_info(),
        "selection_reason": reason
    }

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

async def process_file_to_rag(file_path: Path, filename: str) -> tuple[int, str]:
    """
    Process individual file and add to RAG system
    Returns (chunk_count, processing_info)
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    import uuid
    
    try:
        # 1. 根据文件类型提取文本
        content = ""
        file_ext = file_path.suffix.lower()
        print(f"🔍 Processing file: {filename}, extension: {file_ext}")
        
        if file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif file_ext == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif file_ext == '.py':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 为代码文件添加上下文
                content = f"# Python code file: {filename}\n\n{content}"
        elif file_ext in ['.pdf']:
            # PDF需要额外的库，暂时跳过
            return 0, f"PDF files are currently not supported, please convert to text format"
        elif file_ext in ['.docx']:
            # DOCX需要额外的库，暂时跳过  
            return 0, f"DOCX files are currently not supported, please convert to text format"
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            # 处理图片文件 - 使用OpenAI Vision API提取描述
            print(f"🖼️ Detected image file, starting AI visual analysis: {filename}")
            try:
                import base64
                
                # 读取图片并转换为base64
                with open(file_path, 'rb') as image_file:
                    image_content = image_file.read()
                    image_base64 = base64.b64encode(image_content).decode('utf-8')
                
                print(f"📊 Image size: {len(image_content)} bytes, base64 length: {len(image_base64)}")
                
                # 使用OpenAI Vision API分析图片
                print("🤖 Calling OpenAI Vision API for image analysis...")
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Please provide a detailed description of this image, including any text, objects, people, scenes, or important details that would be useful for a knowledge base. If there's any text in the image, please transcribe it exactly."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{file_ext[1:]};base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1000
                )
                
                # 提取AI的描述作为文档内容
                ai_description = response.choices[0].message.content
                print(f"✅ AI visual analysis completed, description length: {len(ai_description)} characters")
                
                content = f"# Image file: {filename}\n\n## AI Visual Analysis Result:\n\n{ai_description}\n\n## File Information:\n- Filename: {filename}\n- File type: Image ({file_ext})\n- Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
            except Exception as e:
                print(f"❌ Image processing failed: {e}")
                return 0, f"Image analysis failed: {str(e)}"
        else:
            # 尝试作为文本文件读取
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except:
                return 0, f"Unsupported file format: {file_ext}"
        
        if not content.strip():
            return 0, "File content is empty"
        
        # 2. 分割文本为chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(content)
        
        if not chunks:
            return 0, "Text is empty after splitting"
        
        # 3. 创建文档对象并添加元数据
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "file_type": file_ext,
                    "upload_time": str(datetime.now()),
                    "doc_id": str(uuid.uuid4())
                }
            )
            documents.append(doc)
        
        # 4. 添加到Pinecone向量数据库
        try:
            if vectorstore is None:
                print(f"⚠️ Vector database unavailable, skipping vectorization: {filename}")
                return len(documents), f"Document processed but not vectorized (vector database unavailable)"
            
            vectorstore.add_documents(documents)
            print(f"✅ Successfully added {len(documents)} document chunks to vector database: {filename}")
            return len(documents), f"Successfully processed and vectorized {len(documents)} text chunks"
        except Exception as e:
            print(f"❌ Vector database addition failed: {e}")
            return len(documents), f"Document processed but vectorization failed: {str(e)}"
            
    except Exception as e:
        print(f"❌ File processing failed {filename}: {e}")
        return 0, f"Processing failed: {str(e)}"

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Handle file uploads for RAG learning
    Complete file processing and add to vector database for AI learning
    """
    uploaded_filenames = []
    processing_results = []
    total_chunks = 0
    
    for file in files:
        # Save file to upload directory
        file_path = UPLOAD_DIR / file.filename
        
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            uploaded_filenames.append(file.filename)
            print(f"📁 File saved: {file.filename} ({len(content)} bytes)")
            
            # 🚀 完整的RAG处理流程
            chunk_count, process_info = await process_file_to_rag(file_path, file.filename)
            total_chunks += chunk_count
            
            processing_results.append({
                "filename": file.filename,
                "chunks_created": chunk_count,
                "status": "success" if chunk_count > 0 else "failed",
                "info": process_info
            })
            
            # 📝 记录到AI记忆系统
            if chunk_count > 0:
                try:
                    memory_system.store_conversation(
                        session_id="system",
                        user_message=f"User uploaded file: {file.filename}",
                        assistant_response=f"Successfully learned new document '{file.filename}', containing {chunk_count} knowledge segments. I can now answer questions based on this document.",
                        context_used=f"New document: {file.filename}",
                        topics=[file.filename, "Document Learning", "Knowledge Update"]
                    )
                except Exception as memory_error:
                    print(f"Memory system recording failed: {memory_error}")
            
        except Exception as e:
            processing_results.append({
                "filename": file.filename,
                "chunks_created": 0,
                "status": "failed",
                "info": f"Upload failed: {str(e)}"
            })
    
    # 生成详细的响应消息
    success_count = sum(1 for r in processing_results if r["status"] == "success")
    message_parts = [
        f"📤 Uploaded {len(uploaded_filenames)} files",
        f"✅ Successfully processed {success_count} files",
        f"🧠 Learned a total of {total_chunks} knowledge segments"
    ]
    
    if total_chunks > 0:
        message_parts.append("💡 AI can now answer questions based on these new documents!")
    
    return {
        "status": "success" if success_count > 0 else "partial_failure",
        "files_count": len(uploaded_filenames),
        "filenames": uploaded_filenames,
        "processed_count": success_count,
        "total_chunks_created": total_chunks,
        "processing_results": processing_results,
        "message": " | ".join(message_parts)
    }

@app.post("/api/chat_with_image")
async def chat_with_image(
    session_id: str = Form(...),
    message: str = Form(""),
    max_history: int = Form(8),
    image: UploadFile = File(...)
):
    """Chat endpoint with image support using OpenAI Vision"""
    try:
        # Save the uploaded image
        image_path = UPLOAD_DIR / f"chat_{session_id}_{image.filename}"
        image_content = await image.read()
        
        with open(image_path, "wb") as f:
            f.write(image_content)
        
        print(f"💬🖼️ Image chat request - Session: {session_id}, Image: {image.filename}, Message: {message}")
        
        # Prepare messages for OpenAI Vision API
        messages = []
        
        # Add conversation history
        if session_id in SESSIONS:
            history = SESSIONS[session_id][-max_history:]
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Create content for the current message
        content_parts = []
        if message.strip():
            content_parts.append({"type": "text", "text": message})
        
        # Add image
        import base64
        image_base64 = base64.b64encode(image_content).decode('utf-8')
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{image.content_type};base64,{image_base64}"
            }
        })
        
        messages.append({
            "role": "user", 
            "content": content_parts
        })
        
        # Try OpenAI Vision API first, fallback to LLaVA if needed
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use vision-capable model
                messages=messages,
                max_tokens=1000
            )
            answer = response.choices[0].message.content
            
            # Store conversation
            memory_system.store_conversation(
                session_id, message, answer, 
                context_used=f"Image analysis: {image.filename}",
                topics=["image_analysis", "visual_content"]
            )
            
            SESSIONS.setdefault(session_id, []).extend([
                {"role": "user", "content": f"{message} [Image: {image.filename}]"},
                {"role": "assistant", "content": answer}
            ])
            
            return {"answer": answer}
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ OpenAI Vision API error: {e}")
            
            # Try LLaVA as fallback for quota/rate limit errors AND connection errors
            if ("rate_limit_exceeded" in error_msg or "429" in error_msg or "insufficient_quota" in error_msg or 
                "Connection error" in error_msg or "timeout" in error_msg.lower() or "connection" in error_msg.lower()):
                print("🔄 Falling back to local LLaVA model...")
                try:
                    import requests
                    import base64
                    
                    # Convert image to base64 for LLaVA
                    image_base64 = base64.b64encode(image_content).decode('utf-8')
                    
                    # Prepare prompt for LLaVA
                    llava_prompt = message if message.strip() else "Please describe the content of this image."
                    
                    # Call LLaVA via Ollama API
                    ollama_response = requests.post('http://localhost:11434/api/generate', 
                        json={
                            'model': 'llava:7b',
                            'prompt': llava_prompt,
                            'images': [image_base64],
                            'stream': False
                        },
                        timeout=60
                    )
                    
                    if ollama_response.status_code == 200:
                        llava_result = ollama_response.json()
                        answer = llava_result.get('response', '')
                        
                        # Direct response without model indicator
                        if not answer.strip():
                            answer = "I have reviewed your image, but was unable to generate specific analysis content. Please try re-uploading or describe the specific questions you'd like to know about."
                        
                        # Store conversation
                        memory_system.store_conversation(
                            session_id, message, answer, 
                            context_used=f"Local LLaVA analysis: {image.filename}",
                            topics=["image_analysis", "visual_content", "local_model"]
                        )
                        
                        SESSIONS.setdefault(session_id, []).extend([
                            {"role": "user", "content": f"{message} [Image: {image.filename}]"},
                            {"role": "assistant", "content": answer}
                        ])
                        
                        return {"answer": answer}
                    else:
                        print(f"❌ LLaVA API error: {ollama_response.status_code}")
                
                except Exception as llava_error:
                    print(f"❌ LLaVA fallback error: {llava_error}")
                
                # If LLaVA also fails, show user-friendly message
                return {"answer": "🚫 **Image Analysis Temporarily Unavailable**\n\n**Issue:** OpenAI API quota exceeded, local vision model also temporarily unavailable.\n\n**Solutions:**\n- Wait for quota reset (about 8 hours)\n- Upgrade OpenAI payment plan\n- Or input the text content from the image directly\n\n💡 **Tip:** We have installed local LLaVA model as backup, which usually can process images offline"}
            
            elif "Connection error" in error_msg:
                return {"answer": "🌐 **Network Connection Issue**\n\nUnable to connect to OpenAI service for image analysis.\n\n**Suggestions:**\n- Check network connection\n- Try again later\n- Or describe the image content in text form"}
            else:
                return {"answer": f"😔 **Image Processing Failed**\n\nSorry, unable to process your image.\n\n**Error Details:** {error_msg}\n\n**Suggestion:** Please tell me the key information from the image in text form"}
            
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.post("/api/chat_stream_with_image")
async def chat_stream_with_image(
    session_id: str = Form(...),
    message: str = Form(""),
    max_history: int = Form(8),
    stream: str = Form("true"),
    image: UploadFile = File(...)
):
    """Streaming chat endpoint with image support using OpenAI Vision"""
    try:
        # Save the uploaded image
        image_path = UPLOAD_DIR / f"chat_stream_{session_id}_{image.filename}"
        image_content = await image.read()
        
        with open(image_path, "wb") as f:
            f.write(image_content)
        
        print(f"💬🖼️ Streaming image chat - Session: {session_id}, Image: {image.filename}, Message: {message}")
        
        def generate_stream():
            try:
                # Send model info
                yield _sse_format("model_info", '{"model": "gpt-4o-mini", "tier": "vision", "reason": "Image analysis mode"}')
                
                # Prepare messages for OpenAI Vision API
                messages = []
                
                # Add conversation history
                if session_id in SESSIONS:
                    history = SESSIONS[session_id][-max_history:]
                    for msg in history:
                        messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Create content for the current message
                content_parts = []
                if message.strip():
                    content_parts.append({"type": "text", "text": message})
                
                # Add image
                import base64
                image_base64 = base64.b64encode(image_content).decode('utf-8')
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.content_type};base64,{image_base64}"
                    }
                })
                
                messages.append({
                    "role": "user", 
                    "content": content_parts
                })
                
                # Call OpenAI Vision API with streaming, fallback to LLaVA
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        max_tokens=1000,
                        stream=True
                    )
                    
                    full_answer = ""
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_answer += content
                            yield f"data: {content}\n\n"
                    
                    # Store conversation
                    memory_system.store_conversation(
                        session_id, message, full_answer,
                        context_used=f"Image analysis: {image.filename}",
                        topics=["image_analysis", "visual_content"]
                    )
                    
                    SESSIONS.setdefault(session_id, []).extend([
                        {"role": "user", "content": f"{message} [Image: {image.filename}]"},
                        {"role": "assistant", "content": full_answer}
                    ])
                    
                    yield _sse_format("done", "")
                    
                except Exception as openai_error:
                    error_msg = str(openai_error)
                    print(f"❌ OpenAI Vision API error: {openai_error}")
                    
                    # Try LLaVA as fallback for quota/rate limit errors AND connection errors
                    if ("rate_limit_exceeded" in error_msg or "429" in error_msg or "insufficient_quota" in error_msg or 
                        "Connection error" in error_msg or "timeout" in error_msg.lower() or "connection" in error_msg.lower()):
                        print("🔄 Falling back to local LLaVA model for streaming...")
                        try:
                            import requests
                            import base64
                            print(f"📝 LLaVA fallback: Starting image processing...")
                            
                            # Update model info to show LLaVA
                            yield _sse_format("model_info", '{"model": "llava:7b", "tier": "local_vision", "reason": "OpenAI quota exceeded, using local vision model"}')
                            
                            # Convert image to base64 for LLaVA
                            image_base64 = base64.b64encode(image_content).decode('utf-8')
                            
                            # Prepare prompt for LLaVA
                            llava_prompt = message if message.strip() else "Please describe the content of this image."
                            
                            # Call LLaVA via Ollama API with streaming
                            print(f"📡 Calling LLaVA with prompt: {llava_prompt}")
                            ollama_response = requests.post('http://localhost:11434/api/generate', 
                                json={
                                    'model': 'llava:7b',
                                    'prompt': llava_prompt,
                                    'images': [image_base64],
                                    'stream': True
                                },
                                timeout=120,
                                stream=True
                            )
                            print(f"📡 LLaVA response status: {ollama_response.status_code}")
                            
                            if ollama_response.status_code == 200:
                                # Start streaming directly without model indicator
                                full_answer = ""
                                import json
                                for line in ollama_response.iter_lines():
                                    if line:
                                        try:
                                            chunk_data = json.loads(line)
                                            if 'response' in chunk_data:
                                                content = chunk_data['response']
                                                full_answer += content
                                                # Use proper SSE format with double newline
                                                yield f"data: {content}\n\n"
                                        except:
                                            continue
                                
                                # Handle empty response
                                if not full_answer.strip():
                                    yield f"data: I have reviewed your image, but was unable to generate specific analysis content. Please try re-uploading or describe the specific questions you'd like to know about.\n\n"
                                    full_answer = "I have reviewed your image, but was unable to generate specific analysis content. Please try re-uploading or describe the specific questions you'd like to know about."
                                
                                # Store conversation without model indicator
                                memory_system.store_conversation(
                                    session_id, message, full_answer,
                                    context_used=f"Local LLaVA analysis: {image.filename}",
                                    topics=["image_analysis", "visual_content", "local_model"]
                                )
                                
                                SESSIONS.setdefault(session_id, []).extend([
                                    {"role": "user", "content": f"{message} [Image: {image.filename}]"},
                                    {"role": "assistant", "content": full_answer}
                                ])
                                
                                yield _sse_format("done", "")
                                return
                            else:
                                print(f"❌ LLaVA API error: {ollama_response.status_code}")
                        
                        except Exception as llava_error:
                            print(f"❌ LLaVA fallback error: {llava_error}")
                            print(f"❌ LLaVA fallback error type: {type(llava_error)}")
                            import traceback
                            print(f"❌ LLaVA fallback traceback: {traceback.format_exc()}")
                        
                        # If LLaVA also fails, show user-friendly message
                        yield f"data: 🚫 **Image Analysis Temporarily Unavailable**\n\n**Issue:** OpenAI API quota limit reached, and local vision model is also temporarily unavailable.\n\n**Solutions:**\n- Wait for quota reset (about 8 hours)\n- Upgrade OpenAI paid plan\n- Or directly input the text content from the image to me\n\n💡 **Tip:** We have installed local LLaVA model as backup, which can usually process images offline\n\n"
                    
                    elif "Connection error" in error_msg:
                        yield f"data: 🌐 **Network Connection Issue**\n\nUnable to connect to OpenAI service for image analysis.\n\n**Suggestions:**\n- Check network connection\n- Try again later\n- Or describe the image content in text form to me\n\n"
                    else:
                        yield f"data: 😔 **Image Processing Failed**\n\nSorry, unable to process your image.\n\n**Error Details:** {error_msg}\n\n**Suggestion:** Please tell me the key information from the image in text form\n\n"
                    
                    yield _sse_format("done", "")
                    
            except Exception as e:
                print(f"❌ Streaming image processing error: {e}")
                yield f"data: ❌ Error occurred during image processing: {str(e)}\n\n"
                yield _sse_format("done", "")
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except Exception as e:
        print(f"❌ Streaming image setup error: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming image setup failed: {str(e)}")

@app.post("/api/process_existing_files")
async def process_existing_files():
    """
    Batch process existing but unprocessed files in uploads directory
    Let AI learn from previously uploaded but not vectorized files
    """
    if not UPLOAD_DIR.exists():
        return {"status": "no_files", "message": "uploads directory does not exist"}
    
    files = list(UPLOAD_DIR.glob("*"))
    files = [f for f in files if f.is_file()]
    
    if not files:
        return {"status": "no_files", "message": "No files in uploads directory"}
    
    processing_results = []
    total_chunks = 0
    
    for file_path in files:
        print(f"🔄 Processing existing file: {file_path.name}")
        chunk_count, process_info = await process_file_to_rag(file_path, file_path.name)
        total_chunks += chunk_count
        
        processing_results.append({
            "filename": file_path.name,
            "chunks_created": chunk_count,
            "status": "success" if chunk_count > 0 else "failed",
            "info": process_info
        })
        
        # 记录到记忆系统
        if chunk_count > 0:
            try:
                memory_system.store_conversation(
                    session_id="system",
                    user_message=f"Batch processing file: {file_path.name}",
                    assistant_response=f"Relearned document '{file_path.name}', containing {chunk_count} knowledge segments.",
                    context_used=f"Batch processing: {file_path.name}",
                    topics=[file_path.name, "Batch Learning", "Knowledge Update"]
                )
            except Exception as memory_error:
                print(f"Memory system recording failed: {memory_error}")
    
    success_count = sum(1 for r in processing_results if r["status"] == "success")
    
    return {
        "status": "success" if success_count > 0 else "failed",
        "files_processed": len(files),
        "successful_files": success_count,
        "total_chunks_created": total_chunks,
        "processing_results": processing_results,
        "message": f"Batch processing completed: {success_count}/{len(files)} files successful, learned a total of {total_chunks} knowledge segments"
    }

@app.get("/api/uploaded_files")
async def list_uploaded_files():
    """
    List uploaded files and their processing status
    """
    if not UPLOAD_DIR.exists():
        return {"files": [], "message": "uploads directory does not exist"}
    
    files = []
    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.is_file():
            stat = file_path.stat()
            files.append({
                "filename": file_path.name,
                "size": stat.st_size,
                "upload_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "file_type": file_path.suffix.lower()
            })
    
    return {
        "files": files,
        "count": len(files),
        "message": f"Found {len(files)} uploaded files"
    }

# =====================================================
# 🧠 AI成长与学习API (Memory & Personality System)
# =====================================================

class FeedbackRequest(BaseModel):
    conversation_id: str
    feedback_type: str  # 'positive', 'negative', 'correction', 'suggestion'
    content: str

class LearningStatsResponse(BaseModel):
    total_conversations: int
    learned_concepts: int
    total_feedback: int
    top_concepts: List[Dict]
    user_style_analysis: Dict

@app.post("/api/feedback")
async def add_feedback(feedback: FeedbackRequest):
    """User feedback for AI learning and improvement"""
    try:
        memory_system.add_feedback(
            feedback.conversation_id,
            feedback.feedback_type,
            feedback.content
        )
        return {"status": "success", "message": "Thank you for your feedback! I will learn and improve from it."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback storage failed: {str(e)}")

@app.get("/api/learning_stats/{session_id}")
async def get_learning_stats(session_id: str):
    """Get AI learning statistics and personalization analysis"""
    try:
        # 基础学习统计
        base_stats = memory_system.get_learning_stats()
        
        # 用户风格分析
        user_style = personality_system.analyze_user_style(session_id)
        
        # 获取该用户的对话历史统计
        user_conversations = memory_system.get_relevant_memory("", session_id, limit=100)
        
        return LearningStatsResponse(
            total_conversations=base_stats['total_conversations'],
            learned_concepts=base_stats['learned_concepts'],
            total_feedback=base_stats['total_feedback'],
            top_concepts=base_stats['top_concepts'],
            user_style_analysis={
                **user_style,
                "user_conversations_count": len(user_conversations),
                "analysis": {
                    "communication_style": "Formal" if user_style['formality'] > 0.7 else "Casual" if user_style['formality'] < 0.3 else "Moderate",
                    "detail_preference": "Detailed" if user_style['detail_level'] > 0.7 else "Concise" if user_style['detail_level'] < 0.3 else "Moderate",
                    "friendliness_level": "Friendly" if user_style['friendliness'] > 0.5 else "Professional"
                }
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@app.get("/api/memory_search/{session_id}")
async def search_memories(session_id: str, query: str = "", limit: int = 5):
    """Search relevant conversation memory"""
    try:
        memories = memory_system.get_relevant_memory(query, session_id, limit)
        return {
            "memories": memories,
            "count": len(memories),
            "query": query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory search failed: {str(e)}")

@app.get("/api/growth_insights/{session_id}")
async def get_growth_insights(session_id: str):
    """Get AI growth insights and recommendations"""
    try:
        stats = memory_system.get_learning_stats()
        user_style = personality_system.analyze_user_style(session_id)
        memories = memory_system.get_relevant_memory("", session_id, limit=50)
        
        # 分析成长趋势
        insights = {
            "learning_progress": {
                "conversations": stats['total_conversations'],
                "concepts_learned": stats['learned_concepts'],
                "feedback_received": stats['total_feedback'],
                "growth_rate": stats['learned_concepts'] / max(stats['total_conversations'], 1)
            },
            "personalization_level": {
                "style_adaptation": sum(user_style.values()) / len(user_style),
                "conversation_history": len(memories),
                "adaptation_quality": "High" if len(memories) > 20 else "Medium" if len(memories) > 5 else "Low"
            },
            "recommendations": []
        }
        
        # 生成改进建议
        if stats['total_feedback'] < 5:
            insights["recommendations"].append("💬 Please give me more feedback so I can better understand your preferences")
        
        if stats['total_conversations'] > 10 and stats['learned_concepts'] < 5:
            insights["recommendations"].append("🎯 Try asking me more diverse questions so I can learn new concepts")
        
        if user_style['formality'] == 0.5:
            insights["recommendations"].append("🎨 Continue chatting with me, I'm learning to adapt to your communication style")
        
        return insights
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Growth insights failed: {str(e)}")

# 启动服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)