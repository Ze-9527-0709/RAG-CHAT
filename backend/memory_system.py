"""
AI记忆和学习系统
通过对话不断成长的AI助手核心组件
"""
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib
from pathlib import Path

class ConversationMemory:
    """对话记忆系统 - 让AI从每次对话中学习"""
    
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = Path(db_path)
        self.init_database()
    
    def init_database(self):
        """初始化记忆数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 对话记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            user_message TEXT,
            assistant_response TEXT,
            timestamp DATETIME,
            user_feedback TEXT,
            context_used TEXT,
            topics TEXT,  -- JSON array of extracted topics
            sentiment REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 学习到的知识表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS learned_knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept TEXT UNIQUE,
            description TEXT,
            examples TEXT,  -- JSON array
            source_conversations TEXT,  -- JSON array of conversation IDs
            confidence_score REAL DEFAULT 1.0,
            usage_count INTEGER DEFAULT 0,
            last_used DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 用户偏好表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            preference_type TEXT,  -- 'style', 'topic', 'complexity', etc.
            preference_value TEXT,
            confidence REAL DEFAULT 1.0,
            learned_from TEXT,  -- conversation_id
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 反馈表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            feedback_type TEXT,  -- 'positive', 'negative', 'correction', 'suggestion'
            feedback_content TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_conversation(self, session_id: str, user_message: str, 
                          assistant_response: str, context_used: str = None,
                          topics: List[str] = None, sentiment: float = 0.5) -> str:
        """存储对话记录"""
        conversation_id = hashlib.md5(
            f"{session_id}_{user_message}_{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO conversations 
        (id, session_id, user_message, assistant_response, timestamp, 
         context_used, topics, sentiment)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (conversation_id, session_id, user_message, assistant_response,
              datetime.now(), context_used, json.dumps(topics or []), sentiment))
        
        conn.commit()
        conn.close()
        
        # 异步学习处理
        self._learn_from_conversation(conversation_id, user_message, assistant_response, topics)
        
        return conversation_id
    
    def _learn_from_conversation(self, conversation_id: str, user_message: str, 
                               assistant_response: str, topics: List[str] = None):
        """从对话中学习新知识"""
        # 提取概念和模式
        concepts = self._extract_concepts(user_message, assistant_response)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for concept, description in concepts.items():
            cursor.execute('''
            INSERT OR IGNORE INTO learned_knowledge 
            (concept, description, examples, source_conversations)
            VALUES (?, ?, ?, ?)
            ''', (concept, description, json.dumps([user_message]), 
                  json.dumps([conversation_id])))
            
            # 更新已存在的概念
            cursor.execute('''
            UPDATE learned_knowledge 
            SET usage_count = usage_count + 1,
                last_used = ?,
                source_conversations = json_insert(
                    COALESCE(source_conversations, '[]'), 
                    '$[#]', ?
                )
            WHERE concept = ?
            ''', (datetime.now(), conversation_id, concept))
        
        conn.commit()
        conn.close()
    
    def _extract_concepts(self, user_message: str, assistant_response: str) -> Dict[str, str]:
        """从对话中提取概念（简单实现，可用NLP增强）"""
        concepts = {}
        
        # 简单的关键词提取
        keywords = ['如何', '什么是', '怎么', '为什么', '方法', '步骤', '技巧']
        
        for keyword in keywords:
            if keyword in user_message:
                # 提取主题
                topic = user_message.replace(keyword, '').strip()
                if topic:
                    concepts[topic] = assistant_response[:200] + "..."
        
        return concepts
    
    def get_relevant_memory(self, query: str, session_id: str = None, limit: int = 5) -> List[Dict]:
        """获取相关的记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 简单的关键词匹配（实际应用中应使用向量相似度）
        query_words = query.lower().split()
        conditions = []
        params = []
        
        for word in query_words:
            conditions.append("(LOWER(user_message) LIKE ? OR LOWER(assistant_response) LIKE ?)")
            params.extend([f"%{word}%", f"%{word}%"])
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        if session_id:
            where_clause += " AND session_id = ?"
            params.append(session_id)
        
        cursor.execute(f'''
        SELECT * FROM conversations 
        WHERE {where_clause}
        ORDER BY timestamp DESC
        LIMIT ?
        ''', params + [limit])
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'session_id': row[1],
                'user_message': row[2],
                'assistant_response': row[3],
                'timestamp': row[4],
                'context_used': row[5],
                'topics': json.loads(row[6]) if row[6] else []
            })
        
        conn.close()
        return results
    
    def add_feedback(self, conversation_id: str, feedback_type: str, content: str):
        """添加用户反馈"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO feedback (conversation_id, feedback_type, feedback_content)
        VALUES (?, ?, ?)
        ''', (conversation_id, feedback_type, content))
        
        conn.commit()
        conn.close()
        
        # 从反馈中学习
        self._learn_from_feedback(conversation_id, feedback_type, content)
    
    def _learn_from_feedback(self, conversation_id: str, feedback_type: str, content: str):
        """从用户反馈中学习"""
        # 根据反馈调整知识confidence_score
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if feedback_type == 'positive':
            # 增加相关概念的信心分数
            cursor.execute('''
            UPDATE learned_knowledge 
            SET confidence_score = MIN(confidence_score * 1.1, 2.0)
            WHERE source_conversations LIKE ?
            ''', (f'%{conversation_id}%',))
        elif feedback_type == 'negative':
            # 降低相关概念的信心分数
            cursor.execute('''
            UPDATE learned_knowledge 
            SET confidence_score = MAX(confidence_score * 0.9, 0.1)
            WHERE source_conversations LIKE ?
            ''', (f'%{conversation_id}%',))
        
        conn.commit()
        conn.close()
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """获取学习统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 总对话数
        cursor.execute('SELECT COUNT(*) FROM conversations')
        total_conversations = cursor.fetchone()[0]
        
        # 学到的概念数
        cursor.execute('SELECT COUNT(*) FROM learned_knowledge')
        learned_concepts = cursor.fetchone()[0]
        
        # 收到的反馈数
        cursor.execute('SELECT COUNT(*) FROM feedback')
        total_feedback = cursor.fetchone()[0]
        
        # 最活跃的概念
        cursor.execute('''
        SELECT concept, usage_count, confidence_score 
        FROM learned_knowledge 
        ORDER BY usage_count DESC 
        LIMIT 5
        ''')
        top_concepts = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_conversations': total_conversations,
            'learned_concepts': learned_concepts,
            'total_feedback': total_feedback,
            'top_concepts': [
                {'concept': c[0], 'usage': c[1], 'confidence': c[2]}
                for c in top_concepts
            ]
        }

class AdaptivePersonality:
    """自适应个性系统 - 根据用户互动调整回应风格"""
    
    def __init__(self, memory: ConversationMemory):
        self.memory = memory
    
    def analyze_user_style(self, session_id: str) -> Dict[str, float]:
        """分析用户的交流风格"""
        conn = sqlite3.connect(self.memory.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT user_message FROM conversations 
        WHERE session_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 20
        ''', (session_id,))
        
        messages = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not messages:
            return {'formality': 0.5, 'detail_level': 0.5, 'friendliness': 0.5}
        
        # 简单的风格分析
        total_chars = sum(len(msg) for msg in messages)
        avg_length = total_chars / len(messages)
        
        # 正式度分析
        formal_words = ['请', '您', '谢谢', '麻烦']
        informal_words = ['嗯', '哈', '呀', '吧']
        
        formal_score = sum(1 for msg in messages for word in formal_words if word in msg)
        informal_score = sum(1 for msg in messages for word in informal_words if word in msg)
        
        formality = formal_score / (formal_score + informal_score + 1)
        
        return {
            'formality': formality,
            'detail_level': min(avg_length / 50, 1.0),  # 0-1 based on message length
            'friendliness': informal_score / len(messages) if messages else 0.5
        }
    
    def adapt_response_style(self, base_response: str, user_style: Dict[str, float]) -> str:
        """根据用户风格调整回应"""
        adapted_response = base_response
        
        # 根据正式度调整
        if user_style['formality'] > 0.7:
            adapted_response = adapted_response.replace('你', '您')
            adapted_response += "\n\n如有其他需要，请随时告诉我。"
        elif user_style['formality'] < 0.3:
            adapted_response += " 😊"
        
        # 根据详细程度调整
        if user_style['detail_level'] > 0.7:
            adapted_response += "\n\n需要我进一步详细解释任何部分吗？"
        
        return adapted_response

# 使用示例
if __name__ == "__main__":
    # 初始化记忆系统
    memory = ConversationMemory()
    personality = AdaptivePersonality(memory)
    
    # 存储对话
    conv_id = memory.store_conversation(
        "user123", 
        "如何让AI助手变得更智能？", 
        "可以通过多种方法：1. 持续学习对话历史 2. 收集用户反馈...",
        topics=["AI学习", "智能化"]
    )
    
    # 添加反馈
    memory.add_feedback(conv_id, "positive", "回答很有帮助")
    
    # 获取学习统计
    stats = memory.get_learning_stats()
    print("学习统计:", stats)