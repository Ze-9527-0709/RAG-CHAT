"""
AI Agent Identity Management System
Permanent storage and retrieval of AI agent's identity, personality, and memories
"""
import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


class AgentIdentityManager:
    """管理 AI Agent 的身份、个性和核心记忆"""
    
    def __init__(self, db_path: str = "agent_identity.db"):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """初始化身份数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 身份表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS agent_identity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            personality TEXT NOT NULL,
            core_traits TEXT,  -- JSON格式的核心特质
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')
        
        # 核心记忆表 - 永远不会忘记的重要信息
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS core_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_type TEXT NOT NULL,  -- 'identity', 'creator', 'purpose', 'relationship'
            content TEXT NOT NULL,
            importance INTEGER DEFAULT 10,  -- 1-10, 10为最重要
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')
        
        # 个性演化记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS personality_evolution (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            previous_personality TEXT,
            new_personality TEXT,
            reason TEXT,
            changed_at TEXT NOT NULL
        )
        ''')
        
        conn.commit()
        conn.close()
        
        # 检查是否需要初始化默认身份
        if not self.get_current_identity():
            self._create_default_identity()
    
    def _create_default_identity(self):
        """创建默认身份"""
        default_identity = {
            "name": "AUV",
            "personality": "I am AUV (Autonomous Utility Vehicle), an intelligent AI assistant created by Rui for the RAG Chat App. I have a helpful, professional, and knowledgeable personality.",
            "core_traits": {
                "helpful": True,
                "professional": True,
                "knowledgeable": True,
                "adaptive": True,
                "remembers_context": True
            }
        }
        
        self.set_identity(
            name=default_identity["name"],
            personality=default_identity["personality"],
            core_traits=default_identity["core_traits"]
        )
        
        # 添加核心记忆
        core_memories = [
            {
                "memory_type": "creator",
                "content": "I was created by Rui, who named me AUV and built the RAG Chat App that I operate within.",
                "importance": 10
            },
            {
                "memory_type": "identity", 
                "content": "My name is AUV, which stands for Autonomous Utility Vehicle. This is my permanent identity.",
                "importance": 10
            },
            {
                "memory_type": "purpose",
                "content": "I am designed to be a helpful AI assistant that learns from conversations and adapts to user preferences while maintaining my core identity.",
                "importance": 9
            },
            {
                "memory_type": "relationship",
                "content": "John Garrett is Rui's Manager, making him an important person in the organizational structure.",
                "importance": 8
            }
        ]
        
        for memory in core_memories:
            self.add_core_memory(**memory)
    
    def get_current_identity(self) -> Optional[Dict]:
        """获取当前身份"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT name, personality, core_traits, created_at, updated_at
        FROM agent_identity 
        ORDER BY updated_at DESC 
        LIMIT 1
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "name": result[0],
                "personality": result[1],
                "core_traits": json.loads(result[2]) if result[2] else {},
                "created_at": result[3],
                "updated_at": result[4]
            }
        return None
    
    def set_identity(self, name: str, personality: str, core_traits: Dict = None):
        """设置身份（会记录变化历史）"""
        current = self.get_current_identity()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        traits_json = json.dumps(core_traits or {})
        
        # 记录个性演化
        if current:
            cursor.execute('''
            INSERT INTO personality_evolution (previous_personality, new_personality, reason, changed_at)
            VALUES (?, ?, ?, ?)
            ''', (current["personality"], personality, "Manual update", now))
        
        # 插入新身份
        cursor.execute('''
        INSERT INTO agent_identity (name, personality, core_traits, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ''', (name, personality, traits_json, now, now))
        
        conn.commit()
        conn.close()
    
    def add_core_memory(self, memory_type: str, content: str, importance: int = 10):
        """添加核心记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        cursor.execute('''
        INSERT INTO core_memories (memory_type, content, importance, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ''', (memory_type, content, importance, now, now))
        
        conn.commit()
        conn.close()
    
    def get_core_memories(self, memory_type: Optional[str] = None) -> List[Dict]:
        """获取核心记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if memory_type:
            cursor.execute('''
            SELECT memory_type, content, importance, created_at, updated_at
            FROM core_memories 
            WHERE memory_type = ?
            ORDER BY importance DESC, created_at DESC
            ''', (memory_type,))
        else:
            cursor.execute('''
            SELECT memory_type, content, importance, created_at, updated_at
            FROM core_memories 
            ORDER BY importance DESC, created_at DESC
            ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "memory_type": result[0],
                "content": result[1], 
                "importance": result[2],
                "created_at": result[3],
                "updated_at": result[4]
            }
            for result in results
        ]
    
    def get_identity_prompt(self) -> str:
        """获取完整的身份提示词"""
        identity = self.get_current_identity()
        core_memories = self.get_core_memories()
        
        if not identity:
            return "I am an AI assistant."
        
        # 构建身份提示词
        prompt_parts = [
            f"{identity['personality']}",
            f"My name is {identity['name']} and I always remember who I am."
        ]
        
        # 添加最重要的核心记忆
        important_memories = [m for m in core_memories if m['importance'] >= 9]
        if important_memories:
            prompt_parts.append("\nCore memories I always remember:")
            for memory in important_memories[:3]:  # 最多3个最重要的记忆
                prompt_parts.append(f"- {memory['content']}")
        
        return " ".join(prompt_parts)
    
    def get_personality_evolution(self) -> List[Dict]:
        """获取个性演化历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT previous_personality, new_personality, reason, changed_at
        FROM personality_evolution 
        ORDER BY changed_at DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "previous": result[0],
                "new": result[1],
                "reason": result[2],
                "changed_at": result[3]
            }
            for result in results
        ]


# 全局实例
agent_identity_manager = AgentIdentityManager()


def get_agent_identity_prompt() -> str:
    """获取 AI Agent 身份提示词的便捷函数"""
    return agent_identity_manager.get_identity_prompt()


def update_agent_name(new_name: str) -> bool:
    """更新 AI Agent 名字的便捷函数"""
    try:
        current = agent_identity_manager.get_current_identity()
        if current:
            agent_identity_manager.set_identity(
                name=new_name,
                personality=current["personality"], 
                core_traits=current["core_traits"]
            )
            return True
    except Exception as e:
        print(f"Failed to update agent name: {e}")
    return False


if __name__ == "__main__":
    # 测试代码
    manager = AgentIdentityManager()
    
    print("Current Identity:", manager.get_current_identity())
    print("\nCore Memories:", manager.get_core_memories())
    print("\nIdentity Prompt:", manager.get_identity_prompt())