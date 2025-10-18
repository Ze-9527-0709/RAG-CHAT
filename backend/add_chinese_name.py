#!/usr/bin/env python3
"""
Script to add Chinese name to agent identity
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_identity import AgentIdentityManager

def add_chinese_name():
    """Add Chinese name as a core memory"""
    print("Adding Chinese name to agent identity...")
    
    # Initialize the agent identity manager
    agent_identity = AgentIdentityManager("agent_identity.db")
    
    # Add Chinese name as a core memory
    agent_identity.add_core_memory(
        memory_type="identity",
        content="我的中文名字是 诶呦喂 (Ēi yō wēi)。这是一个中文感叹词，表达惊讶或赞叹的情感。这个名字体现了我为用户带来令人惊喜和愉悦回应的能力。",
        importance=10
    )
    
    print("✅ Chinese name added successfully!")
    
    # Verify by getting current identity
    current_identity = agent_identity.get_current_identity()
    if current_identity:
        print(f"Current identity: {current_identity['name']}")
    
    core_memories = agent_identity.get_core_memories()
    print(f"Core memories count: {len(core_memories)}")
    
    # Show all identity-related memories
    identity_memories = [m for m in core_memories if m['memory_type'] == 'identity']
    print(f"Identity memories:")
    for memory in identity_memories:
        print(f"  - {memory['content'][:100]}...")

if __name__ == "__main__":
    add_chinese_name()