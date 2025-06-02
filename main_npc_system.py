#!/usr/bin/env python3
"""
Complete Multi-Agent NPC System with Session Memory Integration
Features realistic memory tracking, spatial awareness, and information propagation
Enhanced with RAG (Retrieval-Augmented Generation) for domain-specific knowledge
"""

import requests
import json
import time
import os
import threading
import queue
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import uuid
import random

# Import the session memory system
from memory.session_memory import MemoryManager, InformationSource, ConfidenceLevel

# Import modular components
from src.core.message_types import NPCRole, Message
from src.core.config_loaders import RoleTemplateLoader, NPCLoader
from src.agents.base_agent import ScalableNPCAgent
from src.agents.npc_factory import NPCFactory
from src.agents.ollama_manager import OllamaRequestManager
from src.coordination.conversational_momentum import ConversationalMomentum
from src.coordination.conversation_manager import EnhancedConversationManager
from src.core.system_manager import create_enhanced_npc_system, show_npc_info

# Import RAG system components
try:
    from src.agents.rag_enhanced_agent import create_rag_enhanced_agent, create_rag_enhanced_team
    RAG_AVAILABLE = True
    print("🔥 RAG system loaded successfully!")
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"⚠️  RAG system not available: {e}")
    print("💡 Install RAG dependencies or run without RAG")


# Global request manager
ollama_manager = OllamaRequestManager()


if __name__ == "__main__":
    print("Enhanced Multi-Agent NPC System with Session Memory Integration")
    if RAG_AVAILABLE:
        print("🔥 RAG-Enhanced NPCs with Domain-Specific Horse Knowledge!")
    else:
        print("📚 Standard NPCs with Session Memory")
    print("\nMake sure Ollama is running: ollama serve")
    print("="*70)
    
    # Initialize system
    manager = create_enhanced_npc_system()
    
    if not manager:
        print("❌ Failed to initialize system. Check your configuration files.")
        exit(1)
    
    # Show system info
    show_npc_info()
    
    print(f"🎭 System ready! Available agents: {', '.join(manager.list_agents())}")
    print(f"📍 Current location: {manager.current_location}")
    if RAG_AVAILABLE:
        print(f"🔥 RAG Status: {manager.get_rag_status()}")
    
    print("\nCommands:")
    print("  - Type message for group discussion")
    print("  - 'AgentName: message' for direct conversation")
    print("  - 'go <location>' to move (barn, arena, paddock, tack_room, office)")
    print("  - 'move <npc> <location>' to move an NPC")
    print("  - 'memory [npc]' to check memories")
    if RAG_AVAILABLE:
        print("  - 'rag status/toggle' for RAG commands")
    print("  - 'info' for system information")
    print("  - 'stats' for statistics")
    print("  - 'reset' to clear conversations")
    print("  - 'quit' to exit")
    
    print(f"\n💬 Test Examples:")
    if manager.rag_enabled:
        print(f"  RAG Test: 'Oskar, what's the best feeding schedule for horses?'")
        print(f"  Expert Knowledge: 'Elin, how do I tell if a horse is stressed?'")
        print(f"  Competition Prep: 'Andy, what's the best way to train for jumping?'")
    else:
        print(f"  Memory test: 'Astrid, Thunder seemed nervous yesterday'")
        print(f"  Location: 'go barn' then ask about what happened")
    print(f"  Recall: 'What do you remember about Thunder?'")
    
    while True:
        try:
            user_input = input(f"\nPlayer ({manager.current_location}): ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'info':
                show_npc_info()
                continue
            elif user_input.lower().startswith('rag '):
                if not RAG_AVAILABLE:
                    print("❌ RAG system not available - check dependencies")
                    continue
                    
                rag_command = user_input[4:].strip().lower()
                if rag_command == 'status':
                    print(f"🔥 RAG Status: {manager.get_rag_status()}")
                    if manager.rag_enabled:
                        print("💡 NPCs can answer horse care questions with expert knowledge")
                    else:
                        print("💡 NPCs use only conversational AI (restart with RAG for expert knowledge)")
                elif rag_command == 'toggle':
                    result = manager.toggle_rag()
                    print(result)
                else:
                    print("Available RAG commands: 'rag status', 'rag toggle'")
                continue
            elif user_input.lower().startswith('go '):
                location = user_input[3:].strip()
                valid_locations = ["stable_yard", "barn", "arena", "paddock", "tack_room", "office"]
                if location in valid_locations:
                    manager.move_player_to_location(location)
                else:
                    print(f"❌ Unknown location. Available: {', '.join(valid_locations)}")
                continue
            elif user_input.lower().startswith('move '):
                parts = user_input[5:].split()
                if len(parts) >= 2:
                    npc_name = parts[0]
                    location = parts[1]
                    if manager.move_npc_to_location(npc_name, location):
                        print(f"✅ Moved {npc_name} to {location}")
                    else:
                        print(f"❌ Could not move {npc_name}")
                else:
                    print("Usage: move <npc> <location>")
                continue
            elif user_input.lower().startswith('memory'):
                parts = user_input.split()
                if len(parts) > 1:
                    manager.show_memory_summary(parts[1])
                else:
                    manager.show_memory_summary()
                continue
            elif user_input.lower() == 'stats':
                stats = manager.get_stats()
                print(f"📊 Total messages: {stats['total_messages']}")
                print(f"📍 Current location: {stats['current_location']}")
                print(f"🔄 Momentum chains: {stats['momentum_chains']}")
                print(f"🧠 Global memories: {stats['memory_system']['global_events']}")
                if RAG_AVAILABLE:
                    print(f"🔥 RAG Status: {stats['rag_status']}")
                
                for agent_name, agent_stats in stats['agents'].items():
                    location = agent_stats['current_location']
                    memory_count = agent_stats['memory_stats'].get('total_memories', 0)
                    rag_info = ""
                    if 'rag_stats' in stats and agent_name in stats['rag_stats']:
                        rag_info = " (RAG-enhanced)"
                    print(f"  {agent_name} ({agent_stats['title']}) at {location}: {memory_count} memories{rag_info}")
                continue
            elif user_input.lower() == 'reset':
                manager.reset_all()
                continue
            
            # Check for direct agent messaging
            if ':' in user_input and user_input.count(':') == 1:
                agent_name, message = user_input.split(':', 1)
                agent_name = agent_name.strip()
                message = message.strip()
                
                if agent_name == "All":
                    responses = manager.send_to_all(message)
                elif agent_name in manager.agents:
                    response, success, response_time = manager.send_to_agent(agent_name, message)
                    if success:
                        agent = manager.agents[agent_name]
                        role_desc = agent.template_data.get('title', agent.npc_role.value)
                        print(f"{agent_name} ({role_desc}): {response}")
                    else:
                        print(f"❌ {agent_name}: {response}")
                else:
                    print(f"❌ Agent '{agent_name}' not found. Available: {', '.join(manager.agents.keys())}")
                continue
            
            # Default to group conversation
            responses = manager.send_to_all(user_input)
            
        except KeyboardInterrupt:
            break
    
    print("\n👋 Thanks for testing the enhanced multi-agent system with session memory!")