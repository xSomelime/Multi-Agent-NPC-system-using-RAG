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
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)

def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True
        return False
    except requests.exceptions.RequestException:
        return False

# Check Ollama service first
if not check_ollama_service():
    print("❌ Ollama service is not running!")
    print("Please start Ollama with 'ollama serve' in a separate terminal")
    sys.exit(1)

print("✅ Ollama service is running")

# Import the session memory system
from memory.session_memory import MemoryManager, InformationSource, ConfidenceLevel

# Import core components first
from src.core.message_types import NPCRole, Message
from src.core.config_loaders import RoleTemplateLoader, NPCLoader
from src.core.system_manager import create_enhanced_npc_system, show_npc_info, get_conversation_system

# Import agent components
from src.agents.base_agent import ScalableNPCAgent
from src.agents.npc_factory import NPCFactory
from src.agents.ollama_manager import OllamaRequestManager, get_ollama_manager

# Import coordination components last
from src.coordination.conversational_momentum import ConversationalMomentum
from src.coordination.conversation_manager import EnhancedConversationManager

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
ollama_manager = get_ollama_manager()

def main():
    """Run the NPC system"""
    print("\n🎭 Multi-Agent NPC System")
    print("=========================")
    
    # Initialize the system in terminal mode
    manager = create_enhanced_npc_system(enable_rag=True, mode="terminal")
    if not manager:
        print("❌ Failed to initialize NPC system")
        return
    
    print("\n💬 Available NPCs:")
    show_npc_info()
    
    # Main conversation loop
    while True:
        try:
            # Update random movement if enabled
            manager.update_random_movement()
            
            current_loc = manager.current_location or "Location not set"
            user_input = input(f"\nPlayer ({current_loc}): ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'info':
                show_npc_info()
                continue
            elif user_input.lower().startswith('random '):
                cmd = user_input[7:].strip().lower()
                if cmd == 'on':
                    success, msg = manager.toggle_random_movement(True)
                elif cmd == 'off':
                    success, msg = manager.toggle_random_movement(False)
                else:
                    print("Usage: random [on|off]")
                    continue
                print(msg)
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
                manager.move_player_to_location(location)
                continue
            elif user_input.lower().startswith('move '):
                parts = user_input[5:].strip().split(' to ')
                if len(parts) == 2:
                    npc_name, location = parts
                    success = manager.move_npc_to_location(npc_name, location)
                    if not success:
                        print(f"❌ Failed to move {npc_name} to {location}")
                else:
                    print("Usage: move <npc_name> to <location>")
                continue
            elif user_input.lower().startswith('where '):
                npc_name = user_input[6:].strip()
                if npc_name == 'all':
                    manager.show_location_history()
                else:
                    manager.show_location_history(npc_name)
                continue
            elif user_input.lower() == 'reset':
                manager.reset_all()
                continue
            elif user_input.lower() == 'stats':
                stats = manager.get_stats()
                print("\n📊 System Statistics:")
                print(f"  Total messages: {stats['total_messages']}")
                print(f"  Current location: {stats['current_location']}")
                print(f"  RAG status: {stats['rag_status']}")
                print(f"  Memory chains: {stats['momentum_chains']}")
                print(f"  Random movement: {'enabled' if manager.random_movement_enabled else 'disabled'}")
                print("\n👥 NPCs:")
                for name, agent_stats in stats['agents'].items():
                    print(f"  {name} ({agent_stats['role']})")
                    print(f"    Location: {agent_stats['current_location']}")
                    print(f"    Messages: {agent_stats['total_messages']}")
                continue
            elif user_input.lower() == 'memory':
                manager.show_memory_summary()
                continue
            elif user_input.lower().startswith('memory '):
                npc_name = user_input[7:].strip()
                manager.show_memory_summary(npc_name)
                continue
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  quit - Exit the system")
                print("  info - Show NPC information")
                print("  random [on|off] - Toggle random NPC movement")
                print("  rag status - Check RAG system status")
                print("  rag toggle - Toggle RAG system")
                print("  go <location> - Move to location")
                print("  move <npc> to <location> - Move NPC")
                print("  where all - Show all NPC locations")
                print("  where <npc> - Show NPC location history")
                print("  reset - Reset all conversations")
                print("  stats - Show system statistics")
                print("  memory - Show all memory summaries")
                print("  memory <npc> - Show NPC memory summary")
                print("  help - Show this help message")
                continue
            
            # Handle conversation
            if ' to ' in user_input:
                npc_name, message = user_input.split(' to ', 1)
                if npc_name.lower() == 'all':
                    responses = manager.send_to_all(message)
                    for npc_name, response, success, response_time in responses:
                        if success:
                            print(f"\n{npc_name}: {response}")
                else:
                    response, success, response_time = manager.send_to_agent(npc_name, message)
                    if success:
                        print(f"\n{npc_name}: {response}")
            else:
                print("Usage: <npc_name> to <message> or all to <message>")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

if __name__ == "__main__":
    main()