#!/usr/bin/env python3
"""
Complete Multi-Agent NPC System with Session Memory Integration
Features realistic memory tracking, spatial awareness, and information propagation
Enhanced with RAG (Retrieval-Augmented Generation) for domain-specific knowledge
"""

import os
# Disable progress bars from transformers and tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

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
import traceback
import argparse

# Configure logging with less verbosity for startup
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors during startup
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import the session memory system
try:
    from memory.session_memory import MemoryManager, InformationSource, ConfidenceLevel
except ImportError as e:
    logger.error(f"Failed to import memory system: {e}")
    sys.exit(1)

# Import core components
try:
    from src.core.message_types import NPCRole, Message
    from src.core.config_loaders import RoleTemplateLoader, NPCLoader
    from src.core.system_manager import create_enhanced_npc_system, show_npc_info, get_conversation_system
except ImportError as e:
    logger.error(f"Failed to import core components: {e}")
    sys.exit(1)

# Import agent components
try:
    from src.agents.base_agent import ScalableNPCAgent
    from src.agents.npc_factory import NPCFactory
    from src.agents.ollama_manager import OllamaRequestManager, get_ollama_manager, start_ollama_service, ensure_ollama_model
except ImportError as e:
    logger.error(f"Failed to import agent components: {e}")
    sys.exit(1)

# Import coordination components
try:
    from src.coordination.conversational_momentum import ConversationalMomentum
    from src.coordination.conversation_manager import EnhancedConversationManager
except ImportError as e:
    logger.error(f"Failed to import coordination components: {e}")
    sys.exit(1)

# Import scenario system
try:
    from src.scenarios.scenario_manager import ScenarioManager
    SCENARIOS_AVAILABLE = True
except ImportError as e:
    SCENARIOS_AVAILABLE = False

# Import RAG system components
try:
    from src.agents.rag_enhanced_agent import create_rag_enhanced_agent, create_rag_enhanced_team
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False

# Initialize Ollama service and model
if not start_ollama_service():
    logger.error("Failed to start Ollama service")
    sys.exit(1)

if not ensure_ollama_model():
    logger.error("Failed to ensure model availability")
    sys.exit(1)

# Global request manager with error handling
try:
    ollama_manager = get_ollama_manager()
except Exception as e:
    logger.error(f"Failed to initialize Ollama manager: {e}")
    sys.exit(1)

def handle_ollama_error(e: Exception) -> str:
    """Handle Ollama errors gracefully with user-friendly messages"""
    if isinstance(e, requests.exceptions.ConnectionError):
        return "Lost connection to Ollama service. Please check if it's still running."
    elif isinstance(e, requests.exceptions.Timeout):
        return "Ollama request timed out. The service might be overloaded."
    elif isinstance(e, requests.exceptions.RequestException):
        return f"Ollama request failed: {str(e)}"
    else:
        return f"Unexpected error: {str(e)}"

def main():
    """Run the NPC system with enhanced error handling"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Agent NPC System')
    parser.add_argument('--no-rag', action='store_true', help='Disable RAG system')
    parser.add_argument('--no-scenarios', action='store_true', help='Disable scenario system')
    args = parser.parse_args()
    
    print("\n*** Multi-Agent NPC System ***")
    print("===============================")
    
    try:
        # Initialize the system in terminal mode with scenarios
        # Check command line args, then environment variable, then default
        use_rag = not args.no_rag and os.getenv('ENABLE_RAG', 'true').lower() == 'true' and RAG_AVAILABLE
        use_scenarios = not args.no_scenarios and SCENARIOS_AVAILABLE
        
        print(f"RAG System: {'Enabled' if use_rag else 'Disabled'}")
        print(f"Scenario System: {'Enabled' if use_scenarios else 'Disabled'}")
        
        manager = create_enhanced_npc_system(enable_rag=use_rag, enable_scenarios=use_scenarios, mode="terminal")
        if not manager:
            logger.error("Failed to initialize NPC system")
            return
        
        show_npc_info()
        
        # Main conversation loop
        while True:
            try:
                # Update random movement if enabled
                manager.update_random_movement()
                
                current_loc = manager.current_location or "Location not set"
                
                # Handle input with EOF protection
                try:
                    user_input = input(f"\nPlayer ({current_loc}): ").strip()
                except EOFError:
                    print("\nNo input available, exiting...")
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == 'quit':
                    print("\nGoodbye!")
                    break
                elif user_input.lower() == 'info':
                    show_npc_info()
                    continue
                elif user_input.lower().startswith('random'):
                    parts = user_input.split()
                    if len(parts) > 1:
                        enabled = parts[1].lower() == 'on'
                        success, msg = manager.toggle_random_movement(enabled)
                        print(f"\n{msg}")
                    else:
                        success, msg = manager.toggle_random_movement()
                        print(f"\n{msg}")
                    continue
                elif user_input.lower() == 'rag status':
                    print(f"\nRAG Status: {manager.get_rag_status()}")
                    continue
                elif user_input.lower() == 'rag toggle':
                    success, message = manager.toggle_rag()
                    print(f"\n{message}")
                    continue
                elif user_input.lower().startswith('go '):
                    location = user_input[3:].strip()
                    manager.move_player_to_location(location)
                    print(f"\nMoved to {location}")
                    continue
                elif user_input.lower().startswith('move '):
                    # Format: move <npc> to <location>
                    parts = user_input[5:].split(' to ', 1)
                    if len(parts) == 2:
                        npc_name, location = parts
                        success = manager.move_npc_to_location(npc_name.strip(), location.strip())
                        if success:
                            print(f"\nMoved {npc_name} to {location}")
                        else:
                            print(f"\nFailed to move {npc_name}")
                    else:
                        print("\nUsage: move <npc> to <location>")
                    continue
                elif user_input.lower() == 'where all':
                    print("\nCurrent NPC locations:")
                    for name, agent in manager.agents.items():
                        base_agent = getattr(agent, 'base_agent', agent)
                        location = getattr(base_agent, 'current_location', 'unknown')
                        print(f"  {name}: {location}")
                    continue
                elif user_input.lower().startswith('where '):
                    npc_name = user_input[6:].strip()
                    if npc_name in manager.agents:
                        # TODO: Show location history
                        base_agent = getattr(manager.agents[npc_name], 'base_agent', manager.agents[npc_name])
                        location = getattr(base_agent, 'current_location', 'unknown')
                        print(f"\n{npc_name} is at {location}")
                    else:
                        print(f"\nNPC '{npc_name}' not found")
                    continue
                elif user_input.lower() == 'reset':
                    manager.reset_all()
                    print("\nReset all conversations")
                    continue
                elif user_input.lower() == 'stats':
                    stats = manager.get_stats()
                    print("\nSystem Statistics:")
                    print(f"  Total messages: {stats['total_messages']}")
                    print(f"  Current location: {stats['current_location']}")
                    print(f"  System type: {stats['system_type']}")
                    print(f"  RAG status: {stats['rag_status']}")
                    print(f"  Momentum chains: {stats['momentum_chains']}")
                    continue
                elif user_input.lower() == 'memory':
                    manager.show_memory_summary()
                    continue
                elif user_input.lower().startswith('memory '):
                    npc_name = user_input[7:].strip()
                    if npc_name in manager.agents:
                        manager.show_memory_summary(npc_name)
                    else:
                        print(f"\nNPC '{npc_name}' not found")
                    continue
                elif user_input.lower() == 'scenarios':
                    scenarios = manager.list_available_scenarios()
                    if scenarios:
                        print("\n🎭 Available Scenarios:")
                        for i, scenario in enumerate(scenarios, 1):
                            print(f"  {i}. {scenario['title']} ({scenario['scenario_id']})")
                            print(f"     {scenario['description']}")
                            print(f"     Duration: {scenario['duration']}, Difficulty: {scenario['difficulty']}")
                    else:
                        print("\nNo scenarios available")
                    continue
                elif user_input.lower().startswith('start '):
                    scenario_id = user_input[6:].strip()
                    success, message = manager.start_scenario(scenario_id)
                    print(f"\n{message}")
                    continue
                elif user_input.lower() == 'scenario':
                    current_scenario = manager.get_current_scenario()
                    if current_scenario:
                        print(f"\n🎭 Current Scenario: {current_scenario['title']}")
                        print(f"Description: {current_scenario['description']}")
                        print(f"Status: {current_scenario.get('status', 'active')}")
                    else:
                        print("\nNo active scenario")
                    continue
                elif user_input.lower() == 'end_scenario':
                    success, message = manager.end_current_scenario()
                    print(f"\n{message}")
                    continue
                elif user_input.lower() == 'help':
                    print("\n🎮 CONVERSATION:")
                    print("  Hello everyone!              - Talk to all NPCs at your location")
                    print("  Oskar: How are you?          - Talk to specific NPC")
                    
                    print("\n🚶 MOVEMENT:")
                    print("  go paddock                   - Move yourself to location")
                    print("  move Astrid to stable        - Move NPC to location")
                    print("  where all                    - Show all NPC locations")
                    
                    print("\n⚙️  SYSTEM:")
                    print("  rag toggle                   - Turn RAG on/off")
                    print("  rag status                   - Check RAG status")
                    print("  info                         - Show NPC information")
                    print("  stats                        - Show system statistics")
                    print("  reset                        - Reset all conversations")
                    print("  quit                         - Exit")
                    
                    if manager.scenarios_enabled:
                        print("\n🎭 SCENARIOS:")
                        print("  scenarios                    - List available scenarios")
                        print("  start winter_planning        - Start a scenario")
                        print("  scenario                     - Show current scenario")
                        print("  end_scenario                 - End current scenario")
                    
                    print("\n🔧 STARTUP OPTIONS:")
                    print("  python main_npc_system.py --no-rag        - Start without RAG")
                    print("  python main_npc_system.py --no-scenarios  - Start without scenarios")
                    continue
                
                # Handle conversation
                if ':' in user_input and user_input.count(':') == 1:
                    # Talking to specific NPC
                    npc_name, message = user_input.split(':', 1)
                    npc_name = npc_name.strip()
                    message = message.strip()
                    
                    # Check for empty message
                    if not message:
                        print(f"\n💬 What would you like to say to {npc_name}? (You didn't include a message)")
                        continue
                    
                    try:
                        response, success, response_time = manager.send_to_agent(npc_name, message)
                        if success:
                            print(f"\n{npc_name}: {response}")
                        else:
                            print(f"\n{npc_name}: {response}")
                    except Exception as e:
                        print(f"\nError in conversation: {str(e)}")
                else:
                    # Check for scenario triggers before processing conversation
                    if manager.scenarios_enabled:
                        triggered_scenario = manager.check_scenario_triggers(user_input)
                        if triggered_scenario:
                            success, message = manager.start_scenario(triggered_scenario)
                            print(f"\n🎭 {message}")
                    
                    # Group conversation (talk to all NPCs at location)
                    responses = manager.send_to_all(user_input, current_loc)
                    # Responses are already displayed by the conversation manager
                    # No need to display them again here
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                logger.debug(traceback.format_exc())
                print(f"An error occurred: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"Critical error in main loop: {str(e)}")
        logger.debug(traceback.format_exc())
        print(f"Critical error: {str(e)}")
        return

if __name__ == "__main__":
    main()