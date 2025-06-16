#!/usr/bin/env python3
"""
Robust System Manager
====================

Handles all complexity: imports, fallbacks, error recovery
Single entry point for conversation system
"""

import sys
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import time
import random

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.npc_factory import NPCFactory
from memory.session_memory import MemoryManager
from src.coordination.conversational_momentum import ConversationalMomentum
from src.coordination.conversation_manager import EnhancedConversationManager
from src.coordination.location_coordinator import LocationCoordinator
from src.coordination.turn_management import TurnManager

# Import scenario system
try:
    from src.scenarios.scenario_manager import ScenarioManager
    SCENARIOS_AVAILABLE = True
except ImportError:
    SCENARIOS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ConversationSystemManager:
    """Manages the overall conversation system with support for both terminal and UE5 modes"""
    
    def __init__(self, mode="terminal"):
        self.mode = mode
        self.memory_manager = MemoryManager()
        self.location_coordinator = LocationCoordinator(self.memory_manager)
        self.turn_manager = TurnManager()
        self.momentum_tracker = ConversationalMomentum(self.memory_manager)
        
        self.conversation_manager = EnhancedConversationManager(
            self.memory_manager,
            self.location_coordinator,
            self.turn_manager,
            self.momentum_tracker
        )
        
        self.agents = {}
        self.rag_enabled = False
        self.system_type = "basic"
        self.current_location = "stable"  # Default starting location
        
        # Scenario system
        self.scenario_manager = None
        self.scenarios_enabled = False
        
        # Random movement settings (terminal mode only)
        self.random_movement_enabled = False
        self.last_random_move_time = time.time()
        self.random_move_interval = 30  # seconds
    
    def initialize(self, enable_rag: bool = True, enable_scenarios: bool = True) -> bool:
        """Initialize the conversation system"""
        try:
            # Create NPCs using core team creation
            npc_factory = NPCFactory()
            team = npc_factory.create_core_team(self.memory_manager, enable_rag=enable_rag)
            
            # Register each NPC in the team
            for npc in team:
                self.agents[npc.name] = npc
                self.conversation_manager.register_agent(npc)
            
            # Set initial player location
            self.current_location = "stable"
            self.conversation_manager.current_location = "stable"
            
            # Initialize scenario system if available and requested
            if SCENARIOS_AVAILABLE and enable_scenarios:
                try:
                    self.scenario_manager = ScenarioManager(
                        memory_manager=self.memory_manager,
                        location_coordinator=self.location_coordinator
                    )
                    self.scenarios_enabled = True
                except Exception as e:
                    logger.warning(f"Failed to initialize scenarios: {e}")
                    self.scenarios_enabled = False
            
            self.rag_enabled = enable_rag
            self.system_type = "full"
            
            return True
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            return False
    
    def toggle_random_movement(self, enabled: bool = None):
        """Toggle random NPC movement (terminal mode only)"""
        if self.mode != "terminal":
            return False, "Random movement only available in terminal mode"
        
        if enabled is None:
            self.random_movement_enabled = not self.random_movement_enabled
        else:
            self.random_movement_enabled = enabled
        
        status = "enabled" if self.random_movement_enabled else "disabled"
        return True, f"Random NPC movement {status}"
    
    def update_random_movement(self):
        """Update random NPC movement if enabled (terminal mode only)"""
        if not self.random_movement_enabled or self.mode != "terminal":
            return
        
        current_time = time.time()
        if current_time - self.last_random_move_time < self.random_move_interval:
            return
        
        # Move a random NPC to a random location
        available_locations = list(self.location_coordinator.locations.keys())
        npc_names = list(self.agents.keys())
        
        npc_to_move = random.choice(npc_names)
        new_location = random.choice(available_locations)
        
        success = self.move_npc_to_location(npc_to_move, new_location)
        if success:
            self.last_random_move_time = current_time
    
    def update_ue5_locations(self, ue5_locations: Dict[str, str]):
        """Update NPC locations from UE5 (UE5 mode only)"""
        if self.mode != "ue5":
            return False, "UE5 location updates only available in UE5 mode"
        
        for npc_name, location in ue5_locations.items():
            if npc_name in self.agents:
                self.move_npc_to_location(npc_name, location)
        
        return True, "Updated NPC locations from UE5"
    
    def send_to_agent(self, agent_name: str, message: str) -> Tuple[str, bool, float]:
        """Send message to specific agent"""
        return self.conversation_manager.send_to_agent(agent_name, message)
    
    def send_to_all(self, message: str, location: str):
        return self.conversation_manager.send_to_all(message, location)
    
    def move_player_to_location(self, location: str):
        """Move player to new location"""
        self.conversation_manager.move_player_to_location(location)
        self.current_location = location
    
    def move_npc_to_location(self, npc_name: str, new_location: str, reason: str = None) -> bool:
        """Move NPC to new location with memory tracking"""
        # First check if NPC exists
        if npc_name not in self.agents:
            return False
            
        # Get current location info
        agent = self.agents[npc_name]
        old_location = getattr(agent, 'current_location', 'unknown')
        
        # Move NPC with memory updates
        success = self.location_coordinator.move_npc(npc_name, new_location, reason)
        if success:
            # Update agent's internal state
            agent.move_to_location(new_location)
            
            # Log the movement
            logger.info(f"🚶 {npc_name} moved from {old_location} to {new_location}" + 
                       (f" ({reason})" if reason else ""))
            
            # If this was triggered by random movement in terminal mode
            if self.mode == "terminal" and self.random_movement_enabled:
                # Get location info for context
                location_info = self.location_coordinator.locations.get(new_location, {})
                activities = location_info.get('typical_activities', [])
                
                if activities:
                    # Choose a random activity for the NPC to do
                    activity = random.choice(activities)
                    npc_memory = self.memory_manager.get_npc_memory(npc_name)
                    if npc_memory:
                        npc_memory.record_activity(new_location, activity)
                        logger.info(f"💭 {npc_name} is {activity} at {new_location}")
            
            return True
        return False
    
    def get_npc_location_info(self, npc_name: str) -> Dict[str, Any]:
        """Get detailed information about an NPC's location history"""
        if npc_name not in self.agents:
            return {"error": "NPC not found"}
            
        npc_memory = self.memory_manager.get_npc_memory(npc_name)
        if not npc_memory:
            return {"error": "No memory found for NPC"}
            
        return {
            "current_location": npc_memory.current_location,
            "recent_locations": npc_memory.get_recent_locations(hours=1),  # Last hour
            "location_knowledge": {
                loc: npc_memory.get_location_familiarity(loc)
                for loc in self.location_coordinator.locations.keys()
            }
        }
    
    # Scenario System Methods
    def list_available_scenarios(self) -> List[Dict[str, Any]]:
        """List all available scenarios"""
        if not self.scenarios_enabled:
            return []
        return self.scenario_manager.list_available_scenarios()
    
    def start_scenario(self, scenario_id: str) -> Tuple[bool, str]:
        """Start a specific scenario"""
        if not self.scenarios_enabled:
            return False, "Scenarios not enabled"
        
        success, message = self.scenario_manager.start_scenario(scenario_id, self.current_location, self.agents)
        if success:
            logger.info(f"🎭 Started scenario: {scenario_id}")
        return success, message
    
    def check_scenario_triggers(self, message: str) -> Optional[str]:
        """Check if current conversation should trigger a scenario"""
        if not self.scenarios_enabled:
            return None
        
        return self.scenario_manager.check_scenario_triggers(
            message, self.current_location, list(self.agents.keys())
        )
    
    def get_current_scenario(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently active scenario"""
        if not self.scenarios_enabled:
            return None
        return self.scenario_manager.get_current_scenario()
    
    def end_current_scenario(self) -> Tuple[bool, str]:
        """End the currently active scenario"""
        if not self.scenarios_enabled:
            return False, "Scenarios not enabled"
        
        success, message = self.scenario_manager.end_scenario()
        if success:
            logger.info("🎭 Scenario ended")
        return success, message
    
    def show_location_history(self, npc_name: str = None):
        """Display location history for one or all NPCs"""
        if npc_name:
            if npc_name not in self.agents:
                print(f"❌ NPC '{npc_name}' not found")
                return
                
            info = self.get_npc_location_info(npc_name)
            print(f"\n📍 Location History for {npc_name}:")
            print(f"  Current: {info['current_location']}")
            print("  Recent locations:")
            for loc in info['recent_locations']:
                familiarity = info['location_knowledge'][loc]
                print(f"    - {loc} (visited {familiarity['visit_count']} times)")
                if familiarity['activities']:
                    print(f"      Activities: {', '.join(familiarity['activities'])}")
        else:
            print("\n📍 All NPC Locations:")
            for name in self.agents.keys():
                info = self.get_npc_location_info(name)
                print(f"\n{name}:")
                print(f"  Current: {info['current_location']}")
                print(f"  Recent: {', '.join(info['recent_locations'])}")
    
    def list_agents(self) -> List[str]:
        """Get list of available agents"""
        return list(self.agents.keys())
    
    def get_rag_status(self) -> str:
        """Get RAG system status"""
        return self.conversation_manager.get_rag_status()
    
    def toggle_rag(self, enabled: bool = None) -> Tuple[bool, str]:
        """Toggle RAG system on/off"""
        if enabled is None:
            enabled = not self.rag_enabled
        
        # Check if RAG is available
        try:
            from src.agents.rag_enhanced_agent import RAG_AVAILABLE
            if enabled and not RAG_AVAILABLE:
                return False, "RAG system not available (missing dependencies)"
        except ImportError:
            if enabled:
                return False, "RAG system not available (missing dependencies)"
        
        # Update system-level flag
        self.rag_enabled = enabled
        
        # Update all agents
        for agent in self.agents.values():
            if hasattr(agent, 'enable_rag'):
                agent.enable_rag = enabled
                if enabled and not hasattr(agent, 'rag_interface'):
                    # Try to initialize RAG for this agent
                    try:
                        from src.knowledge.rag_interface import get_npc_rag_interface
                        agent.rag_interface = get_npc_rag_interface()
                    except Exception as e:
                        logger.warning(f"Failed to enable RAG for {getattr(agent, 'name', 'unknown')}: {e}")
        
        status = "enabled" if enabled else "disabled"
        return True, f"RAG system {status}"
    
    def show_memory_summary(self, npc_name: str = None):
        """Show memory summary for NPC(s)"""
        self.conversation_manager.show_memory_summary(npc_name)
    
    def end_conversation(self, npc_id: str = None):
        """End conversation with specific NPC or all NPCs"""
        self.conversation_manager.end_conversation(npc_id)
    
    def reset_all(self):
        """Reset entire system for new game session"""
        self.conversation_manager.reset_all()
        self.current_location = None
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "total_messages": self.total_messages,
            "current_location": self.current_location,
            "system_type": self.system_type,
            "rag_status": self.get_rag_status(),
            "momentum_chains": self.momentum_tracker.get_chain_count(),
            "memory_system": self.memory_manager.get_system_stats(),
            "agents": {name: agent.get_stats() for name, agent in self.agents.items()}
        }

# Global conversation system instance
_conversation_system = None

def get_conversation_system() -> Optional[ConversationSystemManager]:
    """Get the global conversation system instance"""
    return _conversation_system

def set_conversation_system(system: ConversationSystemManager):
    """Set the global conversation system instance"""
    global _conversation_system
    _conversation_system = system

def create_enhanced_npc_system(enable_rag: bool = True, enable_scenarios: bool = True, mode: str = "terminal") -> Optional[ConversationSystemManager]:
    """Create and initialize the enhanced NPC system
    
    Args:
        enable_rag: Whether to enable RAG features
        enable_scenarios: Whether to enable scenario system
        mode: Operation mode ("terminal" or "ue5")
    """
    try:
        # Create system manager with specified mode
        manager = ConversationSystemManager(mode=mode)
        
        # Initialize the system
        if not manager.initialize(enable_rag, enable_scenarios):
            return None
        
        # Store the global instance
        set_conversation_system(manager)
            
        return manager
    except Exception as e:
        logger.error(f"Failed to create NPC system: {e}")
        return None

def show_npc_info():
    """Show information about available NPCs"""
    system = get_conversation_system()
    if not system:
        print("❌ System not initialized")
        return
    
    print(f"\n🎭 NPC System: {system.mode} mode | RAG: {'✅' if system.rag_enabled else '❌'} | Scenarios: {'✅' if system.scenarios_enabled else '❌'}")
    print(f"📍 You are at: {system.current_location}")
    
    print("\n🎮 Quick Start:")
    print("  Talk to everyone: Hello everyone!")
    print("  Talk to one NPC: Oskar: How are the horses today?")
    print("  Move around: go paddock")
    print("  Move an NPC: move Astrid to stable")
    print("  Toggle features: rag toggle")
    print("  View scenarios: scenarios")
    print("  Get help: help")
    
    print("\n👥 Available NPCs:")
    for name, agent in system.agents.items():
        # Get role from base agent if it's a RAG-enhanced agent
        base_agent = getattr(agent, 'base_agent', agent)
        title = getattr(base_agent, 'template_data', {}).get('title', 'NPC')
        location = base_agent.current_location if hasattr(base_agent, 'current_location') else "Unknown"
        print(f"  {name} ({title}) - at {location}")
    
    print("\n💡 Ready to chat!")

if __name__ == "__main__":
    # Test system
    system = create_enhanced_npc_system()
    if system:
        print(f"Test successful - system type: {system.system_type}")
        
        # Test conversation
        response, success, time = system.send_to_agent("Elin", "Hello!")
        print(f"Test response: {response}")
    else:
        print("Test failed")