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
        
        # Random movement settings (terminal mode only)
        self.random_movement_enabled = False
        self.last_random_move_time = time.time()
        self.random_move_interval = 30  # seconds
    
    def initialize(self, enable_rag: bool = True) -> bool:
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
            
            self.rag_enabled = enable_rag
            self.system_type = "full"
            logger.info(f"✅ Full enhanced NPC system initialized in {self.mode} mode")
            if enable_rag:
                logger.info("🔥 All features available")
            
            # Show initial location info
            npcs_here = self.location_coordinator.get_npcs_at_location("stable", self.agents)
            if npcs_here:
                logger.info(f"👥 NPCs at stable: {', '.join(npcs_here)}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error initializing system: {e}")
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
    
    def send_to_all(self, message: str) -> List[Tuple[str, str, bool, float]]:
        """Send message to all appropriate agents"""
        return self.conversation_manager.send_to_all(message)
    
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

def create_enhanced_npc_system(enable_rag: bool = True, mode: str = "terminal") -> Optional[ConversationSystemManager]:
    """Create and initialize the enhanced NPC system
    
    Args:
        enable_rag: Whether to enable RAG features
        mode: Operation mode ("terminal" or "ue5")
    """
    try:
        # Create system manager with specified mode
        manager = ConversationSystemManager(mode=mode)
        
        # Initialize the system
        if not manager.initialize(enable_rag):
            return None
            
        return manager
    except Exception as e:
        logger.error(f"Failed to create NPC system: {e}")
        return None

def get_conversation_system() -> Optional[ConversationSystemManager]:
    """Get the current conversation system"""
    return _conversation_system

def show_npc_info():
    """Show information about available NPCs"""
    system = get_conversation_system()
    if not system:
        print("❌ System not initialized")
        return
    
    print(f"System type: {system.system_type}")
    if system.rag_enabled:
        print("🔥 All features available")

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