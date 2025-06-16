#!/usr/bin/env python3
"""
Enhanced Conversation Manager
============================

Main orchestrator for multi-agent conversations, integrating memory,
spatial awareness, turn management, and conversational momentum.
Updated for dynamic location system.
"""

import time
import uuid
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from memory.session_memory import MemoryManager, ConfidenceLevel
from src.core.message_types import Message
from .conversational_momentum import ConversationalMomentum
from .turn_management import TurnManager, ConversationType
from .location_coordinator import LocationCoordinator
from src.coordination.group_conversation import GroupConversationManager

logger = logging.getLogger(__name__)

class EnhancedConversationManager:
    """Enhanced ConversationManager with memory integration, RAG support, and dynamic locations"""
    
    def __init__(self, memory_manager: MemoryManager, location_coordinator: LocationCoordinator, 
                 turn_manager: TurnManager, momentum_tracker: ConversationalMomentum):
        self.memory_manager = memory_manager
        self.agents: Dict[str, Any] = {}
        self.conversation_log: List[Message] = []
        self._current_location = "stable_yard"  # Default starting location
        
        # Store coordination components
        self.momentum = momentum_tracker
        self.turn_manager = turn_manager
        self.location_coordinator = location_coordinator
        
        # Initialize group conversation manager
        self.group_manager = GroupConversationManager(memory_manager)
        
        # Track active group conversations
        self.active_group_conversations: Dict[str, Dict] = {}
        
        # RAG configuration
        self.rag_enabled = False
        
        try:
            # Check if RAG is available
            from src.agents.rag_enhanced_agent import RAG_AVAILABLE
            if RAG_AVAILABLE:
                self.rag_enabled = True
                # RAG system initialized
            else:
                print("Running with standard NPCs (RAG dependencies not available)")
        except ImportError:
            print("Running with standard NPCs (no RAG)")
        
        # Location system initialized
    
    @property
    def current_location(self) -> str:
        """Get the current location"""
        return self._current_location
    
    @current_location.setter
    def current_location(self, value: str):
        """Set the current location"""
        self._current_location = value
    
    def toggle_rag(self) -> str:
        """Toggle RAG system on/off (for comparison)"""
        try:
            from src.agents.rag_enhanced_agent import RAG_AVAILABLE
            if not RAG_AVAILABLE:
                return "❌ RAG system not available - check dependencies"
        except ImportError:
            return "❌ RAG system not available - check dependencies"
        
        self.rag_enabled = not self.rag_enabled
        status = "enabled" if self.rag_enabled else "disabled"
        return f"🔄 RAG system {status}. Restart to apply changes."
    
    def get_rag_status(self) -> str:
        """Get RAG system status"""
        if not self.rag_enabled:
            return "disabled"
        try:
            from src.agents.rag_enhanced_agent import RAG_AVAILABLE
            return "enabled and working" if RAG_AVAILABLE else "enabled but unavailable"
        except ImportError:
            return "enabled but missing dependencies"
    
    def register_agent(self, agent: Any):
        """Register an agent"""
        self.agents[agent.name] = agent
        role_desc = getattr(agent, 'template_data', {}).get('title', agent.npc_role.value)
        current_location = getattr(agent, 'current_location', 'unknown')
        # Agent registered silently
    
    def move_player_to_location(self, location: str):
        """Move player to a new location (with dynamic location support)"""
        success, message = self.location_coordinator.move_player(location)
        if success:
            print(f"🚶 {message}")
            
            # Show who's here in this discovered location
            npcs_here = self.location_coordinator.get_npcs_at_location(location, self.agents)
            if npcs_here:
                print(f"👥 NPCs here: {', '.join(npcs_here)}")
            else:
                print(f"No NPCs at {location} currently")
        else:
            print(f"{message}")
    
    def move_npc_to_location(self, npc_name: str, location: str):
        """Move an NPC to a new location (with dynamic location registration)"""
        success, message = self.location_coordinator.move_npc(npc_name, location, self.agents)
        if success:
            print(f"{message}")
            
            # Auto-register this location if it's new
            discovered_locations = self.memory_manager.get_discovered_locations()
            if location not in discovered_locations:
                print(f"New location discovered: {location}")
        else:
            print(f"{message}")
        return success
    
    def send_to_agent(self, agent_name: str, message: str) -> Tuple[str, bool, float]:
        """Enhanced direct messaging with memory recording and dynamic location support"""
        start_time = time.time()
        
        # Try exact match first, then case-insensitive
        agent = self.agents.get(agent_name)
        if not agent:
            # Try case-insensitive lookup
            for name, a in self.agents.items():
                if name.lower() == agent_name.lower():
                    agent = a
                    agent_name = name  # Use the correct case
                    break
        
        if not agent:
            available_npcs = ", ".join(self.agents.keys())
            return f"NPC '{agent_name}' not found. Available NPCs: {available_npcs}", False, 0.0
        
        # Quick location check
        agent_location = getattr(agent, 'current_location', 'unknown')
        current_player_location = self.current_location
        
        if agent_location != current_player_location and agent_location != 'unknown' and current_player_location != 'unknown':
            return f"{agent_name} is not here. They're at {agent_location}, you're at {current_player_location}. Use 'go {agent_location}' to find them.", False, 0.0
        
        # Add to global log (only keep last 10 messages)
        if len(self.conversation_log) > 10:
            self.conversation_log = self.conversation_log[-9:]
        self._add_to_global_log("user", message, "player")
        
        # Get response with minimal context
        response, success, response_time = agent.generate_response(
            message, self.conversation_log[-3:]  # Only use last 3 messages for context
        )
        
        # Log agent response
        if success:
            self._add_to_global_log("assistant", response, agent_name)
            
            # Don't trigger auto-responses for direct conversations - this is a private chat
            # Only process memory updates for nearby NPCs to overhear
            self._process_memory_only(agent_name, response, agent_location)
            
        end_time = time.time()
        return response, success, end_time - start_time
    
    def _process_memory_only(self, agent_name: str, response: str, agent_location: str):
        """Process only memory updates without triggering auto-responses (for direct conversations)"""
        try:
            # Record this as an NPC response that others can overhear
            nearby_npcs = self.location_coordinator.get_npcs_at_location(
                agent_location, self.agents
            )
            nearby_npcs = [name for name in nearby_npcs if name != agent_name]
            
            for nearby_npc in nearby_npcs:
                self.memory_manager.npc_tells_npc(
                    agent_name, nearby_npc,
                    f"conversation with player: {response}",
                    agent_location,
                    ConfidenceLevel.CONFIDENT,
                    ["overheard_conversation"]
                )
                    
        except Exception as e:
            print(f"Warning: Error in memory processing: {e}")

    def _process_memory_and_triggers(self, agent_name: str, response: str, agent_location: str):
        """Process memory updates and conversation triggers separately"""
        try:
            # Record this as an NPC response that others can overhear
            nearby_npcs = self.location_coordinator.get_npcs_at_location(
                agent_location, self.agents
            )
            nearby_npcs = [name for name in nearby_npcs if name != agent_name]
            
            for nearby_npc in nearby_npcs:
                self.memory_manager.npc_tells_npc(
                    agent_name, nearby_npc,
                    f"conversation with player: {response}",
                    agent_location,
                    ConfidenceLevel.CONFIDENT,
                    ["overheard_conversation"]
                )
            
            # Check if other NPCs should auto-respond
            auto_triggers = self.momentum.should_trigger_auto_response(
                response, agent_name, agent_location, self.agents
            )
            
            # Process auto-responses in parallel
            if auto_triggers:
                print("\nOthers want to join the conversation:")
                threads = []
                for other_agent_name, reason, probability in auto_triggers[:2]:  # Limit to 2 auto-responses
                    other_agent = self.agents.get(other_agent_name)
                    if other_agent:
                        role_desc = getattr(other_agent, 'template_data', {}).get('title', other_agent.npc_role.value)
                        print(f"{other_agent_name} ({role_desc}) joining in ({reason})")
                        
                        thread = threading.Thread(
                            target=self._handle_auto_response,
                            args=(other_agent_name, reason, response)
                        )
                        thread.start()
                        threads.append(thread)
                
                # Wait for all responses but not too long
                for thread in threads:
                    thread.join(timeout=5.0)
                    
        except Exception as e:
            print(f"Warning: Error in memory processing: {e}")
    
    def _handle_auto_response(self, npc_name: str, reason: str, original_response: str):
        """Handle auto-response in separate thread"""
        try:
            agent = self.agents.get(npc_name)
            if agent:
                response, success, _ = agent.generate_response(
                    f"Responding to: {original_response}",
                    self.conversation_log[-2:]
                )
                
                if success:
                    self._add_to_global_log("assistant", response, npc_name)
                    print(f"  {npc_name}: {response}")
        except Exception as e:
            print(f"Warning: Auto-response failed for {npc_name}: {e}")
    
    def send_to_all(self, message: str, location: str) -> List[Tuple[str, str]]:
        """Send message to all NPCs at current location, with sequential responses and reactions."""
        responses = []
        npcs_at_location = [npc for npc in self.agents.values() if getattr(npc, 'current_location', 'unknown') == location]
        
        if not npcs_at_location:
            return []
            
        # Add player message to context and conversation log
        self.conversation_log.append(Message(
            id=str(uuid.uuid4()),
            role="user",
            content=message,
            timestamp=time.time(),
            agent_name="Player",
            location=location
        ))
        context = f"Player at {location}: {message}"
        
        # Determine which NPCs want to participate
        participating_npcs = []
        for npc in npcs_at_location:
            if npc.should_participate(message, responses):
                participating_npcs.append(npc)
        
        # Randomize the order to make conversations more natural
        import random
        random.shuffle(participating_npcs)
        
        # Show who will participate
        for npc in participating_npcs:
            # Get NPC's role - handle both RAG-enhanced and regular agents
            if hasattr(npc, 'base_agent'):
                role_desc = getattr(npc.base_agent, 'template_data', {}).get('title', npc.base_agent.npc_role.value)
            else:
                role_desc = getattr(npc, 'template_data', {}).get('title', npc.npc_role.value)
            print(f"{npc.name} ({role_desc}) will respond at {location}")
        
        if not participating_npcs:
            print(f"No NPCs chose to respond to: {message}")
            return []
        
        print(f"{len(participating_npcs)} NPCs will participate in the conversation")
        
        # Track who has spoken to prevent duplicate reactions
        spoken_npcs = set()
        
        # Get responses sequentially, allowing reactions
        for npc in participating_npcs:
            if npc.name in spoken_npcs:
                continue
                
            # Get NPC's role - handle both RAG-enhanced and regular agents
            if hasattr(npc, 'base_agent'):
                # RAG-enhanced agent
                role_desc = getattr(npc.base_agent, 'template_data', {}).get('title', npc.base_agent.npc_role.value)
            else:
                # Regular agent
                role_desc = getattr(npc, 'template_data', {}).get('title', npc.npc_role.value)
                
            # Get NPC's response
            print(f"\n🗣️ {npc.name} ({role_desc}) responding...")
            response, success, response_time = npc.generate_response(context, self.conversation_log)
            spoken_npcs.add(npc.name)
            
            # Only process successful responses
            if not success:
                print(f"{npc.name} failed to respond: {response}")
                continue
            
            # Show response with role
            print(f"💬 {npc.name} ({role_desc}): {response}")
            responses.append((npc.name, response))
            
            # Add response to conversation log
            self.conversation_log.append(Message(
                id=str(uuid.uuid4()),
                role="assistant",
                content=response,
                timestamp=time.time(),
                agent_name=npc.name,
                location=location
            ))
            
            # Let other participating NPCs react to this response (but not for memory questions)
            is_memory_question = any(phrase in message.lower() for phrase in ["do you remember me", "remember me", "do you know me", "have we met", "do i know"])
            
            if not is_memory_question:  # Skip reactions for memory questions
                for other_npc in participating_npcs:
                    if other_npc.name not in spoken_npcs:
                        # Check if they want to react to this specific response
                        if other_npc.should_participate(response, responses):
                            # Get other NPC's role
                            if hasattr(other_npc, 'base_agent'):
                                other_role_desc = getattr(other_npc.base_agent, 'template_data', {}).get('title', other_npc.base_agent.npc_role.value)
                            else:
                                other_role_desc = getattr(other_npc, 'template_data', {}).get('title', other_npc.npc_role.value)
                            
                            print(f"\n💬 {other_npc.name} ({other_role_desc}) reacting...")
                            # Check if the original message was a simple greeting
                            is_greeting = any(greeting in message.lower() for greeting in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"])
                            
                            if is_greeting:
                                # For greetings, just give a simple greeting back
                                reaction, reaction_success, reaction_time = other_npc.generate_response(
                                    f"Player said: {message}\n\nGive a simple greeting back - don't give advice or long responses.",
                                    self.conversation_log
                                )
                            else:
                                # For other topics, use the normal reaction prompt
                                reaction, reaction_success, reaction_time = other_npc.generate_response(
                                    f"Player asked: {message}\n\n{npc.name} just said: {response}\n\nGive YOUR answer to the player's question. You can briefly acknowledge what {npc.name} said, but focus 90% on answering the player's original question with your own perspective.",
                                    self.conversation_log
                                )
                            spoken_npcs.add(other_npc.name)
                            
                            # Only process successful reactions
                            if not reaction_success:
                                print(f"{other_npc.name} failed to react: {reaction}")
                                continue
                            
                            # Show reaction with role
                            print(f"💭 {other_npc.name} ({other_role_desc}): {reaction}")
                            responses.append((other_npc.name, reaction))
                            
                            # Add reaction to conversation log
                            self.conversation_log.append(Message(
                                id=str(uuid.uuid4()),
                                role="assistant",
                                content=reaction,
                                timestamp=time.time(),
                                agent_name=other_npc.name,
                                location=location
                            ))
            
            # Brief pause between responses
            time.sleep(0.5)
        
        return responses
    
    def _extract_tags_from_message(self, message: str) -> List[str]:
        """Extract relevant tags from message for memory categorization"""
        message_lower = message.lower()
        tags = []
        
        topic_keywords = {
            "feeding": ["feed", "hay", "grain", "food", "eat"],
            "training": ["train", "exercise", "practice", "lesson"],
            "health": ["health", "sick", "vet", "medicine", "injury"],
            "grooming": ["brush", "groom", "clean", "wash"],
            "equipment": ["saddle", "bridle", "equipment", "gear"],
            "competition": ["compete", "show", "race", "competition"],
            "behavior": ["behavior", "nervous", "calm", "excited"]
        }
        
        for tag, keywords in topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    def _add_to_global_log(self, role: str, content: str, agent_name: str):
        """Add message to global conversation log"""
        message = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=time.time(),
            agent_name=agent_name,
            location=self.current_location
        )
        self.conversation_log.append(message)
    
    def end_conversation(self, npc_id: str = None):
        """End conversation with specific NPC or all NPCs"""
        if npc_id:
            if npc_id in self.agents:
                self.agents[npc_id].reset_conversation_state()
                print(f"🔄 Reset conversation state with {npc_id}")
        else:
            for agent in self.agents.values():
                agent.reset_conversation_state()
            print("🔄 Reset all conversation states")
    
    def reset_all(self):
        """Reset all conversations and memories (new game)"""
        for agent in self.agents.values():
            agent.reset_conversation()  # Full reset including memories
        self.conversation_log = []
        self.momentum.conversation_chains = []
        self.momentum.chain_count = 0
        self.memory_manager.reset_session()
        print("🔄 Reset all conversations and memories")
    
    def get_stats(self) -> Dict:
        """Get system statistics including memory stats, RAG status, and discovered locations"""
        memory_stats = self.memory_manager.get_system_stats()
        discovered_locations = self.memory_manager.get_discovered_locations()
        
        stats = {
            "total_agents": len(self.agents),
            "total_messages": len(self.conversation_log),
            "momentum_chains": len(self.momentum.conversation_chains),
            "current_location": self.current_location,
            "rag_status": self.get_rag_status(),
            "memory_system": memory_stats,
            "location_stats": self.location_coordinator.get_stats(),
            "turn_management": self.turn_manager.get_conversation_stats(),
            "discovered_locations": discovered_locations,
            "location_count": len(discovered_locations),
            "agents": {}
        }
        
        # Add RAG stats if available
        if self.rag_enabled:
            rag_stats = {}
            for agent_name, agent in self.agents.items():
                if hasattr(agent, 'get_rag_stats'):
                    rag_stats[agent_name] = agent.get_rag_stats()
            if rag_stats:
                stats["rag_stats"] = rag_stats
        
        for agent_name, agent in self.agents.items():
            stats["agents"][agent_name] = agent.get_stats()
        
        return stats
    
    def list_agents(self) -> List[str]:
        """Get list of agent names"""
        return list(self.agents.keys()) + ["All"]
    
    def show_memory_summary(self, npc_name: str = None):
        """Show memory summary for specific NPC or all NPCs, including location info"""
        if npc_name and npc_name in self.agents:
            agent_memory = self.memory_manager.get_npc_memory(npc_name)
            if agent_memory:
                summary = agent_memory.get_memory_summary()
                current_location = getattr(self.agents[npc_name], 'current_location', 'unknown')
                
                print(f"\n🧠 {npc_name}'s Memory Summary:")
                print(f"  Current location: {current_location}")
                print(f"  Total memories: {summary['total_memories']}")
                print(f"  Witnessed events: {summary['witnessed_events']}")
                print(f"  Player told: {summary['player_told']}")
                print(f"  NPC told: {summary['npc_told']}")
                print(f"  Overheard: {summary['overheard']}")
                print(f"  Recent (1h): {summary['recent_memories']}")
                print(f"  Locations visited: {', '.join(summary.get('locations_visited', []))}")
            else:
                print(f"❌ No memory data for {npc_name}")
        else:
            # Show all NPCs with location info
            stats = self.memory_manager.get_system_stats()
            discovered_locations = self.memory_manager.get_discovered_locations()
            
            print(f"\n🧠 Memory System Summary:")
            print(f"  Global events: {stats['global_events']}")
            print(f"  Discovered locations: {', '.join(discovered_locations)}")
            
            for npc_name, npc_stats in stats['npc_stats'].items():
                current_location = npc_stats.get('current_location', 'unknown')
                total_memories = npc_stats.get('total_memories', 0)
                locations_visited = npc_stats.get('locations_visited', [])
                print(f"  {npc_name} at {current_location}: {total_memories} memories, visited {len(locations_visited)} locations")
    
    def get_available_conversation_targets(self) -> Dict[str, Any]:
        """Get NPCs available for conversation at current location with UI metadata"""
        available_npcs = self.turn_manager.get_available_conversation_targets(
            self._current_location, self.agents
        )
        
        # Add location context
        location_info = self.location_coordinator.get_location_atmosphere(
            self._current_location, self.agents
        )
        
        # Get active group conversations at this location
        active_conversations = {
            conv_id: info
            for conv_id, info in self.active_group_conversations.items()
            if info["location"] == self._current_location
        }
        
        return {
            "npcs": available_npcs,
            "location": location_info,
            "active_conversations": active_conversations
        }
    
    def start_group_conversation(self, npc_names: List[str], initial_message: str) -> Dict[str, Any]:
        """Start a group conversation with selected NPCs using the enhanced group manager"""
        if not all(name in self.agents for name in npc_names):
            return {"error": "One or more NPCs not found"}
            
        # Validate all NPCs are in current location
        for name in npc_names:
            agent = self.agents[name]
            if getattr(agent, 'current_location', None) != self._current_location:
                return {"error": f"{name} is not at {self._current_location}"}
        
        # Create conversation ID and start group conversation
        conversation_id = str(uuid.uuid4())
        success = self.group_manager.start_group_conversation(
            conversation_id,
            npc_names,
            initial_message,  # Use as initial topic
            self._current_location
        )
        
        if not success:
            return {"error": "Failed to start group conversation"}
        
        # Add message to start conversation
        response, success, time_taken = self.group_manager.add_message(
            conversation_id,
            "player",
            initial_message,
            is_player=True
        )
        
        if not success:
            self.group_manager.end_conversation(conversation_id)
            return {"error": "Failed to start conversation"}
        
        # Store conversation info
        self.active_group_conversations[conversation_id] = {
            "npc_names": npc_names,
            "location": self._current_location,
            "start_time": time.time(),
            "current_topic": initial_message
        }
        
        return {
            "conversation_id": conversation_id,
            "location": self._current_location,
            "participants": npc_names,
            "initial_response": response
        }
    
    def send_group_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        """Send message to an active group conversation"""
        if conversation_id not in self.active_group_conversations:
            return {"error": "Conversation not found"}
        
        # Add message to conversation
        response, success, time_taken = self.group_manager.add_message(
            conversation_id,
            "player",
            message,
            is_player=True
        )
        
        if not success:
            return {"error": "Failed to process message"}
        
        return {
            "conversation_id": conversation_id,
            "response": response,
            "time_taken": time_taken
        }
    
    def end_group_conversation(self, conversation_id: str):
        """End a group conversation"""
        if conversation_id in self.active_group_conversations:
            # End conversation in group manager
            self.group_manager.end_conversation(conversation_id)
            
            # Clean up local tracking
            del self.active_group_conversations[conversation_id]
            
            return {"status": "success", "message": "Conversation ended"}
        return {"error": "Conversation not found"}