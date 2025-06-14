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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from memory.session_memory import MemoryManager, ConfidenceLevel
from src.core.message_types import Message
from .conversational_momentum import ConversationalMomentum
from .turn_management import TurnManager
from .location_coordinator import LocationCoordinator


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
        
        # Track active group conversations
        self.active_group_conversations: Dict[str, List[str]] = {}
        
        # RAG configuration
        self.rag_enabled = False
        
        try:
            # Check if RAG is available
            from src.agents.rag_enhanced_agent import RAG_AVAILABLE
            if RAG_AVAILABLE:
                self.rag_enabled = True
                print("🔥 RAG system enabled for enhanced horse knowledge!")
            else:
                print("📚 Running with standard NPCs (RAG dependencies not available)")
        except ImportError:
            print("📚 Running with standard NPCs (no RAG)")
        
        print("📍 Dynamic location system ready - will discover locations from UE5")
    
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
        print(f"🎭 Registered {agent.name} ({role_desc}) at {current_location}")
    
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
                print(f"📍 No NPCs at {location} currently")
        else:
            print(f"❌ {message}")
    
    def move_npc_to_location(self, npc_name: str, location: str):
        """Move an NPC to a new location (with dynamic location registration)"""
        success, message = self.location_coordinator.move_npc(npc_name, location, self.agents)
        if success:
            print(f"✅ {message}")
            
            # Auto-register this location if it's new
            discovered_locations = self.memory_manager.get_discovered_locations()
            if location not in discovered_locations:
                print(f"🆕 New location discovered: {location}")
        else:
            print(f"❌ {message}")
        return success
    
    def send_to_agent(self, agent_name: str, message: str) -> Tuple[str, bool, float]:
        """Enhanced direct messaging with memory recording and dynamic location support"""
        start_time = time.time()
        
        agent = self.agents.get(agent_name)
        if not agent:
            return f"Agent '{agent_name}' not found.", False, 0.0
        
        # Quick location check
        agent_location = getattr(agent, 'current_location', 'unknown')
        current_player_location = self.current_location
        
        if agent_location != current_player_location and agent_location != 'unknown' and current_player_location != 'unknown':
            return f"{agent_name} is not here (they're at {agent_location})", False, 0.0
        
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
            
            # Only process memory for successful responses
            self._process_memory_and_triggers(agent_name, response, agent_location)
            
        end_time = time.time()
        return response, success, end_time - start_time
    
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
                print("\n🔄 Others want to join the conversation:")
                threads = []
                for other_agent_name, reason, probability in auto_triggers[:2]:  # Limit to 2 auto-responses
                    other_agent = self.agents.get(other_agent_name)
                    if other_agent:
                        role_desc = getattr(other_agent, 'template_data', {}).get('title', other_agent.npc_role.value)
                        print(f"⚡ {other_agent_name} ({role_desc}) joining in ({reason})")
                        
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
                    print(f"  {npc_name}: {response[:50]}...")
        except Exception as e:
            print(f"Warning: Auto-response failed for {npc_name}: {e}")
    
    def send_to_all(self, message: str) -> List[Tuple[str, str, bool, float]]:
        """Enhanced send_to_all with memory and dynamic location awareness"""
        # Add player message to global log
        self._add_to_global_log("user", message, "player")
        
        # Only include NPCs in current location or unknown locations
        current_player_location = self.current_location
        available_agents = {}
        
        for name, agent in self.agents.items():
            agent_location = getattr(agent, 'current_location', 'unknown')
            # Include NPCs at same location or those with unknown location
            if agent_location == current_player_location or agent_location == 'unknown' or current_player_location == 'unknown':
                available_agents[name] = agent
        
        if not available_agents:
            print(f"📍 No NPCs available for conversation at {current_player_location}")
            return []
        
        print(f"💭 {len(available_agents)} NPCs available for conversation at {current_player_location}")
        
        # Use turn manager to determine participants
        participation_scores = self.turn_manager.select_participants(
            message, available_agents, current_player_location, self.memory_manager
        )
        
        if not participation_scores:
            # Fallback to at least one agent
            first_agent = list(available_agents.values())[0]
            print(f"💭 {first_agent.name} will respond at {current_player_location}")
            participating_agents = [first_agent]
        else:
            # Get conversation type and speaking order
            conversation_type = self.turn_manager.determine_conversation_type(message)
            speaking_order = self.turn_manager.determine_speaking_order(
                participation_scores, conversation_type
            )
            
            participating_agents = [available_agents[name] for name in speaking_order]
            print(f"💭 {len(participating_agents)} NPCs will respond at {current_player_location} ({conversation_type.value} mode)")
        
        # Record that all NPCs in location heard player's message
        npc_names_here = list(available_agents.keys())
        if npc_names_here:
            self.memory_manager.record_witnessed_event(
                f"Player said: {message}",
                current_player_location,
                npc_names_here,
                ["player_statement"] + self._extract_tags_from_message(message)
            )
        
        all_responses = []
        current_context = self.conversation_log.copy()
        
        # Get responses sequentially
        for agent in participating_agents:
            role_desc = getattr(agent, 'template_data', {}).get('title', agent.npc_role.value)
            print(f"⏳ {agent.name} ({role_desc}) responding...")
            
            others_responses = [(resp[0], resp[1]) for resp in all_responses]
            
            response, success, response_time = agent.generate_response(
                message, current_context[-5:], others_responses
            )
            
            if success:
                all_responses.append((agent.name, response, success, response_time))
                
                # Add to context immediately
                response_msg = Message(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content=response,
                    timestamp=time.time(),
                    agent_name=agent.name,
                    location=current_player_location
                )
                current_context.append(response_msg)
                self.conversation_log.append(response_msg)
                
                # Record that other NPCs overheard this response
                other_npcs_here = [name for name in npc_names_here if name != agent.name]
                for other_npc in other_npcs_here:
                    self.memory_manager.npc_tells_npc(
                        agent.name, other_npc,
                        f"response to player: {response}",
                        current_player_location,
                        ConfidenceLevel.CONFIDENT,
                        ["overheard_response"]
                    )
                
                time.sleep(0.5)
        
        # Check for auto-responses using momentum system
        if all_responses:
            last_speaker = all_responses[-1][0]
            last_message = all_responses[-1][1]
            
            # Get list of agents who already responded in this turn
            agents_who_already_responded = {resp[0] for resp in all_responses}
            
            auto_triggers = self.momentum.should_trigger_auto_response(
                last_message, last_speaker, current_player_location, available_agents
            )
            
            if auto_triggers:
                print(f"\n🔄 Auto-responses triggered:")
                
                for agent_name, reason, probability in auto_triggers:
                    # Skip if agent already responded in this turn
                    if agent_name in agents_who_already_responded:
                        print(f"⏩ {agent_name} skipped (already responded)")
                        continue
                    
                    agent = available_agents[agent_name]
                    role_desc = getattr(agent, 'template_data', {}).get('title', agent.npc_role.value)
                    
                    print(f"⚡ {agent_name} ({role_desc}) responding ({reason}, {probability:.1%})")
                    
                    auto_response, success, response_time = self.momentum.execute_auto_response(
                        agent_name, reason, last_message, last_speaker, 
                        current_player_location, self.conversation_log[-5:]
                    )
                    
                    if success:
                        all_responses.append((agent_name, auto_response, success, response_time))
                        self._add_to_global_log("assistant", auto_response, agent_name)
                        agent.add_message("assistant", auto_response)
                        
                        print(f"✅ {agent_name} auto-response generated successfully")
                        time.sleep(0.3)
                    
                    break
        
        # Show results
        print(f"\n💬 Responses at {current_player_location}:")
        for i, (agent_name, response, success, response_time) in enumerate(all_responses):
            if success:
                agent = self.agents[agent_name]
                role_desc = getattr(agent, 'template_data', {}).get('title', agent.npc_role.value)
                print(f"{agent_name} ({role_desc}): {response}")
        
        return all_responses
    
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
        
        return {
            "npcs": available_npcs,
            "location": location_info,
            "active_conversations": self.active_group_conversations
        }
    
    def start_group_conversation(self, npc_names: List[str], initial_message: str) -> Dict[str, Any]:
        """Start a group conversation with selected NPCs"""
        if not all(name in self.agents for name in npc_names):
            return {"error": "One or more NPCs not found"}
            
        # Validate all NPCs are in current location
        for name in npc_names:
            agent = self.agents[name]
            if getattr(agent, 'current_location', None) != self._current_location:
                return {"error": f"{name} is not at {self._current_location}"}
        
        # Create conversation ID
        conversation_id = str(uuid.uuid4())
        self.active_group_conversations[conversation_id] = npc_names
        
        # Get initial responses
        responses = []
        for name in npc_names:
            response, success, time = self.send_to_agent(name, initial_message)
            if success:
                responses.append({
                    "npc": name,
                    "response": response,
                    "role": getattr(self.agents[name], 'npc_role', 'unknown').value,
                    "title": getattr(self.agents[name], 'template_data', {}).get('title', '')
                })
        
        return {
            "conversation_id": conversation_id,
            "location": self._current_location,
            "participants": npc_names,
            "responses": responses
        }
    
    def end_group_conversation(self, conversation_id: str):
        """End a group conversation"""
        if conversation_id in self.active_group_conversations:
            del self.active_group_conversations[conversation_id]