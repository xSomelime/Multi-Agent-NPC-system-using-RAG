#!/usr/bin/env python3
"""
Enhanced Conversation Manager
============================

Main orchestrator for multi-agent conversations, integrating memory,
spatial awareness, turn management, and conversational momentum.
"""

import time
import uuid
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from memory.session_memory import MemoryManager, ConfidenceLevel
from src.core.message_types import Message
from .conversational_momentum import ConversationalMomentum
from .turn_management import TurnManager
from .location_coordinator import LocationCoordinator


class EnhancedConversationManager:
    """Enhanced ConversationManager with memory integration and RAG support"""
    
    def __init__(self, enable_rag: bool = True):
        self.memory_manager = MemoryManager()
        self.agents: Dict[str, Any] = {}
        self.conversation_log: List[Message] = []
        
        # Initialize coordination components
        self.momentum = ConversationalMomentum(self.memory_manager)
        self.turn_manager = TurnManager()
        self.location_coordinator = LocationCoordinator(self.memory_manager)
        
        # RAG configuration
        self.rag_enabled = enable_rag
        
        if self.rag_enabled:
            try:
                # Check if RAG is available
                from src.agents.rag_enhanced_agent import RAG_AVAILABLE
                if not RAG_AVAILABLE:
                    self.rag_enabled = False
                    print("📚 Running with standard NPCs (RAG dependencies not available)")
                else:
                    print("🔥 RAG system enabled for enhanced horse knowledge!")
            except ImportError:
                self.rag_enabled = False
                print("📚 Running with standard NPCs (no RAG)")
        else:
            print("📚 Running with standard NPCs (RAG disabled)")
    
    @property
    def current_location(self) -> str:
        """Get current player location"""
        return self.location_coordinator.player_location
    
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
        """Get current RAG status"""
        try:
            from src.agents.rag_enhanced_agent import RAG_AVAILABLE
            if not RAG_AVAILABLE:
                return "❌ Not Available"
            elif self.rag_enabled:
                return "✅ Enabled"
            else:
                return "🔄 Disabled"
        except ImportError:
            return "❌ Not Available"
    
    def register_agent(self, agent: Any):
        """Register an agent"""
        self.agents[agent.name] = agent
        role_desc = getattr(agent, 'template_data', {}).get('title', agent.npc_role.value)
        print(f"🎭 Registered {agent.name} ({role_desc}) at {agent.current_location}")
    
    def move_player_to_location(self, location: str):
        """Move player to a new location"""
        success, message = self.location_coordinator.move_player(location)
        if success:
            print(f"🚶 {message}")
            
            # Show who's here
            npcs_here = self.location_coordinator.get_npcs_at_location(location, self.agents)
            if npcs_here:
                print(f"👥 NPCs here: {', '.join(npcs_here)}")
        else:
            print(f"❌ {message}")
    
    def move_npc_to_location(self, npc_name: str, location: str):
        """Move an NPC to a new location"""
        success, message = self.location_coordinator.move_npc(npc_name, location, self.agents)
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
        return success
    
    def send_to_agent(self, agent_name: str, message: str) -> Tuple[str, bool, float]:
        """Enhanced direct messaging with memory recording"""
        agent = self.agents.get(agent_name)
        if not agent:
            return f"Agent '{agent_name}' not found.", False, 0.0
        
        # Check if player and agent are in same location
        if agent.current_location != self.current_location:
            return f"{agent_name} is not here (they're at {agent.current_location})", False, 0.0
        
        # Add to global log
        self._add_to_global_log("user", message, "player")
        
        # Get response
        response, success, response_time = agent.generate_response(
            message, self.conversation_log[-5:]
        )
        
        # Log agent response
        if success:
            self._add_to_global_log("assistant", response, agent_name)
            
            # Record this as an NPC response that others can overhear
            nearby_npcs = self.location_coordinator.get_npcs_at_location(
                self.current_location, self.agents
            )
            nearby_npcs = [name for name in nearby_npcs if name != agent_name]
            
            for nearby_npc in nearby_npcs:
                self.memory_manager.npc_tells_npc(
                    agent_name, nearby_npc,
                    f"conversation with player: {response}",
                    self.current_location,
                    ConfidenceLevel.CONFIDENT,
                    ["overheard_conversation"]
                )
            
            # Check if other NPCs should auto-respond
            auto_triggers = self.momentum.should_trigger_auto_response(
                response, agent_name, self.current_location, self.agents
            )
            
            if auto_triggers:
                print(f"\n🔄 Others want to join the conversation:")
                
                for other_agent_name, reason, probability in auto_triggers[:1]:
                    other_agent = self.agents[other_agent_name]
                    role_desc = getattr(other_agent, 'template_data', {}).get('title', other_agent.npc_role.value)
                    
                    print(f"⚡ {other_agent_name} ({role_desc}) joining in ({reason})")
                    
                    auto_response, auto_success, auto_time = self.momentum.execute_auto_response(
                        other_agent_name, reason, response, agent_name, 
                        self.current_location, self.conversation_log[-3:]
                    )
                    
                    if auto_success:
                        self._add_to_global_log("assistant", auto_response, other_agent_name)
                        other_agent.add_message("assistant", auto_response)
                        print(f"  {other_agent_name}: {auto_response}")
        
        return response, success, response_time
    
    def send_to_all(self, message: str) -> List[Tuple[str, str, bool, float]]:
        """Enhanced send_to_all with memory and location awareness"""
        # Add player message to global log
        self._add_to_global_log("user", message, "player")
        
        # Only include NPCs in current location
        available_agents = {name: agent for name, agent in self.agents.items() 
                          if agent.current_location == self.current_location}
        
        if not available_agents:
            print(f"📍 No NPCs at {self.current_location}")
            return []
        
        # Use turn manager to determine participants
        participation_scores = self.turn_manager.select_participants(
            message, available_agents, self.current_location, self.memory_manager
        )
        
        if not participation_scores:
            # Fallback to at least one agent
            first_agent = list(available_agents.values())[0]
            print(f"💭 {first_agent.name} will respond at {self.current_location}")
            participating_agents = [first_agent]
        else:
            # Get conversation type and speaking order
            conversation_type = self.turn_manager.determine_conversation_type(message)
            speaking_order = self.turn_manager.determine_speaking_order(
                participation_scores, conversation_type
            )
            
            participating_agents = [available_agents[name] for name in speaking_order]
            print(f"💭 {len(participating_agents)} NPCs will respond at {self.current_location} ({conversation_type.value} mode)")
        
        # Record that all NPCs in location heard player's message
        npc_names_here = list(available_agents.keys())
        self.memory_manager.record_witnessed_event(
            f"Player said: {message}",
            self.current_location,
            npc_names_here,
            ["player_statement"] + self._extract_tags_from_message(message)
        )
        
        all_responses = []
        current_context = self.conversation_log.copy()
        
        # Get responses sequentially
        for agent in participating_agents:
            role_desc = getattr(agent, 'template_data', {}).get('title', agent.npc_role.value)
            print(f"⏳ {agent.name} ({role_desc}) responding...")
            print(f"🔍 DEBUG: Generating response for {agent.name}")
            
            others_responses = [(resp[0], resp[1]) for resp in all_responses]
            
            response, success, response_time = agent.generate_response(
                message, current_context[-5:], others_responses
            )
            
            if success:
                print(f"🔍 DEBUG: {agent.name} generated response: '{response[:50]}...'")
                all_responses.append((agent.name, response, success, response_time))
                
                # Add to context immediately
                response_msg = Message(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content=response,
                    timestamp=time.time(),
                    agent_name=agent.name,
                    location=self.current_location
                )
                current_context.append(response_msg)
                self.conversation_log.append(response_msg)
                
                # Record that other NPCs overheard this response
                other_npcs_here = [name for name in npc_names_here if name != agent.name]
                for other_npc in other_npcs_here:
                    self.memory_manager.npc_tells_npc(
                        agent.name, other_npc,
                        f"response to player: {response}",
                        self.current_location,
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
                last_message, last_speaker, self.current_location, available_agents
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
                        self.current_location, self.conversation_log[-5:]
                    )
                    
                    if success:
                        all_responses.append((agent_name, auto_response, success, response_time))
                        self._add_to_global_log("assistant", auto_response, agent_name)
                        agent.add_message("assistant", auto_response)
                        
                        print(f"✅ {agent_name} auto-response generated successfully")
                        time.sleep(0.3)
                    
                    break
        
        # Show results
        print(f"\n💬 Responses at {self.current_location}:")
        print(f"🔍 DEBUG: About to display {len(all_responses)} responses")
        for i, (agent_name, response, success, response_time) in enumerate(all_responses):
            print(f"🔍 DEBUG: Response {i+1}: {agent_name} - '{response[:30]}...'")
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
    
    def reset_all(self):
        """Reset all conversations and memory"""
        for agent in self.agents.values():
            agent.reset_conversation()
        self.conversation_log = []
        self.momentum.conversation_chains = []
        self.momentum.chain_count = 0
        self.memory_manager.reset_session()
        print("🔄 All conversations and memories reset")
    
    def get_stats(self) -> Dict:
        """Get system statistics including memory stats and RAG status"""
        memory_stats = self.memory_manager.get_system_stats()
        
        stats = {
            "total_agents": len(self.agents),
            "total_messages": len(self.conversation_log),
            "momentum_chains": len(self.momentum.conversation_chains),
            "current_location": self.current_location,
            "rag_status": self.get_rag_status(),
            "memory_system": memory_stats,
            "location_stats": self.location_coordinator.get_stats(),
            "turn_management": self.turn_manager.get_conversation_stats(),
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
        """Show memory summary for specific NPC or all NPCs"""
        if npc_name and npc_name in self.agents:
            agent_memory = self.memory_manager.get_npc_memory(npc_name)
            if agent_memory:
                summary = agent_memory.get_memory_summary()
                print(f"\n🧠 {npc_name}'s Memory Summary:")
                print(f"  Total memories: {summary['total_memories']}")
                print(f"  Witnessed events: {summary['witnessed_events']}")
                print(f"  Player told: {summary['player_told']}")
                print(f"  NPC told: {summary['npc_told']}")
                print(f"  Overheard: {summary['overheard']}")
                print(f"  Recent (1h): {summary['recent_memories']}")
            else:
                print(f"❌ No memory data for {npc_name}")
        else:
            # Show all NPCs
            stats = self.memory_manager.get_system_stats()
            print(f"\n🧠 Memory System Summary:")
            print(f"  Global events: {stats['global_events']}")
            for npc_name, npc_stats in stats['npc_stats'].items():
                print(f"  {npc_name}: {npc_stats['total_memories']} memories") 