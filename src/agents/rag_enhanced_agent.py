#!/usr/bin/env python3
"""
RAG-Enhanced NPC Agent
Extends the base NPC agent with RAG knowledge retrieval capabilities
"""

import sys
import os
import time
import logging
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check RAG system availability
try:
    from knowledge.rag_system import get_rag_system
    from knowledge.npc_rag_integration import get_npc_rag_interface, NPCKnowledgeContext
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"⚠️  RAG dependencies not available: {e}")

from src.agents.base_agent import ScalableNPCAgent

class RAGEnhancedNPCAgent:
    """
    Enhanced NPC Agent with RAG knowledge integration
    Wraps existing ScalableNPCAgent and adds RAG functionality
    """
    
    def __init__(self, base_agent: ScalableNPCAgent, enable_rag: bool = True):
        """
        Initialize RAG-enhanced agent
        
        Args:
            base_agent: Instance of ScalableNPCAgent
            enable_rag: Whether to enable RAG functionality
        """
        self.base_agent = base_agent
        self.enable_rag = enable_rag
        
        # Initialize RAG components if enabled
        self.rag_interface = None
        if self.enable_rag:
            try:
                self.rag_interface = get_npc_rag_interface()
                logging.info(f"RAG enabled for {self.base_agent.name}")
            except Exception as e:
                logging.warning(f"Failed to initialize RAG for {self.base_agent.name}: {e}")
                self.enable_rag = False
    
    def generate_response(self, user_input: str, conversation_context: List = None, others_responses: List = None) -> Tuple[str, bool, float]:
        """
        Enhanced response generation with RAG knowledge integration
        
        Args:
            user_input: Player's message
            conversation_context: Recent conversation history
            others_responses: Responses from other NPCs
            
        Returns:
            Tuple of (response, success, response_time)
        """
        if not self.enable_rag or not self.rag_interface:
            # Fall back to base agent if RAG unavailable
            return self.base_agent.generate_response(user_input, conversation_context, others_responses)
        
        try:
            # Create knowledge context for RAG
            context = NPCKnowledgeContext(
                npc_name=self.base_agent.name,
                role=self.base_agent.npc_role.value,
                current_topic=user_input,
                conversation_history=[user_input] + ([ctx.content for ctx in conversation_context[-3:]] if conversation_context else []),
                confidence_level="medium",
                max_knowledge_items=3
            )
            
            # Check if RAG should be used for this topic
            use_rag = self.rag_interface.should_use_rag_for_topic(user_input, context.role)
            
            if use_rag:
                # Get RAG-enhanced response
                return self._generate_rag_enhanced_response(user_input, context, conversation_context, others_responses)
            else:
                # Use base agent for non-knowledge topics
                return self.base_agent.generate_response(user_input, conversation_context, others_responses)
                
        except Exception as e:
            logging.error(f"RAG error for {self.base_agent.name}: {e}")
            # Fall back to base agent on RAG failure
            return self.base_agent.generate_response(user_input, conversation_context, others_responses)
    
    def _generate_rag_enhanced_response(self, user_input: str, context: NPCKnowledgeContext, 
                                      conversation_context: List = None, others_responses: List = None) -> Tuple[str, bool, float]:
        """Generate response enhanced with RAG knowledge"""
        start_time = time.time()
        
        # Build enhanced prompt using RAG
        base_prompt = self.base_agent._build_persona()
        enhanced_prompt = self.rag_interface.enhance_npc_prompt(base_prompt, context)
        
        # Get memory context from existing system
        memory_context = self.base_agent.memory_manager.get_relevant_context_for_npc(
            self.base_agent.name, user_input, max_memories=2
        )
        
        # Build conversation type and opinions (existing logic)
        conversation_type = self.base_agent.detect_conversation_type(user_input)
        
        prompt_parts = [
            f"STAY IN CHARACTER: You are {self.base_agent.name}, not Dr. Evelyn or anyone else.",
            enhanced_prompt  # This now includes RAG knowledge
        ]
        
        # Add memory context if available
        if memory_context:
            prompt_parts.append(f"What you remember: {'; '.join(memory_context[:2])}")
        
        # Add limited conversation context
        if conversation_context:
            prompt_parts.append("Recent context:")
            for msg in conversation_context[-2:]:
                if hasattr(msg, 'role'):
                    if msg.role == "user":
                        prompt_parts.append(f"Player: {msg.content}")
                    elif msg.role == "assistant" and msg.agent_name != self.base_agent.name:
                        prompt_parts.append(f"{msg.agent_name}: {msg.content}")
        
        # Conversation type specific instructions
        if conversation_type == "debate":
            opinions = self.base_agent.get_relevant_opinions(user_input)
            if opinions != "No specific opinions on this topic":
                prompt_parts.append(f"Your opinion: {opinions[:100]}")
            
            if others_responses and len(others_responses) > 0:
                last_response = others_responses[-1]
                prompt_parts.append(f"{last_response[0]} just said: {last_response[1]}")
            
            if self.base_agent.npc_role == "rival":
                instruction = f"Give one arrogant sentence as {self.base_agent.name} about why your expensive approach is better."
            else:
                instruction = f"Give one professional sentence as {self.base_agent.name} based on your expertise."
        else:
            if self.base_agent.npc_role == "rival":
                instruction = f"Give one brief, dismissive sentence as {self.base_agent.name}."
            else:
                instruction = f"Give one helpful, friendly sentence as {self.base_agent.name}."
        
        prompt_parts.append(instruction)
        prompt_parts.append(f"Player: {user_input}")
        prompt_parts.append(f"{self.base_agent.name}:")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Use thread-safe request manager
        ollama_manager = self.base_agent.ollama_manager
        agent_response, success = ollama_manager.make_request(
            self.base_agent.name, 
            full_prompt, 
            self.base_agent.model, 
            self.base_agent.temperature, 
            self.base_agent.max_response_length
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if success:
            # Record conversation using base agent methods
            self.base_agent.add_message("user", user_input)
            self.base_agent.add_message("assistant", agent_response)
            
            # Record player telling this NPC something
            self.base_agent.memory_manager.player_tells_npc(
                self.base_agent.name, user_input, self.base_agent.current_location,
                self.base_agent._extract_tags_from_message(user_input)
            )
            
            # Log successful RAG usage
            logging.debug(f"RAG-enhanced response for {self.base_agent.name}: {agent_response[:50]}...")
            
        return agent_response, success, response_time
    
    def should_participate(self, message_content: str, existing_responses: List = None) -> bool:
        """Enhanced participation logic that considers RAG knowledge"""
        
        # First check base agent logic
        base_participation = self.base_agent.should_participate(message_content, existing_responses)
        
        if base_participation:
            return True
        
        # If RAG is enabled, check if we have relevant knowledge
        if self.enable_rag and self.rag_interface:
            try:
                return self.rag_interface.should_use_rag_for_topic(
                    message_content, self.base_agent.npc_role.value
                )
            except Exception as e:
                logging.error(f"RAG participation check failed for {self.base_agent.name}: {e}")
        
        return False
    
    def move_to_location(self, new_location: str):
        """Delegate location movement to base agent"""
        self.base_agent.move_to_location(new_location)
    
    def reset_conversation_state(self):
        """Reset only the conversation state while preserving memories."""
        self.base_agent.reset_conversation_state()
    
    def reset_conversation(self):
        """Reset both conversation state and memories."""
        self.base_agent.reset_conversation()
    
    def get_stats(self) -> Dict:
        """Get agent statistics including RAG stats"""
        base_stats = self.base_agent.get_stats()
        
        # Add RAG-specific stats
        base_stats.update({
            "rag_enabled": self.enable_rag,
            "rag_interface_available": self.rag_interface is not None
        })
        
        return base_stats
    
    # Forward other base agent properties
    @property
    def name(self):
        return self.base_agent.name
    
    @property
    def current_location(self):
        return self.base_agent.current_location
    
    @property
    def npc_role(self):
        return self.base_agent.npc_role
    
    @property
    def conversation_history(self):
        return self.base_agent.conversation_history


def create_rag_enhanced_agent(npc_config_name: str, memory_manager, enable_rag: bool = True):
    """
    Factory function to create RAG-enhanced NPC agents
    
    Args:
        npc_config_name: NPC configuration file name
        memory_manager: Session memory manager
        enable_rag: Whether to enable RAG functionality
        
    Returns:
        RAGEnhancedNPCAgent instance
    """
    # Create base agent without initial location
    base_agent = ScalableNPCAgent(npc_config_name, memory_manager)
    
    # Create RAG-enhanced wrapper
    rag_agent = RAGEnhancedNPCAgent(base_agent, enable_rag)
    
    return rag_agent

def create_rag_enhanced_team(memory_manager, enable_rag: bool = True):
    """Create the core stable team with RAG enhancement"""
    
    team_configs = [
        ("elin_behaviourist", "barn"),
        ("oskar_stable_hand", "stable_yard"),
        ("astrid_stable_hand", "barn"),
        ("chris_rival", "arena"),
        ("andy_trainer", "arena")
    ]
    
    rag_team = []
    
    for config_name, location in team_configs:
        try:
            rag_agent = create_rag_enhanced_agent(config_name, memory_manager, enable_rag)
            rag_team.append(rag_agent)
            logging.info(f"Created RAG-enhanced {rag_agent.base_agent.name}")
        except Exception as e:
            logging.error(f"Failed to create RAG agent for {config_name}: {e}")
            # Fall back to base agent
            base_agent = ScalableNPCAgent(config_name, memory_manager)
            rag_team.append(base_agent)
    
    return rag_team

if __name__ == "__main__":
    # Test RAG-enhanced agent
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing RAG-Enhanced Agent")
    
    # This would normally be run within the main system
    # Just print success message for now
    print("✅ RAG-Enhanced Agent module loaded successfully")
    print("💡 Use create_rag_enhanced_agent() to create enhanced NPCs")
    print("💡 Use create_rag_enhanced_team() to create full enhanced team") 