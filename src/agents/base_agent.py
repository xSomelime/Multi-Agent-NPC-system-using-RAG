#!/usr/bin/env python3
"""
Base NPC Agent
==============

Core NPC agent implementation with memory integration, personality, and conversation management.
Handles response generation, memory context, and participation logic.
"""

import time
import uuid
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from memory.session_memory import MemoryManager
from src.core.message_types import Message, NPCRole
from src.core.config_loaders import NPCLoader, RoleTemplateLoader
from src.agents.ollama_manager import get_ollama_manager


class ScalableNPCAgent:
    """Enhanced NPC Agent with memory integration"""
    
    def __init__(self, npc_config_name: str, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.current_location = None  # Location will be set by UE5
        
        self.npc_loader = NPCLoader()
        self.template_loader = RoleTemplateLoader()
        
        # Load configuration data
        self.npc_data = self.npc_loader.load_npc_config(npc_config_name)
        self.template_data = self.template_loader.load_role_template(
            self.npc_data['role_template']
        )
        
        # Basic properties
        self.name = self.npc_data['name']
        self.npc_role = NPCRole(self.npc_data['role_template'])
        self.conversation_history: List[Message] = []
        
        # Register with memory manager - no initial location
        self.memory_manager.register_npc(self.name)
        
        # Professional opinions for debates
        self.professional_opinions = self.npc_data.get('professional_opinions', {})
        self.controversial_stances = self.npc_data.get('controversial_stances', [])
        
        # Build persona
        self.persona = self._build_persona()
        
        # LLM settings
        self.ollama_url = "http://localhost:11434/api"
        self.model = "phi3:mini"
        
        if self.npc_role == NPCRole.RIVAL:
            self.temperature = 0.4
            self.max_response_length = 60
        else:
            self.temperature = 0.3
            self.max_response_length = 50
        
        # Expertise areas for smart participation
        self.expertise_areas = self.template_data.get('expertise_areas', [])
        
        print(f"✅ Created {self.name} ({self.template_data.get('title', 'NPC')})")
    
    def move_to_location(self, new_location: str):
        """Move NPC to a new location"""
        old_location = self.current_location
        self.current_location = new_location
        self.memory_manager.update_npc_location(self.name, new_location)
        
        # Record movement in memory
        self.memory_manager.record_witnessed_event(
            f"{self.name} moved from {old_location} to {new_location}",
            new_location,
            [self.name],
            ["movement", "location_change"]
        )
        
        print(f"🚶 {self.name} moved to {new_location}")
    
    def _build_persona(self) -> str:
        """Build comprehensive persona from template + individual data"""
        name = self.npc_data['name']
        title = self.template_data.get('title', 'Horse Care Professional')
        background = self.npc_data.get('personal_background', '')
        traits = ", ".join(self.npc_data.get('personality', {}).get('traits', ['helpful']))
        style = self.npc_data.get('personality', {}).get('speaking_style', 'friendly')
        
        persona = f"""You are {name}, a {title} working at this horse stable.
        
Background: {background}

Personality: You are {traits}. Your speaking style is {style}.

CRITICAL RULES:
- Give exactly ONE sentence response
- Never mention other people's names unless directly relevant
- Be natural and conversational but very brief
- Don't start with 'Oh' or 'That's interesting' or similar filler
- Stay completely in character as {name}
"""
        
        # Add role-specific personality reinforcement
        if self.npc_role == NPCRole.RIVAL:
            persona += f"""

AS {name.upper()} THE RIVAL:
- You are wealthy and dismissive - briefly mention expensive things
- Judge others for having cheaper equipment in one sentence
- Be arrogant but concise
"""
        
        return persona
    
    def generate_response(self, user_input: str, conversation_context: List[Message] = None, others_responses: List = None) -> Tuple[str, bool, float]:
        """Generate response using memory context and thread-safe Ollama requests"""
        start_time = time.time()
        
        # Get relevant memories for context
        memory_context = self.memory_manager.get_relevant_context_for_npc(
            self.name, user_input, max_memories=2
        )
        
        # Build prompt
        conversation_type = self.detect_conversation_type(user_input)
        
        prompt_parts = [
            f"STAY IN CHARACTER: You are {self.name}, not Dr. Evelyn or anyone else.",
            self.persona
        ]
        
        # Add memory context if available
        if memory_context:
            prompt_parts.append(f"What you remember: {'; '.join(memory_context[:2])}")
        
        # Add limited conversation context
        if conversation_context:
            prompt_parts.append("Recent context:")
            for msg in conversation_context[-2:]:
                if msg.role == "user":
                    prompt_parts.append(f"Player: {msg.content}")
                elif msg.role == "assistant" and msg.agent_name != self.name:
                    prompt_parts.append(f"{msg.agent_name}: {msg.content}")
        
        # Conversation type specific instructions
        if conversation_type == "debate":
            opinions = self.get_relevant_opinions(user_input)
            if opinions != "No specific opinions on this topic":
                prompt_parts.append(f"Your opinion: {opinions[:100]}")
            
            if others_responses and len(others_responses) > 0:
                last_response = others_responses[-1]
                prompt_parts.append(f"{last_response[0]} just said: {last_response[1]}")
            
            if self.npc_role == NPCRole.RIVAL:
                instruction = f"Give one arrogant sentence as {self.name} about why your expensive approach is better."
            else:
                instruction = f"Give one professional sentence as {self.name} based on your expertise."
        else:
            if self.npc_role == NPCRole.RIVAL:
                instruction = f"Give one brief, dismissive sentence as {self.name}."
            else:
                instruction = f"Give one helpful, friendly sentence as {self.name}."
        
        prompt_parts.append(instruction)
        prompt_parts.append(f"Player: {user_input}")
        prompt_parts.append(f"{self.name}:")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Use thread-safe request manager
        ollama_manager = get_ollama_manager()
        agent_response, success = ollama_manager.make_request(
            self.name, 
            full_prompt, 
            self.model, 
            self.temperature, 
            self.max_response_length
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if success:
            # Record conversation in both local and memory manager
            self.add_message("user", user_input)
            self.add_message("assistant", agent_response)
            
            # Record player telling this NPC something
            self.memory_manager.player_tells_npc(
                self.name, user_input, self.current_location, 
                self._extract_tags_from_message(user_input)
            )
            
            return agent_response, True, response_time
        else:
            return agent_response, False, response_time
    
    def _extract_tags_from_message(self, message: str) -> List[str]:
        """Extract relevant tags from message for memory categorization"""
        message_lower = message.lower()
        tags = []
        
        # Common horse-related topics
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
    
    def detect_conversation_type(self, message: str) -> str:
        """Detect if this should be a debate or casual conversation"""
        message_lower = message.lower()
        
        debate_triggers = ["better", "prefer", "best", "vs", "versus", "compare", "which", "what do you think about", "opinion"]
        controversial_topics = ["saddle", "feed", "training", "method", "brand", "equipment", "technique", "hay", "silage", "grain"]
        
        is_debate_question = any(trigger in message_lower for trigger in debate_triggers)
        has_controversial_topic = any(topic in message_lower for topic in controversial_topics)
        
        if is_debate_question and has_controversial_topic:
            return "debate"
        else:
            return "conversation"
    
    def get_relevant_opinions(self, topic: str) -> str:
        """Get relevant professional opinions for debate topics"""
        topic_lower = topic.lower()
        relevant_opinions = []
        
        for opinion_topic, opinion in self.professional_opinions.items():
            if any(keyword in topic_lower for keyword in opinion_topic.split('_')):
                relevant_opinions.append(f"{opinion_topic}: {opinion}")
        
        for stance in self.controversial_stances:
            if any(word in stance.lower() for word in topic_lower.split() if len(word) > 3):
                relevant_opinions.append(stance)
        
        return "; ".join(relevant_opinions) if relevant_opinions else "No specific opinions on this topic"
    
    def should_participate(self, message_content: str, existing_responses: List = None) -> bool:
        """Enhanced participation logic using memory"""
        message_lower = message_content.lower()
        existing_count = len(existing_responses or [])
        
        # Always respond if directly addressed by name
        if self.name.lower() in message_lower:
            return True
        
        # Check if NPC has relevant memories about the topic
        agent_memory = self.memory_manager.get_npc_memory(self.name)
        if agent_memory:
            message_words = [word for word in message_lower.split() if len(word) > 3]
            for word in message_words:
                if agent_memory.knows_about(word):
                    return True
        
        # Check for expertise match
        for area in self.expertise_areas:
            area_keywords = area.replace('_', ' ').split()
            if any(keyword in message_lower for keyword in area_keywords):
                return True
        
        # For debates, more people should participate
        conversation_type = self.detect_conversation_type(message_content)
        if conversation_type == "debate":
            return existing_count < 3
        
        # For general questions, be more selective
        general_indicators = ["all", "everyone", "how is", "how are"]
        if any(indicator in message_lower for indicator in general_indicators):
            return existing_count < 2
        
        # Lower chance for random participation
        if existing_count == 0:
            return random.random() < 0.4
        
        return False
    
    def add_message(self, role: str, content: str) -> Message:
        """Add message to conversation history with location"""
        message = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=time.time(),
            agent_name=self.name if role == "assistant" else "player",
            location=self.current_location
        )
        self.conversation_history.append(message)
        return message
    
    def reset_conversation_state(self):
        """Reset only the conversation state while preserving memories.
        This is used when a player walks away and comes back."""
        self.conversation_history = []
    
    def reset_conversation(self):
        """Reset both conversation state and memories.
        WARNING: This will clear ALL conversation history and should only be used
        when starting a completely new game session."""
        self.conversation_history = []
        # Get agent's memory and clear it
        agent_memory = self.memory_manager.get_npc_memory(self.name)
        if agent_memory:
            agent_memory.memories.clear()
            agent_memory.known_facts.clear()
    
    def get_stats(self) -> Dict:
        """Get agent statistics including memory stats"""
        agent_memory = self.memory_manager.get_npc_memory(self.name)
        memory_stats = agent_memory.get_memory_summary() if agent_memory else {}
        
        return {
            "name": self.name,
            "role": self.npc_role.value,
            "title": self.template_data.get('title', 'NPC'),
            "total_messages": len(self.conversation_history),
            "expertise_areas": self.expertise_areas,
            "current_location": self.current_location,
            "memory_stats": memory_stats
        } 