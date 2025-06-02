#!/usr/bin/env python3
"""
Conversational Momentum Manager
==============================

Manages automatic NPC-to-NPC conversations with loop prevention.
Handles emotional triggers, expertise-based responses, and natural conversation flow.
"""

import time
import random
import re
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from memory.session_memory import MemoryManager, ConfidenceLevel


class ConversationalMomentum:
    """Manages automatic NPC-to-NPC conversations with loop prevention"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.conversation_chains: List[str] = []
        self.chain_count = 0
        self.max_chain_length = 4
        self.cooldown_period = 30
        self.last_auto_response: Dict[str, float] = defaultdict(float)
        
        # Emotional triggers for empathetic responses
        self.emotional_keywords = {
            "anxiety": ["nervous", "scared", "worried", "anxious", "afraid"],
            "excitement": ["excited", "thrilled", "amazing", "fantastic"],
            "frustration": ["frustrated", "annoying", "difficult", "hard"],
            "sadness": ["sad", "disappointed", "upset", "down"],
            "confidence": ["confident", "ready", "strong", "sure"]
        }
        
        # Question patterns that trigger responses
        self.question_patterns = [
            r"what do you think,?\s+(\w+)",
            r"(\w+),?\s+what.*\?",
            r"how does that (sound|feel),?\s+(\w+)",
            r"(\w+),?\s+do you",
            r"(\w+),?\s+can you",
            r"right,?\s+(\w+)\??",
            r"(\w+),?\s+would you"
        ]
    
    def should_trigger_auto_response(self, message_content: str, speaker_name: str, 
                                current_location: str, all_agents: Dict[str, Any]) -> List[Tuple[str, str, float]]:
        """Enhanced trigger detection using memory and spatial awareness"""
        
        # Reset chain if enough time passed or player spoke
        if speaker_name == "player":
            self.conversation_chains = []
            self.chain_count = 0
        
        # Check chain length limit
        if self.chain_count >= self.max_chain_length:
            print(f"🔄 Chain limit reached ({self.max_chain_length}), waiting for player input")
            return []
        
        triggered_responses = []
        message_lower = message_content.lower()
        current_time = time.time()
        
        for agent_name, agent in all_agents.items():
            if agent_name == speaker_name:
                continue
                
            # Skip if agent not in same location (spatial awareness)
            agent_memory = self.memory_manager.get_npc_memory(agent_name)
            if not agent_memory:
                continue
                
            # Check if agent can hear based on location
            agent_location = self.memory_manager.spatial_awareness.npc_locations.get(agent_name)
            if agent_location != current_location:
                continue
                
            # Check cooldown
            if current_time - self.last_auto_response[agent_name] < self.cooldown_period:
                continue
            
            response_reason = None
            response_probability = 0.0
            
            # 1. Direct mention/question detection (highest priority)
            mentioned_directly = self._check_direct_mention(message_content, agent_name)
            if mentioned_directly:
                response_reason = "mentioned_directly"
                response_probability = 0.95
            
            # 2. Question pattern detection
            elif self._check_question_pattern(message_content, agent_name):
                response_reason = "asked_question"
                response_probability = 0.85
            
            # 3. Memory-triggered response (knows about topic)
            elif self._check_memory_trigger(message_content, agent_name):
                response_reason = "memory_trigger"
                response_probability = 0.7
            
            # 4. Emotional content response
            elif self._should_respond_emotionally(message_content, agent, speaker_name):
                response_reason = "emotional_response"
                response_probability = 0.6
                
            # 5. Professional expertise trigger
            elif self._check_expertise_trigger(message_content, agent):
                response_reason = "expertise_trigger"
                response_probability = 0.4
            
            # 6. Relationship-based response
            elif self._check_relationship_trigger(message_content, agent, speaker_name):
                response_reason = "relationship_response"
                response_probability = 0.3
            
            # Add some randomness and chain length consideration
            if response_reason:
                # Reduce probability based on chain length
                chain_penalty = self.chain_count * 0.2
                final_probability = max(0.1, response_probability - chain_penalty)
                
                # Roll dice
                if random.random() < final_probability:
                    triggered_responses.append((agent_name, response_reason, final_probability))
        
        # Sort by probability (highest first) and limit to 1-2 responses
        triggered_responses.sort(key=lambda x: x[2], reverse=True)
        return triggered_responses[:2]
    
    def _check_memory_trigger(self, message: str, agent_name: str) -> bool:
        """Check if agent has relevant memories about the topic"""
        agent_memory = self.memory_manager.get_npc_memory(agent_name)
        if not agent_memory:
            return False
        
        # Extract key topics from message
        message_words = [word.lower() for word in message.split() if len(word) > 3]
        
        for word in message_words:
            if agent_memory.knows_about(word):
                return True
        
        return False
    
    def execute_auto_response(self, agent_name: str, trigger_reason: str, 
                            original_message: str, speaker_name: str,
                            current_location: str, conversation_context: List) -> Tuple[str, bool, float]:
        """Generate automatic response with memory context"""
        
        # Update tracking
        self.conversation_chains.append(agent_name)
        self.chain_count += 1
        self.last_auto_response[agent_name] = time.time()
        
        # Build context-aware prompt with memory
        memory_context = ""
        agent_memory = self.memory_manager.get_npc_memory(agent_name)
        if agent_memory:
            # Get relevant memories
            relevant_memories = self.memory_manager.get_relevant_context_for_npc(
                agent_name, original_message, max_memories=2
            )
            if relevant_memories:
                memory_context = f"Based on what you know: {'; '.join(relevant_memories)}. "
        
        if trigger_reason == "mentioned_directly":
            instruction = f"You were directly mentioned by {speaker_name}. Give a brief, natural response."
        elif trigger_reason == "asked_question":
            instruction = f"Answer {speaker_name}'s question. Keep it conversational and brief."
        elif trigger_reason == "memory_trigger":
            instruction = f"Share what you know about this topic. Be conversational."
        elif trigger_reason == "emotional_response":
            instruction = f"Respond with empathy to {speaker_name}'s emotional state. Be supportive."
        elif trigger_reason == "expertise_trigger":
            instruction = f"Share your professional insight about what {speaker_name} said. Stay conversational."
        elif trigger_reason == "relationship_response":
            instruction = f"Respond to {speaker_name} based on your relationship with them. Be natural."
        else:
            instruction = f"Continue the conversation naturally with {speaker_name}. One brief sentence."
        
        # Enhanced prompt with memory context
        auto_prompt = f"""CRITICAL: You are {agent_name}. Do NOT impersonate anyone else.

{speaker_name} just said: "{original_message}"

{memory_context}{instruction}

RULES:
- You are {agent_name}, not {speaker_name} or anyone else
- Do not start your response with anyone's name
- Give exactly ONE sentence as {agent_name}
- Be natural and conversational

{agent_name}:"""
        
        # Import here to avoid circular import
        from main_npc_system import ollama_manager
        
        response, success = ollama_manager.make_request(
            agent_name,
            auto_prompt,
            "phi3:mini",
            0.3,
            35
        )
        
        # Record this conversation in memory
        if success and agent_memory:
            # Record that agent overheard this conversation
            self.memory_manager.npc_tells_npc(
                speaker_name, agent_name, 
                f"conversation about: {original_message[:50]}...", 
                current_location,
                ConfidenceLevel.CONFIDENT,
                ["conversation"]
            )
        
        return response, success, 0.5
    
    def _check_direct_mention(self, message: str, agent_name: str) -> bool:
        """Check if agent is directly mentioned or addressed"""
        message_lower = message.lower()
        agent_lower = agent_name.lower()
        
        if agent_lower in message_lower:
            return True
            
        for pattern in self.question_patterns:
            matches = re.findall(pattern, message_lower)
            for match in matches:
                if isinstance(match, tuple):
                    if any(agent_lower == name.lower() for name in match if name):
                        return True
                elif agent_lower == match.lower():
                    return True
        
        return False
    
    def _check_question_pattern(self, message: str, agent_name: str) -> bool:
        """Check if message contains question directed at agent"""
        message_lower = message.lower()
        
        general_questions = [
            "what do you all think", "how does everyone feel",
            "does anyone", "can someone", "who thinks"
        ]
        
        return any(q in message_lower for q in general_questions)
    
    def _should_respond_emotionally(self, message: str, agent: Any, speaker_name: str) -> bool:
        """Check if empathetic agent should respond to emotional content"""
        message_lower = message.lower()
        
        agent_traits = agent.npc_data.get('personality', {}).get('traits', [])
        is_empathetic = any(trait in ['empathetic', 'gentle', 'caring', 'supportive'] 
                           for trait in agent_traits)
        
        if not is_empathetic:
            return False
        
        for emotion_type, keywords in self.emotional_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return True
        
        return False
    
    def _check_expertise_trigger(self, message: str, agent: Any) -> bool:
        """Check if message relates to agent's expertise"""
        message_lower = message.lower()
        expertise_areas = agent.expertise_areas
        
        for area in expertise_areas:
            area_keywords = area.replace('_', ' ').split()
            if any(keyword in message_lower for keyword in area_keywords):
                return True
        
        return False
    
    def _check_relationship_trigger(self, message: str, agent: Any, speaker_name: str) -> bool:
        """Check if agent should respond based on relationship with speaker"""
        relationships = agent.npc_data.get('relationships', {})
        
        speaker_relationship = relationships.get(speaker_name.lower())
        if not speaker_relationship:
            return False
        
        close_relationships = [
            'mentor', 'friend', 'collaboration', 'supportive',
            'looks_up_to', 'protective', 'professional_respect'
        ]
        
        return any(rel in speaker_relationship for rel in close_relationships) 