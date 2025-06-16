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
            self.max_response_length = 150  # Higher limit since we'll control by sentences
            self.max_sentences = 1  # Chris gives one sentence (but might be longer/more arrogant)
        else:
            self.temperature = 0.3
            self.max_response_length = 120  # Higher limit since we'll control by sentences  
            self.max_sentences = 1  # Others give one sentence
        
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
        
        # Movement logged silently
    
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
- You are wealthy, privileged, and look down on "working people"
- Be dismissive and condescending to stable hands and anyone beneath you
- Only speak up to brag about achievements or insult others
- Keep responses SHORT unless showing off or putting someone down
- You don't understand what working for money means
- Only respect people who actually intimidate or impress you
- Be snobby about class differences, not just equipment
- NEVER admit fault - blame horses, trainers, or equipment when you fail
- You'd rather buy a new expensive horse than train a good one properly
- Everything bad is someone else's fault, everything good is your skill
"""
        
        return persona
    
    def generate_response(self, user_input: str, conversation_context: List[Message] = None, others_responses: List = None) -> Tuple[str, bool, float]:
        """Generate response using memory context and thread-safe Ollama requests"""
        start_time = time.time()
        
        # Check if player is asking about what someone said
        is_asking_about_response = any(phrase in user_input.lower() for phrase in ["what did", "what said", "what responded", "what did you hear"])
        
        # Get relevant memories for context
        # If asking about another NPC, search for that NPC's name specifically
        search_term = user_input
        if is_asking_about_response:
            # Extract NPC names from the question
            npc_names = ["chris", "andy", "oskar", "astrid", "elin"]
            for npc_name in npc_names:
                if npc_name in user_input.lower():
                    search_term = npc_name
                    break
        
        # Get more memories for recall questions
        max_memories = 5 if is_asking_about_response else 2
        memory_context = self.memory_manager.get_relevant_context_for_npc(
            self.name, search_term, max_memories=max_memories
        )
        
        # Build prompt
        conversation_type = self.detect_conversation_type(user_input)
        
        # Check if this is a simple greeting or reaction to a greeting
        is_greeting = any(greeting in user_input.lower() for greeting in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"])
        is_greeting_reaction = ("just said:" in user_input or "respond naturally to this statement" in user_input.lower()) and any(greeting in user_input.lower() for greeting in ["good morning", "hello", "hi", "hey", "greetings"])
        
        # Check if this is a memory/recognition question
        is_memory_question = any(phrase in user_input.lower() for phrase in ["do you remember me", "remember me", "do you know me", "have we met", "do i know"])
        
        # Check if this is a simple yes/no question
        is_simple_question = any(phrase in user_input.lower() for phrase in ["do you like", "are you", "can you", "will you", "have you", "did you"])
        
        prompt_parts = [
            f"STAY IN CHARACTER: You are {self.name}, not Dr. Evelyn or anyone else.",
            f"CRITICAL: Focus on what the player is actually asking/saying. Answer THEIR specific question or respond to THEIR specific comment. Don't give generic advice or change the subject.",
            self.persona
        ]
        
        # Add specific instruction for greetings and memory questions
        if (is_greeting or is_greeting_reaction) and not is_memory_question:
            prompt_parts.append("This is a simple greeting - respond naturally and briefly, don't give professional advice.")
        elif is_memory_question:
            prompt_parts.append("CRITICAL: The player is asking if you know/remember them. ONLY answer this specific question. Don't talk about work, training, or anything else. Just say if you remember them or not. Be direct and brief.")
        elif is_simple_question:
            prompt_parts.append("This is a simple question - give a SHORT, enthusiastic answer that shows your personality. Don't use your usual phrases like 'Here's what works' or 'In my experience'. Just answer yes/no with emotion and excitement. Don't give professional advice.")
        
        # Add memory context if available
        if memory_context:
            prompt_parts.append(f"What you remember: {'; '.join(memory_context[:2])}")
        
        # Add conversation context (much more for "what did X say" questions)
        if conversation_context:
            prompt_parts.append("Recent context:")
            context_limit = 15 if is_asking_about_response else 2  # Much larger window for recall questions
            for msg in conversation_context[-context_limit:]:
                if msg.role == "user":
                    prompt_parts.append(f"Player: {msg.content}")
                elif msg.role == "assistant" and msg.agent_name != self.name:
                    prompt_parts.append(f"{msg.agent_name}: {msg.content}")
        
        # Conversation type specific instructions
        if is_asking_about_response:
            # Special handling for "what did X say" questions
            instruction = f"Look at the recent context above and tell me exactly what the person said. Quote their actual words if possible. Don't make anything up - only say what you can see in the conversation history."
        elif is_memory_question:
            # Special handling for memory questions - keep it simple and direct
            if self.npc_role == NPCRole.RIVAL:
                instruction = f"Give exactly ONE sentence as {self.name} - just say if you know them or not. Be dismissive if you don't. Don't talk about anything else."
            else:
                instruction = f"Give exactly ONE sentence as {self.name} - just say if you know them or not. Be polite. Don't talk about work or training."
        elif is_simple_question and not is_memory_question:
            # Get character-specific instructions from JSON config
            conversation_instructions = self.npc_data.get('personality', {}).get('conversation_instructions', {})
            instruction = conversation_instructions.get('simple_questions', 
                f"Give exactly ONE enthusiastic sentence as {self.name} - show your personality and passion!")
        elif (is_greeting or is_greeting_reaction) and not is_memory_question and not is_simple_question:
            # Get character-specific greeting instructions from JSON config
            conversation_instructions = self.npc_data.get('personality', {}).get('conversation_instructions', {})
            instruction = conversation_instructions.get('greetings', 
                f"Give exactly ONE warm, enthusiastic sentence as {self.name} - show your personality and be genuinely excited to interact!")
        elif conversation_type == "debate":
            opinions = self.get_relevant_opinions(user_input)
            if opinions != "No specific opinions on this topic":
                prompt_parts.append(f"Your opinion: {opinions[:100]}")
            
            if others_responses and len(others_responses) > 0:
                last_response = others_responses[-1]
                prompt_parts.append(f"{last_response[0]} just said: {last_response[1]}")
            
            # Get character-specific debate instructions from JSON config
            conversation_instructions = self.npc_data.get('personality', {}).get('conversation_instructions', {})
            instruction = conversation_instructions.get('debates', 
                f"Give exactly ONE passionate sentence as {self.name} - show your expertise with enthusiasm and personality!")
        else:
            # Get character-specific general instructions from JSON config
            conversation_instructions = self.npc_data.get('personality', {}).get('conversation_instructions', {})
            instruction = conversation_instructions.get('general', 
                f"Give exactly ONE enthusiastic sentence as {self.name} that DIRECTLY answers the player's question with personality and passion!")
        
        prompt_parts.append(instruction)
        
        # Add natural event awareness instruction
        # Check if any recent memories mention today's events
        recent_memories = memory_context[:1] if memory_context else []
        event_keywords = ["today", "arriving", "delivery", "competition", "new horse", "planning"]
        if any(any(keyword in memory.lower() for keyword in event_keywords) for memory in recent_memories):
            prompt_parts.append("DAILY EVENTS: You know what's happening today. Answer naturally and briefly - be excited or interested, not explanatory.")
        
        # Add character-specific personality note if available
        conversation_instructions = self.npc_data.get('personality', {}).get('conversation_instructions', {})
        personality_note = conversation_instructions.get('personality_note')
        if personality_note:
            prompt_parts.append(personality_note)
        
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
            # Clean up response to fix weird punctuation from token limits
            cleaned_response = self._clean_response(agent_response)
            
            # Record conversation in both local and memory manager
            self.add_message("user", user_input)
            self.add_message("assistant", cleaned_response)
            
            # Record player telling this NPC something
            self.memory_manager.player_tells_npc(
                self.name, user_input, self.current_location, 
                self._extract_tags_from_message(user_input)
            )
            
            return cleaned_response, True, response_time
        else:
            return agent_response, False, response_time
    
    def _clean_response(self, response: str) -> str:
        """Clean up response and enforce sentence limits"""
        if not response:
            return response
        
        import re
        
        # First, fix duplicate punctuation from token limits
        response = re.sub(r'[!.]{2,}', '.', response)  # "!." or "!!" -> "."
        response = re.sub(r'[?.]{2,}', '?', response)  # "?." -> "?"
        response = re.sub(r'[,!.]{2,}', '.', response)  # ",." or ",!" -> "."
        response = re.sub(r'!\.$', '.', response)  # "!." at end -> "."
        response = re.sub(r'\?[.,]$', '?', response)  # "?." or "?," at end -> "?"
        response = re.sub(r',[.!]$', '.', response)  # ",." or ",!" at end -> "."
        
        response = response.strip()
        
        # Enforce sentence limits
        max_sentences = getattr(self, 'max_sentences', 1)
        sentences = self._split_into_sentences(response)
        
        if len(sentences) > max_sentences:
            # Keep only the first N sentences
            response = ' '.join(sentences[:max_sentences])
        
        # Ensure response ends with proper punctuation
        if response and not response[-1] in '.!?':
            response += '.'
        
        return response
    
    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences, handling common abbreviations"""
        import re
        
        # Simple sentence splitting that handles most cases
        # Split on . ! ? but not on abbreviations like "Dr." or "Mr."
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean up each sentence
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

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
        
        # Always respond if directly addressed by name (by player or other NPCs)
        if self.name.lower() in message_lower:
            return True
        
        # Handle memory questions - everyone should respond since it's directed at all of them
        memory_questions = ["do you remember me", "remember me", "do you know me", "have we met", "do i know"]
        if any(phrase in message_lower for phrase in memory_questions):
            return existing_count < 3  # Allow more NPCs to respond to memory questions
        
        # Handle "what did X say" questions - only 1-2 NPCs should respond (those who heard it)
        recall_questions = ["what did", "what said", "what responded", "what did you hear"]
        if any(phrase in message_lower for phrase in recall_questions):
            return existing_count < 2  # Limit to first 2 NPCs who want to answer
        
        # Handle simple greetings and social interactions - only 1-2 NPCs should respond
        social_greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "how are you"]
        if any(greeting in message_lower for greeting in social_greetings):
            # For greetings, be more selective - only respond if you're among the first 2
            # Chris might respond to greetings but dismissively
            if self.npc_role == NPCRole.RIVAL:
                return existing_count == 0  # Chris only responds if he's first (to establish dominance)
            else:
                return existing_count < 2  # Others respond if there's room
        
        # Special logic for Chris (rival) - he's more selective and snobby
        if self.npc_role == NPCRole.RIVAL:
            # Chris will respond to general questions but only if he can show off or be dismissive
            general_indicators = ["all", "everyone", "how is", "how are", "what do you think", "your opinion", "how has", "how was"]
            if any(indicator in message_lower for indicator in general_indicators):
                return existing_count < 3  # Chris participates in general discussions to show off
            
            # Chris only bothers to respond if:
            # 1. He can show off about competitions/achievements
            competition_topics = ["competition", "win", "race", "show", "champion", "best", "better"]
            if any(topic in message_lower for topic in competition_topics):
                return True
            
            # 2. Someone mentions problems/failures (to blame others)
            blame_topics = ["problem", "fail", "difficult", "trouble", "wrong", "bad", "issue"]
            if any(topic in message_lower for topic in blame_topics):
                return True
            
            # 3. Someone said something he finds stupid (to insult them)
            if existing_responses and len(existing_responses) > 0:
                # Look for opportunities to be condescending
                last_response = existing_responses[-1][1].lower()
                if any(word in last_response for word in ["simple", "basic", "easy", "cheap", "budget", "train", "practice"]):
                    return True
            
            # 4. It's about his expertise areas (but only if he can dominate)
            expertise_match = False
            for area in self.expertise_areas:
                area_keywords = area.replace('_', ' ').split()
                if any(keyword in message_lower for keyword in area_keywords):
                    expertise_match = True
                    break
            
            if expertise_match and existing_count < 2:  # Only if he can be among the first to respond
                return True
            
            # Otherwise, Chris doesn't bother with "peasant talk"
            return False
        
        # Regular logic for other NPCs
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
        
        # For general questions, allow more participation
        general_indicators = ["all", "everyone", "how is", "how are", "what do you think", "your opinion", "how has", "how was"]
        if any(indicator in message_lower for indicator in general_indicators):
            return existing_count < 4  # Allow up to 4 NPCs to respond to general questions
        
        # Higher chance for random participation to create more dynamic conversations
        if existing_count == 0:
            return random.random() < 0.7  # 70% chance first NPC responds
        elif existing_count < 3:
            return random.random() < 0.3  # 30% chance additional NPCs join
        
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