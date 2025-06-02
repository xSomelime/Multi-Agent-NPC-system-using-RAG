#!/usr/bin/env python3
"""
Complete Multi-Agent NPC System with Session Memory Integration
Features realistic memory tracking, spatial awareness, and information propagation
Enhanced with RAG (Retrieval-Augmented Generation) for domain-specific knowledge
"""

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

# Import the session memory system
from memory.session_memory import MemoryManager, InformationSource, ConfidenceLevel

# Import RAG system components
try:
    from src.agents.rag_enhanced_agent import create_rag_enhanced_agent, create_rag_enhanced_team
    RAG_AVAILABLE = True
    print("🔥 RAG system loaded successfully!")
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"⚠️  RAG system not available: {e}")
    print("💡 Install RAG dependencies or run without RAG")

class NPCRole(Enum):
    STABLE_HAND = "stable_hand"
    TRAINER = "trainer"
    BEHAVIOURIST = "behaviourist"
    COMPETITIVE_RIDER = "competitive_rider"
    RIVAL = "rival"

@dataclass
class Message:
    """Represents a conversation message"""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    agent_name: Optional[str] = None
    location: Optional[str] = None  # NEW: Track where message was said
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()

class OllamaRequestManager:
    """Manages sequential requests to Ollama to prevent response mixing"""
    
    def __init__(self):
        self.lock = threading.Lock()
        
    def make_request(self, agent_name: str, prompt: str, model: str, temperature: float, max_tokens: int):
        """Thread-safe request to Ollama"""
        with self.lock:
            try:
                print(f"🔄 {agent_name} making request to Ollama...")
                
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                            "stop": [
                                "\nPlayer:", "Player:", 
                                f"\n{agent_name}:", f"{agent_name}:",
                                "\nHuman:", "Human:",
                                "\n\n", "\\n\\n",
                                "Dr. Evelyn", "embodying",
                                "\n---", "---",
                                "\n## Instruction", "## Instruction"
                            ]
                        }
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    agent_response = result.get('response', '').strip()
                    
                    # Clean up response
                    agent_response = self._clean_response(agent_response, agent_name)
                    
                    print(f"✅ {agent_name} got clean response: {agent_response[:50]}...")
                    return agent_response, True
                else:
                    print(f"❌ {agent_name} request failed: {response.status_code}")
                    return f"I'm having trouble responding right now.", False
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ {agent_name} request error: {e}")
                return "Sorry, I can't respond right now.", False
    
    def _clean_response(self, response: str, agent_name: str) -> str:
        """Clean and validate response"""
        if not response or len(response) < 3:
            return "I see."
        
        # Remove common artifacts
        artifacts_to_remove = [
            "Dr. Evelyn", "embodying", "## Instruction",
            "<|user|>", "<|assistant|>", "Human:", "Player:",
            "\n\n", "\\n\\n", "\n---", "---"
        ]
        
        cleaned = response
        for artifact in artifacts_to_remove:
            cleaned = cleaned.replace(artifact, "")
        
        # Split into sentences and remove duplicates
        sentences = []
        for sent in cleaned.replace('!', '.').replace('?', '.').split('.'):
            sent = sent.strip()
            if sent and len(sent) > 10:
                sentences.append(sent)
        
        # Remove duplicate sentences
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if sentence_lower not in seen and len(sentence) > 10:
                unique_sentences.append(sentence)
                seen.add(sentence_lower)
                if len(unique_sentences) >= 2:  # Max 2 sentences
                    break
        
        if unique_sentences:
            result = '. '.join(unique_sentences)
            if not result.endswith('.'):
                result += '.'
            return result
        
        return "I see."

# Global request manager
ollama_manager = OllamaRequestManager()

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
                                current_location: str, all_agents: Dict[str, any]) -> List[Tuple[str, str, float]]:
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
            
            # 3. NEW: Memory-triggered response (knows about topic)
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
        """NEW: Check if agent has relevant memories about the topic"""
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
        
        # Use the existing request manager
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
    
    # Keep existing helper methods but enhance them
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
    
    def _should_respond_emotionally(self, message: str, agent: any, speaker_name: str) -> bool:
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
    
    def _check_expertise_trigger(self, message: str, agent: any) -> bool:
        """Check if message relates to agent's expertise"""
        message_lower = message.lower()
        expertise_areas = agent.expertise_areas
        
        for area in expertise_areas:
            area_keywords = area.replace('_', ' ').split()
            if any(keyword in message_lower for keyword in area_keywords):
                return True
        
        return False
    
    def _check_relationship_trigger(self, message: str, agent: any, speaker_name: str) -> bool:
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

class RoleTemplateLoader:
    """Loads and manages role template data"""
    
    def __init__(self, template_dir="data/role_templates"):
        self.template_dir = template_dir
        self._templates = {}
    
    def load_role_template(self, role_name: str) -> Dict:
        """Load base knowledge for a role"""
        if role_name in self._templates:
            return self._templates[role_name]
        
        template_file = os.path.join(self.template_dir, f"{role_name}_template.json")
        
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
                self._templates[role_name] = template_data
                return template_data
        except FileNotFoundError:
            print(f"⚠️  Template file not found: {template_file}")
            return self._get_default_template(role_name)
        except json.JSONDecodeError:
            print(f"⚠️  Invalid JSON in template file: {template_file}")
            return self._get_default_template(role_name)
    
    def _get_default_template(self, role_name: str) -> Dict:
        """Fallback template if file not found"""
        return {
            "role": role_name,
            "title": role_name.replace("_", " ").title(),
            "expertise_areas": ["general_horse_knowledge"],
            "common_responsibilities": ["Daily work with horses"]
        }

class NPCLoader:
    """Loads individual NPC configurations"""
    
    def __init__(self, npc_dir="data/npcs"):
        self.npc_dir = npc_dir
        self._npcs = {}
    
    def load_npc_config(self, npc_config_name: str) -> Dict:
        """Load specific NPC configuration"""
        if npc_config_name in self._npcs:
            return self._npcs[npc_config_name]
        
        npc_file = os.path.join(self.npc_dir, f"{npc_config_name}.json")
        
        try:
            with open(npc_file, 'r', encoding='utf-8') as f:
                npc_data = json.load(f)
                self._npcs[npc_config_name] = npc_data
                return npc_data
        except FileNotFoundError:
            print(f"⚠️  NPC file not found: {npc_file}")
            return self._get_default_npc(npc_config_name)
        except json.JSONDecodeError:
            print(f"⚠️  Invalid JSON in NPC file: {npc_file}")
            return self._get_default_npc(npc_config_name)
    
    def _get_default_npc(self, config_name: str) -> Dict:
        """Fallback NPC if file not found"""
        name = config_name.split('_')[0].title()
        role = config_name.split('_')[1] if '_' in config_name else "stable_hand"
        
        return {
            "name": name,
            "role_template": role,
            "personality": {
                "traits": ["helpful", "friendly"],
                "speaking_style": "casual and supportive"
            },
            "personal_background": f"{name} is a dedicated horse care professional",
            "professional_opinions": {},
            "controversial_stances": []
        }

class ScalableNPCAgent:
    """Enhanced NPC Agent with memory integration"""
    
    def __init__(self, npc_config_name: str, memory_manager: MemoryManager, current_location: str = "stable_yard"):
        self.memory_manager = memory_manager
        self.current_location = current_location
        
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
        
        # Register with memory manager
        self.memory_manager.register_npc(self.name, current_location)
        
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
        
        print(f"✅ Created {self.name} ({self.template_data.get('title', 'NPC')}) at {current_location}")
    
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
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
    
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

class NPCFactory:
    """Factory for creating NPCs with memory integration and optional RAG enhancement"""
    
    @staticmethod
    def create_npc(npc_config_name: str, memory_manager: MemoryManager, location: str = "stable_yard", enable_rag: bool = True) -> ScalableNPCAgent:
        """Create NPC from config file name with memory integration and optional RAG"""
        if enable_rag and RAG_AVAILABLE:
            try:
                # Create RAG-enhanced agent
                return create_rag_enhanced_agent(npc_config_name, memory_manager, location, enable_rag=True)
            except Exception as e:
                print(f"⚠️  Failed to create RAG-enhanced {npc_config_name}: {e}")
                print("🔄 Falling back to regular agent")
                # Fall back to regular agent
                return ScalableNPCAgent(npc_config_name, memory_manager, location)
        else:
            # Create regular agent
            return ScalableNPCAgent(npc_config_name, memory_manager, location)
    
    @staticmethod
    def create_core_team(memory_manager: MemoryManager, enable_rag: bool = True):
        """Create the core stable team with realistic starting locations and optional RAG"""
        if enable_rag and RAG_AVAILABLE:
            try:
                # Create RAG-enhanced team
                return create_rag_enhanced_team(memory_manager, enable_rag=True)
            except Exception as e:
                print(f"⚠️  Failed to create RAG-enhanced team: {e}")
                print("🔄 Falling back to regular agents")
        
        # Create regular team
        return [
            NPCFactory.create_npc("elin_behaviourist", memory_manager, "barn", enable_rag=False),
            NPCFactory.create_npc("oskar_stable_hand", memory_manager, "stable_yard", enable_rag=False), 
            NPCFactory.create_npc("astrid_stable_hand", memory_manager, "barn", enable_rag=False),
            NPCFactory.create_npc("chris_rival", memory_manager, "arena", enable_rag=False),
            NPCFactory.create_npc("andy_trainer", memory_manager, "arena", enable_rag=False)
        ]

class EnhancedConversationManager:
    """Enhanced ConversationManager with memory integration and RAG support"""
    
    def __init__(self, enable_rag: bool = True):
        self.memory_manager = MemoryManager()
        self.agents: Dict[str, ScalableNPCAgent] = {}
        self.conversation_log: List[Message] = []
        self.momentum = ConversationalMomentum(self.memory_manager)
        self.current_location = "stable_yard"  # Track player's current location
        self.rag_enabled = enable_rag and RAG_AVAILABLE
        
        if self.rag_enabled:
            print("🔥 RAG system enabled for enhanced horse knowledge!")
        else:
            print("📚 Running with standard NPCs (no RAG)")
    
    def toggle_rag(self) -> str:
        """Toggle RAG system on/off (for comparison)"""
        if not RAG_AVAILABLE:
            return "❌ RAG system not available - check dependencies"
        
        self.rag_enabled = not self.rag_enabled
        status = "enabled" if self.rag_enabled else "disabled"
        return f"🔄 RAG system {status}. Restart to apply changes."
    
    def get_rag_status(self) -> str:
        """Get current RAG status"""
        if not RAG_AVAILABLE:
            return "❌ Not Available"
        elif self.rag_enabled:
            return "✅ Enabled"
        else:
            return "🔄 Disabled"
    
    def register_agent(self, agent: ScalableNPCAgent):
        """Register an agent"""
        self.agents[agent.name] = agent
        role_desc = agent.template_data.get('title', agent.npc_role.value)
        print(f"🎭 Registered {agent.name} ({role_desc}) at {agent.current_location}")
    
    def move_player_to_location(self, location: str):
        """Move player to a new location"""
        old_location = self.current_location
        self.current_location = location
        
        # Record event that NPCs in both locations can witness
        if old_location != location:
            # NPCs in old location see player leave
            old_location_npcs = [name for name, agent in self.agents.items() 
                               if agent.current_location == old_location]
            if old_location_npcs:
                self.memory_manager.record_witnessed_event(
                    f"Player left {old_location}",
                    old_location,
                    old_location_npcs,
                    ["player_movement"]
                )
            
            # NPCs in new location see player arrive
            new_location_npcs = [name for name, agent in self.agents.items() 
                               if agent.current_location == location]
            if new_location_npcs:
                self.memory_manager.record_witnessed_event(
                    f"Player arrived at {location}",
                    location,
                    new_location_npcs,
                    ["player_movement"]
                )
        
        print(f"🚶 You moved to {location}")
        
        # Show who's here
        npcs_here = [agent.name for agent in self.agents.values() 
                    if agent.current_location == location]
        if npcs_here:
            print(f"👥 NPCs here: {', '.join(npcs_here)}")
    
    def move_npc_to_location(self, npc_name: str, location: str):
        """Move an NPC to a new location"""
        if npc_name in self.agents:
            self.agents[npc_name].move_to_location(location)
            return True
        return False
    
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
            nearby_npcs = [name for name, other_agent in self.agents.items() 
                          if other_agent.current_location == self.current_location and name != agent_name]
            
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
                    role_desc = other_agent.template_data.get('title', other_agent.npc_role.value)
                    
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
        
        conversation_type = list(available_agents.values())[0].detect_conversation_type(message) if available_agents else "conversation"
        
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
        
        # Determine participants from available NPCs
        participating_agents = []
        for agent_name, agent in available_agents.items():
            if agent.should_participate(message, []):
                participating_agents.append(agent)
        
        # Direct questions override participation logic
        for name, agent in available_agents.items():
            if name.lower() in message.lower():
                participating_agents = [agent]
                break
        
        if not participating_agents and available_agents:
            participating_agents = [list(available_agents.values())[0]]
        
        # Randomize order
        random.shuffle(participating_agents)
        
        print(f"💭 {len(participating_agents)} NPCs will respond at {self.current_location} ({conversation_type} mode)")
        
        # Get responses sequentially
        for agent in participating_agents:
            role_desc = agent.template_data.get('title', agent.npc_role.value)
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
        
        # Check for auto-responses
        if all_responses:
            last_speaker = all_responses[-1][0]
            last_message = all_responses[-1][1]
            
            auto_triggers = self.momentum.should_trigger_auto_response(
                last_message, last_speaker, self.current_location, available_agents
            )
            
            if auto_triggers:
                print(f"\n🔄 Auto-responses triggered:")
                
                for agent_name, reason, probability in auto_triggers:
                    agent = available_agents[agent_name]
                    role_desc = agent.template_data.get('title', agent.npc_role.value)
                    
                    print(f"⚡ {agent_name} ({role_desc}) responding ({reason}, {probability:.1%})")
                    
                    auto_response, success, response_time = self.momentum.execute_auto_response(
                        agent_name, reason, last_message, last_speaker, 
                        self.current_location, self.conversation_log[-5:]
                    )
                    
                    if success:
                        all_responses.append((agent_name, auto_response, success, response_time))
                        self._add_to_global_log("assistant", auto_response, agent_name)
                        agent.add_message("assistant", auto_response)
                        
                        print(f"  {agent_name}: {auto_response}")
                        time.sleep(0.3)
                    
                    break
        
        # Show results
        print(f"\n💬 Responses at {self.current_location}:")
        for agent_name, response, success, response_time in all_responses:
            if success:
                agent = self.agents[agent_name]
                role_desc = agent.template_data.get('title', agent.npc_role.value)
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
            "rag_available": RAG_AVAILABLE,
            "memory_system": memory_stats,
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

def create_enhanced_npc_system(enable_rag: bool = True):
    """Initialize the enhanced NPC system with memory integration and optional RAG"""
    print("🎭 Initializing Enhanced Multi-Agent NPC System")
    print("="*70)
    
    # Create enhanced conversation manager (includes memory manager)
    manager = EnhancedConversationManager(enable_rag=enable_rag)
    
    # Load NPCs from configurations with different starting locations
    try:
        npcs = NPCFactory.create_core_team(manager.memory_manager, enable_rag=enable_rag)
        
        for npc in npcs:
            manager.register_agent(npc)
        
        # Record initial setup event
        all_npc_names = [npc.name for npc in npcs]
        manager.memory_manager.record_witnessed_event(
            "New stable session started",
            "stable_yard",
            all_npc_names,
            ["session_start", "setup"]
        )
        
        # Show what was loaded
        if enable_rag and RAG_AVAILABLE:
            print(f"✅ Created {len(npcs)} RAG-enhanced NPCs with domain expertise")
        else:
            print(f"✅ Created {len(npcs)} standard NPCs with session memory")
        
    except Exception as e:
        print(f"⚠️  Error loading NPCs: {e}")
        print("💡 Make sure all NPC configuration files exist")
        return None
    
    return manager

def show_npc_info():
    """Display information about available NPCs and new features"""
    print("\n🎭 Enhanced NPC System Features:")
    print("  📚 Session Memory: NPCs remember conversations and events")
    print("  📍 Spatial Awareness: NPCs only respond if in same location") 
    print("  🧠 Information Propagation: NPCs share knowledge with each other")
    print("  ⏰ Memory-triggered Responses: NPCs recall relevant information")
    print("  📊 Confidence Levels: Different reliability for different sources")
    
    if RAG_AVAILABLE:
        print("  🔥 RAG System: Domain-specific horse care knowledge")
        print("  🎯 Anti-Hallucination: Accurate responses or 'I don't know'")
        print("  🔧 Expert Knowledge: Each NPC has specialized expertise")
    
    print("\n📍 Location Commands:")
    print("  - 'go <location>' to move between areas")
    print("  - 'move <npc> <location>' to move an NPC")
    
    print("\n🧠 Memory Commands:")
    print("  - 'memory <npc>' to see what an NPC remembers")
    print("  - 'memory' to see system memory summary")
    
    if RAG_AVAILABLE:
        print("\n🔥 RAG Commands:")
        print("  - 'rag status' to check RAG system status")
        print("  - 'rag toggle' to enable/disable RAG (restart required)")
        print("  - Ask horse care questions to test knowledge!")
    
    print("\n💡 Try asking NPCs about:")
    print("  - Horse feeding schedules and nutrition")
    print("  - Grooming techniques and equipment")
    print("  - Training methods and competition prep")
    print("  - Horse behavior and health signs")
    print("  Type 'info' anytime to see this again\n")

if __name__ == "__main__":
    print("Enhanced Multi-Agent NPC System with Session Memory Integration")
    if RAG_AVAILABLE:
        print("🔥 RAG-Enhanced NPCs with Domain-Specific Horse Knowledge!")
    else:
        print("📚 Standard NPCs with Session Memory")
    print("\nMake sure Ollama is running: ollama serve")
    print("="*70)
    
    # Initialize system
    manager = create_enhanced_npc_system()
    
    if not manager:
        print("❌ Failed to initialize system. Check your configuration files.")
        exit(1)
    
    # Show system info
    show_npc_info()
    
    print(f"🎭 System ready! Available agents: {', '.join(manager.list_agents())}")
    print(f"📍 Current location: {manager.current_location}")
    if RAG_AVAILABLE:
        print(f"🔥 RAG Status: {manager.get_rag_status()}")
    
    print("\nCommands:")
    print("  - Type message for group discussion")
    print("  - 'AgentName: message' for direct conversation")
    print("  - 'go <location>' to move (barn, arena, paddock, tack_room, office)")
    print("  - 'move <npc> <location>' to move an NPC")
    print("  - 'memory [npc]' to check memories")
    if RAG_AVAILABLE:
        print("  - 'rag status/toggle' for RAG commands")
    print("  - 'info' for system information")
    print("  - 'stats' for statistics")
    print("  - 'reset' to clear conversations")
    print("  - 'quit' to exit")
    
    print(f"\n💬 Test Examples:")
    if manager.rag_enabled:
        print(f"  RAG Test: 'Oskar, what's the best feeding schedule for horses?'")
        print(f"  Expert Knowledge: 'Elin, how do I tell if a horse is stressed?'")
        print(f"  Competition Prep: 'Andy, what's the best way to train for jumping?'")
    else:
        print(f"  Memory test: 'Astrid, Thunder seemed nervous yesterday'")
        print(f"  Location: 'go barn' then ask about what happened")
    print(f"  Recall: 'What do you remember about Thunder?'")
    
    while True:
        try:
            user_input = input(f"\nPlayer ({manager.current_location}): ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'info':
                show_npc_info()
                continue
            elif user_input.lower().startswith('rag '):
                if not RAG_AVAILABLE:
                    print("❌ RAG system not available - check dependencies")
                    continue
                    
                rag_command = user_input[4:].strip().lower()
                if rag_command == 'status':
                    print(f"🔥 RAG Status: {manager.get_rag_status()}")
                    if manager.rag_enabled:
                        print("💡 NPCs can answer horse care questions with expert knowledge")
                    else:
                        print("💡 NPCs use only conversational AI (restart with RAG for expert knowledge)")
                elif rag_command == 'toggle':
                    result = manager.toggle_rag()
                    print(result)
                else:
                    print("Available RAG commands: 'rag status', 'rag toggle'")
                continue
            elif user_input.lower().startswith('go '):
                location = user_input[3:].strip()
                valid_locations = ["stable_yard", "barn", "arena", "paddock", "tack_room", "office"]
                if location in valid_locations:
                    manager.move_player_to_location(location)
                else:
                    print(f"❌ Unknown location. Available: {', '.join(valid_locations)}")
                continue
            elif user_input.lower().startswith('move '):
                parts = user_input[5:].split()
                if len(parts) >= 2:
                    npc_name = parts[0]
                    location = parts[1]
                    if manager.move_npc_to_location(npc_name, location):
                        print(f"✅ Moved {npc_name} to {location}")
                    else:
                        print(f"❌ Could not move {npc_name}")
                else:
                    print("Usage: move <npc> <location>")
                continue
            elif user_input.lower().startswith('memory'):
                parts = user_input.split()
                if len(parts) > 1:
                    manager.show_memory_summary(parts[1])
                else:
                    manager.show_memory_summary()
                continue
            elif user_input.lower() == 'stats':
                stats = manager.get_stats()
                print(f"📊 Total messages: {stats['total_messages']}")
                print(f"📍 Current location: {stats['current_location']}")
                print(f"🔄 Momentum chains: {stats['momentum_chains']}")
                print(f"🧠 Global memories: {stats['memory_system']['global_events']}")
                if RAG_AVAILABLE:
                    print(f"🔥 RAG Status: {stats['rag_status']}")
                
                for agent_name, agent_stats in stats['agents'].items():
                    location = agent_stats['current_location']
                    memory_count = agent_stats['memory_stats'].get('total_memories', 0)
                    rag_info = ""
                    if 'rag_stats' in stats and agent_name in stats['rag_stats']:
                        rag_info = " (RAG-enhanced)"
                    print(f"  {agent_name} ({agent_stats['title']}) at {location}: {memory_count} memories{rag_info}")
                continue
            elif user_input.lower() == 'reset':
                manager.reset_all()
                continue
            
            # Check for direct agent messaging
            if ':' in user_input and user_input.count(':') == 1:
                agent_name, message = user_input.split(':', 1)
                agent_name = agent_name.strip()
                message = message.strip()
                
                if agent_name == "All":
                    responses = manager.send_to_all(message)
                elif agent_name in manager.agents:
                    response, success, response_time = manager.send_to_agent(agent_name, message)
                    if success:
                        agent = manager.agents[agent_name]
                        role_desc = agent.template_data.get('title', agent.npc_role.value)
                        print(f"{agent_name} ({role_desc}): {response}")
                    else:
                        print(f"❌ {agent_name}: {response}")
                else:
                    print(f"❌ Agent '{agent_name}' not found. Available: {', '.join(manager.agents.keys())}")
                continue
            
            # Default to group conversation
            responses = manager.send_to_all(user_input)
            
        except KeyboardInterrupt:
            break
    
    print("\n👋 Thanks for testing the enhanced multi-agent system with session memory!")