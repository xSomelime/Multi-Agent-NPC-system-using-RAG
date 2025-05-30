#!/usr/bin/env python3
"""
Complete Multi-Agent NPC System with Conversational Momentum
Features thread-safe requests, response cleaning, and automatic NPC-to-NPC conversations
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
    
    def __init__(self):
        self.conversation_chains: List[str] = []  # Track recent speakers
        self.chain_count = 0  # Current chain length
        self.max_chain_length = 4  # Prevent infinite loops
        self.cooldown_period = 30  # Seconds before agent can auto-respond again
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
                                   all_agents: Dict[str, any]) -> List[Tuple[str, str, float]]:
        """Determine which NPCs should automatically respond and why"""
        
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
            if agent_name == speaker_name:  # Don't respond to yourself
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
            
            # 3. Emotional content response (for empathetic NPCs)
            elif self._should_respond_emotionally(message_content, agent, speaker_name):
                response_reason = "emotional_response"
                response_probability = 0.6
                
            # 4. Professional expertise trigger
            elif self._check_expertise_trigger(message_content, agent):
                response_reason = "expertise_trigger"
                response_probability = 0.4
            
            # 5. Relationship-based response (lower probability)
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
        return triggered_responses[:2]  # Max 2 auto-responses at once
    
    def _check_direct_mention(self, message: str, agent_name: str) -> bool:
        """Check if agent is directly mentioned or addressed"""
        message_lower = message.lower()
        agent_lower = agent_name.lower()
        
        # Direct name mention
        if agent_lower in message_lower:
            return True
            
        # Question patterns with name
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
        
        # General questions that might involve this agent
        general_questions = [
            "what do you all think", "how does everyone feel",
            "does anyone", "can someone", "who thinks"
        ]
        
        return any(q in message_lower for q in general_questions)
    
    def _should_respond_emotionally(self, message: str, agent: any, speaker_name: str) -> bool:
        """Check if empathetic agent should respond to emotional content"""
        message_lower = message.lower()
        
        # Only empathetic agents respond emotionally
        agent_traits = agent.npc_data.get('personality', {}).get('traits', [])
        is_empathetic = any(trait in ['empathetic', 'gentle', 'caring', 'supportive'] 
                           for trait in agent_traits)
        
        if not is_empathetic:
            return False
        
        # Check for emotional keywords
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
        
        # Agents with close relationships are more likely to respond
        close_relationships = [
            'mentor', 'friend', 'collaboration', 'supportive',
            'looks_up_to', 'protective', 'professional_respect'
        ]
        
        return any(rel in speaker_relationship for rel in close_relationships)
    
    def execute_auto_response(self, agent_name: str, trigger_reason: str, 
                            original_message: str, speaker_name: str,
                            conversation_context: List) -> Tuple[str, bool, float]:
        """Generate automatic response with context-aware prompting"""
        
        # Update tracking
        self.conversation_chains.append(agent_name)
        self.chain_count += 1
        self.last_auto_response[agent_name] = time.time()
        
        # Build context-aware prompt based on trigger reason
        prompt_prefix = f"CONTINUE CONVERSATION: Respond naturally to {speaker_name} as {agent_name}."
        
        if trigger_reason == "mentioned_directly":
            instruction = f"You were directly mentioned. Give a brief, natural response to {speaker_name}."
        elif trigger_reason == "asked_question":
            instruction = f"Answer the question directed at you. Keep it conversational and brief."
        elif trigger_reason == "emotional_response":
            instruction = f"Respond with empathy to {speaker_name}'s emotional state. Be supportive."
        elif trigger_reason == "expertise_trigger":
            instruction = f"Share your professional insight briefly. Stay conversational, not lecturing."
        elif trigger_reason == "relationship_response":
            instruction = f"Respond based on your relationship with {speaker_name}. Be natural and caring."
        else:
            instruction = f"Continue the conversation naturally. One brief sentence."
        
        # Create modified prompt for auto-response
        auto_prompt = f"""{prompt_prefix}
        
        Context: {speaker_name} just said: "{original_message}"
        
        {instruction}
        
        Respond as {agent_name} in exactly ONE sentence. Be natural and conversational.
        
        {agent_name}:"""
        
        # Use the existing request manager
        response, success = ollama_manager.make_request(
            agent_name,
            auto_prompt,
            "phi3:mini",  # Use fast model for auto-responses
            0.4,  # Slightly higher temperature for natural flow
            40    # Shorter responses for auto-replies
        )
        
        return response, success, 0.5  # Estimated time

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
    """Enhanced NPC Agent with conversation and debate capabilities"""
    
    def __init__(self, npc_config_name: str):
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
    
    def detect_conversation_type(self, message: str) -> str:
        """Detect if this should be a debate or casual conversation"""
        message_lower = message.lower()
        
        # Debate triggers
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
        
        # Match topics to opinions
        for opinion_topic, opinion in self.professional_opinions.items():
            if any(keyword in topic_lower for keyword in opinion_topic.split('_')):
                relevant_opinions.append(f"{opinion_topic}: {opinion}")
        
        # Add controversial stances if relevant
        for stance in self.controversial_stances:
            if any(word in stance.lower() for word in topic_lower.split() if len(word) > 3):
                relevant_opinions.append(stance)
        
        return "; ".join(relevant_opinions) if relevant_opinions else "No specific opinions on this topic"
    
    def should_participate(self, message_content: str, existing_responses: List = None) -> bool:
        """Determine if this NPC should respond to a message"""
        message_lower = message_content.lower()
        existing_count = len(existing_responses or [])
        
        # Always respond if directly addressed by name
        if self.name.lower() in message_lower:
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
    
    def generate_response(self, user_input: str, conversation_context: List[Message] = None, others_responses: List = None) -> Tuple[str, bool, float]:
        """Generate response using thread-safe Ollama requests"""
        start_time = time.time()
        
        # Build prompt
        conversation_type = self.detect_conversation_type(user_input)
        
        prompt_parts = [
            f"STAY IN CHARACTER: You are {self.name}, not Dr. Evelyn or anyone else.",
            self.persona
        ]
        
        # Add conversation context (very limited to prevent corruption)
        if conversation_context:
            prompt_parts.append("Recent context:")
            for msg in conversation_context[-2:]:  # Only last 2 messages
                if msg.role == "user":
                    prompt_parts.append(f"Player: {msg.content}")
                elif msg.role == "assistant" and msg.agent_name != self.name:
                    prompt_parts.append(f"{msg.agent_name}: {msg.content}")
        
        # Conversation type specific instructions
        if conversation_type == "debate":
            opinions = self.get_relevant_opinions(user_input)
            if opinions != "No specific opinions on this topic":
                prompt_parts.append(f"Your opinion: {opinions[:100]}")  # Truncated
            
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
            # Record conversation
            self.add_message("user", user_input)
            self.add_message("assistant", agent_response)
            return agent_response, True, response_time
        else:
            return agent_response, False, response_time
    
    def add_message(self, role: str, content: str) -> Message:
        """Add message to conversation history"""
        message = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=time.time(),
            agent_name=self.name if role == "assistant" else "player"
        )
        self.conversation_history.append(message)
        return message
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            "name": self.name,
            "role": self.npc_role.value,
            "title": self.template_data.get('title', 'NPC'),
            "total_messages": len(self.conversation_history),
            "expertise_areas": self.expertise_areas
        }

class NPCFactory:
    """Factory for creating NPCs from configuration files"""
    
    @staticmethod
    def create_npc(npc_config_name: str) -> ScalableNPCAgent:
        """Create NPC from config file name"""
        return ScalableNPCAgent(npc_config_name)
    
    @staticmethod
    def create_core_team():
        """Create the core stable team"""
        return [
            NPCFactory.create_npc("elin_behaviourist"),
            NPCFactory.create_npc("oskar_stable_hand"), 
            NPCFactory.create_npc("astrid_stable_hand"),
            NPCFactory.create_npc("chris_rival"),
            NPCFactory.create_npc("andy_trainer")
        ]

class EnhancedConversationManager:
    """Enhanced ConversationManager with conversational momentum"""
    
    def __init__(self):
        self.agents: Dict[str, ScalableNPCAgent] = {}
        self.conversation_log: List[Message] = []
        self.momentum = ConversationalMomentum()
    
    def register_agent(self, agent: ScalableNPCAgent):
        """Register an agent"""
        self.agents[agent.name] = agent
        role_desc = agent.template_data.get('title', agent.npc_role.value)
        print(f"🎭 Registered {agent.name} ({role_desc})")
    
    def send_to_agent(self, agent_name: str, message: str) -> Tuple[str, bool, float]:
        """Enhanced direct messaging with auto-response potential"""
        agent = self.agents.get(agent_name)
        if not agent:
            return f"Agent '{agent_name}' not found.", False, 0.0
        
        # Add to global log
        self._add_to_global_log("user", message, "player")
        
        # Get response
        response, success, response_time = agent.generate_response(
            message, self.conversation_log[-5:]
        )
        
        # Log agent response
        if success:
            self._add_to_global_log("assistant", response, agent_name)
            
            # Check if other NPCs should auto-respond to this exchange
            auto_triggers = self.momentum.should_trigger_auto_response(
                response, agent_name, self.agents
            )
            
            if auto_triggers:
                print(f"\n🔄 Others want to join the conversation:")
                
                for other_agent_name, reason, probability in auto_triggers[:1]:  # Max 1 for direct conversations
                    other_agent = self.agents[other_agent_name]
                    role_desc = other_agent.template_data.get('title', other_agent.npc_role.value)
                    
                    print(f"⚡ {other_agent_name} ({role_desc}) joining in ({reason})")
                    
                    auto_response, auto_success, auto_time = self.momentum.execute_auto_response(
                        other_agent_name, reason, response, agent_name, self.conversation_log[-3:]
                    )
                    
                    if auto_success:
                        self._add_to_global_log("assistant", auto_response, other_agent_name)
                        other_agent.add_message("assistant", auto_response)
                        print(f"  {other_agent_name}: {auto_response}")
        
        return response, success, response_time
    
    def send_to_all(self, message: str) -> List[Tuple[str, str, bool, float]]:
        """Enhanced send_to_all with automatic follow-up responses"""
        # Add player message to global log
        self._add_to_global_log("user", message, "player")
        
        conversation_type = list(self.agents.values())[0].detect_conversation_type(message) if self.agents else "conversation"
        
        all_responses = []
        current_context = self.conversation_log.copy()
        
        # Determine participants
        participating_agents = []
        for agent_name, agent in self.agents.items():
            if agent.should_participate(message, []):
                participating_agents.append(agent)
        
        # For direct questions, only that person responds
        if any(name.lower() in message.lower() for name in self.agents.keys()):
            direct_agent = None
            for name, agent in self.agents.items():
                if name.lower() in message.lower():
                    direct_agent = agent
                    break
            if direct_agent:
                participating_agents = [direct_agent]
        
        if not participating_agents and self.agents:
            participating_agents = [list(self.agents.values())[0]]
        
        # Randomize order
        random.shuffle(participating_agents)
        
        print(f"💭 {len(participating_agents)} NPCs will respond ({conversation_type} mode)")
        
        # Round 1: Get responses sequentially (thread-safe)
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
                    agent_name=agent.name
                )
                current_context.append(response_msg)
                self.conversation_log.append(response_msg)
                
                # Small delay between responses
                time.sleep(0.5)
        
        # Check for auto-responses after initial round
        if all_responses:
            last_speaker = all_responses[-1][0]  # Name of last agent who spoke
            last_message = all_responses[-1][1]  # What they said
            
            # Check if any NPCs should automatically respond
            auto_triggers = self.momentum.should_trigger_auto_response(
                last_message, last_speaker, self.agents
            )
            
            if auto_triggers:
                print(f"\n🔄 Auto-responses triggered:")
                
                for agent_name, reason, probability in auto_triggers:
                    agent = self.agents[agent_name]
                    role_desc = agent.template_data.get('title', agent.npc_role.value)
                    
                    print(f"⚡ {agent_name} ({role_desc}) responding ({reason}, {probability:.1%})")
                    
                    # Generate auto-response
                    auto_response, success, response_time = self.momentum.execute_auto_response(
                        agent_name, reason, last_message, last_speaker, self.conversation_log[-5:]
                    )
                    
                    if success:
                        # Add to responses and log
                        all_responses.append((agent_name, auto_response, success, response_time))
                        self._add_to_global_log("assistant", auto_response, agent_name)
                        agent.add_message("assistant", auto_response)
                        
                        print(f"  {agent_name}: {auto_response}")
                        
                        # Small delay between auto-responses
                        time.sleep(0.3)
                    
                    # Only allow one auto-response chain per turn
                    break
        
        # Show results
        print(f"\n💬 Responses:")
        for agent_name, response, success, response_time in all_responses:
            if success:
                agent = self.agents[agent_name]
                role_desc = agent.template_data.get('title', agent.npc_role.value)
                print(f"{agent_name} ({role_desc}): {response}")
        
        return all_responses
    
    def _add_to_global_log(self, role: str, content: str, agent_name: str):
        """Add message to global conversation log"""
        message = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=time.time(),
            agent_name=agent_name
        )
        self.conversation_log.append(message)
    
    def reset_all(self):
        """Reset all conversations"""
        for agent in self.agents.values():
            agent.reset_conversation()
        self.conversation_log = []
        self.momentum.conversation_chains = []
        self.momentum.chain_count = 0
        print("🔄 All conversations reset")
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        stats = {
            "total_agents": len(self.agents),
            "total_messages": len(self.conversation_log),
            "momentum_chains": len(self.momentum.conversation_chains),
            "agents": {}
        }
        
        for agent_name, agent in self.agents.items():
            stats["agents"][agent_name] = agent.get_stats()
        
        return stats
    
    def list_agents(self) -> List[str]:
        """Get list of agent names"""
        return list(self.agents.keys()) + ["All"]

def create_enhanced_npc_system():
    """Initialize the enhanced NPC system with conversational momentum"""
    print("🎭 Initializing Enhanced Multi-Agent NPC System with Conversational Momentum")
    print("="*70)
    
    # Create enhanced conversation manager
    manager = EnhancedConversationManager()
    
    # Load NPCs from configurations
    try:
        elin = NPCFactory.create_npc("elin_behaviourist")
        oskar = NPCFactory.create_npc("oskar_stable_hand")
        astrid = NPCFactory.create_npc("astrid_stable_hand")
        chris = NPCFactory.create_npc("chris_rival")
        andy = NPCFactory.create_npc("andy_trainer")

        manager.register_agent(elin)
        manager.register_agent(oskar)
        manager.register_agent(astrid)
        manager.register_agent(chris)
        manager.register_agent(andy)
        
    except Exception as e:
        print(f"⚠️  Error loading NPCs: {e}")
        print("💡 Make sure all NPC configuration files exist")
        return None
    
    return manager

def show_npc_info():
    """Display information about available NPCs"""
    print("\n🎭 Enhanced NPC System Information:")
    print("  - Thread-safe Ollama requests prevent response mixing")
    print("  - Improved response cleaning and validation")
    print("  - Automatic NPC-to-NPC conversations with momentum")
    print("  - Loop prevention with chain limits and cooldowns")
    print("  - Emotional, expertise, and relationship-based triggers")
    print("  Type 'info' anytime to see this again\n")

if __name__ == "__main__":
    print("Enhanced Multi-Agent NPC System with Conversational Momentum")
    print("NPCs now automatically respond to each other!")
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
    print("\nCommands:")
    print("  - Type message for group discussion")
    print("  - 'AgentName: message' for direct conversation")
    print("  - 'info' for system information")
    print("  - 'stats' for statistics")
    print("  - 'reset' to clear conversations")
    print("  - 'quit' to exit")
    
    print(f"\n💬 Examples:")
    print(f"  Momentum: 'Astrid, do you think you'll compete?'")
    print(f"  Debate: 'Which saddle brand do you prefer?'")
    print(f"  Casual: 'How is everyone doing today?'")
    
    while True:
        try:
            user_input = input(f"\nPlayer: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'info':
                show_npc_info()
                continue
            elif user_input.lower() == 'stats':
                stats = manager.get_stats()
                print(f"📊 Total messages: {stats['total_messages']}")
                print(f"🔄 Momentum chains: {stats['momentum_chains']}")
                for agent_name, agent_stats in stats['agents'].items():
                    print(f"  {agent_name} ({agent_stats['title']}): {agent_stats['total_messages']} messages")
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
    
    print("\n👋 Thanks for testing the enhanced multi-agent system with conversational momentum!")