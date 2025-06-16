#!/usr/bin/env python3
"""
Group Conversation Manager
=========================

Handles dynamic group conversations with opinion-based interactions,
argument dynamics, and memory sharing between NPCs.
"""

import time
import random
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from memory.session_memory import MemoryManager, ConfidenceLevel
from src.core.message_types import Message
from src.agents.ollama_manager import get_ollama_manager


class OpinionStrength(Enum):
    """Strength of NPC opinions in discussions"""
    PASSIONATE_AGREE = 1.0    # Very strong agreement, emotional investment
    STRONG_AGREE = 0.8        # Strong agreement with conviction
    AGREE = 0.6               # General agreement
    TENTATIVE_AGREE = 0.4     # Agree but with some reservations
    NEUTRAL = 0.5             # No strong opinion either way
    TENTATIVE_DISAGREE = 0.6  # Disagree but open to discussion
    DISAGREE = 0.4            # General disagreement
    STRONG_DISAGREE = 0.2     # Strong disagreement
    PASSIONATE_DISAGREE = 0.0 # Very strong disagreement, emotional investment


class EmotionalState(Enum):
    """NPC emotional states during conversations"""
    EXCITED = "excited"        # High energy, positive
    HAPPY = "happy"           # Positive, content
    CALM = "calm"            # Neutral, relaxed
    THOUGHTFUL = "thoughtful" # Contemplative, considering
    CONCERNED = "concerned"   # Worried, cautious
    FRUSTRATED = "frustrated" # Annoyed, irritated
    PASSIONATE = "passionate" # Emotionally invested
    DEFENSIVE = "defensive"   # Protective of position
    NEUTRAL = "neutral"       # No strong emotional state


class ArgumentStyle(Enum):
    """How NPCs express their opinions"""
    DIPLOMATIC = "diplomatic"      # Polite, considerate
    ASSERTIVE = "assertive"        # Confident, direct
    PASSIONATE = "passionate"      # Emotional, enthusiastic
    ANALYTICAL = "analytical"      # Logical, detailed
    CONFRONTATIONAL = "confrontational"  # Direct, challenging
    TEACHING = "teaching"          # Educational, explanatory
    EXPERIENTIAL = "experiential"  # Based on personal experience
    PROFESSIONAL = "professional"  # Based on expertise


class PersonalityTrait(Enum):
    """Core personality traits that influence NPC behavior"""
    # Communication Style
    DIRECT = "direct"           # Straightforward, no-nonsense
    DIPLOMATIC = "diplomatic"   # Careful, considerate
    ENTHUSIASTIC = "enthusiastic" # Energetic, passionate
    ANALYTICAL = "analytical"   # Logical, methodical
    TEACHING = "teaching"       # Educational, explanatory
    COMPETITIVE = "competitive" # Driven, ambitious
    EMPATHETIC = "empathetic"   # Understanding, caring
    PROFESSIONAL = "professional" # Formal, business-like
    
    # Response Patterns
    DETAIL_ORIENTED = "detail_oriented"     # Provides specific examples
    EXPERIENCE_BASED = "experience_based"   # References personal history
    THEORY_FOCUSED = "theory_focused"       # Explains principles
    PRACTICAL = "practical"                 # Focuses on actionable advice
    CAUTIOUS = "cautious"                   # Conservative in advice
    BOLD = "bold"                          # Confident, assertive
    HUMOROUS = "humorous"                  # Uses wit and humor
    SERIOUS = "serious"                    # No-nonsense, formal


@dataclass
class TopicOpinion:
    """NPC's opinion on a specific topic"""
    topic: str
    strength: OpinionStrength
    style: ArgumentStyle
    reasoning: str
    confidence: float
    source: str  # How they formed this opinion
    emotional_state: EmotionalState = EmotionalState.NEUTRAL
    formed_at: float = field(default_factory=time.time)
    last_expressed: Optional[float] = None
    times_expressed: int = 0
    related_topics: List[str] = field(default_factory=list)


@dataclass
class EmotionalProfile:
    """NPC's emotional profile and current state"""
    base_emotionality: float  # 0.0 (calm) to 1.0 (emotional)
    empathy_level: float      # 0.0 (low) to 1.0 (high)
    current_state: EmotionalState = EmotionalState.NEUTRAL
    emotional_history: List[Tuple[EmotionalState, float]] = field(default_factory=list)
    triggers: Dict[str, EmotionalState] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)


@dataclass
class PersonalityProfile:
    """Detailed personality profile for an NPC, loaded from JSON config"""
    # Core personality data
    name: str
    role_template: str
    personality: Dict[str, Any]  # Raw personality data from JSON
    personal_background: str
    relationships: Dict[str, str]
    professional_opinions: Dict[str, str]
    controversial_stances: List[str]
    
    # Derived personality traits
    primary_traits: List[PersonalityTrait] = field(default_factory=list)
    secondary_traits: List[PersonalityTrait] = field(default_factory=list)
    
    # Communication preferences (derived from personality data)
    response_length: Tuple[int, int] = (40, 80)  # Default, can be overridden
    formality_level: float = 0.5  # Default, can be overridden
    detail_level: float = 0.5  # Default, can be overridden
    
    # Speaking style (derived from personality data)
    common_phrases: List[str] = field(default_factory=list)
    speech_patterns: List[str] = field(default_factory=list)
    vocabulary_style: str = ""
    
    # Response tendencies (derived from personality data)
    question_frequency: float = 0.3
    example_frequency: float = 0.5
    advice_frequency: float = 0.4
    
    # Professional focus (derived from personality data)
    expertise_areas: List[str] = field(default_factory=list)
    confidence_areas: List[str] = field(default_factory=list)
    cautious_areas: List[str] = field(default_factory=list)
    
    # Emotional tendencies (derived from personality data)
    base_emotionality: float = 0.5
    empathy_level: float = 0.5
    emotional_triggers: Dict[str, EmotionalState] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'PersonalityProfile':
        """Create a personality profile from JSON data"""
        profile = cls(
            name=json_data["name"],
            role_template=json_data["role_template"],
            personality=json_data["personality"],
            personal_background=json_data["personal_background"],
            relationships=json_data.get("relationships", {}),
            professional_opinions=json_data.get("professional_opinions", {}),
            controversial_stances=json_data.get("controversial_stances", [])
        )
        
        # Derive personality traits from JSON data
        profile._derive_traits()
        profile._derive_communication_preferences()
        profile._derive_speaking_style()
        profile._derive_professional_focus()
        profile._derive_emotional_tendencies()
        
        return profile
    
    def _derive_traits(self):
        """Derive personality traits from JSON data"""
        traits = self.personality.get("traits", [])
        
        # Map JSON traits to PersonalityTrait enum
        trait_mapping = {
            "tough_love_mentor": PersonalityTrait.TEACHING,
            "charismatic_exterior": PersonalityTrait.ENTHUSIASTIC,
            "loyal_once_earned": PersonalityTrait.EMPATHETIC,
            "performance_focused": PersonalityTrait.COMPETITIVE,
            "darkly_humorous": PersonalityTrait.HUMOROUS,
            "gentle": PersonalityTrait.EMPATHETIC,
            "anxious": PersonalityTrait.CAUTIOUS,
            "empathetic": PersonalityTrait.EMPATHETIC,
            "hardworking": PersonalityTrait.PRACTICAL,
            "self_doubting": PersonalityTrait.CAUTIOUS,
            "animal_loving": PersonalityTrait.EMPATHETIC,
            # Add more mappings as needed
        }
        
        # Handle both dictionary and list formats for traits
        if isinstance(traits, dict):
            # Handle dictionary format with primary/secondary
            primary_traits = traits.get('primary', [])
            secondary_traits = traits.get('secondary', [])
            
            # Convert all traits to lowercase for mapping
            for trait in primary_traits:
                trait_lower = trait.lower()
                if trait_lower in trait_mapping:
                    self.primary_traits.append(trait_mapping[trait_lower])
            
            for trait in secondary_traits:
                trait_lower = trait.lower()
                if trait_lower in trait_mapping:
                    self.secondary_traits.append(trait_mapping[trait_lower])
        else:
            # Handle list format
            for trait in traits:
                trait_lower = trait.lower()
                if trait_lower in trait_mapping:
                    if len(self.primary_traits) < 3:
                        self.primary_traits.append(trait_mapping[trait_lower])
                    else:
                        self.secondary_traits.append(trait_mapping[trait_lower])
    
    def _derive_communication_preferences(self):
        """Derive communication preferences from personality data"""
        speaking_style = self.personality.get("speaking_style", {})
        
        # Handle both dictionary and string formats for speaking style
        if isinstance(speaking_style, dict):
            tone = speaking_style.get('tone', '').lower()
            vocabulary = speaking_style.get('vocabulary_style', '').lower()
            speaking_style_text = f"{tone} {vocabulary}".strip()
        else:
            speaking_style_text = str(speaking_style).lower()
        
        # Set formality level based on speaking style
        if "formal" in speaking_style_text or "professional" in speaking_style_text:
            self.formality_level = 0.8
        elif "casual" in speaking_style_text or "relaxed" in speaking_style_text:
            self.formality_level = 0.3
        else:
            self.formality_level = 0.5
        
        # Set detail level based on personality
        if "detailed" in speaking_style_text or "thorough" in speaking_style_text:
            self.detail_level = 0.8
        elif "concise" in speaking_style_text or "brief" in speaking_style_text:
            self.detail_level = 0.3
        else:
            self.detail_level = 0.5
        
        # Set response length based on role and personality
        if self.role_template == "trainer":
            self.response_length = (60, 120)
        elif self.role_template == "stable_hand":
            self.response_length = (30, 60)
        else:
            self.response_length = (40, 80)
    
    def _derive_speaking_style(self):
        """Derive speaking style from personality data"""
        speaking_style = self.personality.get("speaking_style", {})
        
        # Handle both dictionary and string formats for speaking style
        if isinstance(speaking_style, dict):
            # Extract from dictionary format
            self.common_phrases = speaking_style.get('common_phrases', [])
            self.speech_patterns = speaking_style.get('speech_patterns', [])
            self.vocabulary_style = speaking_style.get('vocabulary_style', '')
        else:
            # Handle string format
            self.common_phrases = []
            self.speech_patterns = [str(speaking_style)]
            self.vocabulary_style = str(speaking_style)
        
        # Set vocabulary style based on role if not set
        if not self.vocabulary_style:
            if self.role_template == "trainer":
                self.vocabulary_style = "practical with training terms"
            elif self.role_template == "stable_hand":
                self.vocabulary_style = "gentle and caring"
            else:
                self.vocabulary_style = "professional and clear"
    
    def _derive_professional_focus(self):
        """Derive professional focus from personality data"""
        # Extract expertise areas from professional opinions
        self.expertise_areas = list(self.professional_opinions.keys())
        
        # Set confidence areas based on role and personality
        if self.role_template == "trainer":
            self.confidence_areas = ["training methods", "competition strategy"]
        elif self.role_template == "stable_hand":
            self.confidence_areas = ["horse care", "gentle handling"]
        
        # Set cautious areas based on personality
        if "anxious" in self.personality.get("traits", []):
            self.cautious_areas = ["advanced training", "competition"]
        elif "self_doubting" in self.personality.get("traits", []):
            self.cautious_areas = ["giving advice", "making decisions"]
    
    def _derive_emotional_tendencies(self):
        """Derive emotional tendencies from personality data"""
        # Set base emotionality based on personality traits
        if "anxious" in self.personality.get("traits", []):
            self.base_emotionality = 0.7
        elif "gentle" in self.personality.get("traits", []):
            self.base_emotionality = 0.4
        else:
            self.base_emotionality = 0.5
        
        # Set empathy level based on personality traits
        if "empathetic" in self.personality.get("traits", []):
            self.empathy_level = 0.8
        elif "animal_loving" in self.personality.get("traits", []):
            self.empathy_level = 0.7
        else:
            self.empathy_level = 0.5
        
        # Set emotional triggers based on personality and relationships
        self.emotional_triggers = {}
        if "fears_and_limitations" in self.__dict__:
            for fear, desc in self.fears_and_limitations.items():
                self.emotional_triggers[fear] = EmotionalState.CONCERNED
        
        # Add relationship-based triggers
        for person, relationship in self.relationships.items():
            if "frustrated" in relationship:
                self.emotional_triggers[person] = EmotionalState.FRUSTRATED
            elif "respect" in relationship:
                self.emotional_triggers[person] = EmotionalState.THOUGHTFUL


# Define distinct personality profiles for each NPC
PERSONALITY_PROFILES = {
    "andy_trainer": PersonalityProfile(
        name="andy_trainer",
        role_template="trainer",
        personality={
            "traits": ["enthusiastic", "teaching", "practical"],
            "speaking_style": "enthusiastic and confident",
            "personal_background": "Ex-racehorse trainer with a passion for horses",
            "relationships": {
                "astrid_vet": "respectful",
                "chris_rival": "frustrated"
            },
            "professional_opinions": {
                "training_philosophy": "Training is key to a horse's success",
                "horse_welfare": "Horses need proper care and nutrition",
                "competition": "Competition brings out the best in horses"
            },
            "controversial_stances": []
        },
        personal_background="Ex-racehorse trainer with a passion for horses",
        relationships={
            "astrid_vet": "respectful",
            "chris_rival": "frustrated"
        },
        professional_opinions={
            "training_philosophy": "Training is key to a horse's success",
            "horse_welfare": "Horses need proper care and nutrition",
            "competition": "Competition brings out the best in horses"
        },
        controversial_stances=[]
    ),
    
    "astrid_vet": PersonalityProfile(
        name="astrid_vet",
        role_template="vet",
        personality={
            "traits": ["analytical", "professional", "cautious"],
            "speaking_style": "professional and thorough",
            "personal_background": "Veterinary science graduate",
            "relationships": {
                "andy_trainer": "respectful",
                "chris_rival": "frustrated"
            },
            "professional_opinions": {
                "equine_health": "Equine health is a top priority",
                "preventive_care": "Preventive care is better than treatment",
                "emergency": "Emergency response is crucial"
            },
            "controversial_stances": []
        },
        personal_background="Veterinary science graduate",
        relationships={
            "andy_trainer": "respectful",
            "chris_rival": "frustrated"
        },
        professional_opinions={
            "equine_health": "Equine health is a top priority",
            "preventive_care": "Preventive care is better than treatment",
            "emergency": "Emergency response is crucial"
        },
        controversial_stances=[]
    ),
    
    "chris_rival": PersonalityProfile(
        name="chris_rival",
        role_template="stable_hand",
        personality={
            "traits": ["competitive", "direct", "bold"],
            "speaking_style": "confident and assertive",
            "personal_background": "Ex-competitive horse rider",
            "relationships": {
                "andy_trainer": "frustrated",
                "astrid_vet": "respectful"
            },
            "professional_opinions": {
                "performance_training": "Performance training is essential",
                "winning_mindset": "Winning mindset is crucial",
                "pressure_handling": "Handling pressure is key to success"
            },
            "controversial_stances": []
        },
        personal_background="Ex-competitive horse rider",
        relationships={
            "andy_trainer": "frustrated",
            "astrid_vet": "respectful"
        },
        professional_opinions={
            "performance_training": "Performance training is essential",
            "winning_mindset": "Winning mindset is crucial",
            "pressure_handling": "Handling pressure is key to success"
        },
        controversial_stances=[]
    )
}

class GroupConversationManager:
    """Manages dynamic group conversations with opinion-based interactions"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.active_conversations: Dict[str, Dict] = {}
        self.topic_opinions: Dict[str, Dict[str, TopicOpinion]] = {}  # npc -> topic -> opinion
        self.conversation_history: Dict[str, List[Message]] = {}
        self.emotional_profiles: Dict[str, EmotionalProfile] = {}
        self.personality_profiles: Dict[str, PersonalityProfile] = {}
        
        # Track emotional states during conversations
        self.emotional_states: Dict[str, Dict[str, float]] = {}  # npc -> emotion -> intensity
        
        # Load Ollama manager for responses
        self.ollama_manager = get_ollama_manager()
        
        # Load personality profiles from JSON files
        self._load_personality_profiles()
    
    def _load_personality_profiles(self):
        """Load personality profiles from JSON files"""
        npc_dir = os.path.join("data", "npcs")
        for filename in os.listdir(npc_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(npc_dir, filename), 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        profile = PersonalityProfile.from_json(json_data)
                        self.personality_profiles[profile.name.lower()] = profile
                except Exception as e:
                    print(f"Error loading personality profile from {filename}: {e}")
    
    def start_group_conversation(self, conversation_id: str, npc_names: List[str], 
                               initial_topic: str, location: str) -> bool:
        """Start a new group conversation"""
        if conversation_id in self.active_conversations:
            return False
        
        # Initialize conversation state
        self.active_conversations[conversation_id] = {
            "npc_names": npc_names,
            "current_topic": initial_topic,
            "location": location,
            "start_time": time.time(),
            "participants": set(npc_names),
            "sub_topics": set(),
            "emotional_intensity": 0.0
        }
        
        # Initialize emotional states for participants
        for npc in npc_names:
            self.emotional_states[npc] = {
                "engagement": 0.5,
                "agreement": 0.5,
                "confidence": 0.5
            }
        
        # Initialize conversation history
        self.conversation_history[conversation_id] = []
        
        return True
    
    def add_message(self, conversation_id: str, speaker: str, content: str, 
                   is_player: bool = False) -> Tuple[str, bool, float]:
        """Add a message to the conversation and trigger NPC responses"""
        if conversation_id not in self.active_conversations:
            return "Conversation not found", False, 0.0
        
        start_time = time.time()
        conv = self.active_conversations[conversation_id]
        
        # Record message
        message = Message(
            role="user" if is_player else "assistant",
            content=content,
            agent_name=speaker,
            location=conv["location"],
            is_group_message=True,
            group_id=conversation_id
        )
        self.conversation_history[conversation_id].append(message)
        
        # Update emotional states based on message
        self._update_emotional_states(conversation_id, speaker, content)
        
        # Get potential responders
        responders = self._get_potential_responders(conversation_id, speaker, content)
        
        # Generate responses
        responses = []
        for npc_name, reason, probability in responders:
            if random.random() < probability:
                response = self._generate_response(
                    conversation_id, npc_name, content, reason
                )
                if response:
                    responses.append(response)
        
        # Sort responses by emotional impact
        responses.sort(key=lambda x: self._calculate_emotional_impact(x[0]), reverse=True)
        
        # Return the most impactful response
        if responses:
            return responses[0]
        return "No response generated", False, 0.0
    
    def _get_potential_responders(self, conversation_id: str, speaker: str, 
                                content: str) -> List[Tuple[str, str, float]]:
        """Get list of NPCs who might respond to the message"""
        conv = self.active_conversations[conversation_id]
        responders = []
        
        for npc in conv["npc_names"]:
            if npc == speaker:
                continue
            
            # Calculate base probability
            probability = 0.5
            
            # Adjust based on emotional state
            emotional_state = self.emotional_states.get(npc, {})
            probability += (emotional_state.get("engagement", 0.5) - 0.5) * 0.3
            
            # Adjust based on topic relevance
            if self._is_topic_relevant(npc, conv["current_topic"]):
                probability += 0.2
            
            # Adjust based on relationship with speaker
            relationship_factor = self._get_relationship_factor(npc, speaker)
            probability += (relationship_factor - 0.5) * 0.2
            
            # Add to responders if probability is high enough
            if probability > 0.3:
                reason = self._determine_response_reason(npc, content)
                responders.append((npc, reason, probability))
        
        return responders
    
    def _generate_response(self, conversation_id: str, npc_name: str, 
                          original_message: str, reason: str) -> Tuple[str, bool, float]:
        """Generate a response for an NPC in the conversation"""
        conv = self.active_conversations[conversation_id]
        
        # Get opinion on current topic
        opinion = self._get_opinion_on_topic(npc_name, conv["current_topic"])
        
        # Build response prompt
        prompt = self._build_response_prompt(
            npc_name, original_message, reason, opinion, 
            conv["current_topic"], self.conversation_history[conversation_id]
        )
        
        # Get response from Ollama
        response, success = self.ollama_manager.make_request(
            npc_name,
            prompt,
            "phi3:mini",
            0.4,  # temperature
            50    # max tokens
        )
        
        if success:
            # Record response in conversation history
            message = Message(
                role="assistant",
                content=response,
                agent_name=npc_name,
                location=conv["location"],
                is_group_message=True,
                group_id=conversation_id
            )
            self.conversation_history[conversation_id].append(message)
            
            # Update emotional states
            self._update_emotional_states(conversation_id, npc_name, response)
            
            # Share memory with other participants
            self._share_memory_with_participants(conversation_id, npc_name, response)
        
        return response, success, time.time() - time.time()
    
    def _build_response_prompt(self, npc_name: str, original_message: str, 
                             reason: str, opinion: Optional[TopicOpinion],
                             current_topic: str, history: List[Message]) -> str:
        """Build a prompt for generating a response with personality context"""
        profile = self.personality_profiles.get(npc_name)
        if not profile:
            return self._build_basic_prompt(npc_name, original_message, reason, opinion, current_topic, history)
            
        # Check if the message requests a detailed explanation
        needs_detail = any(phrase in original_message.lower() for phrase in [
            "explain", "tell me more", "why", "how does", "describe",
            "what do you mean", "can you elaborate", "could you explain"
        ])
        
        # Determine response length based on context
        if needs_detail:
            min_tokens, max_tokens = profile.response_length
        else:
            # Default to shorter responses (1-2 sentences)
            min_tokens, max_tokens = (20, 40)  # Shorter range for normal responses
        
        prompt_parts = [
            f"CRITICAL: You are {npc_name}. Stay in character.",
            f"\nPersonality Profile:",
            f"- Primary traits: {', '.join(t.value for t in profile.primary_traits)}",
            f"- Secondary traits: {', '.join(t.value for t in profile.secondary_traits)}",
            f"- Speaking style: {profile.vocabulary_style}",
            f"- Formality level: {'formal' if profile.formality_level > 0.7 else 'casual' if profile.formality_level < 0.4 else 'moderate'}",
            f"\nCurrent topic: {current_topic}",
            f"\nResponse Guidelines:",
            f"- Keep response between {min_tokens} and {max_tokens} tokens",
            f"- {'Provide detailed explanation' if needs_detail else 'Keep it brief (1-2 sentences)'}",
            f"- Maintain your distinct personality"
        ]
        
        # Add emotional context
        emotional_profile = self.emotional_profiles.get(npc_name)
        if emotional_profile:
            prompt_parts.append(f"\nCurrent emotional state: {emotional_profile.current_state.value}")
        
        # Add opinion context if available
        if opinion:
            prompt_parts.append(f"\nYour opinion: {opinion.reasoning}")
            prompt_parts.append(f"Express this {opinion.style.value}ly")
        
        # Add speaking style guidance
        prompt_parts.append("\nSpeaking Style:")
        prompt_parts.append(f"- Use these phrases naturally: {', '.join(random.sample(profile.common_phrases, min(2, len(profile.common_phrases))))}")
        prompt_parts.append(f"- Follow these patterns: {random.choice(profile.speech_patterns)}")
        
        # Add recent conversation context with emotional states
        if history:
            prompt_parts.append("\nRecent conversation:")
            for msg in history[-2:]:  # Only last 2 messages for context
                speaker_profile = self.emotional_profiles.get(msg.agent_name)
                emotional_context = f" ({speaker_profile.current_state.value})" if speaker_profile else ""
                prompt_parts.append(f"{msg.agent_name}{emotional_context}: {msg.content}")
        
        # Add response instruction based on personality and reason
        if reason == "agreement":
            if "ENTHUSIASTIC" in [t.value for t in profile.primary_traits]:
                prompt_parts.append("\nShow brief enthusiastic agreement.")
            elif "ANALYTICAL" in [t.value for t in profile.primary_traits]:
                prompt_parts.append("\nAgree briefly with your professional insight.")
            else:
                prompt_parts.append("\nAgree briefly while maintaining your personality.")
        elif reason == "disagreement":
            if "DIPLOMATIC" in [t.value for t in profile.primary_traits]:
                prompt_parts.append("\nExpress brief diplomatic disagreement.")
            elif "DIRECT" in [t.value for t in profile.primary_traits]:
                prompt_parts.append("\nState brief direct disagreement.")
            else:
                prompt_parts.append("\nDisagree briefly while staying in character.")
        elif reason == "expertise":
            if current_topic in profile.expertise_areas:
                prompt_parts.append("\nShare a brief expert insight.")
            elif current_topic in profile.cautious_areas:
                prompt_parts.append("\nShare a brief careful thought.")
            else:
                prompt_parts.append("\nShare a brief perspective based on your experience.")
        
        prompt_parts.append(f"\n{npc_name}:")
        return "\n".join(prompt_parts)
    
    def _build_basic_prompt(self, npc_name: str, original_message: str, 
                          reason: str, opinion: Optional[TopicOpinion],
                          current_topic: str, history: List[Message]) -> str:
        """Fallback prompt builder for NPCs without detailed profiles"""
        # ... existing basic prompt building code ...
    
    def _update_emotional_states(self, conversation_id: str, speaker: str, content: str):
        """Update emotional states based on message content and NPC profile"""
        if speaker not in self.emotional_profiles:
            return
        
        profile = self.emotional_profiles[speaker]
        current_time = time.time()
        
        # Analyze content for emotional triggers
        new_state = self._analyze_emotional_content(content, profile)
        
        # Record emotional change
        profile.emotional_history.append((new_state, current_time))
        profile.current_state = new_state
        profile.last_update = current_time
        
        # Trim emotional history if too long
        if len(profile.emotional_history) > 10:
            profile.emotional_history = profile.emotional_history[-10:]
    
    def _analyze_emotional_content(self, content: str, profile: EmotionalProfile) -> EmotionalState:
        """Analyze content to determine emotional state based on profile"""
        # This would be enhanced with proper NLP in a real implementation
        content_lower = content.lower()
        
        # Check for emotional triggers
        for trigger, state in profile.triggers.items():
            if trigger in content_lower:
                return state
        
        # Basic sentiment analysis
        if any(word in content_lower for word in ["!", "amazing", "wonderful", "excellent"]):
            return EmotionalState.EXCITED if profile.base_emotionality > 0.6 else EmotionalState.HAPPY
        
        if any(word in content_lower for word in ["?", "think", "consider", "maybe"]):
            return EmotionalState.THOUGHTFUL
        
        if any(word in content_lower for word in ["worried", "concerned", "careful"]):
            return EmotionalState.CONCERNED
        
        if any(word in content_lower for word in ["must", "should", "need to"]):
            return EmotionalState.ASSERTIVE
        
        # Default to neutral or calm based on profile
        return EmotionalState.CALM if profile.base_emotionality < 0.5 else EmotionalState.NEUTRAL
    
    def _share_memory_with_participants(self, conversation_id: str, speaker: str, content: str):
        """Share memory with other conversation participants"""
        conv = self.active_conversations[conversation_id]
        
        for npc in conv["npc_names"]:
            if npc != speaker:
                self.memory_manager.npc_tells_npc(
                    speaker, npc,
                    content,
                    conv["location"],
                    ConfidenceLevel.CONFIDENT,
                    ["group_conversation", conv["current_topic"]]
                )
    
    def _calculate_emotional_impact(self, response: str) -> float:
        """Calculate the emotional impact of a response"""
        impact = 0.5  # Base impact
        
        # Simple heuristics for impact
        if "!" in response:
            impact += 0.2
        if "?" in response:
            impact += 0.1
        if len(response.split()) > 15:
            impact += 0.1
        
        return min(1.0, impact)
    
    def _is_topic_relevant(self, npc_name: str, topic: str) -> bool:
        """Check if an NPC has relevant knowledge about a topic"""
        # Check memory
        memories = self.memory_manager.get_npc_memory(npc_name)
        if memories and memories.knows_about(topic):
            return True
        
        # Check opinions
        if npc_name in self.topic_opinions and topic in self.topic_opinions[npc_name]:
            return True
        
        return False
    
    def _get_relationship_factor(self, npc1: str, npc2: str) -> float:
        """Get relationship factor between two NPCs"""
        # This would be enhanced with proper relationship tracking
        return 0.5
    
    def _determine_response_reason(self, npc: str, content: str) -> str:
        """Determine why an NPC should respond"""
        # This would be enhanced with proper content analysis
        return "natural"
    
    def _get_opinion_on_topic(self, npc: str, topic: str) -> Optional[TopicOpinion]:
        """Get NPC's opinion on a topic"""
        if npc in self.topic_opinions and topic in self.topic_opinions[npc]:
            return self.topic_opinions[npc][topic]
        return None
    
    def end_conversation(self, conversation_id: str):
        """End a group conversation"""
        if conversation_id in self.active_conversations:
            # Record final state
            conv = self.active_conversations[conversation_id]
            duration = time.time() - conv["start_time"]
            
            # Share final memories
            for npc in conv["npc_names"]:
                self.memory_manager.record_witnessed_event(
                    f"Group conversation about {conv['current_topic']} ended",
                    conv["location"],
                    conv["npc_names"],
                    ["group_conversation", "conclusion", conv["current_topic"]]
                )
            
            # Clean up
            del self.active_conversations[conversation_id]
            del self.conversation_history[conversation_id]
            
            # Reset emotional states for participants
            for npc in conv["npc_names"]:
                if npc in self.emotional_states:
                    del self.emotional_states[npc]
