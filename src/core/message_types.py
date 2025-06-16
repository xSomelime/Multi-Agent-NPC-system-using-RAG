#!/usr/bin/env python3
"""
Core Message Types and Enums
============================

Core data structures and enumerations used throughout the NPC system.
Includes message types, NPC roles, personalities, and scenario management.
"""

import time
import uuid
from typing import Optional, List, Dict, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class NPCRole(Enum):
    """Enumeration of NPC roles in the stable environment"""
    STABLE_HAND = "stable_hand"
    TRAINER = "trainer"
    BEHAVIOURIST = "behaviourist"
    COMPETITIVE_RIDER = "competitive_rider"
    RIVAL = "rival"


class PersonalityTrait(Enum):
    """Core personality traits that influence NPC behavior"""
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    COMPETITIVE = "competitive"
    ANALYTICAL = "analytical"
    PASSIONATE = "passionate"
    CAUTIOUS = "cautious"
    OUTSPOKEN = "outspoken"
    RESERVED = "reserved"


class InformationSource(Enum):
    """Sources of information for memory system"""
    WITNESSED = "witnessed"
    HEARD = "heard"
    PLAYER_TOLD = "player_told"
    NPC_TOLD = "npc_told"
    INFERRED = "inferred"


class ConfidenceLevel(Enum):
    """Confidence levels for memory and information"""
    CERTAIN = 1.0
    CONFIDENT = 0.8
    UNCERTAIN = 0.5
    DOUBTFUL = 0.2


class EmotionalState(Enum):
    """Emotional states for NPCs"""
    HAPPY = "happy"
    EXCITED = "excited"
    CALM = "calm"
    CONCERNED = "concerned"
    THOUGHTFUL = "thoughtful"
    ASSERTIVE = "assertive"
    NEUTRAL = "neutral"


@dataclass
class Personality:
    """NPC personality configuration"""
    traits: Set[PersonalityTrait]
    speaking_style: str  # Template for speaking patterns
    expertise_areas: List[str]
    opinion_strength: float  # 0.0 to 1.0
    response_length: int  # Target response length in tokens
    temperature: float  # 0.1 to 0.5 for response variation


@dataclass
class Scenario:
    """Game scenario configuration"""
    # Required fields (no defaults)
    title: str
    description: str
    initial_locations: Dict[str, str]  # NPC name to location
    context: str  # Background information
    topics: List[str]  # Potential discussion topics
    relationships: Dict[str, Dict[str, float]]  # NPC to NPC relationship scores
    memory_hooks: List[str]  # Key points NPCs should remember
    
    # Optional fields (with defaults)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    active: bool = True


@dataclass
class Message:
    """Represents a conversation message with metadata"""
    # Required fields (no defaults)
    role: str  # "user", "assistant", or "system"
    content: str
    
    # Optional fields (with defaults)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    agent_name: Optional[str] = None
    location: Optional[str] = None
    title: Optional[str] = None
    confidence: float = 1.0
    source: InformationSource = InformationSource.WITNESSED
    is_group_message: bool = False
    group_id: Optional[str] = None
    recipients: List[str] = field(default_factory=list)
    topic: Optional[str] = None
    emotional_tone: Optional[str] = None


@dataclass
class Location:
    """Represents a location in the game world"""
    name: str
    description: str
    connected_locations: List[str]
    proximity_threshold: float  # Distance at which NPCs can interact
    capacity: int  # Maximum number of NPCs that can be present
    current_occupants: List[str] = field(default_factory=list)


@dataclass
class MemoryEvent:
    """Represents a memory with metadata"""
    # Required fields (no defaults)
    content: str
    source: InformationSource
    
    # Optional fields (with defaults)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    confidence: ConfidenceLevel = ConfidenceLevel.CERTAIN
    location: Optional[str] = None
    related_npcs: List[str] = field(default_factory=list)
    topic: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    expires_at: Optional[float] = None  # For memory degradation


@dataclass
class TopicOpinion:
    """Represents an NPC's opinion on a topic"""
    topic: str
    reasoning: str
    style: EmotionalState
    strength: float  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class EmotionalProfile:
    """Profile for NPC emotional responses"""
    base_emotionality: float  # 0.0 to 1.0
    current_state: EmotionalState = EmotionalState.NEUTRAL
    triggers: dict = field(default_factory=dict)
    emotional_history: List[Tuple[EmotionalState, float]] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)


@dataclass
class PersonalityProfile:
    """Detailed personality profile for NPCs"""
    name: str
    role_template: str
    primary_traits: List[str]
    secondary_traits: List[str]
    expertise_areas: List[str]
    cautious_areas: List[str]
    common_phrases: List[str]
    speech_patterns: List[str]
    emotional_triggers: dict = field(default_factory=dict)
    relationship_preferences: dict = field(default_factory=dict)
    personal_background: str = ""
    
    @classmethod
    def from_json(cls, data: dict) -> 'PersonalityProfile':
        """Create a personality profile from JSON data"""
        # Extract personality data
        personality = data.get('personality', {})
        traits = personality.get('traits', {})
        
        # Handle both dictionary and list formats for traits
        if isinstance(traits, dict):
            primary_traits = [t.upper() for t in traits.get('primary', [])]
            secondary_traits = [t.upper() for t in traits.get('secondary', [])]
        else:
            # Handle list format
            all_traits = [t.upper() for t in traits] if isinstance(traits, list) else []
            primary_traits = all_traits[:3]
            secondary_traits = all_traits[3:]
        
        # Extract speaking style
        speaking_style = personality.get('speaking_style', {})
        if isinstance(speaking_style, dict):
            common_phrases = speaking_style.get('common_phrases', [])
            speech_patterns = speaking_style.get('speech_patterns', [])
        else:
            # Handle string format
            common_phrases = []
            speech_patterns = []
            if isinstance(speaking_style, str):
                speech_patterns = [speaking_style]
        
        # Extract professional focus
        professional_focus = data.get('professional_focus', {})
        expertise_areas = professional_focus.get('expertise_areas', [])
        cautious_areas = professional_focus.get('cautious_areas', [])
        
        # Extract emotional triggers
        emotional_triggers = {}
        if 'emotional_triggers' in personality:
            triggers = personality['emotional_triggers']
            if isinstance(triggers, dict):
                for category, category_triggers in triggers.items():
                    if isinstance(category_triggers, dict):
                        emotional_triggers.update(category_triggers)
        
        # Extract relationships
        relationships = data.get('relationships', {})
        
        return cls(
            name=data['name'],
            role_template=data['role_template'],
            primary_traits=primary_traits,
            secondary_traits=secondary_traits,
            expertise_areas=expertise_areas,
            cautious_areas=cautious_areas,
            common_phrases=common_phrases,
            speech_patterns=speech_patterns,
            emotional_triggers=emotional_triggers,
            relationship_preferences=relationships,
            personal_background=data.get('personal_background', '')
        ) 