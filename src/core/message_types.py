#!/usr/bin/env python3
"""
Core Message Types and Enums
============================

Core data structures and enumerations used throughout the NPC system.
"""

import time
import uuid
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class NPCRole(Enum):
    """Enumeration of NPC roles in the stable environment"""
    STABLE_HAND = "stable_hand"
    TRAINER = "trainer"
    BEHAVIOURIST = "behaviourist"
    COMPETITIVE_RIDER = "competitive_rider"
    RIVAL = "rival"


@dataclass
class Message:
    """Represents a conversation message with metadata"""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    agent_name: Optional[str] = None
    location: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time() 