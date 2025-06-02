"""
Coordination Module
=================

This module handles multi-agent coordination, conversation management,
and inter-NPC communication for the enhanced NPC system.

Components:
- ConversationManager: Orchestrates multi-agent conversations
- ConversationalMomentum: Manages automatic NPC-to-NPC interactions
- TurnManager: Handles who speaks when and conversation flow
- LocationCoordinator: Manages spatial awareness and location-based interactions
"""

from .conversation_manager import EnhancedConversationManager
from .conversational_momentum import ConversationalMomentum
from .turn_management import TurnManager
from .location_coordinator import LocationCoordinator

__all__ = [
    'EnhancedConversationManager',
    'ConversationalMomentum', 
    'TurnManager',
    'LocationCoordinator'
] 