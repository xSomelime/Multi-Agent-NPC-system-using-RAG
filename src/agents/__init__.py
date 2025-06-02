"""
Agents Module
=============

This module contains all NPC agent implementations, including base agents,
RAG-enhanced agents, and agent factories.

Components:
- ScalableNPCAgent: Base NPC agent with memory integration
- RAGEnhancedNPCAgent: RAG-enhanced wrapper for domain expertise
- NPCFactory: Factory for creating different types of agents
- OllamaRequestManager: Thread-safe LLM request handling
"""

# Import only essential components to avoid circular dependencies
from .ollama_manager import OllamaRequestManager

__all__ = [
    'OllamaRequestManager'
] 