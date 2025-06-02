#!/usr/bin/env python3
"""
NPC Factory
===========

Factory for creating NPCs with memory integration and optional RAG enhancement.
Handles both individual NPC creation and core team initialization.
"""

from typing import List
from memory.session_memory import MemoryManager
from .base_agent import ScalableNPCAgent

# RAG components (optional import)
try:
    from src.agents.rag_enhanced_agent import create_rag_enhanced_agent, create_rag_enhanced_team
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


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
    def create_core_team(memory_manager: MemoryManager, enable_rag: bool = True) -> List[ScalableNPCAgent]:
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