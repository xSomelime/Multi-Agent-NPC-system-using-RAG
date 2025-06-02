#!/usr/bin/env python3
"""
System Manager
==============

Main system initialization and management functions for the enhanced NPC system.
Handles system creation, configuration, and high-level coordination.
"""

from typing import Optional
from src.coordination.conversation_manager import EnhancedConversationManager
from src.agents.npc_factory import NPCFactory

# RAG system check
try:
    from src.agents.rag_enhanced_agent import RAG_AVAILABLE
except ImportError:
    RAG_AVAILABLE = False


def create_enhanced_npc_system(enable_rag: bool = True) -> Optional[EnhancedConversationManager]:
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