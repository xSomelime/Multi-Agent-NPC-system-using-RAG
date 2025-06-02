#!/usr/bin/env python3
"""
NPC-RAG Integration Layer
Connects the RAG system to individual NPCs for context-aware knowledge retrieval
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .rag_system import get_rag_system, RetrievalResult, RAGKnowledgeBase

@dataclass
class NPCKnowledgeContext:
    """Context for NPC knowledge retrieval"""
    npc_name: str
    role: str  # stable_hand, trainer, behaviourist, rival
    current_topic: str
    conversation_history: List[str]
    confidence_level: str = "medium"  # high, medium, low
    max_knowledge_items: int = 3

class NPCRAGInterface:
    """
    Interface between NPCs and RAG system
    Provides role-appropriate knowledge retrieval and response enhancement
    """
    
    def __init__(self):
        self.rag_system: RAGKnowledgeBase = get_rag_system()
        
        # Role-specific configuration
        self.role_config = {
            "stable_hand": {
                "primary_domains": ["stable_hand"],
                "secondary_domains": ["trainer", "behaviourist"],  # Can access basic info
                "confidence_boost": 0.1,  # Boost for practical knowledge
                "max_technical_depth": "medium"
            },
            "trainer": {
                "primary_domains": ["trainer"],
                "secondary_domains": ["stable_hand", "behaviourist"],
                "confidence_boost": 0.15,  # High expertise
                "max_technical_depth": "high"
            },
            "behaviourist": {
                "primary_domains": ["behaviourist"],
                "secondary_domains": ["stable_hand", "trainer"],
                "confidence_boost": 0.15,  # Scientific expertise
                "max_technical_depth": "high"
            },
            "rival": {
                "primary_domains": ["rival"],
                "secondary_domains": ["trainer"],  # Competition knowledge only
                "confidence_boost": 0.05,  # Less reliable despite wealth
                "max_technical_depth": "high"
            }
        }
    
    def get_knowledge_for_npc(self, context: NPCKnowledgeContext) -> List[str]:
        """
        Retrieve and format relevant knowledge for an NPC
        
        Args:
            context: NPC context including role and topic
            
        Returns:
            List of formatted knowledge strings for prompt inclusion
        """
        role_config = self.role_config.get(context.role, self.role_config["stable_hand"])
        knowledge_items = []
        
        # Search primary domain first
        for domain in role_config["primary_domains"]:
            results = self.rag_system.retrieve(
                context.current_topic,
                domain_filter=domain,
                require_keywords=True
            )
            
            for result in results[:2]:  # Max 2 from primary domain
                if result.is_relevant:
                    formatted_knowledge = self._format_knowledge_for_npc(
                        result, context, is_primary_domain=True
                    )
                    knowledge_items.append(formatted_knowledge)
        
        # If we need more knowledge, search secondary domains
        if len(knowledge_items) < context.max_knowledge_items:
            remaining_slots = context.max_knowledge_items - len(knowledge_items)
            
            for domain in role_config["secondary_domains"]:
                if remaining_slots <= 0:
                    break
                    
                results = self.rag_system.retrieve(
                    context.current_topic,
                    domain_filter=domain,
                    require_keywords=True
                )
                
                for result in results[:remaining_slots]:
                    if result.is_relevant and result.relevance_score > 0.3:  # Higher threshold for secondary
                        formatted_knowledge = self._format_knowledge_for_npc(
                            result, context, is_primary_domain=False
                        )
                        knowledge_items.append(formatted_knowledge)
                        remaining_slots -= 1
        
        return knowledge_items[:context.max_knowledge_items]
    
    def _format_knowledge_for_npc(self, result: RetrievalResult, context: NPCKnowledgeContext, is_primary_domain: bool) -> str:
        """Format knowledge appropriately for the NPC's role and personality"""
        chunk = result.chunk
        role_config = self.role_config.get(context.role, self.role_config["stable_hand"])
        
        # Base knowledge content
        knowledge_text = chunk.content
        
        # Add confidence indicators based on role and domain
        confidence_prefix = ""
        if not is_primary_domain:
            confidence_prefix = "From what I understand, "
        elif chunk.confidence == "medium":
            confidence_prefix = "In my experience, "
        elif chunk.confidence == "high" and context.role in ["trainer", "behaviourist"]:
            confidence_prefix = "Based on established practice, "
        
        # Role-specific formatting
        if context.role == "stable_hand":
            # Practical, straightforward approach
            return f"{confidence_prefix}{knowledge_text}"
            
        elif context.role == "trainer":
            # Technical but accessible
            if chunk.confidence == "high":
                return f"From a training perspective, {knowledge_text}"
            else:
                return f"{confidence_prefix}{knowledge_text}"
                
        elif context.role == "behaviourist":
            # Scientific and analytical
            source_note = f" (from {chunk.source})" if chunk.source != "unknown" else ""
            return f"{confidence_prefix}{knowledge_text}{source_note}"
            
        elif context.role == "rival":
            # Dismissive or superior tone for non-premium topics
            if "expensive" in knowledge_text.lower() or "premium" in knowledge_text.lower():
                return f"Obviously, {knowledge_text}"
            else:
                return f"I suppose {knowledge_text.lower()}"
        
        return f"{confidence_prefix}{knowledge_text}"
    
    def should_use_rag_for_topic(self, topic: str, npc_role: str) -> bool:
        """
        Determine if RAG should be used for this topic and NPC role
        
        Args:
            topic: The topic being discussed
            npc_role: Role of the NPC
            
        Returns:
            True if RAG knowledge should be retrieved
        """
        role_config = self.role_config.get(npc_role, self.role_config["stable_hand"])
        
        # Check if any domain has knowledge about this topic
        for domain in role_config["primary_domains"] + role_config["secondary_domains"]:
            if self.rag_system.has_knowledge_about(topic, domain):
                return True
        
        return False
    
    def get_fallback_response(self, topic: str, npc_role: str, npc_name: str) -> str:
        """
        Get appropriate fallback response when RAG has no relevant knowledge
        
        Args:
            topic: Topic that was asked about
            npc_role: Role of the NPC
            npc_name: Name of the NPC
            
        Returns:
            Fallback response following .cursorrules guidelines
        """
        role_config = self.role_config.get(npc_role, self.role_config["stable_hand"])
        
        # Try to get domain expertise for suggestions
        expertise_areas = []
        for domain in role_config["primary_domains"]:
            expertise_areas.extend(self.rag_system.get_domain_expertise(domain))
        
        # Role-specific fallback responses
        if npc_role == "stable_hand":
            if expertise_areas:
                return f"I'm not sure about {topic}, but I can help with {', '.join(expertise_areas[:3])} and daily horse care."
            else:
                return f"I don't know about {topic} specifically. Maybe ask someone with more expertise?"
                
        elif npc_role == "trainer":
            if expertise_areas:
                return f"That's outside my expertise. I focus more on {', '.join(expertise_areas[:3])} and training techniques."
            else:
                return f"I don't have experience with {topic}. That might be better suited for a specialist."
                
        elif npc_role == "behaviourist":
            if expertise_areas:
                return f"I don't have research on {topic}. My expertise is in {', '.join(expertise_areas[:3])} and behavioral analysis."
            else:
                return f"I haven't studied {topic} specifically. It would need proper research to give you accurate information."
                
        elif npc_role == "rival":
            return f"I don't waste time on {topic}. I focus on the important things - like winning."
        
        return f"I don't have information about {topic} right now."
    
    def enhance_npc_prompt(self, base_prompt: str, context: NPCKnowledgeContext) -> str:
        """
        Enhance NPC prompt with relevant RAG knowledge
        
        Args:
            base_prompt: Original NPC prompt
            context: Knowledge retrieval context
            
        Returns:
            Enhanced prompt with knowledge context
        """
        # Check if RAG should be used
        if not self.should_use_rag_for_topic(context.current_topic, context.role):
            return base_prompt
        
        # Get relevant knowledge
        knowledge_items = self.get_knowledge_for_npc(context)
        
        if not knowledge_items:
            # No relevant knowledge found - add fallback instruction
            fallback_response = self.get_fallback_response(
                context.current_topic, context.role, context.npc_name
            )
            
            enhanced_prompt = f"""{base_prompt}

KNOWLEDGE STATUS: No specific knowledge about "{context.current_topic}" in your expertise area.
FALLBACK RESPONSE: {fallback_response}

Use the fallback response if asked about {context.current_topic} specifically."""
            
            return enhanced_prompt
        
        # Add knowledge to prompt
        knowledge_section = "\n".join([f"- {item}" for item in knowledge_items])
        
        enhanced_prompt = f"""{base_prompt}

RELEVANT KNOWLEDGE for "{context.current_topic}":
{knowledge_section}

Use this knowledge to inform your response, but maintain your character's personality and speaking style. Don't quote the knowledge directly - integrate it naturally into your response."""
        
        return enhanced_prompt
    
    def get_rag_stats(self) -> Dict:
        """Get RAG system statistics"""
        return self.rag_system.get_stats()

# Global NPC-RAG interface instance
_npc_rag_interface: Optional[NPCRAGInterface] = None

def get_npc_rag_interface() -> NPCRAGInterface:
    """Get or create global NPC-RAG interface instance"""
    global _npc_rag_interface
    
    if _npc_rag_interface is None:
        _npc_rag_interface = NPCRAGInterface()
    
    return _npc_rag_interface

if __name__ == "__main__":
    # Test the NPC-RAG integration
    logging.basicConfig(level=logging.INFO)
    
    interface = NPCRAGInterface()
    
    # Test different NPC roles with various topics
    test_contexts = [
        NPCKnowledgeContext(
            npc_name="Oskar",
            role="stable_hand",
            current_topic="horse feeding schedule",
            conversation_history=["Player asks about feeding times"]
        ),
        NPCKnowledgeContext(
            npc_name="Andy",
            role="trainer", 
            current_topic="saddle fitting",
            conversation_history=["Player asks about equipment"]
        ),
        NPCKnowledgeContext(
            npc_name="Elin",
            role="behaviourist",
            current_topic="horse stress signals", 
            conversation_history=["Player mentions horse seems nervous"]
        ),
        NPCKnowledgeContext(
            npc_name="Chris",
            role="rival",
            current_topic="expensive saddles",
            conversation_history=["Player asks about equipment quality"]
        )
    ]
    
    for context in test_contexts:
        print(f"\n=== Testing {context.npc_name} ({context.role}) on '{context.current_topic}' ===")
        
        # Test knowledge retrieval
        knowledge_items = interface.get_knowledge_for_npc(context)
        print(f"Knowledge items found: {len(knowledge_items)}")
        
        for i, item in enumerate(knowledge_items, 1):
            print(f"  {i}. {item}")
        
        # Test fallback if no knowledge
        if not knowledge_items:
            fallback = interface.get_fallback_response(context.current_topic, context.role, context.npc_name)
            print(f"Fallback response: {fallback}")
        
        # Test prompt enhancement
        base_prompt = f"You are {context.npc_name}, a {context.role}."
        enhanced_prompt = interface.enhance_npc_prompt(base_prompt, context)
        print(f"Enhanced prompt length: {len(enhanced_prompt)} characters") 