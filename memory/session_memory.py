#!/usr/bin/env python3
"""
Session Memory System for NPCs
Tracks what each NPC knows based on what they've witnessed, heard, or been told
Includes spatial awareness for realistic information propagation
"""

import time
import uuid
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum


class InformationSource(Enum):
    WITNESSED = "witnessed"          # NPC saw it happen
    HEARD_CONVERSATION = "heard"     # NPC overheard nearby conversation  
    TOLD_BY_PLAYER = "player_told"   # Player directly told NPC
    TOLD_BY_NPC = "npc_told"         # Another NPC told this NPC
    SECONDHAND = "secondhand"        # Heard through chain of NPCs

class ConfidenceLevel(Enum):
    CERTAIN = "certain"              # Witnessed firsthand
    CONFIDENT = "confident"          # Reliable source
    UNCERTAIN = "uncertain"          # Secondhand or player claim
    DOUBTFUL = "doubtful"           # Third-hand information

@dataclass
class MemoryEvent:
    """Represents something an NPC knows or remembers"""
    id: str
    content: str                     # What happened/was said
    timestamp: float
    location: str                    # Where it happened
    source: InformationSource        # How NPC learned about it
    confidence: ConfidenceLevel      # How sure NPC is
    source_npc: Optional[str] = None # Who told them (if applicable)
    witnesses: List[str] = None      # Who else was present
    tags: List[str] = None           # Categorization tags
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()
        if self.witnesses is None:
            self.witnesses = []
        if self.tags is None:
            self.tags = []

class SpatialAwareness:
    """Manages NPC locations and proximity-based information sharing"""
    
    def __init__(self):
        self.npc_locations: Dict[str, str] = {}
        self.location_zones = {
            "stable_yard": {"conversation_range": 15, "visual_range": 30},
            "arena": {"conversation_range": 20, "visual_range": 50},
            "barn": {"conversation_range": 10, "visual_range": 20},
            "paddock": {"conversation_range": 25, "visual_range": 40},
            "tack_room": {"conversation_range": 8, "visual_range": 15},
            "office": {"conversation_range": 6, "visual_range": 12}
        }
    
    def update_npc_location(self, npc_name: str, location: str):
        """Update NPC's current location"""
        self.npc_locations[npc_name] = location
    
    def get_npcs_in_location(self, location: str) -> List[str]:
        """Get all NPCs currently in a specific location"""
        return [npc for npc, loc in self.npc_locations.items() if loc == location]
    
    def can_overhear_conversation(self, speaker: str, listener: str) -> bool:
        """Check if listener can overhear speaker's conversation"""
        speaker_loc = self.npc_locations.get(speaker)
        listener_loc = self.npc_locations.get(listener)
        
        if not speaker_loc or not listener_loc or speaker_loc != listener_loc:
            return False
        
        # In same location - can overhear within conversation range
        zone_info = self.location_zones.get(speaker_loc, {"conversation_range": 10})
        return True  # Simplified - in real game would calculate actual distance
    
    def can_witness_event(self, npc_name: str, event_location: str) -> bool:
        """Check if NPC can witness an event at given location"""
        npc_location = self.npc_locations.get(npc_name)
        return npc_location == event_location

class SessionMemory:
    """Manages what an individual NPC remembers during a session"""
    
    def __init__(self, npc_name: str):
        self.npc_name = npc_name
        self.memories: List[MemoryEvent] = []
        self.known_facts: Set[str] = set()  # Quick lookup for what NPC knows
        
    def add_witnessed_event(self, content: str, location: str, witnesses: List[str] = None, tags: List[str] = None):
        """Add something the NPC directly witnessed"""
        event = MemoryEvent(
            id=str(uuid.uuid4()),
            content=content,
            timestamp=time.time(),
            location=location,
            source=InformationSource.WITNESSED,
            confidence=ConfidenceLevel.CERTAIN,
            witnesses=witnesses or [],
            tags=tags or []
        )
        self.memories.append(event)
        self.known_facts.add(content.lower())
        return event
    
    def add_overheard_info(self, content: str, location: str, source_npc: str, tags: List[str] = None):
        """Add something the NPC overheard from another conversation"""
        event = MemoryEvent(
            id=str(uuid.uuid4()),
            content=content,
            timestamp=time.time(),
            location=location,
            source=InformationSource.HEARD_CONVERSATION,
            confidence=ConfidenceLevel.CONFIDENT,
            source_npc=source_npc,
            tags=tags or []
        )
        self.memories.append(event)
        self.known_facts.add(content.lower())
        return event
    
    def add_player_information(self, content: str, location: str, tags: List[str] = None):
        """Add information told directly by the player"""
        event = MemoryEvent(
            id=str(uuid.uuid4()),
            content=content,
            timestamp=time.time(),
            location=location,
            source=InformationSource.TOLD_BY_PLAYER,
            confidence=ConfidenceLevel.UNCERTAIN,  # Player claims need verification
            tags=tags or []
        )
        self.memories.append(event)
        self.known_facts.add(content.lower())
        return event
    
    def add_npc_information(self, content: str, location: str, source_npc: str, original_confidence: ConfidenceLevel, tags: List[str] = None):
        """Add information told by another NPC"""
        # Confidence degrades when passed between NPCs
        confidence_degradation = {
            ConfidenceLevel.CERTAIN: ConfidenceLevel.CONFIDENT,
            ConfidenceLevel.CONFIDENT: ConfidenceLevel.UNCERTAIN,
            ConfidenceLevel.UNCERTAIN: ConfidenceLevel.DOUBTFUL,
            ConfidenceLevel.DOUBTFUL: ConfidenceLevel.DOUBTFUL
        }
        
        event = MemoryEvent(
            id=str(uuid.uuid4()),
            content=content,
            timestamp=time.time(),
            location=location,
            source=InformationSource.TOLD_BY_NPC,
            confidence=confidence_degradation.get(original_confidence, ConfidenceLevel.DOUBTFUL),
            source_npc=source_npc,
            tags=tags or []
        )
        self.memories.append(event)
        self.known_facts.add(content.lower())
        return event
    
    def knows_about(self, topic: str) -> bool:
        """Check if NPC knows anything about a topic"""
        topic_lower = topic.lower()
        return any(topic_lower in fact for fact in self.known_facts)
    
    def get_memories_about(self, topic: str, max_memories: int = 3) -> List[MemoryEvent]:
        """Get relevant memories about a specific topic"""
        topic_lower = topic.lower()
        relevant_memories = []
        
        for memory in reversed(self.memories):  # Most recent first
            if (topic_lower in memory.content.lower() or 
                any(topic_lower in tag.lower() for tag in memory.tags)):
                relevant_memories.append(memory)
                if len(relevant_memories) >= max_memories:
                    break
        
        return relevant_memories
    
    def get_recent_memories(self, hours: float = 24.0, max_memories: int = 5) -> List[MemoryEvent]:
        """Get recent memories within specified timeframe"""
        cutoff_time = time.time() - (hours * 3600)
        recent_memories = [m for m in self.memories if m.timestamp >= cutoff_time]
        return sorted(recent_memories, key=lambda x: x.timestamp, reverse=True)[:max_memories]
    
    def verify_information(self, claim: str, required_confidence: ConfidenceLevel = ConfidenceLevel.CONFIDENT) -> bool:
        """Check if NPC can verify a claim with sufficient confidence"""
        claim_lower = claim.lower()
        for memory in self.memories:
            if (claim_lower in memory.content.lower() and 
                memory.confidence.value in [ConfidenceLevel.CERTAIN.value, ConfidenceLevel.CONFIDENT.value]):
                return True
        return False
    
    def get_memory_summary(self) -> Dict:
        """Get summary of what NPC remembers"""
        return {
            "total_memories": len(self.memories),
            "witnessed_events": len([m for m in self.memories if m.source == InformationSource.WITNESSED]),
            "player_told": len([m for m in self.memories if m.source == InformationSource.TOLD_BY_PLAYER]),
            "npc_told": len([m for m in self.memories if m.source == InformationSource.TOLD_BY_NPC]),
            "overheard": len([m for m in self.memories if m.source == InformationSource.HEARD_CONVERSATION]),
            "recent_memories": len(self.get_recent_memories(hours=1.0))
        }

class MemoryManager:
    """Manages session memory for all NPCs and handles information propagation"""
    
    def __init__(self):
        self.npc_memories: Dict[str, SessionMemory] = {}
        self.spatial_awareness = SpatialAwareness()
        self.global_events: List[MemoryEvent] = []  # Events that happened in the world
    
    def register_npc(self, npc_name: str, initial_location: str = "stable_yard"):
        """Register an NPC in the memory system"""
        self.npc_memories[npc_name] = SessionMemory(npc_name)
        self.spatial_awareness.update_npc_location(npc_name, initial_location)
    
    def update_npc_location(self, npc_name: str, new_location: str):
        """Update NPC location for spatial awareness"""
        self.spatial_awareness.update_npc_location(npc_name, new_location)
    
    def record_witnessed_event(self, event_content: str, location: str, witnesses: List[str], tags: List[str] = None):
        """Record an event that was witnessed by specific NPCs"""
        global_event = MemoryEvent(
            id=str(uuid.uuid4()),
            content=event_content,
            timestamp=time.time(),
            location=location,
            source=InformationSource.WITNESSED,
            confidence=ConfidenceLevel.CERTAIN,
            witnesses=witnesses,
            tags=tags or []
        )
        self.global_events.append(global_event)
        
        # Add to witnesses' memories
        for npc_name in witnesses:
            if npc_name in self.npc_memories:
                self.npc_memories[npc_name].add_witnessed_event(
                    event_content, location, witnesses, tags
                )
    
    def player_tells_npc(self, npc_name: str, information: str, location: str, tags: List[str] = None):
        """Record player telling something to an NPC"""
        if npc_name in self.npc_memories:
            event = self.npc_memories[npc_name].add_player_information(information, location, tags)
            
            # Check if other NPCs can overhear
            nearby_npcs = self.spatial_awareness.get_npcs_in_location(location)
            for other_npc in nearby_npcs:
                if (other_npc != npc_name and 
                    other_npc in self.npc_memories and
                    self.spatial_awareness.can_overhear_conversation("player", other_npc)):
                    self.npc_memories[other_npc].add_overheard_info(
                        f"Player told {npc_name}: {information}", location, "player", tags
                    )
            return event
        return None
    
    def npc_tells_npc(self, speaker_npc: str, listener_npc: str, information: str, location: str, 
                      original_confidence: ConfidenceLevel = ConfidenceLevel.CONFIDENT, tags: List[str] = None):
        """Record one NPC telling another NPC something"""
        if listener_npc in self.npc_memories:
            event = self.npc_memories[listener_npc].add_npc_information(
                information, location, speaker_npc, original_confidence, tags
            )
            
            # Check if other NPCs can overhear
            nearby_npcs = self.spatial_awareness.get_npcs_in_location(location)
            for other_npc in nearby_npcs:
                if (other_npc not in [speaker_npc, listener_npc] and 
                    other_npc in self.npc_memories and
                    self.spatial_awareness.can_overhear_conversation(speaker_npc, other_npc)):
                    self.npc_memories[other_npc].add_overheard_info(
                        f"{speaker_npc} told {listener_npc}: {information}", location, speaker_npc, tags
                    )
            return event
        return None
    
    def get_npc_memory(self, npc_name: str) -> Optional[SessionMemory]:
        """Get memory system for specific NPC"""
        return self.npc_memories.get(npc_name)
    
    def get_relevant_context_for_npc(self, npc_name: str, current_topic: str, max_memories: int = 3) -> List[str]:
        """Get relevant memory context for NPC's response generation"""
        npc_memory = self.npc_memories.get(npc_name)
        if not npc_memory:
            return []
        
        relevant_memories = npc_memory.get_memories_about(current_topic, max_memories)
        context_strings = []
        
        for memory in relevant_memories:
            confidence_text = ""
            if memory.confidence == ConfidenceLevel.UNCERTAIN:
                confidence_text = " (though I'm not completely sure)"
            elif memory.confidence == ConfidenceLevel.DOUBTFUL:
                confidence_text = " (but I heard this secondhand)"
            
            source_text = ""
            if memory.source == InformationSource.TOLD_BY_PLAYER:
                source_text = "You told me that "
            elif memory.source == InformationSource.TOLD_BY_NPC:
                source_text = f"{memory.source_npc} mentioned that "
            elif memory.source == InformationSource.WITNESSED:
                source_text = "I saw that "
            elif memory.source == InformationSource.HEARD_CONVERSATION:
                source_text = "I overheard that "
            
            context_strings.append(f"{source_text}{memory.content}{confidence_text}")
        
        return context_strings
    
    def reset_session(self):
        """Clear all session memories (new game session)"""
        for npc_memory in self.npc_memories.values():
            npc_memory.memories.clear()
            npc_memory.known_facts.clear()
        self.global_events.clear()
    
    def get_system_stats(self) -> Dict:
        """Get statistics about the memory system"""
        stats = {
            "registered_npcs": len(self.npc_memories),
            "global_events": len(self.global_events),
            "npc_stats": {}
        }
        
        for npc_name, memory in self.npc_memories.items():
            stats["npc_stats"][npc_name] = memory.get_memory_summary()
        
        return stats

# Example usage for testing
if __name__ == "__main__":
    # Initialize memory system
    memory_manager = MemoryManager()
    
    # Register NPCs
    memory_manager.register_npc("Elin", "barn")
    memory_manager.register_npc("Oskar", "stable_yard") 
    memory_manager.register_npc("Astrid", "barn")
    
    # Record witnessed event
    memory_manager.record_witnessed_event(
        "Thunder was acting nervous during grooming", 
        "barn", 
        ["Elin", "Astrid"],
        ["horse_behavior", "thunder"]
    )
    
    # Player tells Oskar something
    memory_manager.player_tells_npc(
        "Oskar", 
        "I think Thunder needs extra hay tonight", 
        "stable_yard",
        ["feeding", "thunder"]
    )
    
    # Check what Elin remembers about Thunder
    elin_memory = memory_manager.get_npc_memory("Elin")
    if elin_memory:
        thunder_memories = elin_memory.get_memories_about("thunder")
        print(f"Elin remembers {len(thunder_memories)} things about Thunder")
        for memory in thunder_memories:
            print(f"- {memory.content} (confidence: {memory.confidence.value})")
    
    # Get system stats
    stats = memory_manager.get_system_stats()
    print(f"System stats: {stats}")