#!/usr/bin/env python3
"""
Location Coordinator
===================

Manages spatial awareness, location-based interactions, and movement
coordination for the multi-agent NPC system.
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from memory.session_memory import MemoryManager


class LocationType(Enum):
    """Types of locations in the stable environment"""
    STABLE_YARD = "stable_yard"
    BARN = "barn"
    ARENA = "arena"
    PADDOCK = "paddock"
    TACK_ROOM = "tack_room"
    OFFICE = "office"


@dataclass
class LocationInfo:
    """Information about a location"""
    name: str
    location_type: LocationType
    capacity: int
    noise_level: str  # quiet, moderate, loud
    typical_activities: List[str]
    connected_locations: List[str]


class LocationCoordinator:
    """Manages location-based coordination and spatial awareness"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.player_location = "stable_yard"
        
        # Define location properties
        self.locations = {
            "stable_yard": LocationInfo(
                name="stable_yard",
                location_type=LocationType.STABLE_YARD,
                capacity=8,
                noise_level="moderate",
                typical_activities=["general_conversation", "equipment_prep", "horse_handling"],
                connected_locations=["barn", "arena", "tack_room", "office"]
            ),
            "barn": LocationInfo(
                name="barn",
                location_type=LocationType.BARN,
                capacity=6,
                noise_level="quiet",
                typical_activities=["feeding", "grooming", "health_checks", "quiet_work"],
                connected_locations=["stable_yard", "paddock"]
            ),
            "arena": LocationInfo(
                name="arena",
                location_type=LocationType.ARENA,
                capacity=4,
                noise_level="loud",
                typical_activities=["training", "riding", "lessons", "competition_prep"],
                connected_locations=["stable_yard", "paddock"]
            ),
            "paddock": LocationInfo(
                name="paddock",
                location_type=LocationType.PADDOCK,
                capacity=3,
                noise_level="quiet",
                typical_activities=["turnout", "observation", "quiet_discussion"],
                connected_locations=["barn", "arena"]
            ),
            "tack_room": LocationInfo(
                name="tack_room",
                location_type=LocationType.TACK_ROOM,
                capacity=3,
                noise_level="quiet",
                typical_activities=["equipment_maintenance", "private_discussion", "planning"],
                connected_locations=["stable_yard"]
            ),
            "office": LocationInfo(
                name="office",
                location_type=LocationType.OFFICE,
                capacity=2,
                noise_level="quiet",
                typical_activities=["administration", "private_meetings", "planning"],
                connected_locations=["stable_yard"]
            )
        }
    
    def move_player(self, new_location: str) -> Tuple[bool, str]:
        """Move player to new location with validation"""
        if new_location not in self.locations:
            return False, f"Unknown location: {new_location}"
        
        old_location = self.player_location
        self.player_location = new_location
        
        # Record movement events for NPCs to witness
        if old_location != new_location:
            self._record_player_movement(old_location, new_location)
        
        return True, f"Moved from {old_location} to {new_location}"
    
    def move_npc(self, npc_name: str, new_location: str, agents: Dict) -> Tuple[bool, str]:
        """Move NPC to new location"""
        if new_location not in self.locations:
            return False, f"Unknown location: {new_location}"
        
        if npc_name not in agents:
            return False, f"Unknown NPC: {npc_name}"
        
        agent = agents[npc_name]
        old_location = agent.current_location
        
        # Update agent location
        agent.move_to_location(new_location)
        
        return True, f"{npc_name} moved from {old_location} to {new_location}"
    
    def get_npcs_at_location(self, location: str, agents: Dict) -> List[str]:
        """Get all NPCs currently at a specific location"""
        return [name for name, agent in agents.items() 
                if getattr(agent, 'current_location', None) == location]
    
    def get_location_context(self, location: str) -> Dict[str, any]:
        """Get contextual information about a location"""
        if location not in self.locations:
            return {}
        
        location_info = self.locations[location]
        
        return {
            "name": location_info.name,
            "type": location_info.location_type.value,
            "noise_level": location_info.noise_level,
            "typical_activities": location_info.typical_activities,
            "capacity": location_info.capacity,
            "connected_locations": location_info.connected_locations
        }
    
    def can_hear_across_locations(self, from_location: str, to_location: str) -> bool:
        """Check if sound carries between locations"""
        # Adjacent locations with loud activity can sometimes be heard
        if from_location not in self.locations or to_location not in self.locations:
            return False
        
        from_info = self.locations[from_location]
        to_info = self.locations[to_location]
        
        # Same location = always can hear
        if from_location == to_location:
            return True
        
        # Adjacent locations with loud noise can be heard
        if (to_location in from_info.connected_locations and 
            from_info.noise_level == "loud"):
            return True
        
        return False
    
    def suggest_location_for_activity(self, activity: str) -> Optional[str]:
        """Suggest best location for a specific activity"""
        activity_lower = activity.lower()
        
        # Map activities to preferred locations
        activity_preferences = {
            "training": ["arena"],
            "riding": ["arena", "paddock"],
            "feeding": ["barn"],
            "grooming": ["barn", "stable_yard"],
            "private": ["office", "tack_room"],
            "meeting": ["office", "tack_room"],
            "equipment": ["tack_room", "stable_yard"],
            "quiet": ["paddock", "barn", "office"],
            "discussion": ["stable_yard", "office"]
        }
        
        for activity_key, preferred_locations in activity_preferences.items():
            if activity_key in activity_lower:
                return preferred_locations[0]
        
        return "stable_yard"  # Default location
    
    def get_movement_suggestions(self, current_location: str) -> List[Tuple[str, str]]:
        """Get movement suggestions from current location"""
        if current_location not in self.locations:
            return []
        
        location_info = self.locations[current_location]
        suggestions = []
        
        for connected_location in location_info.connected_locations:
            connected_info = self.locations[connected_location]
            reason = f"Go to {connected_location} for {', '.join(connected_info.typical_activities[:2])}"
            suggestions.append((connected_location, reason))
        
        return suggestions
    
    def check_location_capacity(self, location: str, agents: Dict) -> Tuple[int, int, bool]:
        """Check if location is near capacity"""
        if location not in self.locations:
            return 0, 0, True
        
        location_info = self.locations[location]
        current_occupancy = len(self.get_npcs_at_location(location, agents))
        
        # Add 1 for player if they're here
        if self.player_location == location:
            current_occupancy += 1
        
        is_available = current_occupancy < location_info.capacity
        
        return current_occupancy, location_info.capacity, is_available
    
    def _record_player_movement(self, from_location: str, to_location: str):
        """Record player movement for NPCs to witness"""
        # NPCs in old location see player leave
        old_location_npcs = self.memory_manager.spatial_awareness.get_npcs_in_location(from_location)
        if old_location_npcs:
            self.memory_manager.record_witnessed_event(
                f"Player left {from_location}",
                from_location,
                old_location_npcs,
                ["player_movement", "departure"]
            )
        
        # NPCs in new location see player arrive
        new_location_npcs = self.memory_manager.spatial_awareness.get_npcs_in_location(to_location)
        if new_location_npcs:
            self.memory_manager.record_witnessed_event(
                f"Player arrived at {to_location}",
                to_location,
                new_location_npcs,
                ["player_movement", "arrival"]
            )
    
    def get_location_atmosphere(self, location: str, agents: Dict) -> Dict[str, any]:
        """Get current atmosphere and social dynamics of a location"""
        npcs_here = self.get_npcs_at_location(location, agents)
        location_info = self.locations.get(location)
        
        if not location_info:
            return {}
        
        atmosphere = {
            "location": location,
            "npcs_present": npcs_here,
            "player_present": self.player_location == location,
            "occupancy": len(npcs_here) + (1 if self.player_location == location else 0),
            "capacity": location_info.capacity,
            "noise_level": location_info.noise_level,
            "typical_activities": location_info.typical_activities,
            "social_dynamics": self._analyze_social_dynamics(npcs_here, agents)
        }
        
        return atmosphere
    
    def _analyze_social_dynamics(self, npc_names: List[str], agents: Dict) -> Dict[str, any]:
        """Analyze social dynamics between NPCs in a location"""
        if len(npc_names) < 2:
            return {"type": "individual" if npc_names else "empty"}
        
        # Analyze relationships and personalities
        personalities = []
        relationships = {}
        
        for npc_name in npc_names:
            if npc_name in agents:
                agent = agents[npc_name]
                if hasattr(agent, 'npc_data'):
                    traits = agent.npc_data.get('personality', {}).get('traits', [])
                    personalities.extend(traits)
                    
                    npc_relationships = agent.npc_data.get('relationships', {})
                    for other_npc in npc_names:
                        if other_npc != npc_name and other_npc.lower() in npc_relationships:
                            relationships[f"{npc_name}-{other_npc}"] = npc_relationships[other_npc.lower()]
        
        # Determine social dynamics
        dynamics = {"type": "mixed"}
        
        if "competitive" in personalities and "rival" in personalities:
            dynamics["type"] = "tense"
        elif "empathetic" in personalities and "gentle" in personalities:
            dynamics["type"] = "supportive"
        elif "confident" in personalities or "assertive" in personalities:
            dynamics["type"] = "dynamic"
        else:
            dynamics["type"] = "casual"
        
        dynamics["personalities"] = list(set(personalities))
        dynamics["relationships"] = relationships
        
        return dynamics
    
    def get_stats(self) -> Dict[str, any]:
        """Get location coordinator statistics"""
        return {
            "total_locations": len(self.locations),
            "player_location": self.player_location,
            "location_types": [loc.value for loc in LocationType],
            "locations": {name: {
                "type": info.location_type.value,
                "capacity": info.capacity,
                "noise_level": info.noise_level
            } for name, info in self.locations.items()}
        } 