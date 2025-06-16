#!/usr/bin/env python3
"""
Location Coordinator - Original structure with dynamic locations
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from memory.session_memory import MemoryManager

@dataclass
class LocationInfo:
    """Information about a location"""
    name: str
    capacity: int
    noise_level: str  # quiet, moderate, loud
    typical_activities: List[str]
    connected_locations: List[str]
    description: str

class LocationCoordinator:
    """Manages NPC locations and proximity-based information sharing"""
    
    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager
        self.player_location = "stable"  # Default starting location
        self.locations = {}
        
        # Initialize default locations with activities
        self.initialize_default_locations()
    
    def initialize_default_locations(self):
        """Initialize default game locations"""
        default_locations = {
            "stable": {
                "name": "Main Stable",
                "noise_level": "moderate",
                "capacity": 8,
                "typical_activities": ["grooming", "feeding", "tacking up", "morning turnout", "evening stabling"],
                "connected_locations": ["paddock", "pasture"],
                "description": "The main stable building where horses are kept, groomed, and prepared for activities"
            },
            "paddock": {
                "name": "Training Paddock",
                "noise_level": "moderate",
                "capacity": 4,
                "typical_activities": ["training", "lunging", "exercise", "ground work", "cooling down"],
                "connected_locations": ["stable", "pasture"],
                "description": "A fenced training area for horse exercise and groundwork"
            },
            "pasture": {
                "name": "Grazing Pasture",
                "noise_level": "quiet",
                "capacity": 12,
                "typical_activities": ["grazing", "resting", "free exercise", "turnout", "socializing"],
                "connected_locations": ["stable", "paddock"],
                "description": "A large open field where horses can graze and socialize freely"
            }
        }
        
        self.locations = default_locations
    
    def register_location(self, location: str, properties: dict = None):
        """Register a new location with properties"""
        # Normalize the location name
        location = self.normalize_location_name(location)
        
        if location in self.locations:
            return
            
        if not properties:
            # Default properties if none provided
            properties = {
                "name": location.title(),
                "noise_level": "moderate",
                "capacity": 6,
                "typical_activities": ["general"],
                "connected_locations": [],
                "description": ""
            }
        
        self.locations[location] = type('LocationInfo', (), properties)
        
        if self.memory_manager:
            self.memory_manager.spatial_awareness.register_location(location)
    
    def move_player(self, new_location: str) -> Tuple[bool, str]:
        """Move player to new location with validation"""
        # Normalize the location name
        new_location = self.normalize_location_name(new_location)
        
        # Auto-register new locations
        self.register_location(new_location)
        
        old_location = self.player_location
        self.player_location = new_location
        
        # Record movement events for NPCs to witness
        if old_location != new_location and old_location != "unknown":
            self._record_player_movement(old_location, new_location)
        
        return True, f"Moved from {old_location} to {new_location}"
    
    def move_npc(self, npc_name: str, new_location: str, reason: str = None) -> bool:
        """Move NPC to new location with memory updates"""
        # Normalize the location name
        new_location = self.normalize_location_name(new_location)
        
        if new_location not in self.locations:
            return False
            
        # Get NPC's memory
        npc_memory = self.memory_manager.get_npc_memory(npc_name) if self.memory_manager else None
        if not npc_memory:
            return False
        
        old_location = npc_memory.current_location
        
        # Check if movement is valid (locations are connected)
        if old_location != "unknown" and old_location != new_location:
            if new_location not in self.locations[old_location]["connected_locations"]:
                # Can still move, but record it was an unusual movement
                reason = f"{reason or 'Moved'} (unusual path)" 
        
        # Update NPC's location memory
        npc_memory.update_location(new_location, reason)
        
        # Inform other NPCs at both locations about the movement
        if self.memory_manager:
            # NPCs at old location see them leave
            if old_location != "unknown":
                npcs_at_old = self.get_npcs_in_location(old_location)
                for witness in npcs_at_old:
                    if witness != npc_name:
                        witness_memory = self.memory_manager.get_npc_memory(witness)
                        if witness_memory:
                            witness_memory.add_witnessed_event(
                                f"{npc_name} left {old_location} and went to {new_location}",
                                old_location,
                                [witness],
                                ["movement", "npc_departure"]
                            )
            
            # NPCs at new location see them arrive
            npcs_at_new = self.get_npcs_in_location(new_location)
            for witness in npcs_at_new:
                if witness != npc_name:
                    witness_memory = self.memory_manager.get_npc_memory(witness)
                    if witness_memory:
                        witness_memory.add_witnessed_event(
                            f"{npc_name} arrived at {new_location}" + 
                            (f" from {old_location}" if old_location != "unknown" else ""),
                            new_location,
                            [witness],
                            ["movement", "npc_arrival"]
                        )
        
        return True
    
    def get_npcs_at_location(self, location: str, agents: Dict) -> List[str]:
        """Get all NPCs currently at a specific location"""
        return [name for name, agent in agents.items() 
                if getattr(agent, 'current_location', None) == location]
    
    def get_location_context(self, location: str) -> Dict[str, any]:
        """Get contextual information about a location"""
        # Auto-register if unknown
        self.register_location(location)
        
        if location not in self.locations:
            return {"error": f"Could not process location: {location}"}
        
        location_info = self.locations[location]
        
        return {
            "name": location_info.name,
            "noise_level": location_info.noise_level,
            "typical_activities": location_info.typical_activities,
            "capacity": location_info.capacity,
            "connected_locations": location_info.connected_locations,
            "description": self._get_location_description(location)
        }
    
    def _get_location_description(self, location: str) -> str:
        """Get description for location context in conversations"""
        if "stable" in location.lower():
            return f"The {location} area - a hub of daily activity and horse care."
        elif "pasture" in location.lower():
            return f"The {location} - open grazing fields where horses roam freely."
        elif "paddock" in location.lower():
            return f"The {location} - training and exercise area for active work."
        else:
            return f"The {location} area"
    
    def normalize_location_name(self, location: str) -> str:
        """Convert target location names to standard location names.
        
        Examples:
            Target_Stable -> stable
            Target_Pasture -> pasture
            Target_Paddock -> paddock
            Target_Arena -> arena
        """
        # If location is already normalized, return as is
        if location.lower() in self.locations:
            return location.lower()
            
        # Remove 'Target_' prefix if present and convert to lowercase
        normalized = location.lower()
        if normalized.startswith('target_'):
            normalized = normalized[7:]
            
        # If the normalized version exists in our locations, use that
        if normalized in self.locations:
            return normalized
            
        # If we don't recognize it, return the original
        return location
    
    def can_hear_across_locations(self, from_location: str, to_location: str) -> bool:
        """Check if sound carries between locations"""
        # Same location = always can hear
        if from_location == to_location:
            return True
        
        # Connected locations might hear each other
        if from_location in self.locations:
            from_info = self.locations[from_location]
            if to_location in from_info.connected_locations and from_info.noise_level == "loud":
                return True
        
        return False
    
    def suggest_location_for_activity(self, activity: str) -> Optional[str]:
        """Suggest best location for a specific activity"""
        activity_lower = activity.lower()
        
        for location_name, location_info in self.locations.items():
            for typical_activity in location_info.typical_activities:
                if activity_lower in typical_activity.lower():
                    return location_name
        
        # Default to first available location
        if self.locations:
            return list(self.locations.keys())[0]
        return None
    
    def get_movement_suggestions(self, current_location: str) -> List[Tuple[str, str]]:
        """Get movement suggestions from current location"""
        if current_location not in self.locations:
            return []
        
        location_info = self.locations[current_location]
        suggestions = []
        
        for connected_location in location_info.connected_locations:
            if connected_location in self.locations:
                connected_info = self.locations[connected_location]
                activities = ", ".join(connected_info.typical_activities[:2])
                reason = f"Go to {connected_location} for {activities}"
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
    
    def get_location_atmosphere(self, location: str, agents: Dict) -> Dict[str, Any]:
        """Get rich information about a location's current state"""
        if location not in self.locations:
            return {}
        
        location_info = self.locations[location].copy()
        npcs_here = []
        
        for name, agent in agents.items():
            if getattr(agent, 'current_location', None) == location:
                npc_memory = self.memory_manager.get_npc_memory(name) if self.memory_manager else None
                familiarity = npc_memory.get_location_familiarity(location) if npc_memory else {}
                
                npcs_here.append({
                    "name": name,
                    "role": getattr(agent, 'npc_role', 'unknown').value,
                    "title": getattr(agent, 'template_data', {}).get('title', ''),
                    "familiarity": familiarity
                })
        
        location_info["present_npcs"] = npcs_here
        return location_info
    
    def get_stats(self) -> Dict[str, any]:
        """Get location coordinator statistics"""
        return {
            "total_locations": len(self.locations),
            "player_location": self.player_location,
            "discovered_locations": list(self.locations.keys()),
            "location_details": {
                name: {
                    "capacity": info.capacity,
                    "noise_level": info.noise_level,
                    "activities": info.typical_activities[:3]
                } for name, info in self.locations.items()
            }
        }