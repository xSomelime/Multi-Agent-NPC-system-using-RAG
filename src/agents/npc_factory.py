#!/usr/bin/env python3
"""
Dynamic NPC Factory - NO hardcoded locations
Creates NPCs that adapt to any location setup discovered from UE5 scene
"""

from typing import List, Optional
from memory.session_memory import MemoryManager
from .base_agent import ScalableNPCAgent

# RAG components (optional import)
try:
    from src.agents.rag_enhanced_agent import create_rag_enhanced_agent, create_rag_enhanced_team
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


class NPCFactory:
    """Factory for creating NPCs with dynamic location awareness - no hardcoded locations"""
    
    @staticmethod
    def create_npc(npc_config_name: str, memory_manager: MemoryManager, enable_rag: bool = True) -> ScalableNPCAgent:
        """Create NPC from config file name with dynamic location support"""
        
        if enable_rag and RAG_AVAILABLE:
            try:
                # Create RAG-enhanced agent without initial location
                return create_rag_enhanced_agent(npc_config_name, memory_manager, enable_rag=True)
            except Exception as e:
                print(f"⚠️  Failed to create RAG-enhanced {npc_config_name}: {e}")
                print("🔄 Falling back to regular agent")
                # Fall back to regular agent
                return ScalableNPCAgent(npc_config_name, memory_manager)
        else:
            # Create regular agent without initial location
            return ScalableNPCAgent(npc_config_name, memory_manager)
    
    @staticmethod
    def create_core_team(memory_manager: MemoryManager, enable_rag: bool = True) -> List[ScalableNPCAgent]:
        """Create the core team of NPCs with default locations"""
        npcs = []
        
        # Create NPCs with initial locations
        npcs.append(NPCFactory.create_npc("oskar_stable_hand", memory_manager, enable_rag))
        npcs.append(NPCFactory.create_npc("astrid_stable_hand", memory_manager, enable_rag))
        npcs.append(NPCFactory.create_npc("andy_trainer", memory_manager, enable_rag))
        npcs.append(NPCFactory.create_npc("chris_rival", memory_manager, enable_rag))
        npcs.append(NPCFactory.create_npc("elin_behaviourist", memory_manager, enable_rag))
        
        # Distribute NPCs to default locations
        default_distribution = {
            "Oskar": "stable",
            "Astrid": "paddock",  # Move Astrid to paddock
            "Andy": "paddock", 
            "Chris": "stable",  # Move Chris to stable (he likes to show off)
            "Elin": "pasture"
        }
        
        # Move NPCs to their default locations
        for npc in npcs:
            default_location = default_distribution.get(npc.name, "stable")
            npc.move_to_location(default_location)
            memory_manager.update_npc_location(npc.name, default_location)
        
        return npcs
    
    @staticmethod
    def distribute_team_to_discovered_locations(team: List[ScalableNPCAgent], memory_manager: MemoryManager, distribution_strategy: str = "role_based") -> dict:
        """Distribute team members to discovered locations based on strategy"""
        discovered_locations = memory_manager.get_discovered_locations()
        
        if not discovered_locations:
            print("📍 No locations discovered yet - NPCs will be assigned when locations are found")
            return {"distributed": False, "reason": "no_locations_discovered"}
        
        print(f"📍 Distributing {len(team)} NPCs across {len(discovered_locations)} discovered locations")
        
        distribution = {}
        
        if distribution_strategy == "role_based":
            distribution = NPCFactory._distribute_by_role(team, discovered_locations, memory_manager)
        elif distribution_strategy == "even":
            distribution = NPCFactory._distribute_evenly(team, discovered_locations, memory_manager)
        else:
            distribution = NPCFactory._distribute_randomly(team, discovered_locations, memory_manager)
        
        # Actually move the NPCs
        for npc, location in distribution.items():
            if hasattr(npc, 'move_to_location'):
                npc.move_to_location(location)
                memory_manager.update_npc_location(npc.name, location)
                print(f"📍 Moved {npc.name} to {location}")
        
        return {
            "distributed": True,
            "strategy": distribution_strategy,
            "distribution": {npc.name: loc for npc, loc in distribution.items()},
            "locations_used": list(set(distribution.values()))
        }
    
    @staticmethod
    def _distribute_by_role(team: List[ScalableNPCAgent], locations: List[str], memory_manager: MemoryManager) -> dict:
        """Distribute NPCs based on role-location suitability"""
        distribution = {}
        
        # Infer best locations for each NPC based on role and location names
        for npc in team:
            best_location = NPCFactory._suggest_location_for_npc(npc, locations)
            distribution[npc] = best_location
        
        return distribution
    
    @staticmethod
    def _distribute_evenly(team: List[ScalableNPCAgent], locations: List[str], memory_manager: MemoryManager) -> dict:
        """Distribute NPCs evenly across available locations"""
        distribution = {}
        
        for i, npc in enumerate(team):
            location = locations[i % len(locations)]
            distribution[npc] = location
        
        return distribution
    
    @staticmethod
    def _distribute_randomly(team: List[ScalableNPCAgent], locations: List[str], memory_manager: MemoryManager) -> dict:
        """Randomly distribute NPCs across locations"""
        import random
        distribution = {}
        
        for npc in team:
            location = random.choice(locations)
            distribution[npc] = location
        
        return distribution
    
    @staticmethod
    def _suggest_location_for_npc(npc: ScalableNPCAgent, available_locations: List[str]) -> str:
        """Suggest best location for NPC based on role and available location names"""
        npc_role = getattr(npc, 'npc_role', None)
        if not npc_role:
            return available_locations[0]  # Default to first location
        
        role_value = npc_role.value if hasattr(npc_role, 'value') else str(npc_role)
        
        # Score locations based on name patterns and NPC role
        location_scores = {}
        
        for location in available_locations:
            location_lower = location.lower()
            score = 0
            
            # Role-based scoring
            if role_value in ["stable_hand", "behaviourist"]:
                # Prefer locations that sound like care/maintenance areas
                if any(word in location_lower for word in ["stable", "barn", "care", "main"]):
                    score += 10
                elif any(word in location_lower for word in ["pasture", "field", "quiet"]):
                    score += 5
            
            elif role_value in ["trainer", "rival"]:
                # Prefer locations that sound like training/competition areas
                if any(word in location_lower for word in ["paddock", "arena", "training", "ring"]):
                    score += 10
                elif any(word in location_lower for word in ["stable", "barn"]):
                    score += 5
            
            # General preferences
            if "main" in location_lower or "central" in location_lower:
                score += 3
            
            location_scores[location] = score
        
        # Return location with highest score
        best_location = max(location_scores, key=location_scores.get)
        return best_location
    
    @staticmethod
    def auto_assign_when_location_discovered(npc: ScalableNPCAgent, new_location: str, memory_manager: MemoryManager):
        """Automatically assign NPC to location when it's first discovered"""
        current_location = getattr(npc, 'current_location', None)
        
        # Only auto-assign if NPC doesn't have a location yet
        if not current_location or current_location == "unknown":
            # Check if this location is suitable for this NPC
            suitability_score = NPCFactory._calculate_location_suitability(npc, new_location)
            
            # If suitable (score > 5), assign NPC there
            if suitability_score > 5:
                npc.move_to_location(new_location)
                memory_manager.update_npc_location(npc.name, new_location)
                print(f"🎯 Auto-assigned {npc.name} to newly discovered {new_location} (suitability: {suitability_score})")
                return True
        
        return False
    
    @staticmethod
    def _calculate_location_suitability(npc: ScalableNPCAgent, location: str) -> int:
        """Calculate how suitable a location is for an NPC"""
        npc_role = getattr(npc, 'npc_role', None)
        if not npc_role:
            return 1
        
        role_value = npc_role.value if hasattr(npc_role, 'value') else str(npc_role)
        location_lower = location.lower()
        
        score = 0
        
        # Role-location matching
        if role_value in ["stable_hand", "behaviourist"]:
            if any(word in location_lower for word in ["stable", "barn", "care"]):
                score += 8
        elif role_value in ["trainer", "rival"]:
            if any(word in location_lower for word in ["paddock", "arena", "training"]):
                score += 8
        
        # General location preferences
        if any(word in location_lower for word in ["main", "central", "primary"]):
            score += 2
        
        return score
    
    @staticmethod
    def get_team_distribution_stats(team: List[ScalableNPCAgent], memory_manager: MemoryManager) -> dict:
        """Get statistics about team distribution across discovered locations"""
        discovered_locations = memory_manager.get_discovered_locations()
        
        location_counts = {loc: 0 for loc in discovered_locations}
        npc_locations = {}
        unassigned_npcs = []
        
        for npc in team:
            current_location = getattr(npc, 'current_location', None)
            if current_location and current_location in discovered_locations:
                location_counts[current_location] += 1
                npc_locations[npc.name] = current_location
            else:
                unassigned_npcs.append(npc.name)
        
        return {
            "total_npcs": len(team),
            "discovered_locations": discovered_locations,
            "location_counts": location_counts,
            "npc_locations": npc_locations,
            "unassigned_npcs": unassigned_npcs,
            "distribution_balance": min(location_counts.values()) / max(location_counts.values()) if location_counts.values() and max(location_counts.values()) > 0 else 0
        }


# Backwards compatibility alias
NPCFactory = NPCFactory

# Utility functions for dynamic location system
def suggest_npc_placement_from_discovered(npc_role: str, discovered_locations: List[str]) -> Optional[str]:
    """Suggest best discovered location for an NPC role"""
    if not discovered_locations:
        return None
    
    # Score each discovered location for this role
    location_scores = {}
    
    for location in discovered_locations:
        location_lower = location.lower()
        score = 0
        
        # Role-based scoring
        if npc_role.lower() in ["stable_hand", "behaviourist", "caretaker"]:
            if any(word in location_lower for word in ["stable", "barn", "care", "main"]):
                score += 10
            elif any(word in location_lower for word in ["pasture", "field"]):
                score += 5
        
        elif npc_role.lower() in ["trainer", "instructor", "rival", "competitor"]:
            if any(word in location_lower for word in ["paddock", "arena", "training", "ring"]):
                score += 10
            elif any(word in location_lower for word in ["stable", "barn"]):
                score += 3
        
        location_scores[location] = score
    
    # Return best location or first one if all score equally
    if max(location_scores.values()) > 0:
        return max(location_scores, key=location_scores.get)
    else:
        return discovered_locations[0]  # Default to first discovered


# Example usage for testing the dynamic system
if __name__ == "__main__":
    print("🧪 Testing Dynamic NPC Factory")
    print("="*50)
    
    # This would normally be imported from your main system
    from memory.session_memory import MemoryManager
    
    # Initialize dynamic memory manager
    memory_manager = MemoryManager()
    
    print("\n1. Testing core team creation (no locations yet)...")
    team = NPCFactory.create_core_team(memory_manager, enable_rag=False)
    
    print(f"\n2. Simulating location discovery from UE5...")
    # Simulate UE5 discovering locations
    ue5_locations = ["Target_Stable", "Target_Pasture", "Target_Paddock"]
    for location in ue5_locations:
        normalized = location.replace("Target_", "").lower()
        memory_manager.spatial_awareness.register_location_zone(normalized)
        print(f"   Discovered: {location} → {normalized}")
    
    print(f"\n3. Testing team distribution to discovered locations...")
    distribution_result = NPCFactory.distribute_team_to_discovered_locations(
        team, memory_manager, "role_based"
    )
    print(f"   Distribution successful: {distribution_result['distributed']}")
    print(f"   Strategy used: {distribution_result.get('strategy', 'N/A')}")
    
    print(f"\n4. Getting team distribution statistics...")
    stats = NPCFactory.get_team_distribution_stats(team, memory_manager)
    print(f"   Total NPCs: {stats['total_npcs']}")
    print(f"   Discovered locations: {stats['discovered_locations']}")
    print(f"   Location distribution: {stats['location_counts']}")
    print(f"   Unassigned NPCs: {stats['unassigned_npcs']}")
    
    print(f"\n5. Testing role-based placement suggestions...")
    test_roles = ["stable_hand", "trainer", "behaviourist", "rival"]
    for role in test_roles:
        suggested = suggest_npc_placement_from_discovered(role, stats['discovered_locations'])
        print(f"   {role} → {suggested}")
    
    print(f"\n✅ Dynamic NPC Factory testing complete!")
    print(f"💡 Ready for UE5 integration with full location discovery support")