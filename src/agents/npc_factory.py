from .behaviourist import BehaviouristAgent
from .trainer import TrainerAgent  # Future
from .stable_hand import StableHandAgent  # Future

class NPCFactory:
    """Factory for creating NPCs from configuration files"""
    
    @staticmethod
    def create_npc(npc_config_name):
        """Create NPC from config file name (e.g., 'elin_behaviourist')"""
        
        # Determine agent type from config
        if 'behaviourist' in npc_config_name:
            return BehaviouristAgent(npc_config_name)
        elif 'trainer' in npc_config_name:
            return TrainerAgent(npc_config_name)
        elif 'stable_hand' in npc_config_name:
            return StableHandAgent(npc_config_name)
        else:
            raise ValueError(f"Unknown NPC type in config: {npc_config_name}")
    
    @staticmethod
    def create_behaviourist(name="Elin"):
        """Convenience method for creating behaviourist with custom name"""
        return BehaviouristAgent("elin_behaviourist")

# Usage:
# elin = NPCFactory.create_npc("elin_behaviourist") 
# eller
# elin = NPCFactory.create_behaviourist()