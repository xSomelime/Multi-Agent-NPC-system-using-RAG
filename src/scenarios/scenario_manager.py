from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set
import json
from pathlib import Path
import logging
import random

from memory.session_memory import MemoryManager
from src.coordination.location_coordinator import LocationCoordinator
from .scenario_state import ScenarioStateTracker, ConflictState, ObjectiveState, SentimentLevel

logger = logging.getLogger(__name__)

class ScenarioState(Enum):
    INACTIVE = "inactive"
    TRIGGERED = "triggered"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"

class ScenarioPhase(Enum):
    SETUP = "setup"
    INTRODUCTION = "introduction"
    DISCUSSION = "discussion"
    RESOLUTION = "resolution"
    CONCLUSION = "conclusion"

@dataclass
class ScenarioContext:
    scenario_id: str
    current_phase: ScenarioPhase
    active_npcs: Set[str]
    location: str
    turn_count: int = 0
    state_tracker: Optional[ScenarioStateTracker] = None

class ScenarioManager:
    def __init__(
        self,
        memory_manager: MemoryManager,
        location_coordinator: LocationCoordinator
    ):
        self.memory_manager = memory_manager
        self.location_coordinator = location_coordinator
        
        # Load scenario definitions
        self.scenarios = self._load_scenario_definitions()
        
        # Active scenario tracking
        self.active_scenario: Optional[ScenarioContext] = None
        self.scenario_states: Dict[str, ScenarioState] = {
            scenario_id: ScenarioState.INACTIVE 
            for scenario_id in self.scenarios.keys()
        }
        
        # Scenario-specific state tracking
        self.scenario_contexts: Dict[str, Dict] = {}

    def _load_scenario_definitions(self) -> Dict:
        """Load scenario definitions from JSON file."""
        scenario_path = Path("data/scenarios/scenario_definitions.json")
        with open(scenario_path, 'r', encoding='utf-8') as f:
            return json.load(f)["scenarios"]

    def check_location_triggers(self, location: str) -> Optional[str]:
        """Check if current location triggers any scenarios."""
        for scenario_id, scenario in self.scenarios.items():
            if location in scenario.get("location_triggers", {}):
                trigger_prob = scenario["location_triggers"][location]
                if random.random() < trigger_prob:
                    return scenario_id
        return None

    def start_scenario(self, scenario_id: str, location: str, available_npcs: Dict) -> tuple[bool, str]:
        """Start a scenario if all requirements are met."""
        if scenario_id not in self.scenarios:
            return False, f"Scenario '{scenario_id}' not found"

        scenario = self.scenarios[scenario_id]
        required_npcs = set(scenario.get("required_npcs", []))
        available_npc_names = set(available_npcs.keys())
        
        # Check if we have all required NPCs
        if not required_npcs.issubset(available_npc_names):
            missing = required_npcs - available_npc_names
            return False, f"Missing required NPCs: {', '.join(missing)}"

        # Initialize scenario context
        self.active_scenario = ScenarioContext(
            scenario_id=scenario_id,
            current_phase=ScenarioPhase.SETUP,
            active_npcs=required_npcs,
            location=location,
            state_tracker=ScenarioStateTracker()
        )

        # Add scenario memories to NPCs
        self._inject_scenario_memories(scenario, available_npcs)
        
        logger.info(f"Started scenario '{scenario['title']}' with {len(required_npcs)} NPCs")
        return True, f"Started scenario: {scenario['title']}"
    
    def _inject_scenario_memories(self, scenario: Dict, available_npcs: Dict):
        """Inject natural 'what's happening today' awareness"""
        scenario_id = self.active_scenario.scenario_id
        
        # Create natural "today's events" memories based on scenario
        todays_events = {
            "winter_turnout_discussion": "We need to plan the winter turnout schedule today",
            "hay_delivery_coordination": "The hay delivery is arriving today",
            "pasture_rotation_planning": "We're planning the pasture rotation today", 
            "pre_competition_preparation": "There's a competition coming up that we need to prepare for",
            "post_competition_debrief": "We just got back from yesterday's competition",
            "new_horse_integration": "There's a new horse arriving today"
        }
        
        event_memory = todays_events.get(scenario_id, "Something important is happening today")
        
        for npc_name in self.active_scenario.active_npcs:
            if npc_name in available_npcs:
                npc_memory = self.memory_manager.get_npc_memory(npc_name)
                if npc_memory:
                    # Just add one simple "what's happening today" memory
                    npc_memory.record_memory(
                        content=event_memory,
                        location=self.active_scenario.location,
                        tags=["daily_events", "today"]
                    )
    
    def list_available_scenarios(self) -> List[Dict]:
        """List all available scenarios"""
        return [
            {
                "scenario_id": scenario_id,
                "title": scenario_data["title"],
                "description": scenario_data["description"],
                "duration": scenario_data["duration"],
                "difficulty": scenario_data["difficulty"],
                "required_npcs": scenario_data.get("required_npcs", []),
                "optional_npcs": scenario_data.get("optional_npcs", [])
            }
            for scenario_id, scenario_data in self.scenarios.items()
        ]
    
    def check_scenario_triggers(self, message: str, location: str, available_npcs: List[str]) -> Optional[str]:
        """Check if a message should trigger a scenario"""
        message_lower = message.lower()
        
        # Simple keyword-based triggering
        trigger_keywords = {
            "winter_turnout_discussion": ["winter", "turnout", "schedule", "cold weather"],
            "hay_delivery_coordination": ["hay", "delivery", "feed", "storage"],
            "pasture_rotation_planning": ["pasture", "rotation", "grazing", "grass"],
            "pre_competition_preparation": ["competition", "prepare", "ready", "event"],
            "post_competition_debrief": ["competition results", "how did", "performance"],
            "new_horse_integration": ["new horse", "arrival", "integration", "introduce"]
        }
        
        for scenario_id, keywords in trigger_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                scenario = self.scenarios.get(scenario_id)
                if scenario:
                    required_npcs = set(scenario.get("required_npcs", []))
                    if required_npcs.issubset(set(available_npcs)):
                        return scenario_id
        
        return None
    
    def get_current_scenario(self) -> Optional[Dict]:
        """Get current active scenario info"""
        if not self.active_scenario:
            return None
            
        scenario_data = self.scenarios[self.active_scenario.scenario_id]
        return {
            "scenario_id": self.active_scenario.scenario_id,
            "title": scenario_data["title"],
            "description": scenario_data["description"],
            "current_phase": self.active_scenario.current_phase.value,
            "active_npcs": list(self.active_scenario.active_npcs),
            "location": self.active_scenario.location,
            "turn_count": self.active_scenario.turn_count
        }
    
    def end_scenario(self) -> tuple[bool, str]:
        """End the current scenario"""
        if not self.active_scenario:
            return False, "No active scenario to end"
            
        scenario_title = self.scenarios[self.active_scenario.scenario_id]["title"]
        self.active_scenario = None
        
        return True, f"Ended scenario: {scenario_title}"

    def process_turn(self, player_input: str, npc_responses: List[Dict]) -> Dict:
        """Process a turn in the active scenario."""
        if not self.active_scenario:
            return {"error": "No active scenario"}

        # Update state tracking
        self.active_scenario.state_tracker.conversation_history.append({
            "player": player_input,
            "npc_responses": npc_responses
        })
        
        # Update team morale and track conflicts
        self.active_scenario.state_tracker.update_team_morale(npc_responses)
        self.active_scenario.state_tracker.track_conflicts(npc_responses)
        self.active_scenario.state_tracker.track_objectives(npc_responses)

        # Update scenario phase based on state
        self._update_scenario_phase()

        # Get current state summary
        state_summary = self.active_scenario.state_tracker.get_state_summary()
        
        # Check for scenario completion
        if self._is_scenario_complete():
            self._conclude_scenario()
            state_summary["status"] = "completed"
        else:
            state_summary["status"] = "in_progress"
            state_summary["current_phase"] = self.active_scenario.current_phase.value

        return state_summary

    def _update_scenario_phase(self) -> None:
        """Update the scenario phase based on current state."""
        if not self.active_scenario:
            return

        state = self.active_scenario.state_tracker
        current_phase = self.active_scenario.current_phase

        # Phase transition logic
        if current_phase == ScenarioPhase.SETUP:
            if state.team_morale > 0.6 and len(state.conflicts) == 0:
                self.active_scenario.current_phase = ScenarioPhase.INTRODUCTION
                
        elif current_phase == ScenarioPhase.INTRODUCTION:
            if state.conversation_history and len(state.conversation_history) > 2:
                self.active_scenario.current_phase = ScenarioPhase.DISCUSSION
                
        elif current_phase == ScenarioPhase.DISCUSSION:
            # Move to resolution if conflicts are being resolved
            if state.conflicts and all(c.resolution_progress > 0.5 for c in state.conflicts.values()):
                self.active_scenario.current_phase = ScenarioPhase.RESOLUTION
                
        elif current_phase == ScenarioPhase.RESOLUTION:
            # Move to conclusion if objectives are mostly complete
            completed = sum(1 for obj in state.objectives.values() if obj.completed)
            if completed / len(state.objectives) > 0.8:
                self.active_scenario.current_phase = ScenarioPhase.CONCLUSION

    def _is_scenario_complete(self) -> bool:
        """Check if the current scenario is complete."""
        if not self.active_scenario:
            return False

        state = self.active_scenario.state_tracker
        scenario = self.scenarios[self.active_scenario.scenario_id]

        # Check if we've reached the conclusion phase
        if self.active_scenario.current_phase != ScenarioPhase.CONCLUSION:
            return False

        # Check if all objectives are complete
        if not all(obj.completed for obj in state.objectives.values()):
            return False

        # Check if all conflicts are resolved
        if state.conflicts and not all(c.resolution_progress > 0.8 for c in state.conflicts.values()):
            return False

        # Check if we've met the minimum duration
        if len(state.conversation_history) < scenario.get("min_duration", 5):
            return False

        return True

    def _conclude_scenario(self) -> None:
        """Handle scenario conclusion and cleanup."""
        if not self.active_scenario:
            return

        # Record final state in memory
        if self.memory_manager:
            state_summary = self.active_scenario.state_tracker.get_state_summary()
            self.memory_manager.record_scenario_conclusion(
                self.active_scenario.scenario_id,
                state_summary
            )

        # Clear active scenario
        self.active_scenario = None

    def _load_rag_context(self, scenario: Dict) -> None:
        """Load relevant RAG context for the scenario."""
        if not self.rag_manager:
            return

        required_topics = scenario.get("rag_context", {}).get("required_topics", [])
        for topic in required_topics:
            self.rag_manager.load_topic_context(topic)

    def get_active_scenario_info(self) -> Optional[Dict]:
        """Get information about the currently active scenario."""
        if not self.active_scenario:
            return None

        scenario = self.scenarios[self.active_scenario.scenario_id]
        state_summary = self.active_scenario.state_tracker.get_state_summary()

        return {
            "scenario_id": self.active_scenario.scenario_id,
            "title": scenario["title"],
            "description": scenario["description"],
            "current_phase": self.active_scenario.current_phase.value,
            "active_npcs": list(self.active_scenario.active_npcs),
            "location": self.active_scenario.location,
            "state": state_summary
        }

    def get_available_scenarios(self, location: str, available_npcs: Set[str]) -> List[Dict]:
        """Get list of available scenarios for current location and NPCs."""
        available = []
        for scenario_id, scenario in self.scenarios.items():
            # Check location trigger
            if location not in scenario.get("location_triggers", {}):
                continue

            # Check NPC requirements
            required_npcs = set(scenario.get("required_npcs", []))
            if not required_npcs.issubset(available_npcs):
                continue

            available.append({
                "scenario_id": scenario_id,
                "title": scenario["title"],
                "description": scenario["description"],
                "required_npcs": list(required_npcs),
                "trigger_probability": scenario["location_triggers"][location]
            })

        return available 