from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set
import re
from collections import defaultdict

class SentimentLevel(Enum):
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2

class ConflictIntensity(Enum):
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.9

@dataclass
class ConflictState:
    topic: str
    participants: Set[str]
    intensity: float
    resolution_progress: float = 0.0
    last_update: str = ""

@dataclass
class ObjectiveState:
    description: str
    completed: bool = False
    progress: float = 0.0
    blockers: List[str] = None

class ScenarioStateTracker:
    def __init__(self):
        self.conflicts: Dict[str, ConflictState] = {}
        self.objectives: Dict[str, ObjectiveState] = {}
        self.team_morale: float = 0.8
        self.conversation_history: List[Dict] = []
        
        # Sentiment analysis patterns
        self.positive_patterns = [
            r"\b(great|excellent|perfect|wonderful|fantastic)\b",
            r"\b(agree|support|like|love|appreciate)\b",
            r"\b(help|assist|cooperate|work together)\b",
            r"\b(thank|thanks|grateful|appreciate)\b",
            r"\b(pleased|happy|satisfied|content)\b"
        ]
        
        self.negative_patterns = [
            r"\b(bad|terrible|awful|horrible|poor)\b",
            r"\b(disagree|against|oppose|hate|dislike)\b",
            r"\b(problem|issue|concern|worry|anxious)\b",
            r"\b(angry|upset|frustrated|annoyed|irritated)\b",
            r"\b(no|never|can't|won't|shouldn't)\b"
        ]
        
        self.intensifiers = [
            r"\b(very|extremely|absolutely|completely|totally)\b",
            r"\b(really|quite|rather|somewhat|slightly)\b"
        ]

    def analyze_sentiment(self, text: str) -> SentimentLevel:
        """Analyze the sentiment of a text using pattern matching."""
        positive_matches = sum(
            len(re.findall(pattern, text.lower()))
            for pattern in self.positive_patterns
        )
        negative_matches = sum(
            len(re.findall(pattern, text.lower()))
            for pattern in self.negative_patterns
        )
        
        # Check for intensifiers
        intensifier_count = sum(
            len(re.findall(pattern, text.lower()))
            for pattern in self.intensifiers
        )
        
        # Calculate base sentiment
        if positive_matches > negative_matches:
            base_sentiment = SentimentLevel.POSITIVE
        elif negative_matches > positive_matches:
            base_sentiment = SentimentLevel.NEGATIVE
        else:
            base_sentiment = SentimentLevel.NEUTRAL
            
        # Apply intensifiers
        if intensifier_count > 0:
            if base_sentiment == SentimentLevel.POSITIVE:
                return SentimentLevel.VERY_POSITIVE
            elif base_sentiment == SentimentLevel.NEGATIVE:
                return SentimentLevel.VERY_NEGATIVE
                
        return base_sentiment

    def update_team_morale(self, responses: List[Dict]) -> None:
        """Update team morale based on response sentiments."""
        sentiment_sum = 0
        for response in responses:
            sentiment = self.analyze_sentiment(response["content"])
            sentiment_sum += sentiment.value
            
        # Calculate average sentiment and update morale
        if responses:
            avg_sentiment = sentiment_sum / len(responses)
            # Convert sentiment range (-2 to 2) to morale (0 to 1)
            morale_change = (avg_sentiment + 2) / 4
            # Apply change with some smoothing
            self.team_morale = 0.7 * self.team_morale + 0.3 * morale_change
            # Ensure morale stays in valid range
            self.team_morale = max(0.0, min(1.0, self.team_morale))

    def track_conflicts(self, responses: List[Dict]) -> None:
        """Track and update conflicts based on responses."""
        for response in responses:
            # Check for conflict indicators in the response
            if self._is_conflict_indicator(response["content"]):
                conflict = self._extract_conflict(response)
                if conflict:
                    conflict_id = f"{conflict.topic}_{'-'.join(sorted(conflict.participants))}"
                    if conflict_id in self.conflicts:
                        # Update existing conflict
                        self._update_conflict(conflict_id, conflict)
                    else:
                        # Add new conflict
                        self.conflicts[conflict_id] = conflict

    def _is_conflict_indicator(self, text: str) -> bool:
        """Check if text indicates a conflict."""
        conflict_indicators = [
            r"\b(disagree|dispute|conflict|argument|issue)\b",
            r"\b(but|however|although|though|yet)\b",
            r"\b(no|never|can't|won't|shouldn't)\b",
            r"\b(problem|concern|worry|anxious|upset)\b"
        ]
        return any(re.search(pattern, text.lower()) for pattern in conflict_indicators)

    def _extract_conflict(self, response: Dict) -> Optional[ConflictState]:
        """Extract conflict information from a response."""
        # This is a simplified version - in practice, you'd want more sophisticated
        # conflict detection and extraction
        if "conflict_topic" in response.get("metadata", {}):
            return ConflictState(
                topic=response["metadata"]["conflict_topic"],
                participants=set(response["metadata"].get("participants", [])),
                intensity=response["metadata"].get("conflict_intensity", 0.5)
            )
        return None

    def _update_conflict(self, conflict_id: str, new_conflict: ConflictState) -> None:
        """Update an existing conflict's state."""
        current = self.conflicts[conflict_id]
        # Update intensity based on new information
        current.intensity = max(current.intensity, new_conflict.intensity)
        # Update resolution progress if there are signs of resolution
        if self._is_resolution_indicator(new_conflict.last_update):
            current.resolution_progress += 0.1
        current.last_update = new_conflict.last_update

    def _is_resolution_indicator(self, text: str) -> bool:
        """Check if text indicates conflict resolution."""
        resolution_indicators = [
            r"\b(agree|compromise|solution|resolve|settle)\b",
            r"\b(understand|accept|work together|cooperate)\b",
            r"\b(okay|fine|alright|good|great)\b"
        ]
        return any(re.search(pattern, text.lower()) for pattern in resolution_indicators)

    def track_objectives(self, responses: List[Dict]) -> None:
        """Track and update scenario objectives based on responses."""
        for response in responses:
            if "objective_progress" in response.get("metadata", {}):
                for obj_id, progress in response["metadata"]["objective_progress"].items():
                    if obj_id in self.objectives:
                        self._update_objective(obj_id, progress)

    def _update_objective(self, obj_id: str, progress: float) -> None:
        """Update an objective's progress."""
        objective = self.objectives[obj_id]
        objective.progress = max(objective.progress, progress)
        if objective.progress >= 1.0:
            objective.completed = True
            objective.blockers = []

    def get_state_summary(self) -> Dict:
        """Get a summary of the current scenario state."""
        return {
            "team_morale": self.team_morale,
            "active_conflicts": len(self.conflicts),
            "conflict_intensity": sum(c.intensity for c in self.conflicts.values()) / max(1, len(self.conflicts)),
            "completed_objectives": sum(1 for obj in self.objectives.values() if obj.completed),
            "total_objectives": len(self.objectives),
            "conversation_turns": len(self.conversation_history)
        } 