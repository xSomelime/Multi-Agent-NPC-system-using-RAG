#!/usr/bin/env python3
"""
Turn Management System
======================

Handles conversation turn-taking, participation logic, and speaking order
for multi-agent conversations.
"""

import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ConversationType(Enum):
    """Types of conversations that affect turn-taking"""
    CASUAL = "casual"
    DEBATE = "debate"
    QUESTION_ANSWER = "question_answer"
    EMERGENCY = "emergency"
    TEACHING = "teaching"


@dataclass
class ParticipationScore:
    """Scores for determining NPC participation"""
    agent_name: str
    base_score: float
    expertise_bonus: float
    relationship_bonus: float
    memory_bonus: float
    final_score: float
    reason: str


class TurnManager:
    """Manages conversation turns and participation decisions"""
    
    def __init__(self):
        self.conversation_history_limit = 5
        self.max_participants_per_turn = 5  # Increased from 3 for more group interaction
        
        # Participation thresholds by conversation type
        self.participation_thresholds = {
            ConversationType.CASUAL: 0.2,  # Lowered for more group participation
            ConversationType.DEBATE: 0.15,
            ConversationType.QUESTION_ANSWER: 0.3,
            ConversationType.EMERGENCY: 0.1,
            ConversationType.TEACHING: 0.25
        }
        
        # Track active conversations for UI highlighting
        self.active_conversations = {}
    
    def get_available_conversation_targets(self, location: str, agents: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get NPCs available for conversation at current location with their status"""
        available_npcs = []
        
        for name, agent in agents.items():
            if getattr(agent, 'current_location', None) == location:
                npc_info = {
                    'name': name,
                    'role': getattr(agent, 'npc_role', 'unknown').value,
                    'is_busy': name in self.active_conversations,
                    'title': getattr(agent, 'template_data', {}).get('title', ''),
                    'current_activity': self._get_npc_current_activity(agent)
                }
                available_npcs.append(npc_info)
        
        return available_npcs
    
    def _get_npc_current_activity(self, agent: Any) -> str:
        """Get NPC's current activity for UI display"""
        if hasattr(agent, 'current_activity'):
            return agent.current_activity
        return "Standing by"
    
    def determine_conversation_type(self, message: str, context: List = None) -> ConversationType:
        """Analyze message to determine conversation type"""
        message_lower = message.lower()
        
        # Emergency indicators
        emergency_words = ["help", "emergency", "urgent", "quickly", "immediately", "danger"]
        if any(word in message_lower for word in emergency_words):
            return ConversationType.EMERGENCY
        
        # Question/answer indicators
        question_words = ["what", "how", "why", "when", "where", "can you", "do you know"]
        has_question_mark = "?" in message
        if has_question_mark or any(word in message_lower for word in question_words):
            return ConversationType.QUESTION_ANSWER
        
        # Debate indicators
        debate_words = ["better", "prefer", "best", "vs", "versus", "compare", "which", "opinion"]
        controversial_topics = ["saddle", "feed", "training", "method", "brand", "equipment"]
        has_debate_trigger = any(word in message_lower for word in debate_words)
        has_controversial_topic = any(topic in message_lower for topic in controversial_topics)
        
        if has_debate_trigger and has_controversial_topic:
            return ConversationType.DEBATE
        
        # Teaching indicators (longer explanations)
        teaching_words = ["explain", "teach", "show me", "how to", "learn", "understand"]
        if any(word in message_lower for word in teaching_words):
            return ConversationType.TEACHING
        
        return ConversationType.CASUAL
    
    def select_participants(self, message: str, available_agents: Dict[str, Any], 
                          current_location: str, memory_manager) -> List[ParticipationScore]:
        """Select which NPCs should participate in the conversation"""
        
        conversation_type = self.determine_conversation_type(message)
        threshold = self.participation_thresholds[conversation_type]
        
        participation_scores = []
        
        for agent_name, agent in available_agents.items():
            # Skip if not in same location
            agent_location = getattr(agent, 'current_location', None)
            if agent_location != current_location:
                continue
            
            score = self._calculate_participation_score(
                agent_name, agent, message, conversation_type, memory_manager
            )
            
            if score.final_score >= threshold:
                participation_scores.append(score)
        
        # Sort by score and limit participants
        participation_scores.sort(key=lambda x: x.final_score, reverse=True)
        return participation_scores[:self.max_participants_per_turn]
    
    def _calculate_participation_score(self, agent_name: str, agent: Any, message: str, 
                                     conversation_type: ConversationType, memory_manager) -> ParticipationScore:
        """Calculate how likely an agent is to participate"""
        
        message_lower = message.lower()
        base_score = 0.1  # Base participation chance
        expertise_bonus = 0.0
        relationship_bonus = 0.0
        memory_bonus = 0.0
        reasons = []
        
        # Direct mention gets highest priority
        if agent_name.lower() in message_lower:
            base_score = 1.0
            reasons.append("directly_mentioned")
        
        # Check expertise match
        if hasattr(agent, 'expertise_areas'):
            for area in agent.expertise_areas:
                area_keywords = area.replace('_', ' ').split()
                if any(keyword in message_lower for keyword in area_keywords):
                    expertise_bonus += 0.3
                    reasons.append(f"expertise_{area}")
        
        # Check memory relevance
        if memory_manager:
            agent_memory = memory_manager.get_npc_memory(agent_name)
            if agent_memory:
                message_words = [word for word in message_lower.split() if len(word) > 3]
                for word in message_words:
                    if agent_memory.knows_about(word):
                        memory_bonus += 0.2
                        reasons.append("relevant_memory")
                        break
        
        # Relationship-based participation
        if hasattr(agent, 'npc_data'):
            personality_traits = agent.npc_data.get('personality', {}).get('traits', [])
            
            # Empathetic agents respond to emotional content
            empathetic_traits = ['empathetic', 'gentle', 'caring', 'supportive']
            if any(trait in personality_traits for trait in empathetic_traits):
                emotional_words = ['nervous', 'scared', 'excited', 'worried', 'happy', 'sad']
                if any(word in message_lower for word in emotional_words):
                    relationship_bonus += 0.2
                    reasons.append("empathetic_response")
            
            # Confident/leadership traits increase participation in debates
            if conversation_type == ConversationType.DEBATE:
                leadership_traits = ['confident', 'assertive', 'competitive', 'leader']
                if any(trait in personality_traits for trait in leadership_traits):
                    relationship_bonus += 0.25
                    reasons.append("leadership_in_debate")
        
        # Conversation type modifiers
        if conversation_type == ConversationType.EMERGENCY:
            base_score += 0.3  # Everyone more likely to help in emergency
        elif conversation_type == ConversationType.QUESTION_ANSWER:
            expertise_bonus *= 1.5  # Boost expertise bonus for Q&A
        
        final_score = min(1.0, base_score + expertise_bonus + relationship_bonus + memory_bonus)
        
        return ParticipationScore(
            agent_name=agent_name,
            base_score=base_score,
            expertise_bonus=expertise_bonus,
            relationship_bonus=relationship_bonus,
            memory_bonus=memory_bonus,
            final_score=final_score,
            reason=" + ".join(reasons) if reasons else "random_participation"
        )
    
    def determine_speaking_order(self, participants: List[ParticipationScore], 
                               conversation_type: ConversationType) -> List[str]:
        """Determine the order in which NPCs should speak"""
        
        if not participants:
            return []
        
        # For emergencies, highest score goes first
        if conversation_type == ConversationType.EMERGENCY:
            participants.sort(key=lambda x: x.final_score, reverse=True)
            return [p.agent_name for p in participants]
        
        # For debates, mix high scores with some randomness for natural flow
        elif conversation_type == ConversationType.DEBATE:
            # Top scorer goes first, then randomize the rest
            top_scorer = max(participants, key=lambda x: x.final_score)
            others = [p for p in participants if p != top_scorer]
            random.shuffle(others)
            return [top_scorer.agent_name] + [p.agent_name for p in others]
        
        # For Q&A, expertise should go first
        elif conversation_type == ConversationType.QUESTION_ANSWER:
            participants.sort(key=lambda x: x.expertise_bonus, reverse=True)
            return [p.agent_name for p in participants]
        
        # For casual and teaching, add some randomness
        else:
            # Weight by score but add randomness
            weighted_participants = []
            for p in participants:
                # Convert score to weight (higher score = higher chance)
                weight = int(p.final_score * 10) + 1
                weighted_participants.extend([p] * weight)
            
            selected_order = []
            available = participants.copy()
            
            for _ in range(len(participants)):
                if not available:
                    break
                
                # Weighted random selection
                weights = [p.final_score for p in available]
                selected = random.choices(available, weights=weights)[0]
                selected_order.append(selected.agent_name)
                available.remove(selected)
            
            return selected_order
    
    def should_limit_responses(self, conversation_type: ConversationType, 
                             current_chain_length: int) -> bool:
        """Determine if we should limit responses to prevent endless chains"""
        
        limits = {
            ConversationType.EMERGENCY: 5,
            ConversationType.DEBATE: 4,
            ConversationType.QUESTION_ANSWER: 2,
            ConversationType.TEACHING: 3,
            ConversationType.CASUAL: 3
        }
        
        return current_chain_length >= limits.get(conversation_type, 3)
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about conversation management"""
        return {
            "max_participants": self.max_participants_per_turn,
            "conversation_types": [ct.value for ct in ConversationType],
            "participation_thresholds": {ct.value: thresh for ct, thresh in self.participation_thresholds.items()}
        } 