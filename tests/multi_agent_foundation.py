#!/usr/bin/env python3
"""
Enhanced Multi-Agent NPC System with Natural Conversation Flow
NPCs respond to player AND react to each other naturally
"""

import requests
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import uuid
import random

class NPCRole(Enum):
    STABLE_HAND = "stable_hand"
    TRAINER = "trainer" 
    HEALTH_MONITOR = "health_monitor"
    PERSONALITY = "personality"

@dataclass
class Message:
    """Represents a single message in conversation"""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    agent_name: Optional[str] = None
    message_type: str = "response"  # "response", "reaction", "follow_up"
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()

class ConversationContext:
    """Manages conversation flow and determines response patterns"""
    
    def __init__(self):
        self.current_topic = None
        self.participants = set()
        self.conversation_energy = "neutral"  # "excited", "concerned", "casual"
    
    def analyze_message(self, message: str) -> Dict:
        """Analyze message to determine conversation characteristics"""
        message_lower = message.lower()
        
        # Detect question types
        question_indicators = {
            "general_question": ["has anyone", "does anyone", "har någon", "vet någon"],
            "experience_sharing": ["how did", "how was", "hur gick", "vad hände"],
            "advice_seeking": ["should i", "how do i", "what do you think", "vad tycker"],
            "news_sharing": ["guess what", "did you hear", "gissa vad", "hörde ni"]
        }
        
        # Detect emotional tone
        emotional_indicators = {
            "excited": ["amazing", "awesome", "fantastic", "superkul", "fantastiskt"],
            "concerned": ["worried", "problem", "hurt", "sick", "orolig", "problem"],
            "casual": ["maybe", "perhaps", "kanske", "lite"],
            "competitive": ["won", "lost", "competition", "better", "vann", "förlorade", "tävling"]
        }
        
        # Detect expertise areas
        expertise_areas = {
            "care": ["feeding", "grooming", "care", "sick", "health", "vård", "mat"],
            "training": ["training", "riding", "jumping", "lesson", "träning", "lektion"],
            "competition": ["competition", "show", "event", "tävling", "uppvisning"],
            "social": ["everyone", "all", "alla", "together", "tillsammans"]
        }
        
        analysis = {
            "question_type": None,
            "emotional_tone": "neutral",
            "expertise_area": None,
            "expects_multiple_responses": False
        }
        
        # Analyze question type
        for q_type, indicators in question_indicators.items():
            if any(indicator in message_lower for indicator in indicators):
                analysis["question_type"] = q_type
                analysis["expects_multiple_responses"] = q_type in ["general_question", "experience_sharing"]
                break
        
        # Analyze emotional tone
        for tone, indicators in emotional_indicators.items():
            if any(indicator in message_lower for indicator in indicators):
                analysis["emotional_tone"] = tone
                break
        
        # Analyze expertise area
        for area, indicators in expertise_areas.items():
            if any(indicator in message_lower for indicator in indicators):
                analysis["expertise_area"] = area
                break
        
        return analysis

class AIAgent:
    """Enhanced AI Agent with natural conversation abilities"""
    
    def __init__(self, name: str, npc_role: NPCRole, persona_prompt: str, 
                 background_info: List[str] = None, temperature: float = 0.3):
        self.name = name
        self.npc_role = npc_role
        self.persona_prompt = persona_prompt
        self.background_info = background_info or []
        self.temperature = temperature
        self.conversation_history: List[Message] = []
        self.ollama_url = "http://localhost:11434/api"
        self.model = "phi3:mini"
        
        # Enhanced response settings
        self.max_response_length = 40  # Mycket kortare för naturlig dialog
        self.response_style = "conversational"
        
        # Conversation participation patterns
        self.participation_patterns = {
            "always_respond": ["health", "safety", "emergency"],
            "often_respond": ["training", "care", "advice"],
            "sometimes_respond": ["social", "general", "news"],
            "react_to_others": True
        }

    def should_participate(self, message_analysis: Dict, existing_responses: List[str] = None) -> Tuple[bool, str]:
        """
        Determine if and how this agent should participate
        Returns: (should_respond, response_type)
        """
        existing_count = len(existing_responses or [])
        
        # Always respond to safety/health issues
        if message_analysis.get("expertise_area") == "care" and self.npc_role == NPCRole.HEALTH_MONITOR:
            return True, "expert_response"
        
        # Always respond to training questions  
        if message_analysis.get("expertise_area") == "training" and self.npc_role == NPCRole.TRAINER:
            return True, "expert_response"
        
        # Always respond to care questions
        if message_analysis.get("expertise_area") == "care" and self.npc_role == NPCRole.STABLE_HAND:
            return True, "expert_response"
        
        # For general questions, let 2-3 people respond
        if message_analysis.get("question_type") == "general_question":
            if existing_count < 3:
                return True, "general_response"
        
        # For experience sharing, participate if you have relevant experience
        if message_analysis.get("question_type") == "experience_sharing":
            if existing_count < 2:
                return True, "experience_share"
        
        # React to others' responses (follow-up conversation)
        if existing_count > 0 and existing_count < 4:
            # Sometimes add supportive comments
            if random.random() < 0.4:  # 40% chance to add follow-up
                return True, "reaction"
        
        return False, "none"

    def build_conversation_prompt(self, original_message: str, conversation_context: List[Message], 
                                response_type: str, message_analysis: Dict) -> str:
        """Build context-aware prompt for natural conversation"""
        
        # Base persona
        prompt_parts = [f"You are {self.name}. {self.persona_prompt}"]
        
        # Add background knowledge if relevant
        if self.background_info and (message_analysis.get("expertise_area") or "competition" in original_message.lower() or "weekend" in original_message.lower()):
            # Filter background info to most relevant
            relevant_info = []
            for info in self.background_info:
                # Always include competition-related info for competition questions
                if "competition" in original_message.lower() or "weekend" in original_message.lower():
                    if any(keyword in info.lower() for keyword in ["competition", "sarah", "jake", "weekend", "won", "place", "thunder", "storm"]):
                        relevant_info.append(info)
                # Include expertise-specific info
                elif message_analysis.get("expertise_area") and message_analysis["expertise_area"] in info.lower():
                    relevant_info.append(info)
            
            if relevant_info:
                prompt_parts.append(f"What you know: {'. '.join(relevant_info[:3])}")  # Max 3 facts
        
        # Add conversation context
        if conversation_context:
            prompt_parts.append("Recent conversation:")
            for msg in conversation_context[-6:]:  # Last 6 messages for context
                if msg.role == "user":
                    prompt_parts.append(f"Player: {msg.content}")
                elif msg.role == "assistant" and msg.agent_name != self.name:
                    prompt_parts.append(f"{msg.agent_name}: {msg.content}")
        
        # Response type specific instructions
        response_instructions = {
            "expert_response": f"Give helpful advice based on your expertise. Be specific and knowledgeable.",
            "general_response": f"Respond naturally as yourself. Share what you think or feel about this.",
            "experience_share": f"If you have relevant experience, share it. Otherwise, show interest and ask questions.",
            "reaction": f"React to what others have said. Be supportive, add your thoughts, or ask follow-up questions."
        }
        
        instruction = response_instructions.get(response_type, "Respond naturally as yourself.")
        prompt_parts.append(f"Instructions: {instruction}")
        
        # Emotional tone guidance
        if message_analysis.get("emotional_tone") != "neutral":
            tone_guide = {
                "excited": "Match the enthusiasm in your response.",
                "concerned": "Be supportive and helpful.",
                "competitive": "Show appropriate interest in the competition aspect."
            }
            if message_analysis["emotional_tone"] in tone_guide:
                prompt_parts.append(tone_guide[message_analysis["emotional_tone"]])
        
        # Final prompt construction with variety
        prompt_parts.append(f"\nPlayer asked: {original_message}")
        
        # Add variety instruction to avoid repetitive starts
        variety_instructions = [
            "Keep your response to 1-2 sentences maximum. Be direct and conversational.",
            "Answer briefly in 1-2 sentences. Don't start with 'Oh' or generic phrases.",
            "Give a short, natural response. Maximum 2 sentences.",
            "Respond naturally in 1-2 sentences. Avoid long explanations."
        ]
        import random
        prompt_parts.append(variety_instructions[random.randint(0, len(variety_instructions)-1)])
        
        prompt_parts.append(f"{self.name}:")
        
        return "\n".join(prompt_parts)

    def generate_response(self, original_message: str, conversation_context: List[Message] = None, 
                         response_type: str = "general_response", message_analysis: Dict = None) -> Tuple[str, bool, float]:
        """Generate contextual response"""
        
        start_time = time.time()
        
        # Build sophisticated prompt
        full_prompt = self.build_conversation_prompt(
            original_message, conversation_context or [], response_type, message_analysis or {}
        )
        
        try:
            response = requests.post(
                f"{self.ollama_url}/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_response_length,
                        "stop": [
                            "\nPlayer:", "\nInstruction:", "\n###", 
                            "Player:", "Instruction:", "###"
                        ]
                    }
                },
                timeout=20
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                agent_response = result.get('response', '').strip()
                
                # Enhanced response cleaning with "Oh" filter and length check
                if len(agent_response) < 10 or agent_response.lower().startswith(('oh ', 'oh,', 'that\'s interesting')):
                    fallback_responses = {
                        "expert_response": f"Let me help with that.",
                        "reaction": "Good point!",
                        "general_response": "What do you think?",
                        "experience_share": "I've seen that before."
                    }
                    agent_response = fallback_responses.get(response_type, "Tell me more.")
                
                # Clean up unwanted patterns
                cleanup_patterns = [
                    "Coach Prestige", "Instruction:", "###", "- As ", " - As ", 
                    "demonstrating confidence", "showcasing superiority", "---"
                ]
                for pattern in cleanup_patterns:
                    if pattern in agent_response:
                        agent_response = agent_response.split(pattern)[0].strip()
                
                # Record the response
                self.add_message("user", original_message)
                self.add_message("assistant", agent_response, self.name)
                
                return agent_response, True, response_time
            else:
                return f"Sorry, I'm having trouble responding. (Error: {response.status_code})", False, response_time
                
        except requests.exceptions.RequestException as e:
            end_time = time.time()
            return "I can't respond right now. Please try again.", False, end_time - start_time
    
    def add_message(self, role: str, content: str, sender_name: str = None) -> Message:
        """Add message to agent's history"""
        message = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=time.time(),
            agent_name=sender_name or (self.name if role == "assistant" else "player")
        )
        self.conversation_history.append(message)
        return message
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        total_messages = len(self.conversation_history)
        user_messages = len([m for m in self.conversation_history if m.role == "user"])
        assistant_messages = len([m for m in self.conversation_history if m.role == "assistant"])
        
        return {
            "name": self.name,
            "role": self.npc_role.value,
            "temperature": self.temperature,
            "total_messages": total_messages,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "model": self.model
        }

class EnhancedNPCFactory:
    """Factory for creating NPCs with background knowledge"""
    
    @staticmethod
    def create_stable_hand(name: str = "Anna") -> AIAgent:
        """Create Stable Hand with care expertise"""
        persona = f"""You are {name}, a friendly and experienced stable hand who has worked with horses for many years. 
        You care deeply about horse welfare and know all the horses personally. You're social, helpful, and love sharing 
        your knowledge about horse care. You get excited when horses are healthy and worried when they're not."""
        
        background = [
            "Horses need 2-3% of their body weight in feed daily, split into 2-3 meals",
            "Daily grooming includes brushing, hoof picking, and health checks", 
            "Thunder is a 500kg warmblood who likes apples and gets nervous in storms",
            "Storm is a spirited mare who needs extra exercise and attention",
            "Morning routine starts at 6 AM with feeding, then grooming and turnout",
            "Last weekend's competition: Sarah won first place with Storm in jumping, Jake came second with Thunder",
            "Competition results: Emma placed third with her horse Lightning, Marcus had a fall but is okay",
            "Anna helped prepare all the horses for competition - groomed them extra well that morning"
        ]
        
        return AIAgent(name, NPCRole.STABLE_HAND, persona, background, temperature=0.25)

    @staticmethod
    def create_trainer(name: str = "Erik") -> AIAgent:
        """Create Trainer with technical expertise"""
        persona = f"""You are {name}, an experienced horse trainer who has been working with riders of all levels for over 10 years. 
        You're passionate about proper technique and horse-rider partnerships. You can be serious about training but also 
        supportive and encouraging. You love seeing progress and celebrating achievements."""
        
        background = [
            "Training should progress gradually from basic groundwork to advanced exercises",
            "Jumping requires proper position, rhythm, and horse preparation",
            "Each horse has different learning styles and physical capabilities", 
            "Competition preparation involves both physical and mental conditioning",
            "Safety gear is non-negotiable for all riding activities",
            "Erik coached Sarah and Jake for last weekend's competition - very proud of their performance",
            "Sarah's winning round was technically perfect - excellent seat and timing over the jumps",
            "Jake made one small mistake in the combination but recovered well for second place",
            "The competition had 15 riders total, our stable sent 4 riders and placed well"
        ]
        
        return AIAgent(name, NPCRole.TRAINER, persona, background, temperature=0.3)

    @staticmethod
    def create_health_monitor(name: str = "Lisa") -> AIAgent:
        """Create Health Monitor focused on wellness"""
        persona = f"""You are {name}, a dedicated health monitor who watches over the horses' wellbeing. 
        You have a keen eye for detecting health issues early and believe prevention is better than cure. 
        You're gentle, observant, and always concerned about the horses' comfort and health."""
        
        background = [
            "Early signs of illness include changes in appetite, behavior, or posture",
            "Lameness can be detected by watching how horses move and stand",
            "Colic is a serious emergency requiring immediate veterinary attention",
            "Regular health checks prevent minor issues from becoming major problems",
            "Each horse has unique health considerations and history",
            "Lisa checked all competition horses before and after the event - everyone came back healthy",
            "Sarah's horse Storm had perfect recovery after the competition, no stress signs",
            "Jake's horse Thunder was slightly tired but bounced back quickly with proper cooling down",
            "Competition medical check: No injuries reported, all riders and horses safe"
        ]
        
        return AIAgent(name, NPCRole.HEALTH_MONITOR, persona, background, temperature=0.2)

    @staticmethod
    def create_competitive_rider(name: str = "Jake") -> AIAgent:
        """Create Competitive Rider with personality"""
        persona = f"""You are {name}, a competitive rider who takes training and competitions seriously. 
        You're ambitious and sometimes a bit intense about winning. You have strong opinions about techniques 
        and aren't afraid to share them. Despite being competitive, you respect good horsemanship."""
        
        background = [
            "Competition success requires dedication, consistent training, and mental preparation",
            "Different competitions require different skills - dressage vs jumping vs cross-country",
            "Horse and rider partnerships take time to develop and refine",
            "Winning feels great, but learning from losses is more important for long-term success",
            "Equipment quality and fit can make or break a competition performance",
            "Jake competed last weekend with Thunder and came second place in jumping",
            "Jake's round had one rail down in the triple combination but was otherwise clean",
            "Sarah beat Jake by just 0.3 seconds - it was a very close competition",
            "Jake is already planning training improvements for the next competition in two weeks"
        ]
        
        return AIAgent(name, NPCRole.PERSONALITY, persona, background, temperature=0.35)

class EnhancedMessageRouter:
    """Advanced message routing with natural conversation flow"""
    
    def __init__(self):
        self.agents: Dict[str, AIAgent] = {}
        self.conversation_log: List[Message] = []
        self.context_analyzer = ConversationContext()
    
    def register_agent(self, agent: AIAgent):
        """Register an agent"""
        self.agents[agent.name] = agent
        print(f"✅ Registered {agent.name} ({agent.npc_role.value})")
    
    def send_message_to_agent(self, agent_name: str, message: str) -> Tuple[str, bool, float]:
        """Send message to specific agent"""
        agent = self.agents.get(agent_name)
        if not agent:
            return f"Agent '{agent_name}' not found.", False, 0.0
        
        # Analyze message for context
        analysis = self.context_analyzer.analyze_message(message)
        
        # Add to global log
        global_message = Message(
            id=str(uuid.uuid4()),
            role="user",
            content=message,
            timestamp=time.time(),
            agent_name="player"
        )
        self.conversation_log.append(global_message)
        
        # Generate response with full context
        response, success, response_time = agent.generate_response(
            message, self.conversation_log[-10:], "expert_response", analysis
        )
        
        # Log agent response
        if success:
            response_message = Message(
                id=str(uuid.uuid4()),
                role="assistant", 
                content=response,
                timestamp=time.time(),
                agent_name=agent_name
            )
            self.conversation_log.append(response_message)
        
        return response, success, response_time
    
    def send_message_to_all_agents(self, message: str) -> List[Tuple[str, str, bool, float]]:
        """Enhanced group conversation with natural flow"""
        print("💬 Analyzing conversation...")
        
        # Analyze the message for conversation characteristics
        analysis = self.context_analyzer.analyze_message(message)
        print(f"💭 Detected: {analysis.get('question_type', 'general')} with {analysis.get('emotional_tone', 'neutral')} tone")
        
        # Add player message to global log
        global_message = Message(
            id=str(uuid.uuid4()),
            role="user",
            content=message,
            timestamp=time.time(),
            agent_name="player"
        )
        self.conversation_log.append(global_message)
        
        # Determine participation order and types
        participating_agents = []
        for agent_name, agent in self.agents.items():
            should_respond, response_type = agent.should_participate(analysis, [])
            if should_respond:
                participating_agents.append((agent_name, agent, response_type))
        
        # If nobody wants to respond, pick someone relevant or random
        if not participating_agents:
            # Pick based on message content or random
            if "health" in message.lower() or "sick" in message.lower():
                if "Lisa" in self.agents:
                    participating_agents = [("Lisa", self.agents["Lisa"], "general_response")]
            elif "training" in message.lower() or "riding" in message.lower():
                if "Erik" in self.agents:
                    participating_agents = [("Erik", self.agents["Erik"], "general_response")]
            elif "care" in message.lower() or "feeding" in message.lower():
                if "Anna" in self.agents:
                    participating_agents = [("Anna", self.agents["Anna"], "general_response")]
            else:
                # Random selection if no clear expertise match
                available_agents = list(self.agents.items())
                if available_agents:
                    import random
                    agent_name, agent = random.choice(available_agents)
                    participating_agents = [(agent_name, agent, "general_response")]
        
        print(f"💭 {len(participating_agents)} NPCs will participate")
        
        # Show who will participate with their roles
        role_descriptions = {
            NPCRole.STABLE_HAND: "Horse Care",
            NPCRole.TRAINER: "Training Expert", 
            NPCRole.HEALTH_MONITOR: "Health Monitor",
            NPCRole.PERSONALITY: "Competitive Rider"
        }
        
        participant_info = []
        for agent_name, agent, response_type in participating_agents:
            role_desc = role_descriptions.get(agent.npc_role, agent.npc_role.value)
            participant_info.append(f"{agent_name} ({role_desc})")
        
        if participant_info:
            print(f"👥 Participating: {', '.join(participant_info)}")
        
        responses = []
        current_context = self.conversation_log.copy()
        
        # Get responses in natural order
        for i, (agent_name, agent, response_type) in enumerate(participating_agents):
            # Show role when responding
            role_descriptions = {
                NPCRole.STABLE_HAND: "Horse Care",
                NPCRole.TRAINER: "Training Expert", 
                NPCRole.HEALTH_MONITOR: "Health Monitor",
                NPCRole.PERSONALITY: "Competitive Rider"
            }
            role_desc = role_descriptions.get(agent.npc_role, agent.npc_role.value)
            print(f"⏳ {agent_name} ({role_desc}) responding...")
            
            # Generate response with current context
            response, success, response_time = agent.generate_response(
                message, current_context[-12:], response_type, analysis
            )
            
            responses.append((agent_name, response, success, response_time))
            
            # Add this response to context for next agents
            if success:
                response_message = Message(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content=response,
                    timestamp=time.time(),
                    agent_name=agent_name,
                    message_type=response_type
                )
                current_context.append(response_message)
                self.conversation_log.append(response_message)
            
            # Natural pause between responses
            if i < len(participating_agents) - 1:
                time.sleep(0.4)
        
        return responses
    
    def get_conversation_summary(self, last_n_messages: int = 10) -> str:
        """Get formatted conversation summary"""
        recent_messages = self.conversation_log[-last_n_messages:]
        
        summary = []
        for msg in recent_messages:
            if msg.role == "user":
                summary.append(f"Player: {msg.content}")
            elif msg.role == "assistant":
                summary.append(f"{msg.agent_name}: {msg.content}")
        
        return "\n".join(summary)
    
    def reset_all_conversations(self):
        """Reset all conversations"""
        for agent in self.agents.values():
            agent.reset_conversation()
        self.conversation_log = []
        print("🔄 All conversations reset")
    
    def list_agents(self) -> List[str]:
        """Get available agent names"""
        return list(self.agents.keys()) + ["All"]
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        stats = {
            "total_agents": len(self.agents),
            "total_global_messages": len(self.conversation_log),
            "agents": {}
        }
        
        for agent_name, agent in self.agents.items():
            stats["agents"][agent_name] = agent.get_stats()
        
        return stats

def create_enhanced_system():
    """Initialize enhanced multi-agent system"""
    router = EnhancedMessageRouter()
    
    # Create NPCs with enhanced personalities and knowledge
    anna = EnhancedNPCFactory.create_stable_hand("Anna")
    erik = EnhancedNPCFactory.create_trainer("Erik") 
    lisa = EnhancedNPCFactory.create_health_monitor("Lisa")
    jake = EnhancedNPCFactory.create_competitive_rider("Jake")
    
    # Register all agents
    router.register_agent(anna)
    router.register_agent(erik)
    router.register_agent(lisa)
    router.register_agent(jake)
    
    return router

def show_npc_roles():
    """Display NPC roles and expertise for player reference"""
    print("\n🎭 Meet the NPCs:")
    print("  Anna (Stable Hand) - Horse care, feeding, grooming, daily routines")
    print("  Erik (Trainer) - Riding techniques, training methods, competitions") 
    print("  Lisa (Health Monitor) - Horse health, injuries, wellness checks")
    print("  Jake (Competitive Rider) - Competition experience, winning strategies")
    print("  Type 'roles' anytime to see this again\n")

if __name__ == "__main__":
    print("Enhanced Multi-Agent NPC System")
    print("Natural conversation with contextual responses and agent interactions")
    print("\nMake sure Ollama is running: ollama serve")
    print("="*60)
    
    # Initialize system
    router = create_enhanced_system()
    
    # Show NPC roles
    show_npc_roles()
    
    print(f"🎭 Enhanced NPCs ready for natural conversation!")
    print(f"Available: {', '.join(router.list_agents())}")
    print("\nCommands:")
    print("  - Type message for group discussion")
    print("  - 'Anna: message' to talk directly to Anna")
    print("  - 'roles' to see NPC expertise areas")
    print("  - 'list' for available NPCs")
    print("  - 'summary' for recent conversation")
    print("  - 'stats' for system statistics") 
    print("  - 'reset' to clear all history")
    print("  - 'quit' to exit")
    
    print(f"\n💬 Ready for conversation! Try: 'Has anyone heard how the competition went last weekend?'")
    
    while True:
        try:
            user_input = input(f"\nPlayer: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'list':
                print(f"Available NPCs: {', '.join(router.list_agents())}")
                continue
            elif user_input.lower() == 'roles':
                show_npc_roles()
                continue
            elif user_input.lower() == 'summary':
                print("\n📝 Recent conversation:")
                print(router.get_conversation_summary(8))
                continue
            elif user_input.lower() == 'stats':
                stats = router.get_system_stats()
                print(f"📊 Global messages: {stats['total_global_messages']}")
                for agent_name, agent_stats in stats['agents'].items():
                    print(f"  {agent_name}: {agent_stats['assistant_messages']} responses")
                continue
            elif user_input.lower() == 'reset':
                router.reset_all_conversations()
                continue
            
            # Check for direct agent messaging
            if ':' in user_input and user_input.count(':') == 1:
                agent_name, message = user_input.split(':', 1)
                agent_name = agent_name.strip()
                message = message.strip()
                
                if agent_name in router.agents:
                    response, success, response_time = router.send_message_to_agent(agent_name, message)
                    if success:
                        # Show role for individual conversations too
                        agent = router.agents[agent_name]
                        role_descriptions = {
                            NPCRole.STABLE_HAND: "Horse Care",
                            NPCRole.TRAINER: "Training Expert", 
                            NPCRole.HEALTH_MONITOR: "Health Monitor",
                            NPCRole.PERSONALITY: "Competitive Rider"
                        }
                        role_desc = role_descriptions.get(agent.npc_role, agent.npc_role.value)
                        print(f"{agent_name} ({role_desc}): {response}")
                    else:
                        print(f"❌ {agent_name}: {response}")
                    continue
                else:
                    print(f"❌ Agent '{agent_name}' not found. Available: {', '.join(router.agents.keys())}")
                    continue
            
            # Group conversation
            responses = router.send_message_to_all_agents(user_input)
            print(f"\n💬 Group conversation:")
            for agent_name, response, success, response_time in responses:
                if success:
                    # Show role in output too
                    agent = router.agents[agent_name]
                    role_descriptions = {
                        NPCRole.STABLE_HAND: "Horse Care",
                        NPCRole.TRAINER: "Training Expert", 
                        NPCRole.HEALTH_MONITOR: "Health Monitor",
                        NPCRole.PERSONALITY: "Competitive Rider"
                    }
                    role_desc = role_descriptions.get(agent.npc_role, agent.npc_role.value)
                    print(f"{agent_name} ({role_desc}): {response}")
                else:
                    print(f"❌ {agent_name}: {response}")
            
        except KeyboardInterrupt:
            break
    
    print("\n👋 Thanks for testing the enhanced conversation system!")