#!/usr/bin/env python3
"""
Complete Multi-Agent NPC System with Conversation and Debate Modes
Supports both casual conversations and structured debates
"""

import requests
import json
import time
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import uuid
import random

class NPCRole(Enum):
    STABLE_HAND = "stable_hand"
    TRAINER = "trainer"
    BEHAVIOURIST = "behaviourist"
    COMPETITIVE_RIDER = "competitive_rider"
    RIVAL = "rival"

@dataclass
class Message:
    """Represents a conversation message"""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    agent_name: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()

class RoleTemplateLoader:
    """Loads and manages role template data"""
    
    def __init__(self, template_dir="data/role_templates"):
        self.template_dir = template_dir
        self._templates = {}
    
    def load_role_template(self, role_name: str) -> Dict:
        """Load base knowledge for a role"""
        if role_name in self._templates:
            return self._templates[role_name]
        
        template_file = os.path.join(self.template_dir, f"{role_name}_template.json")
        
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
                self._templates[role_name] = template_data
                return template_data
        except FileNotFoundError:
            print(f"⚠️  Template file not found: {template_file}")
            return self._get_default_template(role_name)
        except json.JSONDecodeError:
            print(f"⚠️  Invalid JSON in template file: {template_file}")
            return self._get_default_template(role_name)
    
    def _get_default_template(self, role_name: str) -> Dict:
        """Fallback template if file not found"""
        return {
            "role": role_name,
            "title": role_name.replace("_", " ").title(),
            "base_knowledge": [f"I am a {role_name} working with horses"],
            "expertise_areas": ["general_horse_knowledge"],
            "common_responsibilities": ["Daily work with horses"]
        }

class NPCLoader:
    """Loads individual NPC configurations"""
    
    def __init__(self, npc_dir="data/npcs"):
        self.npc_dir = npc_dir
        self._npcs = {}
    
    def load_npc_config(self, npc_config_name: str) -> Dict:
        """Load specific NPC configuration"""
        if npc_config_name in self._npcs:
            return self._npcs[npc_config_name]
        
        npc_file = os.path.join(self.npc_dir, f"{npc_config_name}.json")
        
        try:
            with open(npc_file, 'r', encoding='utf-8') as f:
                npc_data = json.load(f)
                self._npcs[npc_config_name] = npc_data
                return npc_data
        except FileNotFoundError:
            print(f"⚠️  NPC file not found: {npc_file}")
            return self._get_default_npc(npc_config_name)
        except json.JSONDecodeError:
            print(f"⚠️  Invalid JSON in NPC file: {npc_file}")
            return self._get_default_npc(npc_config_name)
    
    def _get_default_npc(self, config_name: str) -> Dict:
        """Fallback NPC if file not found"""
        name = config_name.split('_')[0].title()
        role = config_name.split('_')[1] if '_' in config_name else "stable_hand"
        
        return {
            "name": name,
            "role_template": role,
            "personality": {
                "traits": ["helpful", "friendly"],
                "speaking_style": "casual and supportive",
                "quirks": ["loves working with horses"]
            },
            "personal_background": f"{name} is a dedicated horse care professional",
            "unique_knowledge": [],
            "professional_opinions": {},
            "controversial_stances": []
        }

class ScalableNPCAgent:
    """Enhanced NPC Agent with conversation and debate capabilities"""
    
    def __init__(self, npc_config_name: str):
        self.npc_loader = NPCLoader()
        self.template_loader = RoleTemplateLoader()
        
        # Load configuration data
        self.npc_data = self.npc_loader.load_npc_config(npc_config_name)
        self.template_data = self.template_loader.load_role_template(
            self.npc_data['role_template']
        )
        
        # Basic properties
        self.name = self.npc_data['name']
        self.npc_role = NPCRole(self.npc_data['role_template'])
        self.conversation_history: List[Message] = []
        
        # Combine knowledge sources
        self.knowledge_base = (
            self.template_data.get('base_knowledge', []) + 
            self.npc_data.get('unique_knowledge', [])
        )
        
        # Professional opinions for debates
        self.professional_opinions = self.npc_data.get('professional_opinions', {})
        self.controversial_stances = self.npc_data.get('controversial_stances', [])
        
        # Build persona
        self.persona = self._build_persona()
        
        # LLM settings
        self.ollama_url = "http://localhost:11434/api"
        self.model = "phi3:mini"
        
        if self.npc_role == NPCRole.RIVAL:
            self.temperature = 0.35
            self.max_response_length = 100
        else:
            self.temperature = 0.25
            self.max_response_length = 80
        
        # Expertise areas for smart participation
        self.expertise_areas = self.template_data.get('expertise_areas', [])
        
        print(f"✅ Created {self.name} ({self.template_data.get('title', 'NPC')})")
    
    def _build_persona(self) -> str:
        """Build comprehensive persona from template + individual data"""
        name = self.npc_data['name']
        title = self.template_data.get('title', 'Horse Care Professional')
        background = self.npc_data.get('personal_background', '')
        traits = ", ".join(self.npc_data.get('personality', {}).get('traits', ['helpful']))
        style = self.npc_data.get('personality', {}).get('speaking_style', 'friendly')
        
        persona = f"""You are {name}, a {title} working at this horse stable.
        
        Background: {background}
        
        Personality: You are {traits}. Your speaking style is {style}.
        
        IMPORTANT RULES:
        - Keep responses to 1-2 sentences maximum
        - Never mention other people's names (Elin, Oskar, Astrid, Chris) unless directly relevant
        - The person you're talking to is "the player" or "you" - don't assume their name
        - Be natural and conversational but concise
        - Don't start with 'Oh' or 'That's interesting'
        """
        
        # Add role-specific personality reinforcement
        if self.npc_role == NPCRole.RIVAL:
            persona += f"""
        
        AS {name.upper()} THE RIVAL:
        - You are wealthy and privileged - always mention expensive equipment/horses
        - You believe money = quality and judge others for having cheaper gear
        - You name-drop expensive brands and costs when relevant  
        - You are dismissive of "budget" approaches to horse care
        - You assume your way (expensive way) is obviously superior
        """
        
        return persona
    
    def detect_conversation_type(self, message: str) -> str:
        """Detect if this should be a debate or casual conversation"""
        message_lower = message.lower()
        
        # Debate triggers
        debate_triggers = ["better", "prefer", "best", "vs", "versus", "compare", "which", "what do you think about", "opinion"]
        controversial_topics = ["saddle", "feed", "training", "method", "brand", "equipment", "technique", "hay", "silage", "grain"]
        
        is_debate_question = any(trigger in message_lower for trigger in debate_triggers)
        has_controversial_topic = any(topic in message_lower for topic in controversial_topics)
        
        if is_debate_question and has_controversial_topic:
            return "debate"
        else:
            return "conversation"
    
    def get_relevant_opinions(self, topic: str) -> str:
        """Get relevant professional opinions for debate topics"""
        topic_lower = topic.lower()
        relevant_opinions = []
        
        # Match topics to opinions
        for opinion_topic, opinion in self.professional_opinions.items():
            if any(keyword in topic_lower for keyword in opinion_topic.split('_')):
                relevant_opinions.append(f"{opinion_topic}: {opinion}")
        
        # Add controversial stances if relevant
        for stance in self.controversial_stances:
            if any(word in stance.lower() for word in topic_lower.split() if len(word) > 3):
                relevant_opinions.append(stance)
        
        return "; ".join(relevant_opinions) if relevant_opinions else "No specific opinions on this topic"
    
    def should_participate(self, message_content: str, existing_responses: List = None) -> bool:
        """Determine if this NPC should respond to a message"""
        message_lower = message_content.lower()
        existing_count = len(existing_responses or [])
        
        # Always respond if directly addressed by name
        if self.name.lower() in message_lower:
            return True
        
        # Check for expertise match
        for area in self.expertise_areas:
            area_keywords = area.replace('_', ' ').split()
            if any(keyword in message_lower for keyword in area_keywords):
                return True
        
        # For debates, more people should participate
        conversation_type = self.detect_conversation_type(message_content)
        if conversation_type == "debate":
            return existing_count < 3  # Allow more participants in debates
        
        # For general questions, be more selective
        general_indicators = ["all", "everyone", "how is", "how are"]
        if any(indicator in message_lower for indicator in general_indicators):
            return existing_count < 2  # Fewer participants for casual chat
        
        # Lower chance for random participation
        if existing_count == 0:
            return random.random() < 0.3
        
        return False
    
    def generate_response(self, user_input: str, conversation_context: List[Message] = None, others_responses: List = None) -> Tuple[str, bool, float]:
        """Generate response using persona and knowledge with conversation type awareness"""
        start_time = time.time()
        
        # Detect conversation type
        conversation_type = self.detect_conversation_type(user_input)
        
        # Build context-aware prompt
        prompt_parts = [
        f"STAY IN CHARACTER: You are {self.name}, not Dr. Evelyn or anyone else.",
        self.persona
        ]
        
        # Add relevant knowledge
        relevant_knowledge = self._get_relevant_knowledge(user_input)
        if relevant_knowledge:
            prompt_parts.append(f"What you know: {'. '.join(relevant_knowledge[:2])}")
        
        # Add conversation context
        if conversation_context:
            prompt_parts.append("Recent conversation:")
            for msg in conversation_context[-4:]:
                if msg.role == "user":
                    prompt_parts.append(f"Player: {msg.content}")
                elif msg.role == "assistant" and msg.agent_name != self.name:
                    prompt_parts.append(f"{msg.agent_name}: {msg.content}")
        
        # Different instructions based on conversation type
        if conversation_type == "debate":
            # Add professional opinions for debates
            opinions = self.get_relevant_opinions(user_input)
            if opinions != "No specific opinions on this topic":
                prompt_parts.append(f"Your professional opinions: {opinions}")
            
            if others_responses:
                others_said = [f"{name}: {resp}" for name, resp in others_responses[-2:]]
                prompt_parts.append(f"Others just said: {'; '.join(others_said)}")
                
                if self.npc_role == NPCRole.RIVAL:
                    instruction = f"Defend your position as arrogant {self.name}. Show why your expensive approach is superior. 1-2 sentences."
                else:
                    instruction = f"Give your professional opinion as {self.name}. Explain your position based on experience. 1-2 sentences."
            else:
                if self.npc_role == NPCRole.RIVAL:
                    instruction = f"Give your arrogant opinion as {self.name}. Mention expensive brands or judge cheaper alternatives. 1-2 sentences."
                else:
                    instruction = f"Share your professional opinion as {self.name} based on your expertise. 1-2 sentences."
        else:
            # Casual conversation mode
            if self.npc_role == NPCRole.RIVAL:
                instruction = f"Respond casually as dismissive {self.name}. Keep it brief and uninterested. 1 sentence."
            else:
                instruction = f"Respond naturally and supportively as {self.name}. Build on the conversation. 1-2 sentences."
        
        prompt_parts.append(instruction)
        prompt_parts.append(f"Player: {user_input}")
        prompt_parts.append(f"{self.name}:")
        
        full_prompt = "\n".join(prompt_parts)
        
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
                        "stop": ["\nPlayer:", "Player:", "\n## Instruction", "## Instruction", "\n<|user|>", "<|user|>", "Dr. Evelyn", "embodying"]
                    }
                },
                timeout=20
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                agent_response = result.get('response', '').strip()
                
                # Minimal cleanup
                if len(agent_response) < 3:
                    if self.npc_role == NPCRole.RIVAL:
                        agent_response = "Whatever."
                    else:
                        agent_response = "I see."
                
                # Record conversation
                self.add_message("user", user_input)
                self.add_message("assistant", agent_response)
                
                return agent_response, True, response_time
            else:
                return "I'm having trouble responding right now.", False, response_time
                
        except requests.exceptions.RequestException:
            end_time = time.time()
            return "Sorry, I can't respond right now.", False, end_time - start_time
    
    def _get_relevant_knowledge(self, query: str) -> List[str]:
        """Get knowledge relevant to the query"""
        query_lower = query.lower()
        relevant = []
        
        for knowledge_item in self.knowledge_base:
            if any(word in knowledge_item.lower() for word in query_lower.split() if len(word) > 3):
                relevant.append(knowledge_item)
        
        return relevant[:2]  # Limit to prevent prompt overflow
    
    def add_message(self, role: str, content: str) -> Message:
        """Add message to conversation history"""
        message = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=time.time(),
            agent_name=self.name if role == "assistant" else "player"
        )
        self.conversation_history.append(message)
        return message
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            "name": self.name,
            "role": self.npc_role.value,
            "title": self.template_data.get('title', 'NPC'),
            "total_messages": len(self.conversation_history),
            "expertise_areas": self.expertise_areas
        }

class NPCFactory:
    """Factory for creating NPCs from configuration files"""
    
    @staticmethod
    def create_npc(npc_config_name: str) -> ScalableNPCAgent:
        """Create NPC from config file name"""
        return ScalableNPCAgent(npc_config_name)
    
    @staticmethod
    def create_core_team():
        """Create the core stable team"""
        return [
            NPCFactory.create_npc("elin_behaviourist"),
            NPCFactory.create_npc("oskar_stable_hand"), 
            NPCFactory.create_npc("astrid_stable_hand"),
            NPCFactory.create_npc("chris_rival"),
            NPCFactory.create_npc("andy_trainer")
        ]

class ConversationManager:
    """Manages multi-agent conversations and debates"""
    
    def __init__(self):
        self.agents: Dict[str, ScalableNPCAgent] = {}
        self.conversation_log: List[Message] = []
    
    def register_agent(self, agent: ScalableNPCAgent):
        """Register an agent"""
        self.agents[agent.name] = agent
        role_desc = agent.template_data.get('title', agent.npc_role.value)
        print(f"🎭 Registered {agent.name} ({role_desc})")
    
    def send_to_agent(self, agent_name: str, message: str) -> Tuple[str, bool, float]:
        """Send message to specific agent"""
        agent = self.agents.get(agent_name)
        if not agent:
            return f"Agent '{agent_name}' not found.", False, 0.0
        
        # Add to global log
        self._add_to_global_log("user", message, "player")
        
        # Get response
        response, success, response_time = agent.generate_response(
            message, self.conversation_log[-10:]
        )
        
        # Log agent response
        if success:
            self._add_to_global_log("assistant", response, agent_name)
        
        return response, success, response_time
    
    def send_to_all(self, message: str) -> List[Tuple[str, str, bool, float]]:
        """Send message to all relevant agents with debate/conversation awareness"""
        # Add player message to global log
        self._add_to_global_log("user", message, "player")
        
        # Detect conversation type
        conversation_type = list(self.agents.values())[0].detect_conversation_type(message) if self.agents else "conversation"
        
        all_responses = []
        current_context = self.conversation_log.copy()
        
        # Determine initial participants
        participating_agents = []
        for agent_name, agent in self.agents.items():
            if agent.should_participate(message, []):
                participating_agents.append(agent)
        
        # For direct questions to someone, only that person should respond initially
        if any(name.lower() in message.lower() for name in self.agents.keys()):
            direct_agent = None
            for name, agent in self.agents.items():
                if name.lower() in message.lower():
                    direct_agent = agent
                    break
            if direct_agent:
                participating_agents = [direct_agent]
        
        if not participating_agents and self.agents:
            participating_agents = [list(self.agents.values())[0]]
        
        # Randomize order for more natural conversation
        random.shuffle(participating_agents)
        
        print(f"💭 {len(participating_agents)} NPCs will respond ({conversation_type} mode)")
        
        # Round 1: Initial responses
        for agent in participating_agents:
            role_desc = agent.template_data.get('title', agent.npc_role.value)
            print(f"⏳ {agent.name} ({role_desc}) responding...")
            
            # Pass other responses for debate context
            others_responses = [(resp[0], resp[1]) for resp in all_responses]
            
            response, success, response_time = agent.generate_response(
                message, current_context[-10:], others_responses
            )
            
            if success:
                all_responses.append((agent.name, response, success, response_time))
                
                # Add to context immediately
                response_msg = Message(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content=response,
                    timestamp=time.time(),
                    agent_name=agent.name
                )
                current_context.append(response_msg)
                self.conversation_log.append(response_msg)
                
                time.sleep(0.3)
        
        # Show initial round results
        print(f"\n💬 Round 1 complete:")
        for agent_name, response, success, response_time in all_responses:
            if success:
                agent = self.agents[agent_name]
                role_desc = agent.template_data.get('title', agent.npc_role.value)
                print(f"{agent_name} ({role_desc}): {response}")
        
        # Round 2: Rebuttals (only for debates with multiple participants)
        if conversation_type == "debate" and len(all_responses) > 1:
            print(f"\n💭 Round 2: Rebuttals...")
            round2_responses = []
            
            for agent_name, agent in self.agents.items():
                # 50% chance to give rebuttal in debates
                if random.random() > 0.5:
                    continue
                
                # Don't let everyone speak twice
                if len(round2_responses) >= 2:
                    break
                
                # Create rebuttal prompt based on what others said
                others_responses = [(resp[0], resp[1]) for resp in all_responses if resp[0] != agent_name]
                if others_responses:
                    rebuttal_message = f"Rebut or respond to others' opinions about: {message}"
                    
                    response, success, response_time = agent.generate_response(
                        rebuttal_message, current_context[-8:], others_responses
                    )
                    
                    if success and len(response.strip()) > 5:
                        role_desc = agent.template_data.get('title', agent.npc_role.value)
                        print(f"🗣️ {agent.name} ({role_desc}) rebuts: {response}")
                        
                        round2_responses.append((agent.name, response, success, response_time))
                        
                        # Add to context
                        response_msg = Message(
                            id=str(uuid.uuid4()),
                            role="assistant",
                            content=response,
                            timestamp=time.time(),
                            agent_name=agent.name
                        )
                        current_context.append(response_msg)
                        self.conversation_log.append(response_msg)
                        
                        time.sleep(0.3)
            
            all_responses.extend(round2_responses)
        
        return all_responses
    
    def _add_to_global_log(self, role: str, content: str, agent_name: str):
        """Add message to global conversation log"""
        message = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=time.time(),
            agent_name=agent_name
        )
        self.conversation_log.append(message)
    
    def reset_all(self):
        """Reset all conversations"""
        for agent in self.agents.values():
            agent.reset_conversation()
        self.conversation_log = []
        print("🔄 All conversations reset")
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        stats = {
            "total_agents": len(self.agents),
            "total_messages": len(self.conversation_log),
            "agents": {}
        }
        
        for agent_name, agent in self.agents.items():
            stats["agents"][agent_name] = agent.get_stats()
        
        return stats
    
    def list_agents(self) -> List[str]:
        """Get list of agent names"""
        return list(self.agents.keys()) + ["All"]

def create_npc_system():
    """Initialize the scalable NPC system"""
    print("🎭 Initializing Scalable Multi-Agent NPC System")
    print("="*60)
    
    # Create conversation manager
    manager = ConversationManager()
    
    # Load NPCs from configurations
    try:
        # Load all NPCs
        elin = NPCFactory.create_npc("elin_behaviourist")
        oskar = NPCFactory.create_npc("oskar_stable_hand")
        astrid = NPCFactory.create_npc("astrid_stable_hand")
        chris = NPCFactory.create_npc("chris_rival")
        andy = NPCFactory.create_npc("andy_trainer")

        manager.register_agent(elin)
        manager.register_agent(oskar)
        manager.register_agent(astrid)
        manager.register_agent(chris)
        manager.register_agent(andy)
        
    except Exception as e:
        print(f"⚠️  Error loading NPCs: {e}")
        print("💡 Make sure all NPC configuration files exist")
        return None
    
    return manager

def show_npc_info():
    """Display information about available NPCs"""
    print("\n🎭 NPC System Information:")
    print("  This system supports both casual conversations and structured debates")
    print("  Debate mode triggers: 'Which is better?', 'What do you prefer?', 'Compare X vs Y'")
    print("  Conversation mode: Casual chat, greetings, general questions")
    print("  Role templates: data/role_templates/")
    print("  Individual NPCs: data/npcs/")
    print("  Type 'info' anytime to see this again\n")

if __name__ == "__main__":
    print("Scalable Multi-Agent NPC System")
    print("Advanced system with conversation and debate capabilities")
    print("\nMake sure Ollama is running: ollama serve")
    print("="*60)
    
    # Initialize system
    manager = create_npc_system()
    
    if not manager:
        print("❌ Failed to initialize system. Check your configuration files.")
        exit(1)
    
    # Show system info
    show_npc_info()
    
    print(f"🎭 System ready! Available agents: {', '.join(manager.list_agents())}")
    print("\nCommands:")
    print("  - Type message for group discussion")
    print("  - 'AgentName: message' for direct conversation")
    print("  - 'info' for system information")
    print("  - 'stats' for statistics")
    print("  - 'reset' to clear conversations")
    print("  - 'quit' to exit")
    
    print(f"\n💬 Examples:")
    print(f"  Debate: 'Which saddle brand do you prefer?'")
    print(f"  Casual: 'How is everyone doing today?'")
    
    while True:
        try:
            user_input = input(f"\nPlayer: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'info':
                show_npc_info()
                continue
            elif user_input.lower() == 'stats':
                stats = manager.get_stats()
                print(f"📊 Total messages: {stats['total_messages']}")
                for agent_name, agent_stats in stats['agents'].items():
                    print(f"  {agent_name} ({agent_stats['title']}): {agent_stats['total_messages']} messages")
                continue
            elif user_input.lower() == 'reset':
                manager.reset_all()
                continue
            
            # Check for direct agent messaging
            if ':' in user_input and user_input.count(':') == 1:
                agent_name, message = user_input.split(':', 1)
                agent_name = agent_name.strip()
                message = message.strip()
                
                if agent_name == "All":
                    # Group conversation - results shown in send_to_all
                    responses = manager.send_to_all(message)
                elif agent_name in manager.agents:
                    # Individual conversation
                    response, success, response_time = manager.send_to_agent(agent_name, message)
                    if success:
                        agent = manager.agents[agent_name]
                        role_desc = agent.template_data.get('title', agent.npc_role.value)
                        print(f"{agent_name} ({role_desc}): {response}")
                    else:
                        print(f"❌ {agent_name}: {response}")
                else:
                    print(f"❌ Agent '{agent_name}' not found. Available: {', '.join(manager.agents.keys())}")
                continue
            
            # Default to group conversation
            responses = manager.send_to_all(user_input)
            
        except KeyboardInterrupt:
            break
    
    print("\n👋 Thanks for testing the multi-agent system with debate capabilities!")