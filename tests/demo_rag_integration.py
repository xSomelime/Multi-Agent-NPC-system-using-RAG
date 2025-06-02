#!/usr/bin/env python3
"""
RAG Integration Demo
Demonstrates how to use the RAG-enhanced NPC system
"""

import sys
import logging
from typing import Dict, List

# Add src to path
sys.path.append('src')

def demo_rag_enhanced_system():
    """Demonstrate the RAG-enhanced NPC system"""
    
    print("🚀 RAG-Enhanced NPC System Demo")
    print("=" * 60)
    
    try:
        # Import components
        from memory.session_memory import MemoryManager
        from agents.rag_enhanced_agent import create_rag_enhanced_team
        
        print("✅ Successfully imported RAG components")
        
        # Initialize memory manager
        memory_manager = MemoryManager()
        print("✅ Memory manager initialized")
        
        # Create RAG-enhanced team
        print("\n🎭 Creating RAG-Enhanced NPCs...")
        rag_team = create_rag_enhanced_team(memory_manager, enable_rag=True)
        
        print(f"✅ Created {len(rag_team)} RAG-enhanced NPCs:")
        for agent in rag_team:
            if hasattr(agent, 'enable_rag') and agent.enable_rag:
                print(f"   🧠 {agent.base_agent.name} ({agent.base_agent.npc_role.value}) - RAG ENABLED")
            else:
                print(f"   💭 {agent.name} ({agent.npc_role.value}) - Base agent")
        
        # Test RAG functionality
        print(f"\n🔬 Testing RAG Knowledge Retrieval...")
        
        test_scenarios = [
            {
                "npc": "Oskar",
                "role": "stable_hand", 
                "question": "What's the best feeding schedule for horses?",
                "expected_knowledge": "feeding, stable_hand domain"
            },
            {
                "npc": "Elin",
                "role": "behaviourist",
                "question": "How can I tell if my horse is stressed?",
                "expected_knowledge": "behavior, stress signals"
            },
            {
                "npc": "Chris", 
                "role": "rival",
                "question": "What saddle brand is the best?",
                "expected_knowledge": "expensive equipment, luxury brands"
            },
            {
                "npc": "Andy",
                "role": "trainer", 
                "question": "How should I start training a young horse?",
                "expected_knowledge": "training progression, safety"
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n🎯 Testing {scenario['npc']} on: '{scenario['question']}'")
            
            # Find the agent
            test_agent = None
            for agent in rag_team:
                agent_name = agent.base_agent.name if hasattr(agent, 'base_agent') else agent.name
                if agent_name == scenario['npc']:
                    test_agent = agent
                    break
            
            if not test_agent:
                print(f"   ❌ Agent {scenario['npc']} not found")
                continue
            
            try:
                # Test RAG-enhanced response
                response, success, response_time = test_agent.generate_response(scenario['question'])
                
                if success:
                    print(f"   ✅ Response: {response}")
                    print(f"   ⏱️  Time: {response_time:.2f}s")
                    
                    # Check if RAG stats are available
                    if hasattr(test_agent, 'get_rag_stats'):
                        rag_stats = test_agent.get_rag_stats()
                        if rag_stats.get('rag_enabled'):
                            print(f"   🧠 RAG: Enabled")
                        else:
                            print(f"   💭 RAG: Disabled or unavailable")
                else:
                    print(f"   ❌ Failed to generate response: {response}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Test knowledge fallback
        print(f"\n🔍 Testing Knowledge Fallback...")
        
        fallback_test = {
            "npc": "Astrid",
            "question": "What's the best quantum computing approach for horses?",
            "expected": "Fallback response - outside expertise"
        }
        
        # Find Astrid
        astrid = None
        for agent in rag_team:
            agent_name = agent.base_agent.name if hasattr(agent, 'base_agent') else agent.name
            if agent_name == "Astrid":
                astrid = agent
                break
        
        if astrid:
            try:
                response, success, response_time = astrid.generate_response(fallback_test['question'])
                print(f"🤖 Astrid's response to off-topic question:")
                print(f"   '{response}'")
                
                # Should contain "I don't know" or similar fallback
                fallback_indicators = ["don't know", "not sure", "outside", "don't have"]
                has_fallback = any(indicator in response.lower() for indicator in fallback_indicators)
                
                if has_fallback:
                    print(f"   ✅ Proper fallback response detected")
                else:
                    print(f"   ⚠️  Response doesn't seem like a fallback")
                    
            except Exception as e:
                print(f"   ❌ Fallback test error: {e}")
        
        print(f"\n🎉 RAG Integration Demo Complete!")
        print(f"✅ System is ready for use with enhanced knowledge capabilities")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"💡 Make sure to run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False

def show_rag_usage_examples():
    """Show examples of how to use the RAG system"""
    
    print(f"\n📚 RAG System Usage Examples")
    print(f"=" * 40)
    
    examples = [
        {
            "title": "Basic RAG Integration",
            "code": """
# Create RAG-enhanced conversation manager
from memory.session_memory import MemoryManager
from agents.rag_enhanced_agent import create_rag_enhanced_team

memory_manager = MemoryManager()
rag_team = create_rag_enhanced_team(memory_manager)

# Register with conversation manager
for agent in rag_team:
    conversation_manager.register_agent(agent)
"""
        },
        {
            "title": "Manual RAG Agent Creation", 
            "code": """
# Create individual RAG-enhanced agent
from agents.rag_enhanced_agent import create_rag_enhanced_agent

oskar = create_rag_enhanced_agent(
    "oskar_stable_hand", 
    memory_manager, 
    location="stable_yard",
    enable_rag=True
)
"""
        },
        {
            "title": "Testing RAG Knowledge",
            "code": """
# Test if agent has knowledge about a topic
context = NPCKnowledgeContext(
    npc_name="Elin",
    role="behaviourist", 
    current_topic="horse stress signals",
    conversation_history=[]
)

rag_interface = get_npc_rag_interface()
knowledge_items = rag_interface.get_knowledge_for_npc(context)
"""
        },
        {
            "title": "Fallback Handling",
            "code": """
# Get appropriate fallback when no knowledge available
fallback = rag_interface.get_fallback_response(
    "quantum horse physics",  # Nonsense topic
    "stable_hand",
    "Oskar"
)
# Returns: "I don't know about quantum horse physics specifically..."
"""
        }
    ]
    
    for example in examples:
        print(f"\n📝 {example['title']}:")
        print(f"{example['code']}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    print("🧪 RAG Integration Testing")
    print("Make sure Ollama is running: ollama serve")
    print("=" * 60)
    
    try:
        # Run the demo
        success = demo_rag_enhanced_system()
        
        if success:
            # Show usage examples
            show_rag_usage_examples()
            
            print(f"\n🎯 Next Steps:")
            print(f"1. Install dependencies: pip install -r requirements.txt")
            print(f"2. Test the RAG system: python test_rag_system.py")
            print(f"3. Replace base agents with RAG agents in main_npc_system.py")
            print(f"4. Enjoy incredibly knowledgeable NPCs! 🐴")
        else:
            print(f"\n❌ Demo failed - check error messages above")
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logging.exception("Demo error details:") 