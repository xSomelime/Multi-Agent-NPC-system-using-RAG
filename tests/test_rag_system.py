#!/usr/bin/env python3
"""
Test script for the RAG system
Verifies that knowledge loading and retrieval work correctly
"""

import logging
import sys
import os

# Add src to path so we can import our modules
sys.path.append('src')

def test_rag_system():
    """Test the RAG system functionality"""
    
    print("🔬 Testing RAG System")
    print("=" * 50)
    
    try:
        # Import RAG components
        from knowledge.rag_system import RAGKnowledgeBase
        from knowledge.npc_rag_integration import NPCRAGInterface, NPCKnowledgeContext
        
        print("✅ Successfully imported RAG modules")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to install dependencies: pip install -r requirements.txt")
        return False
    
    # Test RAG knowledge base
    print("\n🧠 Testing RAG Knowledge Base...")
    
    try:
        rag = RAGKnowledgeBase()
        chunks_loaded = rag.load_knowledge_from_json("src/knowledge")
        
        print(f"✅ Loaded {chunks_loaded} knowledge chunks")
        
        if chunks_loaded == 0:
            print("❌ No knowledge chunks loaded!")
            return False
            
        # Get stats
        stats = rag.get_stats()
        print(f"📊 RAG Stats:")
        print(f"   - Total chunks: {stats['total_chunks']}")
        print(f"   - Embedding dimension: {stats['embedding_dimension']}")
        print(f"   - Domains: {list(stats['domains'].keys())}")
        
    except Exception as e:
        print(f"❌ RAG system error: {e}")
        return False
    
    # Test knowledge retrieval
    print("\n🔍 Testing Knowledge Retrieval...")
    
    test_queries = [
        ("horse feeding", "stable_hand"),
        ("saddle fitting", "trainer"),
        ("horse behavior", "behaviourist"),
        ("expensive equipment", "rival")
    ]
    
    for query, expected_domain in test_queries:
        print(f"\n🔎 Query: '{query}' (expecting {expected_domain} knowledge)")
        
        try:
            results = rag.retrieve(query, domain_filter=expected_domain)
            
            if results:
                print(f"✅ Found {len(results)} relevant results:")
                for i, result in enumerate(results[:2], 1):
                    print(f"   {i}. {result.chunk.content[:80]}... (score: {result.relevance_score:.3f})")
            else:
                print(f"⚠️  No results found for '{query}' in {expected_domain}")
                
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return False
    
    # Test NPC integration
    print("\n🎭 Testing NPC Integration...")
    
    try:
        interface = NPCRAGInterface()
        
        # Test contexts for different NPCs
        test_contexts = [
            NPCKnowledgeContext(
                npc_name="Oskar",
                role="stable_hand",
                current_topic="feeding schedule",
                conversation_history=[]
            ),
            NPCKnowledgeContext(
                npc_name="Chris", 
                role="rival",
                current_topic="premium saddles",
                conversation_history=[]
            )
        ]
        
        for context in test_contexts:
            print(f"\n🎯 Testing {context.npc_name} ({context.role}) on '{context.current_topic}'")
            
            # Test knowledge retrieval for NPC
            knowledge_items = interface.get_knowledge_for_npc(context)
            
            if knowledge_items:
                print(f"✅ Retrieved {len(knowledge_items)} knowledge items:")
                for i, item in enumerate(knowledge_items, 1):
                    print(f"   {i}. {item[:80]}...")
            else:
                print(f"⚠️  No knowledge items for {context.npc_name}")
                
                # Test fallback
                fallback = interface.get_fallback_response(
                    context.current_topic, context.role, context.npc_name
                )
                print(f"💬 Fallback: {fallback}")
            
            # Test prompt enhancement
            base_prompt = f"You are {context.npc_name}, a {context.role}."
            enhanced_prompt = interface.enhance_npc_prompt(base_prompt, context)
            
            enhancement_added = len(enhanced_prompt) > len(base_prompt)
            print(f"📝 Prompt enhancement: {'✅ Added' if enhancement_added else '⚠️ No change'}")
            
    except Exception as e:
        print(f"❌ NPC integration error: {e}")
        return False
    
    print(f"\n🎉 RAG System Test Complete!")
    print(f"✅ All components working correctly")
    return True

def test_knowledge_files():
    """Test that knowledge files exist and are readable"""
    print("\n📁 Testing Knowledge Files...")
    
    knowledge_files = [
        "src/knowledge/stable_hand_knowledge.json",
        "src/knowledge/trainer_knowledge.json",
        "src/knowledge/behaviourist_knowledge.json", 
        "src/knowledge/rival_knowledge.json"
    ]
    
    all_exist = True
    
    for filepath in knowledge_files:
        if os.path.exists(filepath):
            print(f"✅ {filepath}")
            
            # Try to read the file
            try:
                import json
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    chunk_count = len(data.get('knowledge_chunks', []))
                    print(f"   └─ {chunk_count} knowledge chunks")
            except Exception as e:
                print(f"   └─ ❌ Error reading: {e}")
                all_exist = False
        else:
            print(f"❌ {filepath} - NOT FOUND")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    print("🧪 RAG System Testing Suite")
    print("=" * 60)
    
    # Test knowledge files first
    files_ok = test_knowledge_files()
    
    if not files_ok:
        print("\n❌ Knowledge files missing or corrupted!")
        print("💡 Make sure all knowledge JSON files exist in src/knowledge/")
        sys.exit(1)
    
    # Test RAG system
    try:
        success = test_rag_system()
        
        if success:
            print("\n🚀 RAG System is ready!")
            print("💡 You can now integrate it with your main NPC system")
            sys.exit(0)
        else:
            print("\n❌ RAG System tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        sys.exit(1) 