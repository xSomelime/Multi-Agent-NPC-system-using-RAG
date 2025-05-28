#!/usr/bin/env python3
"""
Simplified Ollama NPC Test - Manual Model Management
Story 1.0: LLM Foundation & Model Testing

Run this after manually pulling models:
ollama pull phi3:mini
ollama pull gemma2:2b
"""

import requests
import json
import time
from typing import Dict, List, Tuple
import statistics

class SimpleOllamaTest:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        
    def check_ollama_status(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_available_models(self) -> List[str]:
        """Get list of locally available models"""
        try:
            response = requests.get(f"{self.api_url}/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except requests.exceptions.RequestException:
            return []
    
    def test_model_response(self, model_name: str, prompt: str, temperature: float = 0.3) -> Tuple[str, float, bool]:
        """Test model response with timing"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": 100  # Shorter for faster testing
                    }
                },
                timeout=30
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', ''), response_time, True
            else:
                return f"Error: {response.status_code}", response_time, False
                
        except requests.exceptions.RequestException as e:
            end_time = time.time()
            return f"Request failed: {e}", end_time - start_time, False

def run_simple_npc_test():
    """Simple NPC personality test with available models"""
    tester = SimpleOllamaTest()
    
    print("=== Simple Ollama NPC Test ===")
    
    # Check status
    if not tester.check_ollama_status():
        print("❌ Ollama not running. Start with: ollama serve")
        return
    
    print("✓ Ollama is running")
    
    # List available models
    available_models = tester.list_available_models()
    print(f"\nAvailable models: {available_models}")
    
    if not available_models:
        print("\n❌ No models found!")
        print("Download models first:")
        print("  ollama pull phi3:mini")
        print("  ollama pull gemma2:2b")
        return
    
    # Test prompts for NPCs
    test_cases = [
        {
            "name": "Anna (Stable Hand)",
            "prompt": "You are Anna, a practical stable hand. A player asks: 'How do I feed my horse?' Give a brief, helpful answer.",
            "temperature": 0.2
        },
        {
            "name": "Erik (Trainer)", 
            "prompt": "You are Erik, an experienced trainer. A player says: 'My horse won't jump.' Give technical advice.",
            "temperature": 0.3
        },
        {
            "name": "Lisa (Health Monitor)",
            "prompt": "You are Lisa, focused on horse health. A player mentions: 'My horse seems tired.' Respond with health concerns.",
            "temperature": 0.1
        }
    ]
    
    results = {}
    
    # Test each available model
    for model in available_models:
        print(f"\n{'='*40}")
        print(f"Testing: {model}")
        print('='*40)
        
        model_results = {}
        
        for test_case in test_cases:
            print(f"\n🧪 Testing {test_case['name']}...")
            
            response, response_time, success = tester.test_model_response(
                model, 
                test_case['prompt'], 
                test_case['temperature']
            )
            
            model_results[test_case['name']] = {
                'response': response,
                'response_time': response_time,
                'success': success,
                'temperature': test_case['temperature']
            }
            
            if success:
                print(f"⏱️  Response time: {response_time:.2f}s")
                print(f"🤖 Response: {response[:150]}...")
                
                # Quick quality assessment
                if response_time < 2.0:
                    print("✅ Fast enough for real-time NPC")
                elif response_time < 5.0:
                    print("⚠️  Acceptable for turn-based NPC")
                else:
                    print("❌ Too slow for game use")
            else:
                print(f"❌ Failed: {response}")
        
        results[model] = model_results
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY & RECOMMENDATIONS")
    print('='*50)
    
    best_for_realtime = []
    best_for_quality = []
    
    for model_name, model_data in results.items():
        response_times = [data['response_time'] for data in model_data.values() if data['success']]
        success_rate = sum(1 for data in model_data.values() if data['success']) / len(model_data) if model_data else 0
        
        if response_times:
            avg_time = statistics.mean(response_times)
            
            print(f"\n📊 {model_name}:")
            print(f"   Success Rate: {success_rate:.1%}")
            print(f"   Average Time: {avg_time:.2f}s")
            
            if success_rate > 0.8:
                if avg_time < 2.0:
                    best_for_realtime.append((model_name, avg_time))
                    print("   ✅ EXCELLENT for real-time NPCs")
                elif avg_time < 5.0:
                    best_for_quality.append((model_name, avg_time))
                    print("   ✅ GOOD for quality NPCs")
                else:
                    print("   ⚠️  Too slow for game use")
            else:
                print("   ❌ Poor success rate")
    
    # Final recommendation
    print(f"\n🎯 RECOMMENDATION FOR STORY 1.1:")
    if best_for_realtime:
        model, time = min(best_for_realtime, key=lambda x: x[1])
        print(f"   Use {model} for multi-agent NPCs ({time:.2f}s avg)")
    elif best_for_quality:
        model, time = min(best_for_quality, key=lambda x: x[1])
        print(f"   Use {model} for NPCs ({time:.2f}s avg) - consider turn-based")
    else:
        print("   ❌ No suitable models found - try different models or cloud API")
    
    return results

if __name__ == "__main__":
    print("Story 1.0: LLM Foundation & Model Testing")
    print("Make sure you've downloaded models first:")
    print("  ollama pull phi3:mini")
    print("  ollama pull gemma2:2b")
    print("\nStarting NPC personality tests...")
    print("="*60)
    
    results = run_simple_npc_test()
    
    print(f"\n{'='*60}")
    print("✅ Story 1.0 COMPLETE!")
    print("Next: Story 1.1 - Basic Agent Architecture")
    print("="*60)