#!/usr/bin/env python3
"""
Ollama Request Manager
=====================

Thread-safe request manager for Ollama API calls to prevent response mixing
and ensure clean, validated responses from the LLM.
"""

import requests
import threading
from typing import Tuple


class OllamaRequestManager:
    """Manages sequential requests to Ollama to prevent response mixing"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.ollama_url = "http://localhost:11434/api/generate"
        
    def make_request(self, agent_name: str, prompt: str, model: str, temperature: float, max_tokens: int) -> Tuple[str, bool]:
        """Thread-safe request to Ollama"""
        with self.lock:
            try:
                print(f"🔄 {agent_name} making request to Ollama...")
                
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                            "stop": [
                                "\nPlayer:", "Player:", 
                                f"\n{agent_name}:", f"{agent_name}:",
                                "\nHuman:", "Human:",
                                "\n\n", "\\n\\n",
                                "Dr. Evelyn", "embodying",
                                "\n---", "---",
                                "\n## Instruction", "## Instruction"
                            ]
                        }
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    agent_response = result.get('response', '').strip()
                    
                    # Clean up response
                    agent_response = self._clean_response(agent_response, agent_name)
                    
                    print(f"✅ {agent_name} got clean response: {agent_response[:50]}...")
                    return agent_response, True
                else:
                    print(f"❌ {agent_name} request failed: {response.status_code}")
                    return f"I'm having trouble responding right now.", False
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ {agent_name} request error: {e}")
                return "Sorry, I can't respond right now.", False
    
    def _clean_response(self, response: str, agent_name: str) -> str:
        """Clean and validate response"""
        if not response or len(response) < 3:
            return "I see."
        
        # Remove common artifacts
        artifacts_to_remove = [
            "Dr. Evelyn", "embodying", "## Instruction",
            "<|user|>", "<|assistant|>", "Human:", "Player:",
            "\n\n", "\\n\\n", "\n---", "---"
        ]
        
        cleaned = response
        for artifact in artifacts_to_remove:
            cleaned = cleaned.replace(artifact, "")
        
        # Split into sentences and remove duplicates
        sentences = []
        for sent in cleaned.replace('!', '.').replace('?', '.').split('.'):
            sent = sent.strip()
            if sent and len(sent) > 10:
                sentences.append(sent)
        
        # Remove duplicate sentences
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if sentence_lower not in seen and len(sentence) > 10:
                unique_sentences.append(sentence)
                seen.add(sentence_lower)
                if len(unique_sentences) >= 2:  # Max 2 sentences
                    break
        
        if unique_sentences:
            result = '. '.join(unique_sentences)
            if not result.endswith('.'):
                result += '.'
            return result
        
        return "I see."


# Global request manager instance
_ollama_manager = None

def get_ollama_manager() -> OllamaRequestManager:
    """Get or create global Ollama manager instance"""
    global _ollama_manager
    
    if _ollama_manager is None:
        _ollama_manager = OllamaRequestManager()
    
    return _ollama_manager 