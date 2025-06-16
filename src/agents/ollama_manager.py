#!/usr/bin/env python3
"""
Ollama Request Manager
=====================

Thread-safe request manager for Ollama API calls with caching and optimized processing.
"""

import requests
import threading
import time
import subprocess
import os
import sys
import logging
from typing import Tuple, Dict, Optional
import hashlib

logger = logging.getLogger(__name__)

def start_ollama_service() -> bool:
    """Start Ollama service if not running"""
    try:
        # Check if Ollama is already running
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        pass  # Service not running, continue to start it
    
    try:
        # Try to start Ollama service
        if sys.platform == "win32":
            # Windows
            subprocess.Popen(["ollama", "serve"], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            # Unix-like
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        
        # Wait for service to start
        for _ in range(10):  # Try for 10 seconds
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=1)
                if response.status_code == 200:
                    logger.info("✅ Ollama service started successfully")
                    return True
            except requests.exceptions.RequestException:
                time.sleep(1)
                continue
        
        logger.error("❌ Failed to start Ollama service")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error starting Ollama service: {e}")
        return False

def ensure_ollama_model(model_name: str = "phi3:mini") -> bool:
    """Ensure required model is available"""
    try:
        # Check if model exists
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            if any(model['name'] == model_name for model in models):
                return True
        
        # Model not found, try to pull it
        logger.info(f"🔄 Pulling {model_name} model...")
        subprocess.run(["ollama", "pull", model_name], check=True)
        logger.info(f"✅ {model_name} model pulled successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error ensuring model availability: {e}")
        return False

class OllamaRequestManager:
    """Manages sequential requests to Ollama with caching to improve response time"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.ollama_url = "http://localhost:11434/api/generate"
        self.response_cache: Dict[str, Tuple[str, float]] = {}
        self.cache_ttl = 300  # Cache responses for 5 minutes
        self.cache_size = 1000  # Maximum number of cached responses
        
        # Ensure Ollama is running and model is available
        if not start_ollama_service():
            raise RuntimeError("Failed to start Ollama service")
        if not ensure_ollama_model():
            raise RuntimeError("Failed to ensure model availability")
    
    def _get_cache_key(self, prompt: str, model: str, temperature: float) -> str:
        """Generate cache key for request parameters"""
        key_string = f"{prompt}|{model}|{temperature}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if valid"""
        if cache_key in self.response_cache:
            response, timestamp = self.response_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return response
            else:
                del self.response_cache[cache_key]
        return None
        
    def make_request(self, agent_name: str, prompt: str, model: str, temperature: float, max_tokens: int) -> Tuple[str, bool]:
        """Thread-safe request to Ollama with caching"""
        cache_key = self._get_cache_key(prompt, model, temperature)
        
        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response, True
        
        with self.lock:
            try:
                print(f"💭 {agent_name} thinking...")
                
                # Optimize stop sequences for faster processing
                stop_sequences = [
                    "\nPlayer:", "Player:", 
                    f"\n{agent_name}:", f"{agent_name}:",
                    "\nHuman:", "Human:",
                    "\n\n"
                ]
                
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                            "stop": stop_sequences
                        }
                    },
                    timeout=10  # Reduced timeout for faster failure detection
                )
                
                if response.status_code == 200:
                    result = response.json()
                    agent_response = result.get('response', '').strip()
                    
                    # Simplified cleaning for faster processing
                    agent_response = self._clean_response(agent_response, agent_name)
                    
                    # Cache the response
                    if len(self.response_cache) >= self.cache_size:
                        # Remove oldest entry if cache is full
                        oldest_key = min(self.response_cache.keys(), 
                                       key=lambda k: self.response_cache[k][1])
                        del self.response_cache[oldest_key]
                    
                    self.response_cache[cache_key] = (agent_response, time.time())
                    
                    return agent_response, True
                else:
                    print(f"{agent_name} request failed: {response.status_code}")
                    return f"I'm having trouble responding right now.", False
                    
            except requests.exceptions.RequestException as e:
                print(f"{agent_name} request error: {e}")
                return "Sorry, I can't respond right now.", False
    
    def _clean_response(self, response: str, agent_name: str) -> str:
        """Optimized response cleaning"""
        if not response or len(response) < 3:
            return "I see."
        
        # Simplified cleaning for better performance
        cleaned = response.replace("Dr. Evelyn", "").replace("embodying", "")
        
        # Quick sentence extraction
        sentences = cleaned.split('.')
        result = []
        seen = set()
        
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) > 10:
                sent_lower = sent.lower()
                if sent_lower not in seen:
                    result.append(sent)
                    seen.add(sent_lower)
                    if len(result) >= 2:  # Max 2 sentences
                        break
        
        if result:
            final = '. '.join(result)
            return final + ('.' if not final.endswith('.') else '')
        
        return "I see."


# Global request manager instance
_ollama_manager = None

def get_ollama_manager() -> OllamaRequestManager:
    """Get or create global Ollama manager instance"""
    global _ollama_manager
    
    if _ollama_manager is None:
        _ollama_manager = OllamaRequestManager()
    
    return _ollama_manager 