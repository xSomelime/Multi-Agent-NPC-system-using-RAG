#!/usr/bin/env python3
"""
FastAPI Server for Multi-Agent NPC System
=======================================

Provides REST API endpoints for:
- Individual and group conversations with NPCs
- Location management
- System status and control
"""

import os
import sys
import logging
import requests
import json
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True
        return False
    except requests.exceptions.RequestException:
        return False

# Check Ollama service first
if not check_ollama_service():
    print("❌ Ollama service is not running!")
    print("Please start Ollama with 'ollama serve' in a separate terminal")
    sys.exit(1)

print("✅ Ollama service is running")

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.system_manager import create_enhanced_npc_system

# Initialize the NPC system in UE5 mode
print("🔄 Initializing NPC system in UE5 mode...")
manager = create_enhanced_npc_system(enable_rag=True, mode="ue5")
if not manager:
    print("❌ Failed to initialize NPC system")
    sys.exit(1)

print("✅ NPC system initialized successfully")
print("👥 Available NPCs:", manager.list_agents())

# Create FastAPI app
app = FastAPI(
    title="Multi-Agent NPC System",
    description="REST API for horse management game NPCs with dynamic locations",
    version="2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint - system status"""
    return {
        "status": "running",
        "version": "2.0",
        "features": ["dynamic_locations", "memory", "rag"]
    }

@app.post("/conversation")
async def send_message(data: Dict = Body(...)):
    """Send message to specific NPC"""
    try:
        # Log raw request data with more detail
        logger.info("Received request data:")
        logger.info(f"Type: {type(data)}")
        logger.info(f"Content: {data}")
        
        # If data is string, try to clean and parse it
        if isinstance(data, str):
            try:
                # Clean the string of any potential problematic characters
                cleaned_data = data.strip().replace('\x00', '')
                data = json.loads(cleaned_data)
                logger.info(f"Parsed string data into: {data}")
            except json.JSONDecodeError as je:
                logger.error(f"JSON parsing error: {je}")
                return {
                    "error": "Invalid JSON format",
                    "details": str(je),
                    "received": data
                }
        
        # Try to get npc_id and message, being lenient with casing
        npc_id = None
        message = None
        location = None
        is_end = False
        
        # Check common variations of field names
        for key in data:
            key_lower = key.lower()
            if key_lower in ['npc_id', 'npcid', 'npc']:
                npc_id = str(data[key]).lower()
            elif key_lower in ['message', 'msg', 'text', 'player_message']:
                message = str(data[key])
            elif key_lower == 'location':
                location = str(data[key])
            elif key_lower == 'is_end':
                is_end = bool(data[key])

        # Log parsed values
        logger.info(f"Parsed values - NPC: {npc_id}, Message: {message}, Location: {location}")

        # Update location if provided
        if location:
            try:
                manager.move_player_to_location(location)
                logger.info(f"Updated player location to: {location}")
            except Exception as e:
                logger.warning(f"Failed to update location: {e}")

        # Get list of available NPCs and create mapping
        available_npcs = manager.list_agents()
        npc_map = {}
        
        # Map both full names and short names
        for full_name in available_npcs:
            npc_map[full_name.lower()] = full_name
            # Map short name (e.g., "oskar" -> "oskar_stable_hand")
            short_name = full_name.split('_')[0].lower()
            npc_map[short_name] = full_name
        
        if not npc_id:
            return {
                "npc": "unknown",
                "message": f"Please specify an NPC. Available NPCs: {list(available_npcs)}",
                "success": False,
                "time": 0
            }
            
        # Try to find the full NPC name
        actual_npc_id = npc_map.get(npc_id.lower())
        if not actual_npc_id:
            return {
                "npc": npc_id,
                "message": f"NPC '{npc_id}' not found. Available NPCs: {list(available_npcs)}",
                "success": False,
                "time": 0
            }
        
        # Get response from NPC using the full name
        response, success, response_time = manager.send_to_agent(actual_npc_id, message or "")
        
        result = {
            "npc": actual_npc_id,
            "message": response,
            "success": success,
            "time": response_time
        }
        return result
        
    except Exception as e:
        logger.error(f"Error in conversation: {str(e)}")
        logger.error(f"Request data was: {data}")
        return {
            "error": "Failed to process request",
            "details": str(e),
            "received_data": str(data)
        }

@app.get("/npcs")
async def list_npcs():
    """Get list of available NPCs"""
    try:
        npcs = manager.list_agents()
        return {"npcs": npcs}
    except Exception as e:
        logger.error(f"Error listing NPCs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation_targets")
async def get_conversation_targets():
    """Get NPCs available for conversation at current location"""
    try:
        return manager.get_available_conversation_targets()
    except Exception as e:
        logger.error(f"Error getting conversation targets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/group_conversation/start")
async def start_group_conversation(data: Dict = Body(...)):
    """Start a group conversation with selected NPCs"""
    try:
        npc_names = data.get('npc_names', [])
        message = data.get('message', '')
        
        if not npc_names or not message:
            raise HTTPException(status_code=400, detail="Missing npc_names or message")
            
        result = manager.start_group_conversation(npc_names, message)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    except Exception as e:
        logger.error(f"Error starting group conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/group_conversation/{conversation_id}/message")
async def send_group_message(conversation_id: str, data: Dict = Body(...)):
    """Send message to an active group conversation"""
    try:
        message = data.get('message', '')
        if not message:
            raise HTTPException(status_code=400, detail="Missing message")
            
        if conversation_id not in manager.active_group_conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        npc_names = manager.active_group_conversations[conversation_id]
        responses = []
        
        for name in npc_names:
            response, success, time = manager.send_to_agent(name, message)
            if success:
                responses.append({
                    "npc": name,
                    "response": response,
                    "time": time
                })
        
        return {"responses": responses}
    except Exception as e:
        logger.error(f"Error in group conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/group_conversation/{conversation_id}/end")
async def end_group_conversation(conversation_id: str):
    """End a group conversation"""
    try:
        manager.end_group_conversation(conversation_id)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error ending group conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_locations")
async def update_locations(data: Dict = Body(...)):
    """Update NPC locations from UE5"""
    try:
        locations = data.get('locations', {})
        success, message = manager.update_ue5_locations(locations)
        if not success:
            raise HTTPException(status_code=400, detail=message)
        return {"status": "success", "message": message}
    except Exception as e:
        logger.error(f"Error updating locations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)