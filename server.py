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
import traceback
import time
from typing import Dict, List, Optional, Tuple, Any
from fastapi import FastAPI, HTTPException, Body, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import core components
try:
    from src.core.message_types import NPCRole, Message
    from src.core.config_loaders import RoleTemplateLoader, NPCLoader
    from src.core.system_manager import create_enhanced_npc_system, show_npc_info, get_conversation_system
except ImportError as e:
    logger.error(f"❌ Failed to import core components: {e}")
    sys.exit(1)

# Import agent components
try:
    from src.agents.base_agent import ScalableNPCAgent
    from src.agents.npc_factory import NPCFactory
    from src.agents.ollama_manager import OllamaRequestManager, get_ollama_manager, start_ollama_service, ensure_ollama_model
except ImportError as e:
    logger.error(f"❌ Failed to import agent components: {e}")
    sys.exit(1)

# Import coordination components
try:
    from src.coordination.conversational_momentum import ConversationalMomentum
    from src.coordination.conversation_manager import EnhancedConversationManager
except ImportError as e:
    logger.error(f"❌ Failed to import coordination components: {e}")
    sys.exit(1)

# Initialize Ollama service and model
logger.info("🔄 Starting Ollama service...")
if not start_ollama_service():
    logger.error("❌ Failed to start Ollama service")
    sys.exit(1)

logger.info("🔄 Ensuring model availability...")
if not ensure_ollama_model():
    logger.error("❌ Failed to ensure model availability")
    sys.exit(1)

# Global request manager with error handling
try:
    ollama_manager = get_ollama_manager()
except Exception as e:
    logger.error(f"❌ Failed to initialize Ollama manager: {e}")
    sys.exit(1)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.core.system_manager import create_enhanced_npc_system
except ImportError as e:
    logger.error(f"❌ Failed to import system manager: {e}")
    sys.exit(1)

# Initialize the NPC system in UE5 mode
logger.info("🔄 Initializing NPC system in UE5 mode...")
try:
    manager = create_enhanced_npc_system(enable_rag=True, mode="ue5")
    if not manager:
        logger.error("❌ Failed to initialize NPC system")
        sys.exit(1)
except Exception as e:
    logger.error(f"❌ Error initializing NPC system: {e}")
    logger.debug(traceback.format_exc())
    sys.exit(1)

logger.info("✅ NPC system initialized successfully")
logger.info(f"👥 Available NPCs: {manager.list_agents()}")

# Pydantic models for request/response validation
class ConversationRequest(BaseModel):
    npc_name: str = Field(..., description="Name of the NPC to talk to")
    message: str = Field(..., description="Message to send to the NPC")
    location: Optional[str] = Field(None, description="Current player location")

class LocationUpdate(BaseModel):
    npc_name: str = Field(..., description="Name of the NPC to move")
    location: str = Field(..., description="New location for the NPC")
    reason: Optional[str] = Field(None, description="Reason for the movement")

class SystemStatus(BaseModel):
    status: str = Field(..., description="System status")
    version: str = Field(..., description="API version")
    features: List[str] = Field(..., description="Available features")
    npcs: List[str] = Field(..., description="Available NPCs")
    current_location: str = Field(..., description="Current player location")

# Create FastAPI app
app = FastAPI(
    title="NPC Conversation System",
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

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions"""
    logger.error(f"❌ Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )

@app.get("/", response_model=SystemStatus)
async def root():
    """Root endpoint - system status"""
    try:
        return {
            "status": "running",
            "version": "2.0",
            "features": ["dynamic_locations", "memory", "rag"],
            "npcs": manager.list_agents(),
            "current_location": manager.current_location or "unknown"
        }
    except Exception as e:
        logger.error(f"Error in root endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/npcs")
async def get_available_npcs():
    """Get list of NPCs available for conversation at current location"""
    try:
        targets = manager.conversation_manager.get_available_conversation_targets()
        return {
            "location": targets["location"],
            "npcs": targets["npcs"],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting available NPCs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversation")
async def send_message(request: ConversationRequest):
    """Send message to specific NPC and get their response.
    This is the main endpoint used by Unreal Engine for NPC interactions."""
    try:
        # Validate NPC exists
        if request.npc_name not in manager.agents:
            raise HTTPException(
                status_code=404,
                detail=f"NPC '{request.npc_name}' not found"
            )
            
        # Update player location if provided
        if request.location:
            try:
                manager.move_player_to_location(request.location)
            except Exception as e:
                logger.warning(f"Failed to update player location: {e}")
        
        # If no message, end conversation
        if not request.message:
            manager.conversation_manager.end_conversation(request.npc_name)
            return {
                "npc": request.npc_name,
                "message": "",
                "success": True,
                "conversation_ended": True,
                "location": manager.current_location
            }
        
        # Get response from NPC
        response, success, time_taken = manager.conversation_manager.send_to_agent(
            request.npc_name,
            request.message
        )
        
        if not success:
            # Handle Ollama errors
            if isinstance(response, Exception):
                error_msg = str(response)
                if isinstance(response, requests.exceptions.ConnectionError):
                    error_msg = "Lost connection to Ollama service"
                elif isinstance(response, requests.exceptions.Timeout):
                    error_msg = "Ollama request timed out"
                raise HTTPException(
                    status_code=503,
                    detail=f"Ollama service error: {error_msg}"
                )
        
        return {
            "npc": request.npc_name,
            "message": response,
            "success": success,
            "time": time_taken,
            "conversation_ended": False,
            "location": manager.current_location
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in conversation: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/location")
async def update_location(update: LocationUpdate):
    """Update NPC location (used by UE5 for NPC movement)"""
    try:
        # Validate NPC exists
        if update.npc_name not in manager.agents:
            raise HTTPException(
                status_code=404,
                detail=f"NPC '{update.npc_name}' not found"
            )
        
        # Move NPC
        success = manager.move_npc_to_location(
            update.npc_name,
            update.location,
            update.reason
        )
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to move {update.npc_name} to {update.location}"
            )
        
        return {
            "success": True,
            "npc": update.npc_name,
            "location": update.location,
            "message": f"Moved {update.npc_name} to {update.location}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating location: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Ollama service
        ollama_ok = ollama_manager.check_service()
        
        # Check system status
        system_ok = manager is not None and len(manager.agents) > 0
        
        status = "healthy" if ollama_ok and system_ok else "degraded"
        
        return {
            "status": status,
            "system": "npc_conversation",
            "ollama": "ok" if ollama_ok else "error",
            "npcs": len(manager.agents) if manager else 0
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "error",
            "system": "npc_conversation",
            "error": str(e)
        }

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )