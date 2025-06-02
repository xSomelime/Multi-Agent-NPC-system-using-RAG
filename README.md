# Multi-Agent NPC System
### An intelligent NPC coordination system for horse management games, built as part of AI/ML coursework.
#### Project Overview
This project implements a multi-agent AI system where intelligent NPCs coordinate to assist players in horse care, training, and management scenarios. Each NPC has specialized knowledge and distinct personalities, working together through natural language coordination.
#
#### Key Features
- 4 Specialized AI Agents: (Example) Stable Hand, Trainer, Health Monitor, and personality-driven NPCs
- RAG Knowledge System: Curated knowledge base with domain-specific expertise for each NPC role
- Anti-Hallucination Safety: Temperature-controlled responses and strict prompting to ensure accurate horse care information
- Intelligent Coordination: NPCs collaborate on daily routines, crisis response, and resource management
- Hybrid Architecture: Pre-defined responses for common scenarios + RAG for edge cases
#
#### Technical Stack
- Local AI: Ollama with Phi-3-mini/Llama3.1 models
- Knowledge Management: Vector storage with curated horse care content
- Safety First: Multiple layers to prevent misinformation about animal care
- Game-Ready: Designed for integration with Unreal Engine (terminal demo included)
#
#### Academic Context
Developed for AI/ML engineering coursework, combining Assignment 2 (multi-agent systems) with advanced project requirements. Demonstrates practical application of RAG, multi-agent coordination, and safe AI deployment in domain-specific contexts.
Getting Started
#
[Installation and setup instructions will be added as development progresses]
#### Project Status
- In Development - 3-week development cycle planned
- Project Tracking: https://amandamartensson-1741249939440.atlassian.net/jira/software/projects/MANS/boards/133

---

## 🔥 RAG-Enhanced NPCs - "Tough to Break" Knowledge System

The system includes a **Retrieval-Augmented Generation (RAG)** component that makes NPCs incredibly difficult to break with technical questions while maintaining their distinct personalities.

### What RAG Adds
- ✅ **Accurate horse care knowledge** from verified sources  
- ✅ **Domain-specific expertise** per NPC role
- ✅ **Anti-hallucination protection** - NPCs say "I don't know" rather than make up facts
- ✅ **Personality-consistent responses** - knowledge delivered in character
- ✅ **Session memory integration** - NPCs remember conversations with factual backing

### Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the Enhanced System**
```bash
python main_npc_system.py
```

3. **Test RAG Knowledge**
```bash
# Try these example queries:
"Oskar, what's the best feeding schedule for horses?"
"Elin, how do I tell if a horse is stressed?"
"Andy, what's the best way to train for jumping?"
```

### NPC Knowledge Specialization

| NPC Role | Primary Expertise | Example Knowledge |
|----------|------------------|-------------------|
| **Stable Hand (Oskar/Astrid)** | Daily horse care | Feeding schedules, grooming, maintenance |
| **Trainer (Andy)** | Training techniques | Riding progression, equipment selection |
| **Behaviourist (Elin)** | Horse psychology | Stress signals, natural horsemanship |
| **Rival (Chris)** | Competition/luxury | Expensive equipment, market values |

### Anti-Hallucination Features

**Strict Relevance Filtering**: 0.4 minimum similarity threshold for knowledge retrieval

**Graceful Fallbacks**: When NPCs don't know something:
```
Oskar: "I'm not sure about that, but I can help with feeding, 
       grooming, and daily horse care."

Elin: "I don't have research on that topic. My expertise is in 
      behavior, stress detection, and welfare analysis."
```

**Source Attribution**: All knowledge includes confidence levels and domain verification

### System Commands

- `rag status` - Check RAG system status
- `rag toggle` - Enable/disable RAG (restart required)
- `memory [npc]` - View what NPCs remember
- `go <location>` - Move between areas (barn, arena, paddock, etc.)
- `stats` - Show system statistics including RAG usage

### Performance
- **~50ms** RAG overhead per enhanced response
- **~10MB** memory usage for full knowledge base
- **Thread-safe** operation with existing memory system
- **Unreal Engine ready** with clean API design

The result: NPCs that provide accurate, role-appropriate horse care knowledge while maintaining their distinct personalities, making them extremely difficult to break with technical questions! 🐴
