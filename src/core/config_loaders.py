#!/usr/bin/env python3
"""
Configuration Loaders
=====================

Utilities for loading NPC configurations and role templates from JSON files.
Handles file not found errors and provides fallback configurations.
"""

import os
import json
from typing import Dict


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
                "speaking_style": "casual and supportive"
            },
            "personal_background": f"{name} is a dedicated horse care professional",
            "professional_opinions": {},
            "controversial_stances": []
        } 