"""
Core Module
===========

Core system components including message types, enums, and configuration loading.

Components:
- Message: Core message dataclass
- NPCRole: Role enumeration
- ConfigLoaders: JSON configuration loading utilities
"""

from .message_types import Message, NPCRole
from .config_loaders import RoleTemplateLoader, NPCLoader

__all__ = [
    'Message',
    'NPCRole', 
    'RoleTemplateLoader',
    'NPCLoader'
] 