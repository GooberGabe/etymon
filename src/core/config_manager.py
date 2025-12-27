"""Configuration management for Etymon."""

import json
import os
from typing import Dict, Any


class ConfigManager:
    """Manages application configuration with dependency injection support."""
    
    def __init__(self, config_path: str = "config/world_generation.json"):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to the main configuration file
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                self._config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key (e.g., 'world.width')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section dictionary
        """
        return self._config.get(section, {})
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot-notation key.
        
        Args:
            key: Configuration key (e.g., 'world.width')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.load_config()
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)