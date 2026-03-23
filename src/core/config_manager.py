"""Configuration management for Etymon."""

import json
import os
from typing import Dict, Any, List, Union, Set


class ConfigManager:
    """Manages application configuration with dependency injection support."""

    def __init__(self, config_paths: Union[str, List[str], None] = None):
        """Initialize configuration manager.

        Args:
            config_paths: One or more configuration file paths. When omitted,
                defaults to the split config files (simulation.json,
                rendering.json, ui.json) and falls back to world_generation.json
                when those are not present.
        """
        normalized = self._normalize_paths(config_paths)
        self.config_paths: List[str] = normalized
        # Keep a legacy-friendly attribute for callers that expect a single path
        self.config_path: str = self.config_paths[0] if self.config_paths else "config/world_generation.json"
        self._config: Dict[str, Any] = {}
        self.load_config()

    def _normalize_paths(self, config_paths: Union[str, List[str], None]) -> List[str]:
        """Normalize provided paths and choose sensible defaults."""
        if config_paths is None:
            default_paths = [
                "config/simulation.json",
                "config/rendering.json",
                "config/ui.json",
            ]
            existing_defaults = [p for p in default_paths if os.path.exists(p)]
            if existing_defaults:
                return existing_defaults
            return ["config/world_generation.json"]
        if isinstance(config_paths, str):
            return [config_paths]
        return list(config_paths)

    def _merge_dicts(self, base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-merge two dictionaries, with `incoming` taking precedence."""
        for key, value in incoming.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = self._merge_dicts(base[key], value)
            else:
                base[key] = value
        return base

    def _load_single_config(self, path: str, visited: Set[str]) -> Dict[str, Any]:
        abs_path = os.path.abspath(path)
        if abs_path in visited:
            raise ValueError(f"Cyclic config include detected at {path}")
        visited.add(abs_path)

        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {path}: {e}")

        merged: Dict[str, Any] = {}
        includes = data.get('includes')
        if isinstance(includes, list):
            for include_path in includes:
                resolved = include_path
                if not os.path.isabs(include_path):
                    resolved = os.path.join(os.path.dirname(path), include_path)
                included_data = self._load_single_config(resolved, visited)
                merged = self._merge_dicts(merged, included_data)

        # Merge the current file after includes so it can override
        data_without_includes = {k: v for k, v in data.items() if k != 'includes'}
        merged = self._merge_dicts(merged, data_without_includes)
        return merged

    def load_config(self) -> None:
        """Load and merge configuration from configured file paths."""
        merged: Dict[str, Any] = {}
        visited: Set[str] = set()
        loaded_any = False

        for path in self.config_paths:
            data = self._load_single_config(path, visited)
            merged = self._merge_dicts(merged, data)
            loaded_any = True

        if not loaded_any:
            raise FileNotFoundError("No configuration files could be loaded")

        self._config = merged

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
        """Save current configuration to file.

        Note: saving is only supported when a single config path is in use.
        """
        if len(self.config_paths) != 1:
            raise NotImplementedError("Saving merged configs is not supported; supply a single path")
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)