"""Persistence helpers for saving and loading complete world states."""

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy will exist in runtime env, but keep fallback
    np = None

from src.core.config_manager import ConfigManager
from src.world.world_data import World


class WorldSaveManager:
    """Handles serialization and deserialization of world saves."""

    def __init__(self, config: ConfigManager):
        self.config = config
        config_path = Path(config.config_path).resolve()
        self._root_dir = config_path.parent.parent
        self._save_dir = self._root_dir / "saves"
        self._save_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_name(self, save_name: Optional[str]) -> str:
        if not save_name:
            return self.config.get('simulation.save.default_name', 'autosave')
        return save_name

    def _build_path(self, save_name: str) -> Path:
        base = Path(save_name)
        if not base.suffix:
            base = base.with_suffix('.json')
        if base.is_absolute():
            return base
        return self._save_dir / base

    @staticmethod
    def _json_default(obj: Any):
        if np is not None:
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, tuple):
            return list(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def save_world(self, world: World, save_name: Optional[str] = None) -> Path:
        name = self._normalize_name(save_name)
        append_timestamp = bool(self.config.get('simulation.save.append_timestamp', False))
        if append_timestamp:
            name = f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        path = self._build_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = world.to_save_payload()
        with path.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, default=self._json_default)
        print(f"[save] World saved to {path}")
        return path

    def load_world(self, save_name: str) -> World:
        if not save_name:
            raise ValueError("Save name must be provided to load a world.")
        path = self._build_path(save_name)
        if not path.exists():
            raise FileNotFoundError(f"Save file not found: {path}")
        with path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        world = World.from_save_payload(payload, self.config)
        print(f"[save] World loaded from {path}")
        return world
