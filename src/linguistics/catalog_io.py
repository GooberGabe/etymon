"""Utility helpers for loading and saving language catalogs.

These routines keep JSON handling in one place so console tools and tests can
focus on the linguistics logic instead of file plumbing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List
import json

from .models import LanguageCatalog

JSON_EXTENSIONS = {".json"}


def _coerce_path(path_like: Path | str) -> Path:
    path = path_like if isinstance(path_like, Path) else Path(path_like)
    return path.expanduser().resolve()


def load_catalog_from_json(path_like: Path | str) -> LanguageCatalog:
    path = _coerce_path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    catalog = LanguageCatalog.from_dict(payload)
    catalog.metadata.setdefault("source_file", str(path))
    catalog.metadata.setdefault("generation", 1)
    catalog.metadata.setdefault("root_name", catalog.name)
    return catalog


def save_catalog_to_json(catalog: LanguageCatalog, path_like: Path | str) -> Path:
    path = _coerce_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(catalog.to_dict(), handle, ensure_ascii=False, indent=2)
    return path


def iter_catalog_paths(directory_like: Path | str) -> Iterable[Path]:
    directory = _coerce_path(directory_like)
    if not directory.exists():
        raise FileNotFoundError(f"Catalog directory not found: {directory}")
    for candidate in sorted(directory.iterdir()):
        if candidate.suffix.lower() in JSON_EXTENSIONS and candidate.is_file():
            yield candidate


def load_all_catalogs(directory_like: Path | str) -> Dict[str, LanguageCatalog]:
    catalogs: Dict[str, LanguageCatalog] = {}
    for path in iter_catalog_paths(directory_like):
        catalog = load_catalog_from_json(path)
        catalogs[catalog.name] = catalog
    return catalogs


def format_catalog_summary(catalog: LanguageCatalog, max_examples: int = 5) -> str:
    """Produce a lightweight summary string for console output."""
    parts: List[str] = []
    parts.append(f"{catalog.name}: {catalog.word_count()} words")
    parts.append(f"Categories: {', '.join(catalog.categories())}")
    sample = ", ".join(word.phonetic_form for word in catalog.words[:max_examples])
    if sample:
        parts.append(f"Sample: {sample}")
    return " | ".join(parts)
