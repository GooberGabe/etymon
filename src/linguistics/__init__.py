"""Linguistics subsystem bootstrap for Etymon."""

from .models import LanguageCatalog, SoundChange, Typology, Word
from .catalog_io import (
    format_catalog_summary,
    iter_catalog_paths,
    load_all_catalogs,
    load_catalog_from_json,
    save_catalog_to_json,
)
from .phonotactics import enforce_phonotactics_form, simplify_orthography
from .compounding import build_compound, derive_surface_form, build_compound_with_gloss
from .typology_blender import blend_typologies
from .time_evolution import apply_time_shift
from .pipeline import (
    apply_adstratum_influence,
    apply_substrate_influence,
    evolve_catalog,
    transform_language,
)
from .console import evolve_catalog_once, preview_catalog_evolution

__all__ = [
    "LanguageCatalog",
    "SoundChange",
    "Typology",
    "Word",
    "format_catalog_summary",
    "iter_catalog_paths",
    "load_all_catalogs",
    "load_catalog_from_json",
    "save_catalog_to_json",
    "enforce_phonotactics_form",
    "simplify_orthography",
    "build_compound",
    "build_compound_with_gloss",
    "derive_surface_form",
    "blend_typologies",
    "apply_time_shift",
    "evolve_catalog",
    "transform_language",
    "apply_substrate_influence",
    "apply_adstratum_influence",
    "evolve_catalog_once",
    "preview_catalog_evolution",
]
