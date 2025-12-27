"""High-level orchestration helpers for language transformations (Phase 6)."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Tuple
import random

from .models import LanguageCatalog, Typology
from .time_evolution import apply_time_shift
from .typology_blender import blend_typologies


def _generate_language_name(metadata: Dict[str, object], base_name: str) -> str:
    """Generate a formal language name based on metadata lineage."""
    root_name = metadata.get("root_name", base_name)
    generation = metadata.get("generation", 1)
    
    if generation == 1:
        return root_name
    else:
        return f"{root_name} (Gen {generation})"


def evolve_catalog(
    catalog: LanguageCatalog,
    target_typology: Typology | None = None,
    *,
    rng: Optional[random.Random] = None,
    label_suffix: str = "",
    metadata_updates: Optional[Dict[str, object]] = None,
) -> LanguageCatalog:
    """Apply the requested typology to a catalog and return a transformed copy."""

    rng = rng or random.Random()
    working_typology = target_typology or catalog.typology
    working = LanguageCatalog(
        name=catalog.name,
        words=[word.clone() for word in catalog.words],
        typology=working_typology,
        metadata=dict(catalog.metadata),
    )
    evolved = working.apply_sound_changes(rng=rng)
    
    # Update metadata for lineage tracking
    evolved_metadata = dict(evolved.metadata)
    if metadata_updates:
        evolved_metadata.update(metadata_updates)
    
    # Handle generation tracking
    current_gen = evolved_metadata.get("generation", 1)
    is_transformation = metadata_updates and "transformation" in metadata_updates
    if is_transformation:  # If this is a transformation, increment generation
        evolved_metadata["generation"] = current_gen + 1
        if "root_name" not in evolved_metadata:
            evolved_metadata["root_name"] = catalog.name
    
    # Generate new name
    new_name = _generate_language_name(evolved_metadata, catalog.name)
    
    return replace(evolved, name=new_name, metadata=evolved_metadata)


def transform_language(
    input_catalog: LanguageCatalog,
    transformation_type: str = "time_evolution",
    *,
    time_depth: int = 1,
    substrate_typology: Typology | None = None,
    influence_strength: float = 0.3,
    peer_catalog: LanguageCatalog | None = None,
    bidirectional_weight: float = 0.5,
    rng: Optional[random.Random] = None,
) -> LanguageCatalog | Tuple[LanguageCatalog, LanguageCatalog]:
    """Main entry point for time evolution, substrate, adstratum, or creolization scenarios."""

    rng = rng or random.Random()
    mode = (transformation_type or "time_evolution").lower()

    if mode == "time_evolution":
        evolved_typology = apply_time_shift(input_catalog.typology, time_depth, rng=rng)
        return evolve_catalog(
            input_catalog,
            target_typology=evolved_typology,
            rng=rng,
            metadata_updates={
                "transformation": "time_evolution",
                "time_depth": time_depth,
            },
        )

    if mode == "substrate":
        if substrate_typology is None:
            raise ValueError("substrate_typology is required for substrate transformation")
        blended = blend_typologies(
            primary_typology=input_catalog.typology,
            influence_typology=substrate_typology,
            influence_weight=influence_strength,
            rng=rng,
            substrate_mode=True,
        )
        # For substrate influence, we change the typology but don't apply sound changes
        # since substrate effects are synchronic contact phenomena, not historical evolution
        return LanguageCatalog(
            name=_generate_language_name(input_catalog.metadata, input_catalog.name),
            words=[word.clone() for word in input_catalog.words],  # Keep original words
            typology=blended,
            metadata=dict(input_catalog.metadata, **{
                "transformation": "substrate",
                "influence_strength": influence_strength,
                "generation": input_catalog.metadata.get("generation", 1) + 1,
            }),
        )

    if mode == "adstratum":
        if peer_catalog is None:
            raise ValueError("peer_catalog is required for adstratum transformation")
        evolved_a, _ = apply_adstratum_influence(
            input_catalog,
            peer_catalog,
            bidirectional_weight=bidirectional_weight,
            rng=rng,
        )
        return evolved_a

    if mode == "creolization":
        if peer_catalog is None:
            raise ValueError("peer_catalog is required for creolization transformation")
        # Creolization: heavy mixing with simplification
        blended = blend_typologies(
            primary_typology=input_catalog.typology,
            influence_typology=peer_catalog.typology,
            influence_weight=0.7,  # Heavy influence
            rng=rng,
        )
        # Apply some simplification
        blended.max_onset_complexity = min(blended.max_onset_complexity, 2)
        blended.max_coda_complexity = min(blended.max_coda_complexity, 2)
        return evolve_catalog(
            input_catalog,
            target_typology=blended,
            rng=rng,
            metadata_updates={
                "transformation": "creolization",
                "peer_language": peer_catalog.name,
            },
        )

    raise ValueError(f"Unknown transformation_type: {transformation_type}")


def apply_substrate_influence(
    superstrate_catalog: LanguageCatalog,
    substrate_typology: Typology,
    *,
    influence_strength: float = 0.2,
    rng: Optional[random.Random] = None,
) -> LanguageCatalog:
    """Convenience wrapper for substrate-driven contact scenarios."""

    return transform_language(
        superstrate_catalog,
        transformation_type="substrate",
        substrate_typology=substrate_typology,
        influence_strength=influence_strength,
        rng=rng,
    )


def apply_adstratum_influence(
    catalog_a: LanguageCatalog,
    catalog_b: LanguageCatalog,
    *,
    bidirectional_weight: float = 0.5,
    rng: Optional[random.Random] = None,
) -> Tuple[LanguageCatalog, LanguageCatalog]:
    """Apply mutual influence between peer languages (trade, bilingual zones)."""

    rng = rng or random.Random()
    blended_a = blend_typologies(
        primary_typology=catalog_a.typology,
        influence_typology=catalog_b.typology,
        influence_weight=bidirectional_weight,
        rng=rng,
    )
    blended_b = blend_typologies(
        primary_typology=catalog_b.typology,
        influence_typology=catalog_a.typology,
        influence_weight=bidirectional_weight,
        rng=rng,
    )
    evolved_a = evolve_catalog(
        catalog_a,
        target_typology=blended_a,
        rng=rng,
        metadata_updates={
            "transformation": "adstratum",
            "peer_language": catalog_b.name,
        },
    )
    evolved_b = evolve_catalog(
        catalog_b,
        target_typology=blended_b,
        rng=rng,
        metadata_updates={
            "transformation": "adstratum",
            "peer_language": catalog_a.name,
        },
    )
    return evolved_a, evolved_b
