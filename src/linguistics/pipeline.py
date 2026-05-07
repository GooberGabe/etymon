"""High-level orchestration helpers for language transformations (Phase 6)."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Tuple
import random

from .models import LanguageCatalog, Typology
from .time_evolution import apply_time_shift
from .typology_blender import blend_typologies



def _generate_language_name(metadata: Dict[str, object], base_name: str) -> str:
    """Generate a formal language name using culture adjectives and lineage."""
    # Try to use constituent cultures if available
    from src.world import world_data
    cultures = metadata.get("constituent_cultures")
    if isinstance(cultures, (list, tuple)) and cultures:
        adjectives = []
        for c in cultures:
            adj = None
            try:
                # Use the world_data adjective logic
                adj = world_data.MapRenderer._derive_culture_adjective(world_data.MapRenderer, c, c)
            except Exception:
                adj = c
            if adj:
                adjectives.append(adj)
        if adjectives:
            lang_name = "-".join(adjectives)
        else:
            lang_name = base_name
    else:
        # Fallback: use base_name with adjective
        try:
            lang_name = world_data.MapRenderer._derive_culture_adjective(world_data.MapRenderer, base_name, base_name)
        except Exception:
            lang_name = base_name
    generation = metadata.get("generation", 1)
    if generation > 1:
        lang_name = f"{lang_name} (Gen {generation})"
    return lang_name


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
    substrate_catalog: LanguageCatalog | None = None,
    contact_context: Dict[str, object] | None = None,
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
        effective_substrate = substrate_catalog.typology if substrate_catalog else substrate_typology
        if effective_substrate is None:
            raise ValueError("substrate_typology is required for substrate transformation")

        ctx = contact_context or {}
        primary_share = float(ctx.get("primary_share", 0.0))
        substrate_share = float(ctx.get("substrate_share", 0.0))
        dominance_ratio = float(ctx.get("dominance_ratio", 0.5))  # portion held by primary vs total
        primary_generation = int(ctx.get("primary_generation", input_catalog.metadata.get("generation", 1)))
        substrate_generation = int(ctx.get("substrate_generation", (substrate_catalog.metadata if substrate_catalog else {}).get("generation", 1)))
        primary_has_polity = ctx.get("primary_has_polity")
        substrate_has_polity = ctx.get("substrate_has_polity")

        # Start from caller influence; adjust based on situational dominance/age/polity presence
        modifier = 1.0
        if dominance_ratio >= 0.7:
            modifier *= 0.85  # dominant primary resists substrate spread
        elif dominance_ratio <= 0.55:
            modifier *= 1.05  # closer to parity, a bit more substrate bleed

        if substrate_generation < primary_generation:
            modifier *= 1.1  # older substrate carries prestige/gravitas
        elif substrate_generation > primary_generation + 1:
            modifier *= 0.95  # notably younger substrate loses ground

        if primary_has_polity is False and substrate_has_polity is True:
            modifier *= 1.05  # substrate tied to a polity has more leverage
        elif primary_has_polity is True and substrate_has_polity is False:
            modifier *= 0.95  # primary polity dampens substrate

        effective_influence = max(0.05, min(0.95, influence_strength * modifier))

        blended = blend_typologies(
            primary_typology=input_catalog.typology,
            influence_typology=effective_substrate,
            influence_weight=effective_influence,
            rng=rng,
            substrate_mode=True,
        )


        def borrow_words(primary, substrate, influence, ctx, rng):
            print("[DEBUG] Starting lexical borrowing...")
            if substrate:
                print(f"[DEBUG] Substrate catalog: {getattr(substrate, 'name', None)} | Words: {len(getattr(substrate, 'words', []))}")
                if getattr(substrate, 'words', []):
                    print(f"[DEBUG] Sample substrate word: {substrate.words[0].gloss} [{substrate.words[0].category}]: {substrate.words[0].phonetic_form}")
            primary_words = list(primary.words)
            substrate_words = list(substrate.words) if substrate else []
            if not substrate_words:
                print("[DEBUG] No substrate words available.")
                return primary_words
            from collections import defaultdict
            cat_to_words = defaultdict(list)
            for w in substrate_words:
                cat_to_words[(w.category or "uncategorized")].append(w)
            polity_bias = ctx.get("primary_has_polity") is True and ctx.get("substrate_has_polity") is False
            military_cats = {"military", "warfare", "government", "administration"}
            basic_cats = {"body", "kinship", "nature", "everyday", "basic"}
            borrow_probs = {}
            for cat in cat_to_words:
                if polity_bias and cat.lower() in military_cats:
                    borrow_probs[cat] = min(1.0, influence + 0.4)
                elif not polity_bias and cat.lower() in basic_cats:
                    borrow_probs[cat] = min(1.0, influence + 0.2)
                else:
                    borrow_probs[cat] = influence
            new_words = []
            used_substrate_ids = set()
            for w in primary_words:
                cat = w.category or "uncategorized"
                candidates = [sw for sw in cat_to_words[cat] if sw.gloss == w.gloss and id(sw) not in used_substrate_ids]
                if not candidates:
                    candidates = [sw for sw in cat_to_words[cat] if id(sw) not in used_substrate_ids]
                prob = borrow_probs.get(cat, influence)
                # Make replacement borrowing less aggressive: use influence squared
                replacement_prob = prob ** 2
                if candidates and rng.random() < replacement_prob:
                    chosen = rng.choice(candidates)
                    used_substrate_ids.add(id(chosen))
                    borrowed = chosen.clone(etymology=f"borrowed_from_{substrate.name}_DEBUG")
                    print(f"[DEBUG] Borrowed (replacement): {borrowed.gloss} [{borrowed.category}] -> {borrowed.phonetic_form}")
                    new_words.append(borrowed)
                else:
                    new_words.append(w.clone())
            supplement_prob = min(0.2, influence)
            for sw in substrate_words:
                if id(sw) not in used_substrate_ids and rng.random() < supplement_prob:
                    borrowed = sw.clone(etymology=f"borrowed_from_{substrate.name}_DEBUG")
                    print(f"[DEBUG] Borrowed (supplement): {borrowed.gloss} [{borrowed.category}] -> {borrowed.phonetic_form}")
                    new_words.append(borrowed)
            borrowed_count = len([w for w in new_words if getattr(w, 'etymology', None) and 'borrowed_from_' in str(w.etymology)])
            print(f"[DEBUG] Total borrowed words: {borrowed_count}")
            return new_words

        # Perform lexical borrowing
        new_words = borrow_words(input_catalog, substrate_catalog, effective_influence, ctx, rng)

        # Apply blended sound changes to the new lexicon
        from src.linguistics.models import LanguageCatalog as LC
        temp_catalog = LC(
            name=_generate_language_name(input_catalog.metadata, input_catalog.name),
            words=new_words,
            typology=blended,
            metadata=dict(input_catalog.metadata, **{
                "transformation": "substrate",
                "influence_strength_requested": influence_strength,
                "influence_strength_effective": effective_influence,
                "contact_bias": {
                    "primary_share": primary_share,
                    "substrate_share": substrate_share,
                    "dominance_ratio": dominance_ratio,
                    "primary_generation": primary_generation,
                    "substrate_generation": substrate_generation,
                    "primary_has_polity": primary_has_polity,
                    "substrate_has_polity": substrate_has_polity,
                    "modifier": modifier,
                },
                "generation": input_catalog.metadata.get("generation", 1) + 1,
            }),
        )
        # Use the built-in apply_sound_changes to apply all blended sound changes
        final_catalog = temp_catalog.apply_sound_changes(rng=rng, generation=input_catalog.metadata.get("generation", 1) + 1)
        return final_catalog

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
    substrate_typology: Typology | None = None,
    *,
    contact_context: Dict[str, object] | None = None,
    substrate_catalog: LanguageCatalog | None = None,
    influence_strength: float = 0.2,
    rng: Optional[random.Random] = None,
) -> LanguageCatalog:
    """Convenience wrapper for substrate-driven contact scenarios."""

    return transform_language(
        superstrate_catalog,
        transformation_type="substrate",
        substrate_typology=substrate_typology,
        substrate_catalog=substrate_catalog,
        contact_context=contact_context,
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
