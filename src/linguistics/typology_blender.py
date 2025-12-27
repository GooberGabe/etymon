"""Utilities for blending typologies during contact scenarios."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Optional, Sequence
import random

from .models import SoundChange, Typology


def blend_typologies(
    primary_typology: Typology,
    influence_typology: Typology,
    influence_weight: float = 0.3,
    rng: Optional[random.Random] = None,
    *,
    substrate_mode: bool = False,
) -> Typology:
    """Combine two typologies, weighting the influence according to the spec."""

    weight = max(0.0, min(1.0, influence_weight))
    rng = rng or random.Random()
    
    # In substrate mode, phonological features are more strongly influenced
    phonological_weight = weight * 1.5 if substrate_mode else weight
    phonological_weight = min(1.0, phonological_weight)

    consonants = _merge_unique(primary_typology.consonants, influence_typology.consonants)
    vowels = _merge_unique(primary_typology.vowels, influence_typology.vowels)

    max_onset_complexity = _weighted_complexity(
        primary_typology.max_onset_complexity,
        influence_typology.max_onset_complexity,
        phonological_weight,
    )
    max_coda_complexity = _weighted_complexity(
        primary_typology.max_coda_complexity,
        influence_typology.max_coda_complexity,
        phonological_weight,
    )

    allowed_onsets = _merge_unique(primary_typology.allowed_onsets, influence_typology.allowed_onsets)
    allowed_codas = _merge_unique(primary_typology.allowed_codas, influence_typology.allowed_codas)
    syllable_patterns = _merge_unique(
        primary_typology.syllable_patterns,
        influence_typology.syllable_patterns,
    )

    stress_pattern = _select_stress_pattern(
        primary_typology.stress_pattern,
        influence_typology.stress_pattern,
        phonological_weight,
        rng,
    )

    has_tone = _blended_bool(primary_typology.has_tone, influence_typology.has_tone, weight, rng)
    vowel_harmony = _blended_bool(
        primary_typology.vowel_harmony,
        influence_typology.vowel_harmony,
        phonological_weight,
        rng,
    )
    consonant_mutation = _blended_bool(
        primary_typology.consonant_mutation,
        influence_typology.consonant_mutation,
        phonological_weight,
        rng,
    )

    sound_changes = _combine_sound_changes(primary_typology, influence_typology, substrate_mode)

    lenition_contexts = _merge_unique(
        primary_typology.lenition_contexts,
        influence_typology.lenition_contexts,
    )
    fortition_contexts = _merge_unique(
        primary_typology.fortition_contexts,
        influence_typology.fortition_contexts,
    )

    return Typology(
        consonants=consonants,
        vowels=vowels,
        max_onset_complexity=max_onset_complexity,
        max_coda_complexity=max_coda_complexity,
        allowed_onsets=allowed_onsets,
        allowed_codas=allowed_codas,
        syllable_patterns=syllable_patterns,
        stress_pattern=stress_pattern,
        has_tone=has_tone,
        sound_changes=sound_changes,
        lenition_contexts=lenition_contexts,
        fortition_contexts=fortition_contexts,
        vowel_harmony=vowel_harmony,
        consonant_mutation=consonant_mutation,
    )


def _merge_unique(primary: Sequence[str], influence: Sequence[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for source in (primary, influence):
        for value in source:
            if value in seen:
                continue
            seen.add(value)
            merged.append(value)
    return merged


def _weighted_complexity(primary_value: int, influence_value: int, weight: float) -> int:
    """Blend syllable complexity values conservatively to prevent unrealistic jumps.

    In real languages, syllable structure changes very gradually. This function
    limits changes to at most 1 complexity level per contact event, regardless
    of influence strength.
    """
    if primary_value == influence_value:
        return primary_value

    # Limit maximum change to 1 level per contact event
    if influence_value > primary_value:
        # Can increase by at most 1
        return min(primary_value + 1, influence_value)
    else:
        # Can decrease by at most 1
        return max(primary_value - 1, influence_value)


def _select_stress_pattern(
    primary_pattern: str,
    influence_pattern: str,
    weight: float,
    rng: random.Random,
) -> str:
    if not influence_pattern:
        return primary_pattern
    if not primary_pattern:
        return influence_pattern
    return influence_pattern if rng.random() < weight else primary_pattern


def _blended_bool(
    primary_value: bool,
    influence_value: bool,
    weight: float,
    rng: random.Random,
) -> bool:
    if influence_value == primary_value:
        return primary_value
    if influence_value and not primary_value:
        return rng.random() < weight
    # influence lacks the trait; only keep it if weight is low
    retain_threshold = max(0.0, 1.0 - weight)
    return primary_value and rng.random() < retain_threshold


def _combine_sound_changes(primary: Typology, influence: Typology, substrate_mode: bool = False) -> List[SoundChange]:
    combined: List[SoundChange] = []
    if substrate_mode:
        # In substrate mode, substrate sound changes come first (higher priority)
        for change in influence.sound_changes:
            combined.append(replace(change))
        for change in primary.sound_changes:
            combined.append(replace(change))
    else:
        # In other modes, primary comes first
        for change in primary.sound_changes:
            combined.append(replace(change))
        for change in influence.sound_changes:
            combined.append(replace(change))
    return combined
