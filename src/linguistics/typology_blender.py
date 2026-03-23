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

    consonants = _aggressive_merge(primary_typology.consonants, influence_typology.consonants, phonological_weight, rng)
    vowels = _aggressive_merge(primary_typology.vowels, influence_typology.vowels, phonological_weight, rng)

    max_onset_complexity = _aggressive_weighted_complexity(
        primary_typology.max_onset_complexity,
        influence_typology.max_onset_complexity,
        phonological_weight,
        substrate_mode
    )
    max_coda_complexity = _aggressive_weighted_complexity(
        primary_typology.max_coda_complexity,
        influence_typology.max_coda_complexity,
        phonological_weight,
        substrate_mode
    )

    allowed_onsets = _aggressive_merge(primary_typology.allowed_onsets, influence_typology.allowed_onsets, phonological_weight, rng)
    allowed_codas = _aggressive_merge(primary_typology.allowed_codas, influence_typology.allowed_codas, phonological_weight, rng)
    syllable_patterns = _aggressive_merge(
        primary_typology.syllable_patterns,
        influence_typology.syllable_patterns,
        phonological_weight,
        rng
    )

    stress_pattern = _select_stress_pattern(
        primary_typology.stress_pattern,
        influence_typology.stress_pattern,
        phonological_weight,
        rng,
    )

    has_tone = _aggressive_blended_bool(primary_typology.has_tone, influence_typology.has_tone, phonological_weight, rng, substrate_mode)
    vowel_harmony = _aggressive_blended_bool(
        primary_typology.vowel_harmony,
        influence_typology.vowel_harmony,
        phonological_weight,
        rng,
        substrate_mode
    )
    consonant_mutation = _aggressive_blended_bool(
        primary_typology.consonant_mutation,
        influence_typology.consonant_mutation,
        phonological_weight,
        rng,
        substrate_mode
    )

    sound_changes = _combine_sound_changes(primary_typology, influence_typology, substrate_mode)

    lenition_contexts = _aggressive_merge(
        primary_typology.lenition_contexts,
        influence_typology.lenition_contexts,
        phonological_weight,
        rng
    )
    fortition_contexts = _aggressive_merge(
        primary_typology.fortition_contexts,
        influence_typology.fortition_contexts,
        phonological_weight,
        rng
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



def _aggressive_merge(primary: Sequence[str], influence: Sequence[str], weight: float, rng: random.Random) -> List[str]:
    """Interleave/weight elements from both lists, not just append unique."""
    pool = list(primary) + list(influence)
    unique = list(dict.fromkeys(pool))  # preserve order, remove dups
    result = []
    for item in unique:
        if item in primary and item in influence:
            # Present in both, always include
            result.append(item)
        elif item in primary:
            # Only in primary, include with probability 1-weight
            if rng.random() > weight:
                result.append(item)
        elif item in influence:
            # Only in influence, include with probability = weight
            if rng.random() < weight:
                result.append(item)
    rng.shuffle(result)
    return result



def _aggressive_weighted_complexity(primary_value: int, influence_value: int, weight: float, substrate_mode: bool) -> int:
    """Allow larger jumps in complexity if substrate influence is strong."""
    if primary_value == influence_value:
        return primary_value
    if substrate_mode and weight > 0.5:
        # If substrate is strong, jump closer to influence
        return int(round(primary_value * (1-weight) + influence_value * weight))
    # Otherwise, allow up to 2 steps if weight is moderate, else 1
    max_step = 2 if weight > 0.4 else 1
    if influence_value > primary_value:
        return min(primary_value + max_step, influence_value)
    else:
        return max(primary_value - max_step, influence_value)


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



def _aggressive_blended_bool(
    primary_value: bool,
    influence_value: bool,
    weight: float,
    rng: random.Random,
    substrate_mode: bool = False
) -> bool:
    if influence_value == primary_value:
        return primary_value
    if substrate_mode:
        # In substrate mode, much more likely to adopt substrate trait
        if influence_value:
            return rng.random() < (weight + 0.3)  # boost adoption
        else:
            return rng.random() > (weight - 0.2)  # more likely to lose trait if substrate lacks it
    # Non-substrate: still more likely to adopt if weight is high
    if influence_value and not primary_value:
        return rng.random() < (weight + 0.1)
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
