"""Stochastic helpers for diachronic typology drift (Phase 5)."""
from __future__ import annotations

from copy import deepcopy
from typing import List, Optional
import random

from .models import SoundChange, Typology

LENITION_STEPS = [
    ("p", "f"),
    ("t", "θ"),
    ("k", "x"),
    ("b", "v"),
    ("d", "ð"),
    ("g", "ɣ"),
]

VOWEL_CHAIN = ["i", "e", "a", "o", "u"]

STRESS_PATTERNS = [
    "initial",
    "penultimate",
    "ultimate",
    "antepenultimate",
]


def apply_time_shift(
    typology: Typology,
    time_depth: int,
    *,
    rng: Optional[random.Random] = None,
) -> Typology:
    """Return a new typology that incorporates probabilistic diachronic drift."""

    steps = int(abs(time_depth))
    if steps == 0:
        return deepcopy(typology)

    rng = rng or random.Random()
    evolved = deepcopy(typology)
    pending_changes: List[SoundChange] = []

    for step in range(steps):
        pending_changes.extend(_apply_lenition_drift(evolved, step, rng))
        pending_changes.extend(_apply_vowel_drift(evolved, rng))
        _adjust_complexity(evolved, rng)
        _maybe_shift_stress(evolved, rng)
        _maybe_toggle_harmony(evolved, rng)

    evolved.sound_changes = evolved.sound_changes + pending_changes
    return evolved


def _apply_lenition_drift(
    typology: Typology,
    step: int,
    rng: random.Random,
) -> List[SoundChange]:
    rate = min(0.75, 0.2 + 0.1 * step)
    produced: List[SoundChange] = []
    for source, target in LENITION_STEPS:
        if source not in typology.consonants:
            continue
        if rng.random() > rate:
            continue
        change = SoundChange(
            name=f"Lenition {source}>{target}",
            source=source,
            target=target,
            probability=1.0,
        )
        produced.append(change)
        if target not in typology.consonants:
            typology.consonants.append(target)
    return produced


def _apply_vowel_drift(typology: Typology, rng: random.Random) -> List[SoundChange]:
    if not typology.vowels:
        return []
    produced: List[SoundChange] = []
    selected = rng.choice(typology.vowels)
    if selected not in VOWEL_CHAIN:
        return produced
    idx = VOWEL_CHAIN.index(selected)
    direction = rng.choice([-1, 1])
    neighbor_idx = max(0, min(len(VOWEL_CHAIN) - 1, idx + direction))
    neighbor = VOWEL_CHAIN[neighbor_idx]
    if neighbor == selected:
        return produced
    change = SoundChange(
        name=f"Vowel drift {selected}>{neighbor}",
        source=selected,
        target=neighbor,
        probability=0.9,
    )
    produced.append(change)
    if neighbor not in typology.vowels:
        typology.vowels.append(neighbor)
    return produced


def _adjust_complexity(typology: Typology, rng: random.Random) -> None:
    delta = rng.choice([-1, 0, 1])
    typology.max_onset_complexity = max(1, typology.max_onset_complexity + delta)
    delta_coda = rng.choice([-1, 0, 1])
    typology.max_coda_complexity = max(1, typology.max_coda_complexity + delta_coda)


def _maybe_shift_stress(typology: Typology, rng: random.Random) -> None:
    if rng.random() < 0.2:
        typology.stress_pattern = rng.choice(STRESS_PATTERNS)


def _maybe_toggle_harmony(typology: Typology, rng: random.Random) -> None:
    if rng.random() < 0.15:
        typology.vowel_harmony = not typology.vowel_harmony
    if rng.random() < 0.15:
        typology.consonant_mutation = not typology.consonant_mutation