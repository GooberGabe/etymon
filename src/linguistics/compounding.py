"""Controlled compound word builder for polity naming and lexical blending."""
from __future__ import annotations

import random
import re
from typing import Iterable, Optional, Sequence

from .phonotactics import enforce_phonotactics_form, simplify_orthography

# Default linkers that tend to be cross-linguistically tolerable
DEFAULT_LINKERS: Sequence[str] = ("a", "e", "i", "o")


def derive_surface_form(token: str, *, max_len: int = 8) -> str:
    """Return a shortened, compound-safe surface form without mutating the root.

    - trims whitespace/delimiters
    - reduces repeated letters and trailing vowels when the token is long
    - keeps a minimum of 3 characters
    """
    if not token:
        return ""

    working = re.sub(r"[^A-Za-z]+", "", token)
    if not working:
        return ""

    # Remove excessive repeats (e.g., cooouncil -> coouncil)
    working = re.sub(r"(\w)\1{2,}", r"\1\1", working)

    # Trim a trailing vowel on long tokens (lande -> land)
    if len(working) > 5 and working[-1] in "aeiouy":
        working = working[:-1]

    # If still long, drop the penultimate vowel to tighten (saxon -> saxn)
    if len(working) > max_len:
        match = re.search(r"[aeiouy](?=[^aeiouy]{1,2}$)", working)
        if match:
            idx = match.start()
            working = working[:idx] + working[idx + 1 :]

    if len(working) > max_len:
        working = working[:max_len]

    if len(working) < 3:
        return working

    return working


def count_syllables(form: str) -> int:
    """Crude syllable counter by vowel groups."""
    if not form:
        return 0
    groups = re.findall(r"[aeiouy]+", form.lower())
    return max(1, len(groups)) if groups else 1


def build_compound(
    left: str,
    right: str,
    typology,
    *,
    max_chars: int = 14,
    max_syllables: int = 4,
    linkers: Sequence[str] = DEFAULT_LINKERS,
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """Create a compact, phonotactically-repaired compound.

    Returns None on failure (too long or empty components).
    """
    if rng is None:
        rng = random

    surf_left = derive_surface_form(left)
    surf_right = derive_surface_form(right)
    if not surf_left or not surf_right:
        return None

    # Avoid duplicate or identical parts
    if surf_left.lower() == surf_right.lower():
        return None

    linker = _select_linker(surf_left, surf_right, linkers, rng)
    joined = f"{surf_left}{linker}{surf_right}"

    repaired = enforce_phonotactics_form(joined, typology)
    repaired = simplify_orthography(repaired)

    if not repaired:
        return None

    if len(repaired) > max_chars:
        return None
    if count_syllables(repaired) > max_syllables:
        return None

    return repaired


def build_compound_with_gloss(
    left: str,
    left_gloss: Optional[str],
    right: str,
    right_gloss: Optional[str],
    typology,
    *,
    max_chars: int = 14,
    max_syllables: int = 4,
    linkers: Sequence[str] = DEFAULT_LINKERS,
    rng: Optional[random.Random] = None,
) -> Optional[tuple[str, Optional[str]]]:
    """Create a compound and return (surface_form, gloss) when possible."""
    compound = build_compound(
        left,
        right,
        typology,
        max_chars=max_chars,
        max_syllables=max_syllables,
        linkers=linkers,
        rng=rng,
    )
    if not compound:
        return None

    gloss_left = (left_gloss or "").strip()
    gloss_right = (right_gloss or "").strip()
    gloss = None
    if gloss_left or gloss_right:
        gloss = f"{gloss_left} {gloss_right}".strip()

    return compound, gloss


def _select_linker(
    left: str,
    right: str,
    linkers: Sequence[str],
    rng: random.Random,
) -> str:
    if not linkers:
        linkers = DEFAULT_LINKERS

    left_end = left[-1].lower()
    right_start = right[0].lower()
    vowels = set("aeiouy")

    # No linker when vowel-consonant boundary is already smooth
    if (left_end in vowels and right_start not in vowels) or (left_end not in vowels and right_start in vowels):
        return ""

    # Prefer linker when consonant cluster would be heavy
    return rng.choice(linkers)
