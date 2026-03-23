"""Lightweight phonotactic helpers for enforcing typological constraints."""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set, TYPE_CHECKING

DEFAULT_VOWELS: Sequence[str] = ("a", "e", "i", "o", "u")

if TYPE_CHECKING:  # pragma: no cover - avoid circular imports at runtime
    from .models import Typology


def enforce_phonotactics_form(word_form: str, typology: "Typology") -> str:
    """Return a repaired phonetic form that honors simple onset/coda limits."""
    if not word_form:
        return word_form

    vowel_chars = _build_vowel_char_set(typology)
    default_vowel = (typology.vowels[0] if typology.vowels else DEFAULT_VOWELS[0]) or DEFAULT_VOWELS[0]

    # First, check if the word already fits allowed syllable patterns and cluster limits
    def fits_typology(form: str) -> bool:
        # Check for illegal onset/coda clusters
        clusters = re.findall(r'[^aeiouy\s-]+', form)
        for cl in clusters:
            if len(cl) > max(typology.max_onset_complexity, typology.max_coda_complexity):
                return False
        # Check for allowed syllable patterns if defined
        if typology.syllable_patterns:
            for pat in typology.syllable_patterns:
                if re.fullmatch(pat.replace('C', '[^aeiouy\s-]').replace('V', '[aeiouy]'), form):
                    return True
            return False
        return True

    if fits_typology(word_form):
        return word_form

    # Otherwise, minimally repair as before
    result: List[str] = []
    cluster: List[str] = []
    stage = "onset"

    for ch in word_form:
        if _is_boundary(ch):
            if cluster:
                result.extend(_repair_cluster(cluster, stage, typology, default_vowel))
                cluster = []
            result.append(ch)
            stage = "onset"
            continue
        if _is_vowel(ch, vowel_chars):
            if cluster:
                result.extend(_repair_cluster(cluster, stage, typology, default_vowel))
                cluster = []
            result.append(ch)
            stage = "coda"
        else:
            cluster.append(ch)

    if cluster:
        terminal_stage = stage if stage == "coda" else "onset"
        result.extend(_repair_cluster(cluster, terminal_stage, typology, default_vowel))

    smoothed = "".join(result)
    return simplify_orthography(smoothed)


_SEQUENCE_PATTERNS = {
    1: re.compile(r"([A-Za-z])\\1{2,}", flags=re.IGNORECASE),
    2: re.compile(r"((?:[A-Za-z]{2}))\\1{2,}", flags=re.IGNORECASE),
    3: re.compile(r"((?:[A-Za-z]{3}))\\1{2,}", flags=re.IGNORECASE),
}

_VOWEL_CLUSTER_PATTERN = re.compile(r"([aeiouy]{3,})", flags=re.IGNORECASE)


def simplify_orthography(word_form: str) -> str:
    """Condense extreme letter patterns while keeping the word recognizable."""
    if not word_form:
        return word_form
    simplified = word_form
    for length, pattern in _SEQUENCE_PATTERNS.items():
        simplified = pattern.sub(lambda match: match.group(1) * 2, simplified)
    simplified = _VOWEL_CLUSTER_PATTERN.sub(_trim_vowel_cluster, simplified)
    return simplified


def _trim_vowel_cluster(match: re.Match[str]) -> str:
    cluster = match.group(1)
    if len(cluster) <= 3:
        return cluster
    return cluster[:2] + cluster[-1]


def _is_boundary(ch: str) -> bool:
    return ch.isspace() or ch == "-"


def _build_vowel_char_set(typology: "Typology") -> Set[str]:
    characters: Set[str] = set()
    for vowel in (typology.vowels or DEFAULT_VOWELS):
        if not vowel:
            continue
        characters.add(vowel)
        for char in vowel:
            characters.add(char)
    return characters


def _is_vowel(char: str, vowel_chars: Set[str]) -> bool:
    return char in vowel_chars


def _repair_cluster(
    cluster: Iterable[str],
    stage: str,
    typology: "Typology",
    default_vowel: str,
) -> List[str]:
    cluster_list = list(cluster)
    if not cluster_list:
        return []

    limit = typology.max_onset_complexity if stage == "onset" else typology.max_coda_complexity
    limit = max(limit, 1)
    if len(cluster_list) <= limit:
        return cluster_list

    allowed = typology.allowed_onsets if stage == "onset" else typology.allowed_codas
    normalized = _match_allowed_cluster(cluster_list, allowed)
    if normalized is not None:
        cluster_list = normalized

    if len(cluster_list) <= limit:
        return cluster_list

    if stage == "onset":
        return _repair_onset(cluster_list, limit, default_vowel)
    return _repair_coda(cluster_list, limit)


def _match_allowed_cluster(cluster: List[str], allowed: Sequence[str]) -> List[str] | None:
    if not allowed:
        return None
    joined = "".join(cluster)
    if joined in allowed:
        return cluster
    reversed_joined = joined[::-1]
    if reversed_joined in allowed:
        return list(reversed(cluster))
    for candidate in allowed:
        if candidate and candidate in joined:
            return list(candidate)
    return None


def _repair_onset(cluster: List[str], limit: int, default_vowel: str) -> List[str]:
    repaired: List[str] = []
    working = cluster[:]
    while len(working) > limit:
        repaired.extend(working[:limit])
        working = working[limit:]
        repaired.append(default_vowel)
    repaired.extend(working)
    return repaired


def _repair_coda(cluster: List[str], limit: int) -> List[str]:
    return cluster[-limit:]
