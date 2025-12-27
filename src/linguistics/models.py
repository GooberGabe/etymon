"""Core data structures for the Etymon linguistics system.

This module intentionally starts small.  It focuses on the data
representations that downstream components (sound change engine, blending
logic, console tooling) will rely on.  The goal for this first iteration is
simple: make it easy to load, inspect, and transform language catalogs in
memory without yet worrying about the full simulation pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional
import random
import re

from .phonotactics import enforce_phonotactics_form


@dataclass(slots=True)
class Word:
    """Individual lexical item with optional metadata."""

    phonetic_form: str
    gloss: str
    category: Optional[str] = None
    is_irregular: bool = False
    historical_forms: List[Dict[str, Any]] = field(default_factory=list)  # Track evolution
    etymology: Optional[str] = None  # Source language/family
    semantic_shift: Optional[str] = None  # How meaning changed over time

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "phonetic_form": self.phonetic_form,
            "gloss": self.gloss,
        }
        if self.category is not None:
            data["category"] = self.category
        if self.is_irregular:
            data["is_irregular"] = self.is_irregular
        if self.historical_forms:
            data["historical_forms"] = self.historical_forms
        if self.etymology is not None:
            data["etymology"] = self.etymology
        if self.semantic_shift is not None:
            data["semantic_shift"] = self.semantic_shift
        return data

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Word":
        return cls(
            phonetic_form=str(payload.get("phonetic_form", "")),
            gloss=str(payload.get("gloss", "")),
            category=payload.get("category"),
            is_irregular=bool(payload.get("is_irregular", False)),
            historical_forms=list(payload.get("historical_forms", [])),
            etymology=payload.get("etymology"),
            semantic_shift=payload.get("semantic_shift"),
        )

    def clone(self, **overrides: object) -> "Word":
        return replace(self, **overrides)

    def apply_sound_change(self, change: SoundChange, generation: int, rng: random.Random) -> "Word":
        """Apply a sound change and track the historical form."""
        old_form = self.phonetic_form
        new_word = change.apply(self, rng)

        if new_word.phonetic_form != old_form:
            # Record the historical change
            historical_entry = {
                "generation": generation,
                "change_name": change.name,
                "old_form": old_form,
                "new_form": new_word.phonetic_form,
                "timestamp": None  # Could add simulation time
            }
            new_word.historical_forms = self.historical_forms + [historical_entry]

        return new_word


@dataclass(slots=True)
class SoundChange:
    """Represents a single phonological transformation rule.

    This implementation purposely supports only a limited subset of the
    feature set described in LINGUISTICS.md.  Patterns are interpreted as
    Python regular-expressions.  Environments are spliced in as look-arounds
    when possible; if a given pattern is incompatible with regex look-arounds
    (e.g., variable-length expressions), the engine falls back to a simple
    substring replacement.  This gives us a practical starting point that can
    be extended later with a richer rule engine or natural-class support.
    """

    name: str
    source: str
    target: str
    env_before: str = ""
    env_after: str = ""
    probability: float = 1.0
    condition: Optional[Callable[[Word], bool]] = None

    _REGEX_CACHE: Dict[str, re.Pattern] = field(default_factory=dict, init=False, repr=False)

    def _build_pattern(self) -> Optional[re.Pattern]:
        pattern = self.source
        if self.env_before:
            pattern = f"(?<={self.env_before}){pattern}"
        if self.env_after:
            pattern = f"{pattern}(?={self.env_after})"
        try:
            return re.compile(pattern)
        except re.error:
            return None

    def apply(self, word: Word, rng: random.Random) -> Word:
        if self.condition and not self.condition(word):
            return word
        if self.probability < 1.0 and rng.random() > self.probability:
            return word
        regex = self._build_pattern()
        updated = word.phonetic_form
        if regex is not None:
            updated = regex.sub(self.target, updated)
        else:
            # Fallback: naive global replacement.
            updated = updated.replace(self.source, self.target)
        if updated == word.phonetic_form:
            return word
        return word.clone(phonetic_form=updated)


@dataclass(slots=True)
class Typology:
    """Encapsulates phonological/phonotactic settings for a language."""

    consonants: List[str] = field(default_factory=list)
    vowels: List[str] = field(default_factory=list)
    max_onset_complexity: int = 1
    max_coda_complexity: int = 1
    allowed_onsets: List[str] = field(default_factory=list)
    allowed_codas: List[str] = field(default_factory=list)
    syllable_patterns: List[str] = field(default_factory=list)
    stress_pattern: str = "initial"
    has_tone: bool = False
    sound_changes: List[SoundChange] = field(default_factory=list)
    lenition_contexts: List[str] = field(default_factory=list)
    fortition_contexts: List[str] = field(default_factory=list)
    vowel_harmony: bool = False
    consonant_mutation: bool = False

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Typology":
        sound_changes = [
            SoundChange(
                name=item.get("name", "Unnamed Change"),
                source=item.get("source", ""),
                target=item.get("target", ""),
                env_before=item.get("env_before", ""),
                env_after=item.get("env_after", ""),
                probability=float(item.get("probability", 1.0)),
            )
            for item in payload.get("sound_changes", [])
        ]
        return cls(
            consonants=list(payload.get("consonants", [])),
            vowels=list(payload.get("vowels", [])),
            max_onset_complexity=int(payload.get("max_onset_complexity", 1)),
            max_coda_complexity=int(payload.get("max_coda_complexity", 1)),
            allowed_onsets=list(payload.get("allowed_onsets", [])),
            allowed_codas=list(payload.get("allowed_codas", [])),
            syllable_patterns=list(payload.get("syllable_patterns", [])),
            stress_pattern=str(payload.get("stress_pattern", "initial")),
            has_tone=bool(payload.get("has_tone", False)),
            sound_changes=sound_changes,
            lenition_contexts=list(payload.get("lenition_contexts", [])),
            fortition_contexts=list(payload.get("fortition_contexts", [])),
            vowel_harmony=bool(payload.get("vowel_harmony", False)),
            consonant_mutation=bool(payload.get("consonant_mutation", False)),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "consonants": list(self.consonants),
            "vowels": list(self.vowels),
            "max_onset_complexity": self.max_onset_complexity,
            "max_coda_complexity": self.max_coda_complexity,
            "allowed_onsets": list(self.allowed_onsets),
            "allowed_codas": list(self.allowed_codas),
            "syllable_patterns": list(self.syllable_patterns),
            "stress_pattern": self.stress_pattern,
            "has_tone": self.has_tone,
            "sound_changes": [
                {
                    "name": sc.name,
                    "source": sc.source,
                    "target": sc.target,
                    "env_before": sc.env_before,
                    "env_after": sc.env_after,
                    "probability": sc.probability,
                }
                for sc in self.sound_changes
            ],
            "lenition_contexts": list(self.lenition_contexts),
            "fortition_contexts": list(self.fortition_contexts),
            "vowel_harmony": self.vowel_harmony,
            "consonant_mutation": self.consonant_mutation,
        }


@dataclass(slots=True)
class LanguageCatalog:
    """In-memory representation of a language snapshot."""

    name: str
    words: List[Word]
    typology: Typology
    metadata: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "LanguageCatalog":
        typology = Typology.from_dict(payload.get("typology", {}))
        words = [Word.from_dict(item) for item in payload.get("words", [])]
        return cls(
            name=str(payload.get("name", "Unnamed Language")),
            words=words,
            typology=typology,
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "metadata": dict(self.metadata),
            "typology": self.typology.to_dict(),
            "words": [word.to_dict() for word in self.words],
        }

    def apply_sound_changes(self, rng: Optional[random.Random] = None, generation: int = 1) -> "LanguageCatalog":
        """Produce a new catalog with ordered sound changes applied.

        This keeps the transformation logic intentionally conservative for
        now—only the sound change list is honored.  Phonotactic repairs,
        stress assignment, and irregular handling will be bolted on in later
        phases.
        """

        rng = rng or random.Random()
        transformed_words: List[Word] = []
        for word in self.words:
            updated = word
            for change in self.typology.sound_changes:
                updated = updated.apply_sound_change(change, generation, rng)
            repaired_form = enforce_phonotactics_form(updated.phonetic_form, self.typology)
            if repaired_form != updated.phonetic_form:
                # Record phonotactic repair as a historical change
                historical_entry = {
                    "generation": generation,
                    "change_name": "phonotactic_repair",
                    "old_form": updated.phonetic_form,
                    "new_form": repaired_form,
                    "timestamp": None
                }
                updated = updated.clone(
                    phonetic_form=repaired_form,
                    historical_forms=updated.historical_forms + [historical_entry]
                )
            transformed_words.append(updated)
        return LanguageCatalog(
            name=f"{self.name} (evolved)",
            words=transformed_words,
            typology=self.typology,
            metadata=dict(self.metadata, derived_from=self.name, evolution_generation=generation),
        )

    def word_count(self) -> int:
        return len(self.words)

    def categories(self) -> List[str]:
        return sorted({word.category or "uncategorized" for word in self.words})

    def iter_words_by_category(self, category: str) -> Iterable[Word]:
        for word in self.words:
            if (word.category or "").lower() == category.lower():
                yield word
