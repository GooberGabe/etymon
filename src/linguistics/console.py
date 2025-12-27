"""Console helpers for quick linguistics experiments."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Sequence

from .catalog_io import format_catalog_summary, load_all_catalogs
from .models import LanguageCatalog, Word


def evolve_catalog_once(catalog: LanguageCatalog, rng: random.Random) -> LanguageCatalog:
    """Apply the catalog's sound change list once using the shared RNG."""
    return catalog.apply_sound_changes(rng=rng)


def _format_word_diff(baseline: Word, evolved: Word) -> str:
    gloss = baseline.gloss or "(no gloss)"
    return f"    Δ {gloss}: {baseline.phonetic_form} -> {evolved.phonetic_form}"


def _iter_word_diffs(
    baseline_words: Iterable[Word],
    evolved_words: Iterable[Word],
    *,
    limit: int | None,
) -> List[str]:
    lines: List[str] = []
    diff_count = 0
    for base_word, evolved_word in zip(baseline_words, evolved_words):
        if base_word.phonetic_form == evolved_word.phonetic_form:
            continue
        lines.append(_format_word_diff(base_word, evolved_word))
        diff_count += 1
        if limit is not None and diff_count >= limit:
            break
    if not lines:
        lines.append("    Δ No phonetic changes detected")
    return lines


def preview_catalog_evolution(
    catalog_dir: Path | str,
    *,
    seed: int | None = None,
    max_examples: int = 5,
    limit: int | None = None,
    show_diffs: bool = False,
    diff_limit: int | None = 5,
) -> List[str]:
    """Return formatted summary lines for base and evolved catalogs."""
    catalogs = load_all_catalogs(catalog_dir)
    if not catalogs:
        return [f"No catalogs found in {catalog_dir}"]

    rng = random.Random(seed)
    lines: List[str] = []
    processed = 0
    for name in sorted(catalogs.keys()):
        catalog = catalogs[name]
        evolved = evolve_catalog_once(catalog, rng)
        lines.append(f"[BASE]    {format_catalog_summary(catalog, max_examples)}")
        lines.append(f"[EVOLVED] {format_catalog_summary(evolved, max_examples)}")
        if show_diffs:
            lines.extend(_iter_word_diffs(catalog.words, evolved.words, limit=diff_limit))
        processed += 1
        if limit is not None and processed >= limit:
            break
    return lines


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load catalogs from disk, apply their configured sound changes, and "
            "print concise summaries for manual inspection."
        )
    )
    parser.add_argument(
        "--catalog-dir",
        default=Path("init_catalogs"),
        type=Path,
        help="Directory that stores *.json language catalogs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic sampling when probabilistic changes are used",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="How many lexical items to display per catalog summary",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of catalogs to preview",
    )
    parser.add_argument(
        "--show-diffs",
        action="store_true",
        help="Display per-word phonetic changes below each catalog summary",
    )
    parser.add_argument(
        "--diff-limit",
        type=int,
        default=5,
        help="Maximum number of changed words to display per catalog",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    lines = preview_catalog_evolution(
        args.catalog_dir,
        seed=args.seed,
        max_examples=args.max_examples,
        limit=args.limit,
        show_diffs=args.show_diffs,
        diff_limit=args.diff_limit,
    )
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
