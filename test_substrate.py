#!/usr/bin/env python3
"""Command line tool for testing substrate blending between language catalogs."""

import argparse
import random
from pathlib import Path
from typing import Sequence

from src.linguistics.catalog_io import load_all_catalogs, format_catalog_summary
from src.linguistics.pipeline import transform_language


def test_substrate_blending(
    catalog_dir: Path,
    primary_name: str,
    substrate_name: str,
    influence_strength: float = 0.5,
    seed: int | None = None,
) -> list[str]:
    """Test substrate blending between two catalogs."""
    catalogs = load_all_catalogs(catalog_dir)
    if not catalogs:
        return [f"No catalogs found in {catalog_dir}"]

    if primary_name not in catalogs:
        return [f"Primary catalog '{primary_name}' not found. Available: {sorted(catalogs.keys())}"]

    if substrate_name not in catalogs:
        return [f"Substrate catalog '{substrate_name}' not found. Available: {sorted(catalogs.keys())}"]

    primary = catalogs[primary_name]
    substrate = catalogs[substrate_name]

    rng = random.Random(seed)

    lines = [
        f"Testing substrate blending:",
        f"  Primary (superstrate): {primary_name}",
        f"  Substrate: {substrate_name}",
        f"  Influence strength: {influence_strength}",
        f"  Random seed: {seed}",
        "",
        f"BEFORE BLENDING:",
        f"Primary: {format_catalog_summary(primary, 3)}",
        f"Substrate: {format_catalog_summary(substrate, 3)}",
        "",
    ]

    # Perform substrate blending
    blended = transform_language(
        primary,
        transformation_type="substrate",
        substrate_typology=substrate.typology,
        substrate_catalog=substrate,
        influence_strength=influence_strength,
        rng=rng,
    )


    import json
    lines.append(f"AFTER SUBSTRATE BLENDING:")
    lines.append(f"Result: {format_catalog_summary(blended, 5)}")
    lines.append("")

    # Show borrowed words (including all supplements) and percentage
    borrowed = []
    for w in getattr(blended, 'words', []):
        ety = getattr(w, 'etymology', None)
        if ety and f"borrowed_from_{substrate.name}" in str(ety):
            borrowed.append(w)
    total_words = len(getattr(blended, 'words', []))
    percent_borrowed = (len(borrowed) / total_words * 100) if total_words else 0
    lines.append(f"DEBUG: {len(borrowed)} of {total_words} words ({percent_borrowed:.1f}%) were borrowed from the substrate.")
    if borrowed:
        lines.append(f"Words borrowed from substrate ({substrate.name}):")
        for w in borrowed:
            lines.append(f"  {w.gloss} [{w.category}]: {w.phonetic_form}")
        lines.append("")
    else:
        lines.append("No words were borrowed from the substrate.")
        lines.append("")

    # Show words changed by sound changes or phonotactic repair
    changed = [w for w in blended.words if getattr(w, 'historical_forms', None)]
    if changed:
        lines.append("Words changed by sound changes or phonotactic repair:")
        for w in changed:
            lines.append(f"  {w.gloss} [{w.category}]: {w.phonetic_form}")
            for h in w.historical_forms:
                lines.append(f"    - {h['change_name']}: {h['old_form']} → {h['new_form']}")
        lines.append("")
    else:
        lines.append("No words were changed by sound changes or phonotactic repair.")
        lines.append("")

    lines.append("FULL BLENDED CATALOG (JSON):")
    lines.append(json.dumps(blended.to_dict(), ensure_ascii=False, indent=2))
    lines.append("")
    lines.append(f"Typology comparison:")
    lines.append(f"  Primary consonants: {len(primary.typology.consonants)} types")
    lines.append(f"  Substrate consonants: {len(substrate.typology.consonants)} types")
    lines.append(f"  Blended consonants: {len(blended.typology.consonants)} types")
    lines.append("")
    lines.append(f"  Primary syllable patterns: {primary.typology.syllable_patterns[:3]}")
    lines.append(f"  Substrate syllable patterns: {substrate.typology.syllable_patterns[:3]}")
    lines.append(f"  Blended syllable patterns: {blended.typology.syllable_patterns[:3]}")
    lines.append("")
    lines.append(f"  Primary max onset complexity: {primary.typology.max_onset_complexity}")
    lines.append(f"  Substrate max onset complexity: {substrate.typology.max_onset_complexity}")
    lines.append(f"  Blended max onset complexity: {blended.typology.max_onset_complexity}")

    return lines


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test substrate blending between two language catalogs"
    )
    parser.add_argument(
        "--catalog-dir",
        default=Path("init_catalogs"),
        type=Path,
        help="Directory that stores *.json language catalogs",
    )
    parser.add_argument(
        "--primary",
        required=True,
        help="Name of the primary (superstrate) catalog",
    )
    parser.add_argument(
        "--substrate",
        required=True,
        help="Name of the substrate catalog",
    )
    parser.add_argument(
        "--influence",
        type=float,
        default=0.5,
        help="Strength of substrate influence (0.0-1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    lines = test_substrate_blending(
        args.catalog_dir,
        args.primary,
        args.substrate,
        args.influence,
        args.seed,
    )
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())