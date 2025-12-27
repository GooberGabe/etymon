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
        influence_strength=influence_strength,
        rng=rng,
    )

    lines.extend([
        f"AFTER SUBSTRATE BLENDING:",
        f"Result: {format_catalog_summary(blended, 5)}",
        "",
        f"Typology comparison:",
        f"  Primary consonants: {len(primary.typology.consonants)} types",
        f"  Substrate consonants: {len(substrate.typology.consonants)} types",
        f"  Blended consonants: {len(blended.typology.consonants)} types",
        "",
        f"  Primary syllable patterns: {primary.typology.syllable_patterns[:3]}",
        f"  Substrate syllable patterns: {substrate.typology.syllable_patterns[:3]}",
        f"  Blended syllable patterns: {blended.typology.syllable_patterns[:3]}",
        "",
        f"  Primary max onset complexity: {primary.typology.max_onset_complexity}",
        f"  Substrate max onset complexity: {substrate.typology.max_onset_complexity}",
        f"  Blended max onset complexity: {blended.typology.max_onset_complexity}",
    ])

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