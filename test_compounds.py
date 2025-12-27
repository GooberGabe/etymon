#!/usr/bin/env python3
"""Test script for compound word generation improvements."""

import random
from pathlib import Path
from src.world.world_data import World
from src.core.config_manager import ConfigManager
from src.linguistics.catalog_io import load_all_catalogs


def test_compound_words():
    """Test compound word generation with improved logic."""
    print("Testing Compound Word Generation Improvements")
    print("=" * 50)

    # Load configuration
    config_path = Path("config/world_generation.json")
    config_manager = ConfigManager(config_path)

    # Create world instance with proper dimensions
    width = config_manager.get('world.width', 1000)
    height = config_manager.get('world.height', 700)
    world = World(width, height, config_manager)

    # Load language catalogs and set up culture languages
    catalogs = load_all_catalogs(Path("init_catalogs"))
    world.culture_languages = catalogs

    # Set up a deterministic seed for reproducible results
    rng = random.Random(42)
    world._language_rng = rng

    # Test compound generation for different cultures
    test_cultures = ["Proto-Germanic", "Proto-Celtic", "Proto-Italic", "Proto-Greek"]

    print("Testing compound word generation for different cultures:")
    print()

    for culture_name in test_cultures:
        print(f"Culture: {culture_name}")
        print("-" * 30)

        # Generate several compound names to test
        compounds = []
        for i in range(5):
            compound = world._generate_compound_polity_name(
                culture_name, "kingdom", "Kingdom"
            )
            if compound:
                compounds.append(compound)

        if compounds:
            print("Generated compounds:")
            for compound in compounds:
                print(f"  - {compound}")

            # Check for duplicates in individual compounds
            duplicates_found = []
            for compound in compounds:
                # Simple check: if compound contains repeated sequences
                words = compound.lower().split()
                if len(words) == 1:  # Single word compounds
                    # Check for repeated syllables (rough approximation)
                    if len(compound) > 6:
                        half_len = len(compound) // 2
                        first_half = compound[:half_len].lower()
                        second_half = compound[half_len:].lower()
                        if first_half in second_half or second_half in first_half:
                            duplicates_found.append(compound)

            if duplicates_found:
                print(f"⚠️  Potential duplicates found: {duplicates_found}")
            else:
                print("✅ No obvious duplicates detected")

        else:
            print("  No compounds generated")

        print()


if __name__ == "__main__":
    test_compound_words()