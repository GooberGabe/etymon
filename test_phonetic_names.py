#!/usr/bin/env python3
"""Test script to verify settlement name phonetic storage and language evolution."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.config_manager import ConfigManager
from world.world_generator import WorldGenerator

def test_name_generation():
    """Test that settlement name generation returns both display and phonetic forms."""
    print("Testing settlement name generation...")

    # Load config and generate world
    config = ConfigManager()
    generator = WorldGenerator(config)
    world = generator.generate_world()

    # Generate a settlement name
    display_name, phonetic_name = world._generate_settlement_name(0)
    print(f"Generated: display='{display_name}', phonetic='{phonetic_name}'")

    assert display_name is not None
    assert phonetic_name is not None
    assert len(display_name) > 0
    assert len(phonetic_name) > 0
    print("✓ Name generation test passed")

def test_world_serialization():
    """Test that world serialization includes phonetic names."""
    print("\nTesting world serialization with phonetic names...")

    # Load config and generate world
    config = ConfigManager()
    generator = WorldGenerator(config)
    world = generator.generate_world()

    # Add a test population center with phonetic name
    from world.world_data import PopulationCenter
    test_center = PopulationCenter(
        tile_index=0,
        name="Test City",
        original_threshold=1000,
        established_year=0,
        established_tick=0,
        phonetic_name="tɛst sɪti"
    )
    world.population_centers.append(test_center)

    # Serialize the world
    payload = world.to_save_payload()

    # Check that population_centers data includes phonetic_name
    centers_data = payload.get('population_centers', [])
    assert len(centers_data) > 0, "No population centers in serialized data"

    # Find our test center
    test_center_data = None
    for center_data in centers_data:
        if center_data.get('name') == "Test City":
            test_center_data = center_data
            break

    assert test_center_data is not None, "Test center not found in serialized data"
    assert 'phonetic_name' in test_center_data, "phonetic_name not in serialized center data"
    assert test_center_data['phonetic_name'] == "tɛst sɪti", f"phonetic_name mismatch: {test_center_data['phonetic_name']}"

    print("✓ World serialization test passed")

    # Test deserialization
    restored_world = world.from_save_payload(payload, config)
    restored_centers = restored_world.population_centers

    # Find restored test center
    restored_test_center = None
    for center in restored_centers:
        if center.name == "Test City":
            restored_test_center = center
            break

    assert restored_test_center is not None, "Test center not found in restored world"
    assert restored_test_center.phonetic_name == "tɛst sɪti", f"phonetic_name not restored correctly: {restored_test_center.phonetic_name}"

    print("✓ World deserialization test passed")

if __name__ == "__main__":
    test_name_generation()
    test_world_serialization()
    print("\n🎉 All tests passed!")