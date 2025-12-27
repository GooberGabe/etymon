#!/usr/bin/env python3
"""
Basic test script for Etymon components.

Run this to test core functionality without the full pygame interface.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config_manager import ConfigManager
from src.world.world_generator import WorldGenerator


def test_config_manager():
    """Test configuration management."""
    print("Testing ConfigManager...")
    
    try:
        config = ConfigManager()
        print(f"  World width: {config.get('world.width')}")
        print(f"  World height: {config.get('world.height')}")
        print(f"  Num points: {config.get('world.num_points')}")
        print(f"  Sea level: {config.get('world.sea_level')}")
        print("  ✓ Config loaded successfully")
        return config
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return None


def test_world_generation(config):
    """Test world generation."""
    print("\nTesting WorldGenerator...")
    
    try:
        generator = WorldGenerator(config)
        world = generator.generate_world()
        
        print(f"  Generated {len(world.tiles)} tiles")
        print(f"  World dimensions: {world.width}x{world.height}")
        print(f"  Water tiles: {world.water_tiles}")
        print(f"  Land tiles: {world.land_tiles}")
        
        stats = world.calculate_statistics()
        print(f"  Water percentage: {stats['water_percentage']:.1f}%")
        print(f"  Elevation range: {stats['min_elevation']:.2f} - {stats['max_elevation']:.2f}")
        
        print("  ✓ World generation successful")
        return world
    except Exception as e:
        print(f"  ✗ World generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run basic tests."""
    print("Etymon Basic Test Suite")
    print("=" * 30)
    
    # Test configuration
    config = test_config_manager()
    if not config:
        print("\nConfiguration test failed. Exiting.")
        return 1
    
    # Test world generation
    world = test_world_generation(config)
    if not world:
        print("\nWorld generation test failed. Exiting.")
        return 1
    
    print("\n✓ All tests passed! You can now run 'python main.py' to start the full application.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
