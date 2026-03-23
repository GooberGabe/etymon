"""World generation system using Voronoi diagrams and Perlin noise."""

import numpy as np
from scipy.spatial import Voronoi
from noise import pnoise2
from typing import List, Tuple, Dict, Any, Optional, Set
import random
import math
from collections import deque

from src.core.config_manager import ConfigManager
from src.world.world_data import World, Tile, Region, Polity, Leader, Culture, River


class WorldGenerator:
    def generate_world(self) -> 'World':
        # DEBUG: Print config file path(s) and tectonic config summary
        if self.verbose_logging:
            paths = getattr(self.config, 'config_paths', [getattr(self.config, 'config_path', 'unknown')])
            print(f"[WorldGen][DEBUG] ConfigManager paths: {paths}")
            tectonics = self.config.get_section('world.tectonics')
            print("[WorldGen][DEBUG] Tectonic config at worldgen:")
            for k, v in tectonics.items():
                print(f"    {k}: {v}")
        # ...existing code...
    
    def __init__(self, config: ConfigManager):
        """Initialize world generator.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.verbose_logging = bool(config.get('generation.verbose_logging', False))
        self.width = config.get('world.width', 800)
        self.height = config.get('world.height', 600)
        self.num_points = config.get('world.num_points', 1000)
        self.sea_level = config.get('world.sea_level', 0.3)
        
        # Noise configuration
        self.noise_scale = config.get('world.noise.scale', 100.0)
        self.noise_octaves = config.get('world.noise.octaves', 6)
        self.noise_persistence = config.get('world.noise.persistence', 0.5)
        self.noise_lacunarity = config.get('world.noise.lacunarity', 2.0)
        
        # Handle seed: if -1, generate random seed
        configured_seed = config.get('world.noise.seed', 42)
        if configured_seed == -1:
            import time
            self.noise_seed = int(time.time() * 1000) % 1000000  # Use timestamp for random seed
            if self.verbose_logging:
                print(f"Generated random seed: {self.noise_seed}")
        else:
            self.noise_seed = configured_seed
        
        # Height curve configuration
        self.height_curve_coefficient = config.get('world.height_curve.coefficient', 1.0)
        self.height_curve_exponent = config.get('world.height_curve.exponent', 1.0)
        
        # Climate configuration
        self.equator_position = config.get('world.climate.equator_position', 0.5)
        self.pole_distance = config.get('world.climate.pole_distance', 0.5)
        self.temp_equator_base = config.get('world.climate.temperature_range.equator_base', 0.9)
        self.temp_pole_base = config.get('world.climate.temperature_range.pole_base', 0.1)
        self.temp_altitude_effect = config.get('world.climate.temperature_range.altitude_effect', 0.3)
        
        # Rainfall noise configuration
        self.rainfall_scale = config.get('world.climate.rainfall.scale', 150.0)
        self.rainfall_octaves = config.get('world.climate.rainfall.octaves', 4)
        self.rainfall_persistence = config.get('world.climate.rainfall.persistence', 0.6)
        self.rainfall_lacunarity = config.get('world.climate.rainfall.lacunarity', 2.0)
        
        # Color configuration
        self.colors = config.get_section('colors')
        self.test_setup = config.get('simulation.test_setup', 'default')
        self.uniform_population_value = config.get('simulation.uniform_population_value', 20)
        self.base_culture_name = config.get('simulation.test_setup.base_culture_name', 'TestCulture')
        self.base_culture_color = tuple(config.get('simulation.test_setup.base_culture_color', [150, 100, 200]))
        self._load_river_config()
    def _update_config_values(self) -> None:
        """Update configuration values from reloaded config."""
        self.verbose_logging = bool(self.config.get('generation.verbose_logging', False))
        self.width = self.config.get('world.width', 800)
        self.height = self.config.get('world.height', 600)
        self.num_points = self.config.get('world.num_points', 1000)
        self.sea_level = self.config.get('world.sea_level', 0.3)
        
        # Noise configuration
        self.noise_scale = self.config.get('world.noise.scale', 100.0)
        self.noise_octaves = self.config.get('world.noise.octaves', 6)
        self.noise_persistence = self.config.get('world.noise.persistence', 0.5)
        self.noise_lacunarity = self.config.get('world.noise.lacunarity', 2.0)
        
        # Handle seed: if -1, generate random seed
        configured_seed = self.config.get('world.noise.seed', 42)
        if configured_seed == -1:
            import time
            self.noise_seed = int(time.time() * 1000) % 1000000
            if self.verbose_logging:
                print(f"Generated random seed: {self.noise_seed}")
        else:
            self.noise_seed = configured_seed
        
        # Height curve configuration
        self.height_curve_coefficient = self.config.get('world.height_curve.coefficient', 1.0)
        self.height_curve_exponent = self.config.get('world.height_curve.exponent', 1.0)
        
        # Climate configuration
        self.equator_position = self.config.get('world.climate.equator_position', 0.5)
        self.pole_distance = self.config.get('world.climate.pole_distance', 0.5)
        self.temp_equator_base = self.config.get('world.climate.temperature_range.equator_base', 1.0)
        self.temp_pole_base = self.config.get('world.climate.temperature_range.pole_base', 0.2)
        self.temp_altitude_effect = self.config.get('world.climate.temperature_range.altitude_effect', 0.3)
        self.temp_altitude_threshold = self.config.get('world.climate.temperature_range.altitude_threshold', 0.6)
        
        # Rainfall noise configuration  
        self.rainfall_scale = self.config.get('world.climate.rainfall.scale', 150.0)
        self.rainfall_octaves = self.config.get('world.climate.rainfall.octaves', 4)
        self.rainfall_persistence = self.config.get('world.climate.rainfall.persistence', 0.6)
        self.rainfall_lacunarity = self.config.get('world.climate.rainfall.lacunarity', 2.0)
        self.rainfall_contrast = self.config.get('world.climate.rainfall.contrast', 1.2)
        self.rainfall_base_offset = self.config.get('world.climate.rainfall.base_offset', 0.0)
        
        # Regions configuration
        self.tiles_per_region = self.config.get('world.climate.regions.tiles_per_region', 25)
        self.min_region_size = self.config.get('world.climate.regions.min_region_size', 8)
        self.max_region_size = self.config.get('world.climate.regions.max_region_size', 20)
        
        # Color configuration
        self.colors = self.config.get_section('colors')
        self.test_setup = self.config.get('simulation.test_setup', 'default')
        self.uniform_population_value = self.config.get('simulation.uniform_population_value', 20)
        self.base_culture_name = self.config.get('simulation.test_setup.base_culture_name', 'TestCulture')
        self.base_culture_color = tuple(self.config.get('simulation.test_setup.base_culture_color', [150, 100, 200]))
        self._load_river_config()

    def _load_river_config(self) -> None:
        """Load river generation parameters from configuration."""
        self.river_max_count = self.config.get('world.rivers.max_count', 80)
        self.river_min_source_elevation = self.config.get('world.rivers.min_source_elevation', 0.55)
        self.river_min_source_rainfall = self.config.get('world.rivers.min_source_rainfall', 0.45)
        self.river_min_flux = self.config.get('world.rivers.min_flux', 0.12)
        self.river_min_length = int(self.config.get('world.rivers.min_length', 5))
        self.river_flow_decay = self.config.get('world.rivers.flow_decay', 0.12)
        self.river_equal_altitude_tolerance = self.config.get('world.rivers.equal_altitude_tolerance', 0.01)
        self.river_lake_altitude_tolerance = self.config.get('world.rivers.lake_altitude_tolerance', 0.02)
        self.river_rainfall_flow_scale = self.config.get('world.rivers.rainfall_flow_scale', 0.9)
        self.river_base_source = self.config.get('world.rivers.base_source', 0.08)
        self.river_candidate_multiplier = self.config.get('world.rivers.candidate_multiplier', 1.5)
        self.river_rainfall_slope_bias = self.config.get('world.rivers.rainfall_slope_bias', 0.08)
        self.river_altitude_weight = self.config.get('world.rivers.altitude_weight', 1.25)
        self.river_erosion_threshold = float(self.config.get('world.rivers.erosion_height_threshold', 0.03))
        self.river_erosion_lowering_margin = float(self.config.get('world.rivers.erosion_lowering_margin', 0.002))
        self.river_erosion_min_clearance = 1e-4
        self.river_coast_distance_bias = float(self.config.get('world.rivers.coast_distance_bias', 0.01))
        self.river_branching_min_flux = float(self.config.get('world.rivers.branching_min_flux', 0.2))
        self.river_branching_max_per_tile = int(self.config.get('world.rivers.branching_max_per_tile', 1))
        self.river_branching_score_tolerance = float(self.config.get('world.rivers.branching_score_tolerance', 0.02))
    
    def generate_world(self) -> World:
        """Generate a complete world.
        
        Returns:
            Generated world instance
        """
        # Reload configuration for real-time changes
        self.config.reload_config()
        self._update_config_values()
        
        # Create world instance
        world = World(self.width, self.height, self.config)
        world.set_world_seed(self.noise_seed)
        language_dir = self.config.get('linguistics.catalog_dir', 'init_catalogs')
        language_seed_value = self.config.get('linguistics.seed')
        language_seed = None
        if language_seed_value is not None:
            try:
                language_seed = int(language_seed_value)
            except (TypeError, ValueError):
                language_seed = None
        if language_seed is None:
            language_seed = self.noise_seed
        world.initialize_language_system(language_dir, seed=language_seed)
        world.sea_level = self.sea_level
        
        # Generate Voronoi points
        points = self._generate_points()
        
        # Create Voronoi diagram
        vor = Voronoi(points)
        
        # Generate elevation map
        elevations = self._generate_elevations(points)
        if self.verbose_logging:
            # Diagnostics for elevation
            print("[WorldGen] Elevation diagnostics:")
            print(f"  Elevation min: {np.min(elevations):.4f}, max: {np.max(elevations):.4f}, mean: {np.mean(elevations):.4f}, std: {np.std(elevations):.4f}")
            sea_level = self.sea_level
            land = np.sum(elevations > 0)
            water = np.sum(elevations == 0)
            print(f"  Land tiles: {land}, Water tiles: {water}, Land %: {land/(land+water)*100:.2f}")
            print(f"  Elevation bands:")
            print(f"    >0.9 (mountainous): {np.sum(elevations > 0.9)}")
            print(f"    >0.8 (alpine): {np.sum((elevations > 0.8) & (elevations <= 0.9))}")
            print(f"    0.4-0.8 (hills/upland): {np.sum((elevations >= 0.4) & (elevations <= 0.8))}")
            print(f"    <0.4 (lowlands): {np.sum(elevations < 0.4)}")
        
        # Create tiles from Voronoi regions
        tiles = self._create_tiles_from_voronoi(vor, elevations)
        
        # Add tiles to world
        for tile in tiles:
            world.add_tile(tile)
        self._generate_rivers(world)
        world.rebuild_coastal_tile_cache()
        
        # Generate regions for land tiles
        self._generate_regions(world)
        
        # Generate starting polities/scenarios
        self._generate_starting_setup(world)
        world.initialize_leader_system()
        
        return world

    def _generate_starting_setup(self, world: 'World') -> None:
        """Generate whichever starting scenario is configured."""
        setup = (self.test_setup or 'default').lower()
        if setup == 'uniform_population':
            return
        if setup == 'war':
            if not self._generate_war_starting_setup(world):
                self._generate_test_polity(world)
            return
        if setup == 'cultural':
            if not self._generate_cultural_test_setup(world):
                if self.verbose_logging:
                    print("Cultural test setup unavailable; falling back to default setup.")
                self._generate_test_polity(world)
            return
        if setup == 'diaspora':
            self._generate_diaspora_setup(world)
            return
        if setup == 'civ_test':
            self._generate_civ_test_setup(world)
            return
        else:
            self._generate_test_polity(world)

    def _generate_cultural_test_setup(self, world: 'World') -> bool:
        """Create a cultural stress-test polity spanning 12 tiles."""
        target_tile_count = 12
        land_tiles = [i for i, tile in enumerate(world.tiles) if not tile.is_water]
        if len(land_tiles) < target_tile_count:
            return False
        seed_idx = random.choice(land_tiles)
        selected = self._gather_contiguous_tiles(world, seed_idx, target_tile_count)
        if len(selected) < target_tile_count:
            remaining = [idx for idx in land_tiles if idx not in selected]
            random.shuffle(remaining)
            selected.extend(remaining[:target_tile_count - len(selected)])
        if len(selected) < target_tile_count:
            return False
        majority_culture = self.config.get('simulation.test_setup.majority_culture', 'Dominant')
        minority_culture = self.config.get('simulation.test_setup.minority_culture', 'Minor')
        self._ensure_culture_exists(world, majority_culture, (80, 150, 230), selected[0])
        self._ensure_culture_exists(world, minority_culture, (230, 200, 90), selected[-1])
        polity_id = len(world.polities)
        leader = Leader(
            name=f"{majority_culture} Marshal",
            age=random.randint(30, 60),
            culture=majority_culture
        )
        polity = Polity(
            id=polity_id,
            name=f"{majority_culture} Cultural Test",
            color=(60, 110, 190),
            leader=leader,
            primary_culture=majority_culture,
            tile_indices=list(selected),
            capital_tile_index=seed_idx
        )
        for tile_idx in selected:
            tile = world.tiles[tile_idx]
            old_polity = tile.polity_id
            tile.polity_id = polity_id
            tile.control_level = 100 if tile_idx == seed_idx else 90
            tile.population = random.randint(1, 500)
            tile.development = max(tile.development, tile.population * 0.5)
            minority_share = random.uniform(0.08, 0.2)
            tile.cultural_makeup = {
                majority_culture: 1.0 - minority_share,
                minority_culture: minority_share
            }
            if hasattr(world, '_on_tile_polity_changed'):
                world._on_tile_polity_changed(tile_idx, old_polity, tile.polity_id)
        world.polities.append(polity)
        world.assign_polity_language_name(polity)
        if hasattr(world, '_ensure_relationships_for_polity'):
            world._ensure_relationships_for_polity(polity.id)
        if hasattr(world, '_ensure_population_center_entry'):
            world._ensure_population_center_entry(polity.capital_tile_index, name=f"{polity.name} Capital", threshold=1)
        if self.verbose_logging:
            print(f"Cultural test polity created with {len(selected)} tiles")
        return True

    def _generate_war_starting_setup(self, world: 'World') -> bool:
        """Spawn two hostile polities with fortified population centers for war tests."""
        land_tiles = [i for i, tile in enumerate(world.tiles) if not tile.is_water]
        if len(land_tiles) < 10:
            return False

        target_tiles = 8

        def collect_tiles(seed: int, forbidden: Set[int]) -> List[int]:
            """Gather contiguous land tiles while avoiding the forbidden set."""
            if seed in forbidden:
                return []
            gathered: List[int] = []
            visited = set()
            queue = deque([seed])
            while queue and len(gathered) < target_tiles:
                current = queue.popleft()
                if current in visited or current in forbidden:
                    continue
                visited.add(current)
                if current < 0 or current >= len(world.tiles):
                    continue
                tile = world.tiles[current]
                if tile.is_water:
                    continue
                gathered.append(current)
                for neighbor_idx in tile.neighbors:
                    if neighbor_idx not in visited:
                        queue.append(neighbor_idx)
            return gathered

        random.shuffle(land_tiles)
        seed_a = land_tiles[0]
        tiles_a = collect_tiles(seed_a, set())
        if len(tiles_a) < 4:
            return False

        land_set = set(land_tiles)
        forbidden: Set[int] = set(tiles_a)
        neighbor_candidates: List[int] = []
        for idx in tiles_a:
            for neighbor in world.tiles[idx].neighbors:
                if neighbor in land_set and neighbor not in forbidden:
                    neighbor_candidates.append(neighbor)
        if neighbor_candidates:
            seed_b = random.choice(neighbor_candidates)
        else:
            remaining = [idx for idx in land_tiles if idx not in forbidden]
            if not remaining:
                return False
            seed_b = remaining[0]
        tiles_b = collect_tiles(seed_b, forbidden)
        if len(tiles_b) < 4:
            return False

        name_pool = [
            "Ironclad Dominion",
            "Azure Coalition",
            "Verdant Host",
            "Crimson Banner",
            "Golden Compact",
        ]
        random.shuffle(name_pool)
        polity_a_name = name_pool[0]
        polity_b_name = name_pool[1] if len(name_pool) > 1 else f"{polity_a_name} Rival"

        color_pool = [
            (200, 70, 70),
            (70, 130, 200),
            (210, 160, 60),
            (90, 190, 140),
        ]
        random.shuffle(color_pool)
        color_a = color_pool[0]
        color_b = color_pool[1] if len(color_pool) > 1 else (60, 60, 60)

        capital_a = tiles_a[0]
        capital_b = tiles_b[0]
        # Create cultures with dynamic names based on their home regions
        self._ensure_culture_exists(world, "", color_a, capital_a, dynamic_name=True)
        self._ensure_culture_exists(world, "", color_b, capital_b, dynamic_name=True)
        
        # Get the culture names that were generated
        culture_a = world.cultures[-2].name  # Second to last culture added
        culture_b = world.cultures[-1].name  # Last culture added

        polity_a_id = len(world.polities)
        leader_a = Leader(name=f"Commander {polity_a_name.split()[0]}", age=random.randint(28, 55), culture=culture_a)
        polity_a = Polity(
            id=polity_a_id,
            name=polity_a_name,
            color=color_a,
            leader=leader_a,
            primary_culture=culture_a,
            tile_indices=list(tiles_a),
            capital_tile_index=capital_a,
        )
        world.polities.append(polity_a)
        world.assign_polity_language_name(polity_a)

        polity_b_id = len(world.polities)
        leader_b = Leader(name=f"Commander {polity_b_name.split()[0]}", age=random.randint(28, 55), culture=culture_b)
        polity_b = Polity(
            id=polity_b_id,
            name=polity_b_name,
            color=color_b,
            leader=leader_b,
            primary_culture=culture_b,
            tile_indices=list(tiles_b),
            capital_tile_index=capital_b,
        )
        world.polities.append(polity_b)
        world.assign_polity_language_name(polity_b)

        def assign_tiles(polity: Polity, tile_indices: List[int], culture_name: str, capital_idx: int) -> None:
            for idx in tile_indices:
                tile = world.tiles[idx]
                old_polity = tile.polity_id
                tile.polity_id = polity.id
                tile.control_level = 100 if idx == capital_idx else max(80, tile.control_level)
                base_population = random.randint(6000, 9000)
                tile.population = max(tile.population, base_population)
                tile.development = max(tile.development, tile.population * random.uniform(0.55, 0.85))
                tile.cultural_makeup = {culture_name: 1.0}
                if hasattr(world, '_on_tile_polity_changed'):
                    world._on_tile_polity_changed(idx, old_polity, polity.id)

        assign_tiles(polity_a, tiles_a, culture_a, capital_a)
        assign_tiles(polity_b, tiles_b, culture_b, capital_b)

        def create_population_centers(polity: Polity, city_tiles: List[int]) -> None:
            pop_values = [22000, 16000]
            for i, tile_idx in enumerate(city_tiles[:2]):
                tile = world.tiles[tile_idx]
                pop_value = pop_values[i]
                tile.population = max(tile.population, pop_value)
                tile.development = max(tile.development, pop_value * 0.75)
                if hasattr(world, '_ensure_population_center_entry'):
                    world._ensure_population_center_entry(
                        tile_idx,
                        name=f"{polity.name} City {i + 1}",
                        threshold=int(pop_value * 0.8),
                    )

        create_population_centers(polity_a, tiles_a)
        create_population_centers(polity_b, tiles_b)

        if hasattr(world, '_ensure_relationships_for_polity'):
            world._ensure_relationships_for_polity(polity_a.id)
            world._ensure_relationships_for_polity(polity_b.id)

        relationship = world._get_relationship(polity_a.id, polity_b.id, create=True)
        if relationship:
            relationship.met = True
            relationship.ticking_modifiers[polity_a.id] = -50.0
            relationship.ticking_modifiers[polity_b.id] = -50.0
            shared_pairs = set()
            for idx in tiles_a:
                for neighbor in world.tiles[idx].neighbors:
                    if neighbor in tiles_b:
                        shared_pairs.add(tuple(sorted((idx, neighbor))))
            relationship.shared_border_tiles = len(shared_pairs)
        world._relationship_borders_initialized = False

        if hasattr(world, '_log_event'):
            world._log_event(
                'war',
                f"War scenario initialized between {polity_a.name} and {polity_b.name} (baseline -50 relations)",
            )
        return True

    def _generate_diaspora_setup(self, world: 'World') -> None:
        """Generate a diaspora starting setup with 1000 population on a random land tile."""
        land_tiles = [i for i, tile in enumerate(world.tiles) if not tile.is_water]
        if not land_tiles:
            if self.verbose_logging:
                print("No land tiles available for diaspora setup")
            return

        # Select a random land tile for the diaspora population
        diaspora_tile_idx = random.choice(land_tiles)
        diaspora_tile = world.tiles[diaspora_tile_idx]

        # Set the diaspora population (development will start at 0)
        diaspora_tile.population = 1000
        # Note: development starts at 0 for diaspora, will grow naturally over time

        if self.verbose_logging:
            print(f"Diaspora setup: 1000 population placed on tile {diaspora_tile_idx}")

    def _generate_civ_test_setup(self, world: 'World') -> None:
        """Spawn 5 nearby land tiles with 1000 population each for quick civ testing."""
        land_tiles = [i for i, tile in enumerate(world.tiles) if not tile.is_water]
        if len(land_tiles) < 5:
            if self.verbose_logging:
                print("civ_test setup aborted: not enough land tiles")
            return

        seed_idx = random.choice(land_tiles)
        cluster = self._gather_contiguous_tiles(world, seed_idx, target_count=40)
        if len(cluster) < 5:
            remaining = [idx for idx in land_tiles if idx not in cluster]
            random.shuffle(remaining)
            cluster.extend(remaining[: max(0, 5 - len(cluster))])

        # Spread selections across the cluster to avoid tight clumping
        selected: List[int] = []
        if cluster:
            stride = max(1, len(cluster) // 5)
            for i in range(0, len(cluster), stride):
                selected.append(cluster[i])
                if len(selected) >= 5:
                    break
        selected = selected[:5]
        if len(selected) < 5:
            # Fill any gaps from remaining cluster entries
            for idx in cluster:
                if idx not in selected:
                    selected.append(idx)
                    if len(selected) >= 5:
                        break

        for idx in selected:
            tile = world.tiles[idx]
            tile.population = 1000
            tile.development = 1000.0

        # Skip diaspora delays for this test setup by advancing past the delay window
        delay_years = self.config.get('simulation.diaspora.population_center_delay_years', 0)
        if delay_years and world.current_year <= delay_years:
            world.current_year = delay_years + 1

        if self.verbose_logging:
            print(f"civ_test setup: seeded 5 tiles {selected} with population 1000")

    def _gather_contiguous_tiles(self, world: 'World', start_index: int, target_count: int) -> List[int]:
        """Collect up to target_count land tiles contiguous from a seed."""
        if start_index >= len(world.tiles) or world.tiles[start_index].is_water:
            return []
        visited = {start_index}
        order = [start_index]
        queue = deque([start_index])
        while queue and len(order) < target_count:
            current = queue.popleft()
            for neighbor_idx in world.tiles[current].neighbors:
                if neighbor_idx in visited or neighbor_idx >= len(world.tiles):
                    continue
                neighbor_tile = world.tiles[neighbor_idx]
                if neighbor_tile.is_water:
                    continue
                visited.add(neighbor_idx)
                order.append(neighbor_idx)
                queue.append(neighbor_idx)
                if len(order) >= target_count:
                    break
        return order

    def _ensure_culture_exists(self, world: 'World', name: str, color: Tuple[int, int, int], origin_tile: int, dynamic_name: bool = False) -> None:
        """Add a culture to the world if it is missing."""
        if any(culture.name == name for culture in world.cultures):
            return
        home_region_id = -1
        if 0 <= origin_tile < len(world.tiles):
            region_id = world.tiles[origin_tile].region_id
            home_region_id = region_id if region_id is not None else -1
        
        # Use temporary name if dynamic naming is requested
        culture_name = name
        if dynamic_name and home_region_id >= 0:
            culture_name = world._next_culture_name('R')
        
        new_culture = Culture(
            name=culture_name,
            color=color,
            heritage={},
            origin_tile_index=origin_tile,
            birth_year=world.current_year,
            home_region_id=home_region_id,
            immunity_end_year=None,
            is_initial=True
        )
        if hasattr(world, '_ensure_culture_ideas'):
            world._ensure_culture_ideas(new_culture)
        world.cultures.append(new_culture)
        if hasattr(world, '_register_culture_name'):
            world._register_culture_name(new_culture.name)
        
        # Generate dynamic name from home region if requested
        if dynamic_name and home_region_id >= 0:
            region_name = world.ensure_region_language_name(home_region_id, new_culture.name)
            if region_name and not region_name.startswith("Region "):
                # Generate proper culture name (demonym) from region name
                culture_name = world._generate_culture_demonym_from_region(region_name, origin_tile)
                if culture_name:
                    # Rename culture
                    old_name = new_culture.name
                    new_culture.name = culture_name
                    # Update name registry
                    if hasattr(world, '_register_culture_name'):
                        world._culture_name_registry.discard(world._normalize_culture_name_token(old_name))
                        world._register_culture_name(new_culture.name)
                world._log_event(
                    "culture",
                    f"[culture] Renamed culture from '{old_name}' to '{new_culture.name}' (home region name)"
                )
        
        # Assign language after final name is set
        world.assign_base_language(new_culture)
        
        if hasattr(world, 'ensure_region_name_for_culture'):
            world.ensure_region_name_for_culture(new_culture)
    
    def _generate_points(self) -> np.ndarray:
        """Generate random points for Voronoi diagram.
        
        Returns:
            Array of point coordinates
        """
        # Use configured seed for reproducible results
        random.seed(self.noise_seed)
        np.random.seed(self.noise_seed)
        
        points = np.random.random((self.num_points, 2))
        points[:, 0] *= self.width
        points[:, 1] *= self.height
        
        return points
    
    def _generate_elevations(self, points: np.ndarray) -> np.ndarray:
        """Generate elevation values using Perlin noise.

        Args:
            points: Array of point coordinates

        Returns:
            Array of elevation values
        """
        elevations = np.zeros(len(points))
        
        # Generate simple multi-octave Perlin noise
        for i, (x, y) in enumerate(points):
            # Normalize coordinates to 0-1 range for noise function
            nx = x / self.width
            ny = y / self.height
            
            # Generate multi-octave Perlin noise
            elevation = 0.0
            amplitude = 1.0
            frequency = self.noise_scale / max(self.width, self.height)
            
            for octave in range(self.noise_octaves):
                elevation += amplitude * pnoise2(
                    nx * frequency, 
                    ny * frequency, 
                    base=self.noise_seed + octave * 1000
                )
                amplitude *= self.noise_persistence
                frequency *= self.noise_lacunarity
            
            elevations[i] = elevation

        # Apply height curve transformation
        elevations = self.height_curve_coefficient * np.power(np.abs(elevations), self.height_curve_exponent) * np.sign(elevations)

        # Percentile-based normalization for target land/ocean ratio
        cutoff = np.percentile(elevations, self.sea_level * 100)
        # Water: elevation <= cutoff, Land: elevation > cutoff
        land_mask = elevations > cutoff
        water_mask = ~land_mask
        # Set water to 0, rescale land to 0-1
        land_elev = elevations[land_mask]
        if len(land_elev) > 0:
            land_min = np.min(land_elev)
            land_max = np.max(land_elev)
            # Avoid division by zero
            if land_max > land_min:
                elevations[land_mask] = (land_elev - land_min) / (land_max - land_min)
            else:
                elevations[land_mask] = 1.0
        elevations[water_mask] = 0.0

        return elevations
    
    def _create_tiles_from_voronoi(self, vor: Voronoi, 
                                  elevations: np.ndarray) -> List[Tile]:
        """Create tiles from Voronoi diagram.
        
        Args:
            vor: Voronoi diagram
            elevations: Elevation values for each point
            
        Returns:
            List of generated tiles
        """
        tiles = []
        
        for point_idx, region_idx in enumerate(vor.point_region):
            if region_idx == -1:
                continue
                
            region = vor.regions[region_idx]
            if not region or -1 in region:
                continue
            
            # Get vertices of the region
            vertices = []
            for vertex_idx in region:
                vertex = vor.vertices[vertex_idx]
                # Clip to world bounds
                x = max(0, min(self.width, vertex[0]))
                y = max(0, min(self.height, vertex[1]))
                vertices.append((x, y))
            
            if len(vertices) < 3:
                continue
            
            # Get center point
            center = tuple(vor.points[point_idx])
            
            # Get elevation
            elevation = elevations[point_idx]
            is_water = elevation <= 0.0  # 0 is now sea level
            
            # Calculate climate for land tiles
            if not is_water:
                temperature = self._calculate_temperature(center[0], center[1], elevation)
                rainfall = self._calculate_rainfall(center[0], center[1], elevation)
            else:
                # Water tiles don't need climate calculation
                temperature = 0.0
                rainfall = 0.0
            
            # Determine color and biome
            color, biome_name = self._get_biome_color(elevation, temperature, rainfall, is_water)
            
            # Create tile
            tile = Tile(
                vertices=vertices,
                center=center,
                elevation=elevation,
                is_water=is_water,
                color=color,
                neighbors=[],
                temperature=temperature,
                rainfall=rainfall,
                biome=biome_name,
                population=self._generate_base_population(elevation, is_water, False, temperature, rainfall),
                development=self._generate_base_development(elevation, is_water, False),
                cultural_makeup={}
            )

            if (self.test_setup or '').lower() == 'uniform_population' and not tile.is_water:
                tile.population = int(self.uniform_population_value)
            
            tiles.append(tile)
        
        # Calculate neighbor relationships
        self._calculate_neighbors(tiles, vor)
        
        return tiles
    
    def _calculate_neighbors(self, tiles: List, vor) -> None:
        """Calculate neighbor relationships between tiles using Voronoi ridge information.
        
        Args:
            tiles: List of tiles to update
            vor: Voronoi diagram from scipy
        """
        # Create mapping from point index to tile index
        point_to_tile = {}
        for tile_idx, tile in enumerate(tiles):
            # Find the point index that corresponds to this tile's center
            for point_idx, point in enumerate(vor.points):
                if abs(point[0] - tile.center[0]) < 0.001 and abs(point[1] - tile.center[1]) < 0.001:
                    point_to_tile[point_idx] = tile_idx
                    break
        
        # Use Voronoi ridges to find neighbors
        for ridge_points in vor.ridge_points:
            point1, point2 = ridge_points
            
            # Get corresponding tile indices
            tile1_idx = point_to_tile.get(point1)
            tile2_idx = point_to_tile.get(point2)
            
            if tile1_idx is not None and tile2_idx is not None:
                # Add each as neighbor of the other
                if tile2_idx not in tiles[tile1_idx].neighbors:
                    tiles[tile1_idx].neighbors.append(tile2_idx)
                if tile1_idx not in tiles[tile2_idx].neighbors:
                    tiles[tile2_idx].neighbors.append(tile1_idx)

    def _generate_rivers(self, world: World) -> None:
        """Create major rivers based on rainfall, elevation, and drainage directions."""
        if not world.tiles:
            return
        land_indices = [idx for idx, tile in enumerate(world.tiles) if not tile.is_water]
        if not land_indices:
            world.rivers = []
            world.river_lakes = set()
            return
        # Reset river metadata on tiles
        world.rivers = []
        world.river_lakes = set()
        for tile in world.tiles:
            tile.river_ids = []
            tile.river_neighbors = {}
            tile.river_flux = 0.0
            tile.is_river_lake = False

        coast_distances = self._compute_coastal_distance(world)
        drainage_targets, branch_map = self._build_drainage_targets(world, land_indices, coast_distances)
        land_indices = [idx for idx in land_indices if idx < len(world.tiles) and not world.tiles[idx].is_water]
        self._accumulate_river_flow(world, land_indices, drainage_targets)
        self._trace_major_rivers(world, drainage_targets, branch_map, land_indices)

    def _compute_coastal_distance(self, world: World) -> List[int]:
        """Return tile-distance from each tile to the nearest water tile."""
        distances: List[Optional[int]] = [None] * len(world.tiles)
        queue: deque = deque()
        for idx, tile in enumerate(world.tiles):
            if tile.is_water:
                distances[idx] = 0
                queue.append(idx)
        while queue:
            current = queue.popleft()
            current_distance = distances[current]
            if current_distance is None:
                continue
            for neighbor_idx in world.tiles[current].neighbors:
                if neighbor_idx < 0 or neighbor_idx >= len(world.tiles):
                    continue
                if distances[neighbor_idx] is not None:
                    continue
                distances[neighbor_idx] = current_distance + 1
                queue.append(neighbor_idx)
        fallback = max((d for d in distances if d is not None), default=0) + 1
        return [d if d is not None else fallback for d in distances]

    def _build_drainage_targets(
        self,
        world: World,
        land_indices: List[int],
        coast_distances: List[int],
    ) -> Tuple[List[Optional[int]], Dict[int, List[int]]]:
        """Determine drainage routing and optional branch candidates for each tile."""
        targets: List[Optional[int]] = [None] * len(world.tiles)
        branch_map: Dict[int, List[int]] = {}
        tolerance = max(0.0, float(self.river_equal_altitude_tolerance))
        lake_tolerance = max(tolerance, float(self.river_lake_altitude_tolerance))
        rainfall_bias = max(0.0, float(self.river_rainfall_slope_bias))
        distance_bias = max(0.0, float(self.river_coast_distance_bias))
        ordered_land = sorted(land_indices, key=lambda idx: world.tiles[idx].elevation, reverse=True)
        for idx in ordered_land:
            tile = world.tiles[idx]
            # Direct outlet to existing water
            direct_target = next(
                (neighbor_idx for neighbor_idx in tile.neighbors
                 if 0 <= neighbor_idx < len(world.tiles) and world.tiles[neighbor_idx].is_water),
                None,
            )
            if direct_target is not None:
                targets[idx] = direct_target
                continue

            best_idx: Optional[int] = None
            best_score = float('inf')
            candidate_scores: Dict[int, float] = {}
            candidates: List[int] = []
            for neighbor_idx in tile.neighbors:
                if neighbor_idx < 0 or neighbor_idx >= len(world.tiles):
                    continue
                neighbor = world.tiles[neighbor_idx]
                drop = tile.elevation - neighbor.elevation
                if drop < -lake_tolerance and not neighbor.is_water:
                    continue
                score = neighbor.elevation - neighbor.rainfall * rainfall_bias
                if distance_bias > 0.0:
                    score += coast_distances[neighbor_idx] * distance_bias
                candidate_scores[neighbor_idx] = score
                candidates.append(neighbor_idx)
                if drop > tolerance and score < best_score:
                    best_score = score
                    best_idx = neighbor_idx

            if best_idx is None and candidates:
                for neighbor_idx in sorted(candidates, key=lambda n: candidate_scores[n]):
                    drop = tile.elevation - world.tiles[neighbor_idx].elevation
                    if drop >= -lake_tolerance:
                        best_idx = neighbor_idx
                        best_score = candidate_scores[neighbor_idx]
                        break

            if best_idx is None:
                best_idx = self._attempt_river_erosion_escape(world, idx, coast_distances)

            if best_idx is None:
                self._convert_tile_to_lake(world, idx)
                targets[idx] = None
                continue

            targets[idx] = best_idx

            if self.river_branching_max_per_tile <= 0 or not candidates:
                continue
            branchable = []
            for neighbor_idx in sorted(candidates, key=lambda n: candidate_scores[n]):
                if neighbor_idx == best_idx:
                    continue
                if candidate_scores[neighbor_idx] - best_score <= self.river_branching_score_tolerance:
                    branchable.append(neighbor_idx)
                if len(branchable) >= self.river_branching_max_per_tile:
                    break
            if branchable:
                branch_map[idx] = branchable

        return targets, branch_map

    def _attempt_river_erosion_escape(
        self,
        world: World,
        tile_idx: int,
        coast_distances: List[int],
    ) -> Optional[int]:
        """Lower a neighboring tile so trapped flow can escape when slopes are shallow."""
        if self.river_erosion_threshold <= 0 or not self.river_erosion_lowering_margin:
            return None
        tile = world.tiles[tile_idx]
        lowest_idx: Optional[int] = None
        lowest_score = float('inf')
        for neighbor_idx in tile.neighbors:
            if neighbor_idx < 0 or neighbor_idx >= len(world.tiles):
                continue
            neighbor = world.tiles[neighbor_idx]
            if neighbor.is_water:
                continue
            elevation_gap = neighbor.elevation - tile.elevation
            if elevation_gap > self.river_erosion_threshold:
                continue
            score = neighbor.elevation
            if self.river_coast_distance_bias > 0.0:
                score += coast_distances[neighbor_idx] * self.river_coast_distance_bias
            if score < lowest_score:
                lowest_score = score
                lowest_idx = neighbor_idx
        if lowest_idx is None:
            return None
        sea_clearance = world.sea_level + self.river_erosion_min_clearance
        max_drop = tile.elevation - sea_clearance
        if max_drop <= self.river_erosion_min_clearance:
            return None
        drop_amount = min(self.river_erosion_lowering_margin, max_drop)
        if drop_amount <= 0.0:
            return None
        neighbor_tile = world.tiles[lowest_idx]
        neighbor_tile.elevation = tile.elevation - drop_amount
        if neighbor_tile.elevation <= world.sea_level:
            neighbor_tile.elevation = sea_clearance
        neighbor_tile.is_water = neighbor_tile.elevation < world.sea_level
        return lowest_idx

    def _convert_tile_to_lake(self, world: World, tile_idx: int) -> None:
        """Turn a stranded basin into an inland lake so other rivers can terminate in it."""
        tile = world.tiles[tile_idx]
        tile.is_water = True
        tile.is_river_lake = True
        if tile.elevation > world.sea_level:
            tile.elevation = max(world.sea_level * 0.99, tile.elevation - 0.01)
        world.river_lakes.add(tile_idx)

    def _accumulate_river_flow(
        self,
        world: World,
        land_indices: List[int],
        drainage_targets: List[Optional[int]],
    ) -> None:
        """Accumulate river flux from rainfall and upstream contributions."""
        incoming: List[float] = [0.0] * len(world.tiles)
        ordered = sorted(land_indices, key=lambda idx: world.tiles[idx].elevation, reverse=True)
        rainfall_threshold = float(self.river_min_source_rainfall)
        base_source = float(self.river_base_source)
        rainfall_scale = float(self.river_rainfall_flow_scale)
        decay = max(0.0, min(0.95, float(self.river_flow_decay)))
        min_source_elevation = float(self.river_min_source_elevation)
        min_flux = float(self.river_min_flux)
        altitude_weight = max(0.1, float(self.river_altitude_weight))
        for idx in ordered:
            tile = world.tiles[idx]
            rainfall_surplus = max(0.0, tile.rainfall - rainfall_threshold)
            altitude_delta = tile.elevation - min_source_elevation
            if altitude_delta > 0:
                altitude_pressure = altitude_delta
                elevation_bonus = max(0.25, 1.0 + altitude_pressure * altitude_weight)
                elevation_bonus *= 1.0 + min(1.5, altitude_pressure * 0.6)
            else:
                altitude_pressure = 0.0
                elevation_bonus = max(0.2, 1.0 + altitude_delta * 0.4)
            local_input = rainfall_surplus * rainfall_scale * elevation_bonus
            if altitude_pressure > 0.0 or tile.rainfall >= rainfall_threshold:
                altitude_source = base_source * (1.0 + altitude_pressure * altitude_weight)
                local_input += altitude_source * elevation_bonus
            flow = local_input + incoming[idx]
            tile.river_flux = flow
            target = drainage_targets[idx]
            if target is None or target < 0 or target >= len(world.tiles):
                if flow >= min_flux:
                    tile.is_river_lake = True
                    world.river_lakes.add(idx)
                continue
            target_tile = world.tiles[target]
            if target_tile.is_water:
                continue
            carry = flow * (1.0 - decay)
            if carry > 0.0:
                incoming[target] += carry

    def _trace_major_rivers(
        self,
        world: World,
        drainage_targets: List[Optional[int]],
        branch_map: Dict[int, List[int]],
        land_indices: List[int],
    ) -> None:
        """Trace polylines for significant rivers and attach them to tiles."""
        min_flux = float(self.river_min_flux)
        if min_flux <= 0.0:
            return
        candidates = [idx for idx in land_indices if world.tiles[idx].river_flux >= min_flux]
        if not candidates:
            return
        candidates.sort(key=lambda idx: world.tiles[idx].river_flux, reverse=True)
        limit = int(max(1, round(self.river_max_count * self.river_candidate_multiplier)))
        river_tile_registry: Set[int] = set()
        tasks: deque = deque((idx, None) for idx in candidates[:limit])
        processed: Set[Tuple[int, Optional[int]]] = set()
        while tasks and len(world.rivers) < self.river_max_count:
            start_idx, forced_target = tasks.popleft()
            key = (start_idx, forced_target)
            if key in processed:
                continue
            processed.add(key)
            if start_idx < 0 or start_idx >= len(world.tiles):
                continue
            start_tile = world.tiles[start_idx]
            if start_tile.is_water or start_tile.river_flux < min_flux:
                continue
            path: List[int] = []
            visited: Set[int] = set()
            current = start_idx
            local_forced = forced_target
            terminates_in_sea = False
            terminates_in_lake = False
            max_flux = 0.0
            previous_idx = None
            while current is not None and current not in visited:
                visited.add(current)
                if current < 0 or current >= len(world.tiles):
                    break
                current_tile = world.tiles[current]
                if current_tile.is_water:
                    if previous_idx is not None:
                        path.append(current)
                        terminates_in_sea = True
                    break
                if local_forced is None and current in river_tile_registry:
                    break
                path.append(current)
                river_tile_registry.add(current)
                max_flux = max(max_flux, current_tile.river_flux)
                if (
                    current_tile.river_flux >= self.river_branching_min_flux
                    and local_forced is None
                    and current in branch_map
                ):
                    for branch_target in branch_map[current]:
                        tasks.append((current, branch_target))
                target = None
                if local_forced is not None:
                    target = local_forced
                    local_forced = None
                else:
                    target = drainage_targets[current]
                if target is None or target < 0 or target >= len(world.tiles):
                    current_tile.is_river_lake = True
                    world.river_lakes.add(current)
                    terminates_in_lake = True
                    break
                if target == previous_idx:
                    break
                target_tile = world.tiles[target]
                if target_tile.is_water:
                    path.append(target)
                    terminates_in_sea = True
                    break
                previous_idx = current
                current = target
            if len(path) < max(2, self.river_min_length):
                continue
            river_id = len(world.rivers)
            points = [world.tiles[p_idx].center for p_idx in path if 0 <= p_idx < len(world.tiles)]
            river = River(
                id=river_id,
                tile_indices=list(path),
                points=points,
                flux=max_flux,
                terminates_in_sea=terminates_in_sea,
                terminates_in_lake=terminates_in_lake,
            )
            world.rivers.append(river)
            for tile_idx in path:
                if tile_idx < 0 or tile_idx >= len(world.tiles):
                    continue
                tile = world.tiles[tile_idx]
                if tile.is_water:
                    continue
                if river_id not in tile.river_ids:
                    tile.river_ids.append(river_id)
            for a, b in zip(path, path[1:]):
                if a < 0 or b < 0 or a >= len(world.tiles) or b >= len(world.tiles):
                    continue
                tile_a = world.tiles[a]
                tile_b = world.tiles[b]
                if tile_a.is_water or tile_b.is_water:
                    continue
                flux_value = max(tile_a.river_flux, tile_b.river_flux, max_flux)
                tile_a.river_neighbors[b] = max(tile_a.river_neighbors.get(b, 0.0), flux_value)
                tile_b.river_neighbors[a] = max(tile_b.river_neighbors.get(a, 0.0), flux_value)
    
    def _generate_regions(self, world: 'World') -> None:
        """Generate regions using geography-based detection with elevation boundaries.
        
        This creates regions based on natural geographical features rather than
        arbitrary size divisions, using high-elevation areas as boundaries.
        
        Args:
            world: World instance to add regions to
        """
        from src.world.world_data import Region
        import random
        
        # Only create regions for land tiles
        land_tiles = [i for i, tile in enumerate(world.tiles) if not tile.is_water]
        
        if not land_tiles:
            return
        
        # Use elevation-based region detection
        regions = self._create_geography_based_regions(world, land_tiles)
        
        # Assign regions to world
        for region_data in regions:
            region_color = self._generate_region_color(region_data['id'])
            
            region = Region(
                id=region_data['id'],
                name="",  # Dynamic name based on dominant culture
                color=region_color,
                tile_indices=region_data['tiles'],
                center_tile_index=region_data['center']
            )
            
            # Assign region ID to all tiles in this region
            for tile_idx in region_data['tiles']:
                world.tiles[tile_idx].region_id = region_data['id']
            
            world.regions.append(region)
        
        # Final verification: ensure all land tiles have region assignments
        for tile_idx in land_tiles:
            if world.tiles[tile_idx].region_id == -1:
                # Find nearest region and assign
                nearest_region = self._find_nearest_region(world, tile_idx, regions)
                if nearest_region:
                    world.tiles[tile_idx].region_id = nearest_region['id']
                    # Also add to region's tile list if not already there
                    if tile_idx not in nearest_region['tiles']:
                        nearest_region['tiles'].append(tile_idx)
    
    def _create_geography_based_regions(self, world: 'World', land_tiles: List[int]) -> List[Dict[str, Any]]:
        """Create regions based on geographical features and elevation boundaries.
        
        Args:
            world: World instance
            land_tiles: List of land tile indices
            
        Returns:
            List of region dictionaries with id, tiles, and center
        """
        # Define elevation thresholds for natural boundaries
        # High mountains act as barriers
        mountain_threshold = 0.85  # Tiles above this are considered barriers
        hill_threshold = 0.7       # Hills can be minor barriers
        
        # Find connected land areas separated by elevation barriers
        regions = []
        visited = set()
        region_id = 0
        
        for tile_idx in land_tiles:
            if tile_idx in visited:
                continue
            
            # Start a new region from this unvisited land tile
            region_tiles = self._grow_geographical_region(
                world, tile_idx, visited, land_tiles, 
                mountain_threshold, hill_threshold
            )
            
            if region_tiles:
                # Calculate region center (centroid of tiles)
                center_x = sum(world.tiles[idx].center[0] for idx in region_tiles) / len(region_tiles)
                center_y = sum(world.tiles[idx].center[1] for idx in region_tiles) / len(region_tiles)
                
                # Find closest tile to centroid as center
                center_tile = min(region_tiles, 
                                key=lambda idx: ((world.tiles[idx].center[0] - center_x)**2 + 
                                               (world.tiles[idx].center[1] - center_y)**2))
                
                regions.append({
                    'id': region_id,
                    'tiles': region_tiles,
                    'center': center_tile
                })
                
                region_id += 1
        
        # Merge very small regions with neighbors
        regions = self._merge_small_regions(world, regions, land_tiles)
        
        # Reassign consecutive IDs after merging to avoid gaps
        for i, region in enumerate(regions):
            region['id'] = i
        
        # Update tile region assignments to match new consecutive IDs
        for region in regions:
            for tile_idx in region['tiles']:
                world.tiles[tile_idx].region_id = region['id']
        
        # Ensure all land tiles are assigned to regions
        assigned_tiles = set()
        for region in regions:
            assigned_tiles.update(region['tiles'])
        
        unassigned_tiles = [idx for idx in land_tiles if idx not in assigned_tiles]
        
        # Assign unassigned tiles to nearest regions
        for tile_idx in unassigned_tiles:
            nearest_region = self._find_nearest_region(world, tile_idx, regions)
            if nearest_region:
                nearest_region['tiles'].append(tile_idx)
                # Also assign the region ID to this tile
                world.tiles[tile_idx].region_id = nearest_region['id']
        
        return regions
    
    def _grow_geographical_region(self, world: 'World', start_idx: int, visited: set,
                                land_tiles: List[int], mountain_threshold: float, 
                                hill_threshold: float) -> List[int]:
        """Grow a region from a seed tile, preferring geographical continuity.
        
        Uses elevation as a preference rather than absolute barrier, allowing
        regions to expand across minor terrain features while respecting major mountains.
        
        Args:
            world: World instance
            start_idx: Starting tile index
            visited: Set of visited tiles
            land_tiles: All land tile indices
            mountain_threshold: Elevation above which tiles are strong barriers
            hill_threshold: Elevation for minor barriers
            
        Returns:
            List of tile indices in this region
        """
        from collections import deque
        
        region_tiles = []
        queue = deque([(start_idx, 0)])  # (tile_idx, barrier_penalty)
        start_elevation = world.tiles[start_idx].elevation
        
        while queue:
            current_idx, penalty = queue.popleft()
            
            if current_idx in visited or current_idx not in land_tiles:
                continue
            
            current_tile = world.tiles[current_idx]
            current_elevation = current_tile.elevation
            
            # Check if this tile can be added (with penalty system)
            can_add = True
            
            # Very high mountains are absolute barriers
            if current_elevation > mountain_threshold + 0.1:
                can_add = False
            # High mountains with penalty
            elif current_elevation > mountain_threshold:
                if penalty > 2:  # Too much penalty accumulated
                    can_add = False
                else:
                    penalty += 2
            # Hills with smaller penalty
            elif current_elevation > hill_threshold:
                if penalty > 3:  # Allow some hill crossing
                    can_add = False
                else:
                    penalty += 1
            
            if not can_add:
                continue
            
            # Add tile to region
            visited.add(current_idx)
            region_tiles.append(current_idx)
            
            # Stop if region is getting too large
            if len(region_tiles) > self.max_region_size:
                break
            
            # Add neighboring land tiles with updated penalties
            for neighbor_idx in current_tile.neighbors:
                if neighbor_idx not in visited and neighbor_idx in land_tiles:
                    neighbor_tile = world.tiles[neighbor_idx]
                    neighbor_elevation = neighbor_tile.elevation
                    
                    # Calculate elevation-based penalty for this neighbor
                    elevation_diff = abs(neighbor_elevation - start_elevation)
                    neighbor_penalty = penalty
                    
                    # Add penalty based on elevation difference from region start
                    if elevation_diff > 0.3:
                        neighbor_penalty += 2
                    elif elevation_diff > 0.2:
                        neighbor_penalty += 1
                    
                    # Prefer tiles closer to original elevation
                    queue.append((neighbor_idx, neighbor_penalty))
        
        return region_tiles
    
    def _find_nearest_region(self, world: 'World', tile_idx: int, regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the nearest region to a tile for assignment.
        
        Args:
            world: World instance
            tile_idx: Tile index to assign
            regions: List of region dictionaries
            
        Returns:
            Nearest region dictionary, or None if no regions exist
        """
        if not regions:
            return None
        
        tile_pos = world.tiles[tile_idx].center
        min_distance = float('inf')
        nearest_region = None
        
        for region in regions:
            # Check distance to region's center tile
            center_pos = world.tiles[region['center']].center
            distance = ((tile_pos[0] - center_pos[0])**2 + 
                       (tile_pos[1] - center_pos[1])**2)**0.5
            
            if distance < min_distance:
                min_distance = distance
                nearest_region = region
        
        return nearest_region
    
    def _merge_small_regions(self, world: 'World', regions: List[Dict[str, Any]], 
                           land_tiles: List[int]) -> List[Dict[str, Any]]:
        """Merge very small regions with their neighbors to prevent tiny regions.
        
        Args:
            world: World instance
            regions: List of region dictionaries
            land_tiles: All land tile indices
            
        Returns:
            Updated list of regions
        """
        min_region_size = self.min_region_size  # Use configured minimum region size
        
        # Find small regions
        small_regions = [r for r in regions if len(r['tiles']) < min_region_size]
        
        for small_region in small_regions:
            if small_region not in regions:
                continue
            
            # Find best neighbor to merge with
            best_neighbor = None
            best_shared_border = 0
            
            for other_region in regions:
                if other_region == small_region:
                    continue
                
                # Count shared border tiles
                shared_border = 0
                for tile_idx in small_region['tiles']:
                    tile = world.tiles[tile_idx]
                    for neighbor_idx in tile.neighbors:
                        if neighbor_idx in other_region['tiles']:
                            shared_border += 1
                
                if shared_border > best_shared_border:
                    best_shared_border = shared_border
                    best_neighbor = other_region
            
            if best_neighbor:
                # Merge small region into neighbor
                best_neighbor['tiles'].extend(small_region['tiles'])
                regions.remove(small_region)
        
        return regions
    
    def _find_connected_landmasses(self, world: 'World', land_tiles: List[int]) -> List[List[int]]:
        """Find connected components of land tiles to identify separate landmasses.
        
        Args:
            world: World instance
            land_tiles: List of land tile indices
            
        Returns:
            List of landmasses, each containing tile indices
        """
        visited = set()
        landmasses = []
        
        for tile_idx in land_tiles:
            if tile_idx not in visited:
                # Start new landmass with flood fill
                landmass = []
                queue = [tile_idx]
                
                while queue:
                    current_idx = queue.pop(0)
                    if current_idx in visited:
                        continue
                    
                    visited.add(current_idx)
                    landmass.append(current_idx)
                    
                    # Add unvisited land neighbors
                    for neighbor_idx in world.tiles[current_idx].neighbors:
                        if (neighbor_idx not in visited and 
                            not world.tiles[neighbor_idx].is_water):
                            queue.append(neighbor_idx)
                
                landmasses.append(landmass)
        
        return landmasses
    
    def _generate_regions_for_landmass(self, world: 'World', landmass: List[int], start_region_id: int) -> int:
        """Generate regions for a single connected landmass.
        
        Args:
            world: World instance
            landmass: List of tile indices in this landmass
            start_region_id: Starting region ID for this landmass
            
        Returns:
            Next available region ID
        """
        from src.world.world_data import Region
        import random
        
        # Determine number of regions for this landmass
        if len(landmass) < self.min_region_size:
            # Small islands get one region
            num_regions = 1
        else:
            num_regions = max(1, len(landmass) // self.tiles_per_region)
        
        visited = set()
        region_id = start_region_id
        
        for _ in range(num_regions):
            if len(visited) >= len(landmass):
                break
            
            # Find best seed for next region
            seed_idx = self._find_best_region_seed(world, landmass, visited)
            if seed_idx is None:
                break
            
            # Generate region color
            region_color = self._generate_region_color(region_id)
            
            # Create region
            region = Region(
                id=region_id,
                name=f"Region {region_id + 1}",
                color=region_color,
                tile_indices=[],
                center_tile_index=seed_idx
            )
            
            # Grow region with strong contiguity preference
            self._grow_contiguous_region(world, region, seed_idx, visited, landmass)
            
            world.regions.append(region)
            region_id += 1
        
        # Assign any remaining tiles to nearest regions
        self._assign_remaining_landmass_tiles(world, landmass, visited, start_region_id, region_id)
        
        return region_id
    
    def _find_best_region_seed(self, world: 'World', landmass: List[int], visited: set) -> int:
        """Find the best tile to start a new region from.
        
        Args:
            world: World instance
            landmass: Available tiles in this landmass
            visited: Already assigned tiles
            
        Returns:
            Best seed tile index, or None if no valid seeds
        """
        import random
        
        available = [idx for idx in landmass if idx not in visited]
        if not available:
            return None
        
        # For small landmasses, just pick randomly
        if len(available) <= self.min_region_size:
            return random.choice(available)
        
        # For larger landmasses, prefer tiles that are far from existing regions
        if not visited:
            # First region - pick randomly
            return random.choice(available)
        
        # Find tile that's furthest from any assigned tile
        best_seed = None
        max_distance = -1
        
        for candidate in available[:min(20, len(available))]:  # Sample to avoid slow computation
            min_dist_to_assigned = float('inf')
            candidate_pos = world.tiles[candidate].center
            
            for assigned_idx in visited:
                if assigned_idx in landmass:  # Only consider assigned tiles in this landmass
                    assigned_pos = world.tiles[assigned_idx].center
                    dist = ((candidate_pos[0] - assigned_pos[0])**2 + 
                           (candidate_pos[1] - assigned_pos[1])**2)**0.5
                    min_dist_to_assigned = min(min_dist_to_assigned, dist)
            
            if min_dist_to_assigned > max_distance:
                max_distance = min_dist_to_assigned
                best_seed = candidate
        
        return best_seed or random.choice(available)
    
    def _grow_contiguous_region(self, world: 'World', region: 'Region', seed_idx: int, 
                              visited: set, landmass: List[int]) -> None:
        """Grow a region with strong preference for contiguous expansion.
        
        Args:
            world: World instance
            region: Region to grow
            seed_idx: Starting tile index
            visited: Set of already visited tile indices
            landmass: Available tiles in this landmass
        """
        import random
        from collections import deque
        
        target_size = random.randint(self.min_region_size, self.max_region_size)
        
        # Use BFS for more even expansion
        queue = deque([seed_idx])
        region_tiles = set()
        
        while queue and len(region.tile_indices) < target_size:
            current_idx = queue.popleft()
            
            if current_idx in visited or current_idx not in landmass:
                continue
            
            visited.add(current_idx)
            region.tile_indices.append(current_idx)
            region_tiles.add(current_idx)
            world.tiles[current_idx].region_id = region.id
            
            # Add neighbors in order of preference
            neighbors = list(world.tiles[current_idx].neighbors)
            random.shuffle(neighbors)  # Add some randomness
            
            for neighbor_idx in neighbors:
                if (neighbor_idx not in visited and 
                    neighbor_idx in landmass and
                    neighbor_idx not in queue):
                    
                    # Strong preference for tiles that create contiguous regions
                    neighbor_tile = world.tiles[neighbor_idx]
                    current_tile = world.tiles[current_idx]
                    
                    # Check if this neighbor would help maintain contiguity
                    if self._should_add_to_region(current_tile, neighbor_tile, region_tiles, world):
                        queue.append(neighbor_idx)
    
    def _should_add_to_region(self, current_tile: 'Tile', neighbor_tile: 'Tile', 
                            region_tiles: set, world: 'World') -> bool:
        """Determine if a neighbor should be added to maintain good region shape.
        
        Args:
            current_tile: Current tile in region
            neighbor_tile: Potential neighbor to add
            region_tiles: Current tiles in region
            world: World instance
            
        Returns:
            True if neighbor should be added
        """
        import random
        
        # Always add if region is small
        if len(region_tiles) <= 3:
            return True
        
        # Count how many of this neighbor's neighbors are already in the region
        neighbor_idx = None
        for idx, tile in enumerate(world.tiles):
            if tile.center == neighbor_tile.center:
                neighbor_idx = idx
                break
        
        if neighbor_idx is None:
            return random.random() < 0.5
        
        region_neighbor_count = sum(1 for n_idx in world.tiles[neighbor_idx].neighbors 
                                  if n_idx in region_tiles)
        
        # Strong preference for tiles that have multiple connections to the region
        if region_neighbor_count >= 2:
            return random.random() < 0.9
        elif region_neighbor_count == 1:
            # Check biome compatibility for edge expansion
            biome_match = current_tile.biome == neighbor_tile.biome
            elevation_similar = abs(current_tile.elevation - neighbor_tile.elevation) < 0.15
            
            if biome_match:
                return random.random() < 0.8
            elif elevation_similar:
                return random.random() < 0.6
            else:
                return random.random() < 0.3
        else:
            # Avoid creating disconnected components
            return random.random() < 0.1
    
    def _assign_remaining_landmass_tiles(self, world: 'World', landmass: List[int], 
                                       visited: set, start_region_id: int, end_region_id: int) -> None:
        """Assign remaining unvisited tiles in landmass to nearest regions.
        
        Args:
            world: World instance
            landmass: Tiles in this landmass
            visited: Already assigned tiles
            start_region_id: First region ID for this landmass
            end_region_id: Last region ID for this landmass
        """
        unassigned = [idx for idx in landmass if idx not in visited]
        
        for tile_idx in unassigned:
            # Find nearest region by checking neighbors iteratively
            best_region_id = self._find_nearest_region_in_landmass(
                world, tile_idx, start_region_id, end_region_id)
            
            if best_region_id >= 0:
                world.tiles[tile_idx].region_id = best_region_id
                # Find the region and add this tile
                for region in world.regions:
                    if region.id == best_region_id:
                        region.tile_indices.append(tile_idx)
                        break
    
    def _find_nearest_region_in_landmass(self, world: 'World', tile_idx: int, 
                                        start_region_id: int, end_region_id: int) -> int:
        """Find nearest region for a tile within the same landmass.
        
        Args:
            world: World instance
            tile_idx: Tile to assign
            start_region_id: First valid region ID
            end_region_id: Last valid region ID
            
        Returns:
            Best region ID or -1 if none found
        """
        # BFS to find nearest assigned tile
        visited = set()
        queue = [tile_idx]
        
        while queue:
            current_idx = queue.pop(0)
            if current_idx in visited:
                continue
            visited.add(current_idx)
            
            current_region = world.tiles[current_idx].region_id
            if start_region_id <= current_region < end_region_id:
                return current_region
            
            # Add unvisited neighbors
            for neighbor_idx in world.tiles[current_idx].neighbors:
                if neighbor_idx not in visited and not world.tiles[neighbor_idx].is_water:
                    queue.append(neighbor_idx)
        
        # Fallback to first region if BFS fails
        return start_region_id if start_region_id < end_region_id else -1

    
    def _generate_region_color(self, region_id: int) -> Tuple[int, int, int]:
        """Generate a distinct color for a region.
        
        Args:
            region_id: Region identifier
            
        Returns:
            RGB color tuple
        """
        # Use HSV color space to generate distinct colors
        import colorsys
        
        # Generate evenly spaced hues
        hue = (region_id * 0.618033988749895) % 1.0  # Golden ratio for better distribution
        saturation = 0.6 + (region_id % 3) * 0.15     # Vary saturation slightly
        value = 0.7 + (region_id % 2) * 0.2           # Vary brightness slightly
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    
    def _generate_test_polity(self, world: 'World') -> None:
        """Generate a starting polity with just one tile and 100 population.
        
        Args:
            world: World instance to add starting polity to
        """
        import random
        
        # Find a suitable starting tile for the test polity
        land_tiles = [i for i, tile in enumerate(world.tiles) if not tile.is_water]
        
        if len(land_tiles) < 6:
            return  # Not enough land tiles
        
        # Find a tile with several land neighbors for good expansion
        seed_candidates = []
        for tile_idx in land_tiles:
            land_neighbors = sum(1 for n_idx in world.tiles[tile_idx].neighbors 
                               if not world.tiles[n_idx].is_water)
            if land_neighbors >= 3:  # Good connectivity
                seed_candidates.append(tile_idx)
        
        if not seed_candidates:
            seed_candidates = land_tiles[:10]  # Fallback
        
        seed_idx = random.choice(seed_candidates)
        
        # Create test leader
        test_leader = Leader(
            name="Test Ruler",
            age=45,
            culture="TestCulture"
        )
        
        # Create test polity
        test_polity = Polity(
            id=0,
            name=self._generate_placeholder_polity_name(),
            color=(200, 50, 50),  # Red color for easy identification
            leader=test_leader,
            primary_culture=None,
            tile_indices=[],
            capital_tile_index=seed_idx
        )
        
        # Create polity with just the single starting tile
        test_polity.tile_indices.append(seed_idx)
        
        # Set polity control on the single tile with population of exactly 1000
        old_polity = world.tiles[seed_idx].polity_id
        world.tiles[seed_idx].polity_id = test_polity.id
        world.tiles[seed_idx].control_level = 100  # Capital control
        world.tiles[seed_idx].population = 1000   # Exactly 1000 population
        world.tiles[seed_idx].development = 100.0  # High development level
        if hasattr(world, '_on_tile_polity_changed'):
            world._on_tile_polity_changed(seed_idx, old_polity, test_polity.id)
        
        # Add polity to world
        world.polities.append(test_polity)
        world.assign_polity_language_name(test_polity)
        if hasattr(world, '_ensure_relationships_for_polity'):
            world._ensure_relationships_for_polity(test_polity.id)
        if hasattr(world, '_ensure_population_center_entry'):
            world._ensure_population_center_entry(test_polity.capital_tile_index, name=f"{test_polity.name} Capital", threshold=1)
        
        if self.verbose_logging:
            print("\nTest Polity Created:")
            print(f"  Name: {test_polity.name}")
            print(f"  Leader: {test_polity.leader.name} (age {test_polity.leader.age})")
            print(f"  Culture: {test_polity.primary_culture}")
            print(f"  Tiles Controlled: {len(test_polity.tile_indices)}")
            print(f"  Color: {test_polity.color}")
    
    def _generate_placeholder_polity_name(self) -> str:
        """Generate a placeholder name for a polity.
        
        Returns:
            Placeholder polity name
        """
        import random
        
        # Simple placeholder name components
        prefixes = [
            "Kingdom of", "Duchy of", "Republic of", "Empire of", "Principality of",
            "State of", "Realm of", "Domain of", "Territory of", "Federation of"
        ]
        
        place_names = [
            "Alderia", "Bretonia", "Calthara", "Draven", "Esmeria", "Falshire", 
            "Galandor", "Havenmoor", "Ironhold", "Jorthak", "Kelthara", "Luminar",
            "Morgrim", "Nordmark", "Oathland", "Pyrrhia", "Quendel", "Ravenspire",
            "Stormhaven", "Thornwick", "Ulthara", "Valorian", "Westmarch", "Xerion",
            "Ysgard", "Zephyria"
        ]
        
        prefix = random.choice(prefixes)
        place = random.choice(place_names)
        
        return f"{prefix} {place}"
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple.
        
        Args:
            hex_color: Hex color string (e.g., '#FF0000')
            
        Returns:
            RGB tuple
        """
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _interpolate_color(self, color1: Tuple[int, int, int], 
                          color2: Tuple[int, int, int], 
                          ratio: float) -> Tuple[int, int, int]:
        """Interpolate between two colors.
        
        Args:
            color1: First color (RGB)
            color2: Second color (RGB)
            ratio: Interpolation ratio (0-1)
            
        Returns:
            Interpolated color (RGB)
        """
        ratio = max(0, min(1, ratio))  # Clamp to 0-1
        
        r = int(color1[0] + (color2[0] - color1[0]) * ratio)
        g = int(color1[1] + (color2[1] - color1[1]) * ratio)
        b = int(color1[2] + (color2[2] - color1[2]) * ratio)
        
        return (r, g, b)
    
    def _apply_height_curve(self, elevation: float) -> float:
        """Apply height curve transformation to elevation value.
        
        Args:
            elevation: Raw elevation value (0-1)
            
        Returns:
            Curve-transformed elevation value (0-1)
        """
        # Clamp input to 0-1 range
        elevation = max(0.0, min(1.0, elevation))
        
        # Apply curve transformation: y = coefficient * x^exponent
        if self.height_curve_coefficient == 1.0 and self.height_curve_exponent == 1.0:
            # Optimization: skip calculation for linear curve with no amplification
            return elevation
        else:
            # Apply curve transformation
            transformed = self.height_curve_coefficient * pow(elevation, self.height_curve_exponent)
            # Ensure result stays in 0-1 range (clamp if coefficient > 1)
            return max(0.0, min(1.0, transformed))
    
    def _generate_continental_plates(self, points: np.ndarray) -> List[Dict[str, Any]]:
        """Generate tectonic plates for continental formation (configurable)."""
        tectonics = self.config.get_section('world').get('tectonics', {})
        num_plates_min = tectonics.get('num_plates_min', 15)
        num_plates_max = tectonics.get('num_plates_max', 22)
        if self.verbose_logging:
            print("[WorldGen][DEBUG] Tectonic config loaded:")
            for k, v in tectonics.items():
                print(f"    {k}: {v}")
        # Safety: enforce at least 1 plate
        if num_plates_min < 1 or num_plates_max < 1:
            if self.verbose_logging:
                print(f"[WorldGen][WARNING] num_plates_min ({num_plates_min}) or num_plates_max ({num_plates_max}) < 1. Forcing both to 1.")
            num_plates_min = max(1, num_plates_min)
            num_plates_max = max(1, num_plates_max)
        num_plates = random.randint(num_plates_min, num_plates_max)
        # Use grid layout for smoother, more regular plates
        grid_size = int(math.sqrt(num_plates))
        if grid_size ** 2 < num_plates:
            grid_size += 1
        num_plates = min(num_plates, grid_size ** 2)  # Adjust to fit grid
        plates = []
        for i in range(num_plates):
            row = i // grid_size
            col = i % grid_size
            # Place centers on grid with small jitter for irregularity
            jitter_x = random.uniform(-self.width / grid_size / 4, self.width / grid_size / 4)
            jitter_y = random.uniform(-self.height / grid_size / 4, self.height / grid_size / 4)
            center_x = (col + 0.5) * self.width / grid_size + jitter_x
            center_y = (row + 0.5) * self.height / grid_size + jitter_y
            size = random.uniform(0.2, 0.5)  # Could be made configurable if needed
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(0.1, 0.3)
            move_x = np.cos(angle) * speed
            move_y = np.sin(angle) * speed
            plate_types = ['continental', 'continental', 'continental', 'oceanic', 'mixed']
            plates.append({
                'id': i,
                'center': (center_x, center_y),
                'size': size,
                'movement': (move_x, move_y),
                'type': random.choice(plate_types)
            })
        return plates
    
    def _generate_continental_elevation(self, points: np.ndarray, plates: List[Dict[str, Any]]) -> np.ndarray:
        """Generate base continental elevation from tectonic plates (configurable)."""
        tectonics = self.config.get_section('world.tectonics')
        influence_radius_scale = tectonics.get('influence_radius_scale', 0.45)
        weight_exponent = tectonics.get('weight_exponent', 2.0)
        continental_base = tectonics.get('continental_base', 0.55)
        continental_var = tectonics.get('continental_var', 0.08)
        oceanic_base = tectonics.get('oceanic_base', 0.18)
        oceanic_var = tectonics.get('oceanic_var', 0.03)
        mixed_base = tectonics.get('mixed_base', 0.38)
        mixed_var = tectonics.get('mixed_var', 0.06)
        elevations = np.zeros(len(points))
        for i, (x, y) in enumerate(points):
            elevation = 0.0
            total_weight = 0.0
            for plate in plates:
                dx = x - plate['center'][0]
                dy = y - plate['center'][1]
                distance = np.sqrt(dx*dx + dy*dy)
                influence_radius = plate['size'] * min(self.width, self.height) * influence_radius_scale
                weight = max(0, 1 - (distance / influence_radius))
                weight = weight ** weight_exponent
                if plate['type'] == 'continental':
                    plate_elevation = continental_base  # Fixed for smoother elevation
                elif plate['type'] == 'oceanic':
                    plate_elevation = oceanic_base
                else:
                    plate_elevation = mixed_base
                elevation += plate_elevation * weight
                total_weight += weight
            if total_weight > 0:
                elevations[i] = elevation / total_weight
            else:
                elevations[i] = 0.4
        return elevations
    
    def _generate_mountain_ranges(self, points: np.ndarray, plates: List[Dict[str, Any]]) -> np.ndarray:
        """Generate mountain ranges along plate boundaries.
        
        Args:
            points: Array of point coordinates
            plates: List of tectonic plates
            
        Returns:
            Array of mountain elevation values
        """
        elevations = np.zeros(len(points))
        
        for i, (x, y) in enumerate(points):
            mountain_elevation = 0.0
            
            # Check proximity to plate boundaries
            min_boundary_distance = float('inf')
            
            for plate1 in plates:
                for plate2 in plates:
                    if plate1['id'] >= plate2['id']:
                        continue
                    
                    # Calculate distance to boundary between plates
                    boundary_distance = self._distance_to_plate_boundary(
                        (x, y), plate1, plate2
                    )
                    
                    min_boundary_distance = min(min_boundary_distance, boundary_distance)
            
            # Generate mountains near boundaries (even smaller zone, gentler)
            if min_boundary_distance < 7:
                # Mountain height decreases with distance from boundary
                mountain_factor = max(0, 1 - (min_boundary_distance / 7))
                mountain_factor = mountain_factor ** 3.5  # Even softer falloff

                # Add some randomness
                noise = pnoise2(
                    x / 60.0, y / 60.0,  # Even broader
                    octaves=1,
                    persistence=0.2,
                    base=self.noise_seed + 2000
                )
                noise = (noise + 1) / 2  # 0-1

                # Much lower mountain elevation
                mountain_elevation = mountain_factor * 0.005 * (0.7 + 0.3 * noise)
                elevations[i] = mountain_elevation
            else:
                mountain_elevation = 0.0
        
        return elevations
    
    def _generate_volcanic_islands(self, points: np.ndarray, plates: List[Dict[str, Any]]) -> np.ndarray:
        """Generate volcanic islands in ocean areas.
        
        Args:
            points: Array of point coordinates
            
        Returns:
            Array of volcanic island elevation values
        """
        elevations = np.zeros(len(points))
        
        # Generate very few volcanic islands
        num_chains = random.randint(0, 2)  # Very few chains
        
        for chain_id in range(num_chains):
            # Random chain parameters
            chain_length = random.randint(1, 4)  # Shorter chains
            
            # Position chain in ocean areas (away from continental plates)
            # Find a spot far from continental plates
            best_distance = 0
            start_x, start_y = 0, 0
            
            for attempt in range(20):  # Try multiple positions
                test_x = random.uniform(0, self.width)
                test_y = random.uniform(0, self.height)
                
                min_dist_to_continental = float('inf')
                for plate in plates:
                    if plate['type'] == 'continental':
                        dx = test_x - plate['center'][0]
                        dy = test_y - plate['center'][1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist_to_continental = min(min_dist_to_continental, dist)
                
                if min_dist_to_continental > best_distance:
                    best_distance = min_dist_to_continental
                    start_x, start_y = test_x, test_y
            
            # Chain direction
            angle = random.uniform(0, 2 * np.pi)
            spacing = random.uniform(40, 80)  # Larger spacing
            
            for island_idx in range(chain_length):
                # Position along chain
                dx = np.cos(angle) * spacing * island_idx
                dy = np.sin(angle) * spacing * island_idx
                
                island_x = start_x + dx
                island_y = start_y + dy
                
                # Add some randomness to position
                island_x += random.uniform(-15, 15)
                island_y += random.uniform(-15, 15)
                
                # Smaller, lower islands
                island_size = random.uniform(5, 15)  # Much smaller
                island_height = random.uniform(0.1, 0.25)  # Much lower
                
                # Apply to nearby points
                for i, (x, y) in enumerate(points):
                    dx = x - island_x
                    dy = y - island_y
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance < island_size:
                        # Elevation falls off with distance
                        elevation_factor = max(0, 1 - (distance / island_size))
                        elevation_factor = elevation_factor * elevation_factor
                        
                        elevations[i] = max(elevations[i], island_height * elevation_factor)
        
        return elevations
    
    def _distance_to_plate_boundary(self, point: Tuple[float, float], 
                                  plate1: Dict[str, Any], 
                                  plate2: Dict[str, Any]) -> float:
        """Calculate distance from point to boundary between two plates.
        
        Args:
            point: (x, y) coordinates
            plate1: First plate dictionary
            plate2: Second plate dictionary
            
        Returns:
            Distance to plate boundary
        """
        # Simple approximation: distance to line between plate centers
        p1 = plate1['center']
        p2 = plate2['center']
        
        # Vector from p1 to p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Vector from p1 to point
        px = point[0] - p1[0]
        py = point[1] - p1[1]
        
        # Project point onto line
        line_length = dx*dx + dy*dy
        if line_length == 0:
            return np.sqrt(px*px + py*py)  # Plates at same location
        
        t = max(0, min(1, (px*dx + py*dy) / line_length))
        
        # Closest point on line segment
        closest_x = p1[0] + t * dx
        closest_y = p1[1] + t * dy
        
        # Distance to closest point
        dist_x = point[0] - closest_x
        dist_y = point[1] - closest_y
        
        return np.sqrt(dist_x*dist_x + dist_y*dist_y)
    
    def _calculate_temperature(self, x: float, y: float, elevation: float) -> float:
        """Calculate temperature based on latitude and altitude.
        
        Args:
            x: X coordinate
            y: Y coordinate  
            elevation: Elevation value (0-1)
            
        Returns:
            Temperature value (0-1)
        """
        # Calculate distance from equator (normalized to 0-1)
        equator_y = self.equator_position * self.height
        distance_from_equator = abs(y - equator_y) / (self.height * self.pole_distance)
        distance_from_equator = min(1.0, distance_from_equator)
        
        # Base temperature from latitude
        base_temp = self.temp_equator_base - (self.temp_equator_base - self.temp_pole_base) * distance_from_equator
        
        # Altitude effect (higher = colder) - only significant at high elevations
        # Use a strict threshold approach where altitude has NO effect below threshold
        if elevation > self.temp_altitude_threshold:
            # Strong effect above threshold - more dramatic cooling for mountain peaks
            excess_altitude = (elevation - self.temp_altitude_threshold) / (1.0 - self.temp_altitude_threshold)
            # Use cubic scaling for even more dramatic high-altitude cooling
            altitude_reduction = excess_altitude * excess_altitude * excess_altitude * self.temp_altitude_effect
        else:
            # NO altitude effect below threshold
            altitude_reduction = 0.0
        
        # Final temperature
        temperature = base_temp - altitude_reduction
        return max(0.0, min(1.0, temperature))
    
    def _calculate_rainfall(self, x: float, y: float, elevation: float) -> float:
        """Calculate rainfall using Perlin noise with altitude effects.
        
        Args:
            x: X coordinate
            y: Y coordinate
            elevation: Elevation value (affects high-altitude rainfall)
            
        Returns:
            Rainfall value (0-1)
        """
        # Generate rainfall noise (using different seed offset to avoid correlation with elevation)
        rainfall_noise = pnoise2(
            x / self.rainfall_scale,
            y / self.rainfall_scale,
            octaves=self.rainfall_octaves,
            persistence=self.rainfall_persistence,
            lacunarity=self.rainfall_lacunarity,
            base=self.noise_seed + 1000  # Offset seed for different pattern
        )
        
        # Normalize to 0-1 range
        base_rainfall = (rainfall_noise + 1) / 2
        
        # Apply contrast to stretch the range toward extremes
        # Values closer to 0.5 get pushed toward 0 or 1
        if self.rainfall_contrast != 1.0:
            # Center around 0.5, apply contrast, then re-center
            centered = base_rainfall - 0.5
            stretched = centered * self.rainfall_contrast
            base_rainfall = stretched + 0.5
        
        # Apply base offset to shift overall rainfall levels
        base_rainfall += self.rainfall_base_offset
        
        # Clamp to valid range before altitude effects
        base_rainfall = max(0.0, min(1.0, base_rainfall))
        
        # Apply altitude effect - high altitude reduces rainfall significantly (same threshold as temperature)
        if elevation > self.temp_altitude_threshold:  # Use same threshold as temperature (0.6)
            # Strong effect above threshold - more dramatic rainfall reduction for high altitudes
            excess_altitude = (elevation - self.temp_altitude_threshold) / (1.0 - self.temp_altitude_threshold)
            # Use quadratic scaling for significant high-altitude rainfall reduction
            altitude_reduction_factor = excess_altitude * excess_altitude * 0.8  # Reduce by up to 80%
            base_rainfall *= (1.0 - altitude_reduction_factor)
        
        # Final clamp to ensure valid range
        return max(0.0, min(1.0, base_rainfall))
    
    def _get_biome_color(self, elevation: float, temperature: float, rainfall: float, is_water: bool) -> Tuple[Tuple[int, int, int], str]:
        """Determine tile color and biome based on configurable biome classification.
        
        Args:
            elevation: Elevation value (0-1)
            temperature: Temperature value (0-1)
            rainfall: Rainfall value (0-1)
            is_water: Whether tile is water
            
        Returns:
            Tuple of (RGB color tuple, biome name)
        """
        if is_water:
            # Water color based on depth (unchanged)
            depth_ratio = elevation / self.sea_level if self.sea_level > 0 else 0
            deep_color = self._hex_to_rgb(self.colors.get('deep_water', '#024B86'))
            shallow_color = self._hex_to_rgb(self.colors.get('shallow_water', '#8CF6FF'))
            water_color = self._interpolate_color(deep_color, shallow_color, depth_ratio)
            return water_color, "Water"
        
        # Find matching biome from config
        biome_name = self._classify_biome_from_config(temperature, rainfall)
        
        # Get biome configuration
        biomes = self.colors.get('biomes', {})
        biome_config = biomes.get(biome_name, {'color': '#808080', 'name': 'Unknown'})
        
        # Get base color for biome
        base_color = self._hex_to_rgb(biome_config.get('color', '#808080'))
        
        # Apply altitude lightness effect
        adjusted_color = self._apply_altitude_lightness(base_color, elevation)
        
        return adjusted_color, biome_config.get('name', 'Unknown')
    
    def _classify_biome_from_config(self, temperature: float, rainfall: float) -> str:
        """Classify biome based on configurable conditions.
        
        Args:
            temperature: Temperature value (0-1)
            rainfall: Rainfall value (0-1)
            
        Returns:
            Biome key string
        """
        biomes = self.colors.get('biomes', {})
        
        # Check each biome's conditions
        for biome_key, biome_config in biomes.items():
            conditions = biome_config.get('conditions', {})
            
            # Check temperature conditions
            temp_min = conditions.get('temperature_min', 0.0)
            temp_max = conditions.get('temperature_max', 1.0)
            if not (temp_min <= temperature <= temp_max):
                continue
            
            # Check rainfall conditions
            rain_min = conditions.get('rainfall_min', 0.0)
            rain_max = conditions.get('rainfall_max', 1.0)
            if not (rain_min <= rainfall <= rain_max):
                continue
            
            # All conditions met
            return biome_key
        
        # Fallback if no biome matches
        return 'unknown'
    
    def _apply_altitude_lightness(self, base_color: Tuple[int, int, int], elevation: float) -> Tuple[int, int, int]:
        """Apply altitude-based lightness adjustment to color.
        
        Args:
            base_color: Base RGB color
            elevation: Elevation value (0-1)
            
        Returns:
            Adjusted RGB color
        """
        r, g, b = base_color
        
        # Calculate lightness adjustment (higher = lighter)
        # Max adjustment is +30% lightness for highest elevations
        lightness_factor = 1.0 + (elevation * 0.3)
        
        # Apply adjustment while keeping colors in valid range
        r = min(255, int(r * lightness_factor))
        g = min(255, int(g * lightness_factor))
        b = min(255, int(b * lightness_factor))
        
        return (r, g, b)
    
    def _generate_base_population(self, elevation: float, is_water: bool, is_controlled: bool,
                                temperature: float, rainfall: float) -> int:
        """Generate base population for a tile based on environmental factors.
        
        Args:
            elevation: Tile elevation
            is_water: Whether tile is water
            temperature: Temperature value
            rainfall: Rainfall value
            
        Returns:
            Base population count
        """
        if is_water or not is_controlled:
            return 0
        
        # Get config values
        base_factor = self.config.get('simulation.initial_generation.population_base_factor', 500)
        suitability_threshold = self.config.get('simulation.initial_generation.population_suitability_threshold', 0.2)
        
        # Population is higher in moderate climates and lower elevations
        # Temperature preference: moderate (0.4-0.8)
        temp_factor = 1.0 - abs(temperature - 0.6)  # Peak at 0.6
        temp_factor = max(0.0, temp_factor)
        
        # Rainfall preference: moderate to high (0.3-0.9)
        rain_factor = min(1.0, max(0.0, rainfall - 0.1) / 0.8)
        
        # Elevation preference: lower is better for large populations
        elev_factor = max(0.1, 1.0 - elevation)
        
        # Combine factors with some randomness
        combined_factor = temp_factor * rain_factor * elev_factor
        
        # Only generate population if conditions are minimally suitable
        if combined_factor < suitability_threshold:  # Very harsh conditions
            return 0
        
        population = int(combined_factor * random.randint(50, base_factor))
        
        return max(0, population)
    
    def _generate_base_development(self, elevation: float, is_water: bool, is_controlled: bool) -> float:
        """Generate base development level for a tile.
        
        Args:
            elevation: Tile elevation
            is_water: Whether tile is water
            
        Returns:
            Development level (uncapped, typically 0-1000+)
        """
        if is_water or not is_controlled:
            return 0.0
        
        # Get config values
        undeveloped_chance = self.config.get('simulation.initial_generation.undeveloped_chance', 0.3)
        extreme_high_threshold = self.config.get('simulation.initial_generation.elevation_thresholds.extreme_high', 0.9)
        high_threshold = self.config.get('simulation.initial_generation.elevation_thresholds.high', 0.8)
        medium_threshold = self.config.get('simulation.initial_generation.elevation_thresholds.medium', 0.6)
        
        high_elev_range = self.config.get('simulation.initial_generation.development_ranges.high_elevation', [5, 20])
        medium_elev_range = self.config.get('simulation.initial_generation.development_ranges.medium_elevation', [10, 40])
        low_elev_range = self.config.get('simulation.initial_generation.development_ranges.low_elevation', [20, 80])
        
        # Development starts low and grows with population
        # Very high elevations are harder to develop
        if elevation > extreme_high_threshold:  # Extremely high elevations
            return 0.0
        elif elevation > high_threshold:
            base_dev = random.randint(high_elev_range[0], high_elev_range[1])
        elif elevation > medium_threshold:
            base_dev = random.randint(medium_elev_range[0], medium_elev_range[1])
        else:
            base_dev = random.randint(low_elev_range[0], low_elev_range[1])
        
        # Random chance for completely undeveloped areas
        if random.random() < undeveloped_chance:
            return 0.0
        
        return float(base_dev)