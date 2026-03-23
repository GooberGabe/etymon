"""Map rendering system for Etymon."""

import hashlib
import math
import time
import pygame
from collections import deque
from typing import List, Tuple, Optional, Dict, Any, Callable

from src.core.config_manager import ConfigManager
from src.world.world_data import World, Tile, Region


class MapRenderer:
    """Handles rendering of the world map."""
    
    def __init__(self, config: ConfigManager, screen: pygame.Surface):
        """Initialize map renderer.
        
        Args:
            config: Configuration manager
            screen: Pygame screen surface
        """
        self.config = config
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()
        
        # Camera system
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.zoom = 1.0
        
        # Camera settings from config
        self.pan_speed = config.get('camera.pan_speed', 5.0)
        self.zoom_speed = config.get('camera.zoom_speed', 0.1)
        self.min_zoom = config.get('camera.min_zoom', 0.2)
        self.max_zoom = config.get('camera.max_zoom', 4.0)
        
        # For tracking current colors
        self.current_map_mode = "biomes"
        profiling_default = config.get('simulation.debug.profile_render', False) if config else False
        history_size = config.get('simulation.debug.profile_render_history', 120) if config else 120
        try:
            history_size = max(1, int(history_size))
        except (TypeError, ValueError):
            history_size = 120
        self.render_profiling_enabled = bool(profiling_default)
        self.render_profile_print = bool(config.get('simulation.debug.profile_render_print', True)) if config else True
        self._render_profile_history: deque = deque(maxlen=history_size)
        self._render_profile_sections: Dict[str, float] = {}
        self._render_profile_start_time = 0.0
    
    def render_world(self, world: World) -> None:
        """Render the entire world.
        
        Args:
            world: World instance to render
        """
        self._begin_render_profile()
        # Update colors for dynamic map modes (population, development) annually
        # and for all modes when the mode changes
        dynamic_modes = {"population", "development", "control"}
        mode_changed = world.current_map_mode != self.current_map_mode
        needs_refresh = (
            mode_changed
            or world.current_map_mode == "cultures"
            or world.current_map_mode == "linguistics"
            or world.current_map_mode == "war"
            or (world.current_map_mode in dynamic_modes and world.current_season == 0)
        )
        if needs_refresh:
            self._profile_render_section('update_tile_colors', self.update_tile_colors, world)
            
        # Render all tiles
        self._profile_render_section('tiles', self._render_all_tiles, world)
        
        # Render rivers before overlays
        self._profile_render_section('rivers', self.render_rivers, world)
        
        # Render polity borders on top
        self._profile_render_section('polity_borders', self.render_polity_borders, world)
        
        # Render population centers (dots and settlement names)
        self._profile_render_section('population_centers', self.render_population_centers, world)
        
        # Render polity text labels on top of everything else
        self._profile_render_section('polity_labels', self.render_polity_text, world)

        if world.current_map_mode == "regions":
            self._profile_render_section('region_labels', self.render_region_labels, world)
        self._complete_render_profile(world)
    
    def set_render_profiling_enabled(self, enabled: bool) -> None:
        """Enable or disable render profiling at runtime."""
        self.render_profiling_enabled = bool(enabled)

    def get_recent_render_profiles(self, count: int = 5) -> List[Dict[str, Any]]:
        """Return recent render profiling entries."""
        if count <= 0:
            return []
        return list(self._render_profile_history)[-count:]

    def _begin_render_profile(self) -> None:
        if not self.render_profiling_enabled:
            return
        self._render_profile_sections = {}
        self._render_profile_start_time = time.perf_counter()

    def _profile_render_section(self, label: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        if not self.render_profiling_enabled:
            return func(*args, **kwargs)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            self._render_profile_sections[label] = self._render_profile_sections.get(label, 0.0) + elapsed

    def _complete_render_profile(self, world: World) -> None:
        if not self.render_profiling_enabled:
            return
        total_duration = time.perf_counter() - self._render_profile_start_time
        entry = {
            'tick': getattr(world, 'total_ticks', None),
            'map_mode': world.current_map_mode,
            'total_seconds': total_duration,
            'sections': dict(self._render_profile_sections),
        }
        self._render_profile_history.append(entry)
        if self.render_profile_print:
            sections_sorted = sorted(
                self._render_profile_sections.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if sections_sorted:
                section_bits = ", ".join(
                    f"{name}={duration * 1000:.1f}ms" for name, duration in sections_sorted
                )
                print(
                    f"[Profiler] Render total={total_duration * 1000:.1f}ms (mode={world.current_map_mode}) :: {section_bits}"
                )
            else:
                print(
                    f"[Profiler] Render total={total_duration * 1000:.1f}ms (mode={world.current_map_mode})"
                )
        self._render_profile_sections = {}

    def _render_all_tiles(self, world: World) -> None:
        for tile in world.tiles:
            self.render_tile(tile)

    def render_rivers(self, world: World) -> None:
        """Render all river polylines on top of terrain."""
        rivers = getattr(world, 'rivers', None)
        if not rivers:
            return
        color_values = self.config.get('rendering.rivers.color', [70, 130, 200]) if self.config else [70, 130, 200]
        color = (
            int(color_values[0]) if len(color_values) > 0 else 70,
            int(color_values[1]) if len(color_values) > 1 else 130,
            int(color_values[2]) if len(color_values) > 2 else 200,
        )
        min_width = float(self.config.get('rendering.rivers.min_width', 1.0)) if self.config else 1.0
        max_width = float(self.config.get('rendering.rivers.max_width', 4.0)) if self.config else 4.0
        width_power = float(self.config.get('rendering.rivers.width_power', 0.6)) if self.config else 0.6
        flux_cap = float(self.config.get('rendering.rivers.max_flux_for_width', 1.0)) if self.config else 1.0
        max_width = max(min_width, max_width)
        for river in rivers:
            points = getattr(river, 'points', None) or []
            if len(points) < 2:
                continue
            screen_points = [self.world_to_screen(x, y) for x, y in points]
            if len(screen_points) < 2:
                continue

            tile_indices = getattr(river, 'tile_indices', None) or []
            local_flux_samples: List[float] = []
            if tile_indices and hasattr(world, 'tiles'):
                for tile_idx in tile_indices:
                    if 0 <= tile_idx < len(world.tiles):
                        local_flux_samples.append(getattr(world.tiles[tile_idx], 'river_flux', getattr(river, 'flux', 0.0)))
            if not local_flux_samples:
                local_flux_samples = [getattr(river, 'flux', 0.0)]

            max_local_index = len(local_flux_samples) - 1
            for i in range(len(screen_points) - 1):
                sample_idx = min(i, max_local_index)
                segment_flux = max(0.0, local_flux_samples[sample_idx])
                normalized_flux = max(0.0, min(flux_cap, segment_flux))
                if flux_cap > 0.0:
                    width_factor = (normalized_flux / flux_cap) ** max(0.01, width_power)
                else:
                    width_factor = 0.0
                width = min_width + (max_width - min_width) * width_factor
                scaled_width = max(1, int(width * self.zoom))
                try:
                    pygame.draw.line(
                        self.screen,
                        color,
                        (int(screen_points[i][0]), int(screen_points[i][1])),
                        (int(screen_points[i + 1][0]), int(screen_points[i + 1][1])),
                        scaled_width,
                    )
                except Exception:
                    continue

    def update_tile_colors(self, world: World) -> None:
        """Update tile colors based on current map mode.
        
        Args:
            world: World instance to update
        """
        self.current_map_mode = world.current_map_mode

        war_state = {}
        if world.current_map_mode == "war":
            war_state = self._build_polity_war_state(world)

        linguistics_state = None
        if world.current_map_mode == "linguistics":
            linguistics_state = self._build_linguistics_state(world)

        elevation_state = None
        if world.current_map_mode == "elevation":
            elevation_state = self._build_elevation_palette_state(world)
        
        for tile in world.tiles:
            if world.current_map_mode == "regions":
                self._color_tile_by_region(tile, world)
            elif world.current_map_mode == "elevation":
                self._color_tile_by_elevation(tile, world, elevation_state)
            elif world.current_map_mode == "development":
                self._color_tile_by_development(tile, world)
            elif world.current_map_mode == "population":
                self._color_tile_by_population(tile, world)
            elif world.current_map_mode == "control":
                self._color_tile_by_control(tile, world)
            elif world.current_map_mode == "cultures":
                self._color_tile_by_culture(tile, world)
            elif world.current_map_mode == "linguistics":
                self._color_tile_by_linguistics(tile, world, linguistics_state)
            elif world.current_map_mode == "war":
                self._color_tile_by_war_state(tile, world, war_state)
            else:
                # Default to biomes mode
                self._color_tile_by_biome(tile, world)
    
    def _get_biome_color_from_name(self, biome_name: str) -> Tuple[int, int, int]:
        """Get RGB color for a biome from its name.
        
        Args:
            biome_name: Name of the biome
            
        Returns:
            RGB color tuple
        """
        # This should ideally come from config, but for now use hardcoded mapping
        biome_colors = {
            'Ice': '#D0E0F0',
            'Tundra': '#A8A8A8', 
            'Taiga': '#4A7C59',
            'Temperate Forest': '#5A8A5A',
            'Grasslands': '#7A9A4A',
            'Plains': '#7A9A4A',
            'Steppe': '#9A9A5A',
            'Desert': '#C4A67A',
            'Tropical Rainforest': '#2A5A2A',
            'Savanna': '#B8A55A',
            'Hot Desert': '#C4A67A'
        }
        
        hex_color = biome_colors.get(biome_name, '#808080')  # Default to gray
        return self._hex_to_rgb(hex_color)
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple.
        
        Args:
            hex_color: Hex color string (e.g., '#FF0000')
            
        Returns:
            RGB tuple
        """
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _interpolate_color(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int], 
                          factor: float) -> Tuple[int, int, int]:
        """Interpolate between two colors.
        
        Args:
            color1: First color (RGB)
            color2: Second color (RGB)
            factor: Interpolation factor (0.0 = color1, 1.0 = color2)
            
        Returns:
            Interpolated color
        """
        factor = max(0.0, min(1.0, factor))
        r = int(color1[0] + (color2[0] - color1[0]) * factor)
        g = int(color1[1] + (color2[1] - color1[1]) * factor)
        b = int(color1[2] + (color2[2] - color1[2]) * factor)
        return (r, g, b)

    def _evaluate_gradient(self, value: float, stops: List[Tuple[float, Tuple[int, int, int]]]) -> Tuple[int, int, int]:
        """Sample a color gradient defined by ordered stops."""
        if not stops:
            return (128, 128, 128)
        value = max(0.0, min(1.0, value))
        previous_value, previous_color = stops[0]
        if value <= previous_value:
            return previous_color
        for stop_value, stop_color in stops[1:]:
            if value <= stop_value:
                span = max(stop_value - previous_value, 1e-6)
                factor = (value - previous_value) / span
                return self._interpolate_color(previous_color, stop_color, factor)
            previous_value, previous_color = stop_value, stop_color
        return stops[-1][1]

    def _normalize_value(self, value: float, minimum: float, maximum: float) -> float:
        """Clamp and normalize a value within the provided range."""
        if maximum <= minimum:
            return 0.0
        return max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))

    def _resolve_map_mode_setting(self, world: World, mode: str, fallback: str) -> str:
        """Fetch the selected sub-setting for a mode, falling back to default."""
        if hasattr(world, 'get_map_mode_setting'):
            try:
                value = world.get_map_mode_setting(mode)
            except Exception:
                value = None
            if isinstance(value, str) and value:
                return value
        return fallback

    def _get_culture_color(self, world: World, culture_name: Optional[str]) -> Tuple[int, int, int]:
        """Return the configured color for a named culture when available."""
        if not culture_name:
            return (128, 128, 128)
        for culture in getattr(world, 'cultures', []):
            if culture and getattr(culture, 'name', None) == culture_name:
                color = getattr(culture, 'color', None)
                if isinstance(color, tuple) and len(color) == 3:
                    return color
        return (128, 128, 128)
    
    def _apply_altitude_lightness(self, base_color: Tuple[int, int, int], 
                                elevation: float) -> Tuple[int, int, int]:
        """Apply altitude-based lightness variation to a color.
        
        Args:
            base_color: Base RGB color
            elevation: Elevation value (0-1)
            
        Returns:
            Color with altitude lightness applied
        """
        # Lighten at higher elevations
        lightness_factor = 1.0 + (elevation * 0.3)  # Up to 30% lighter
        
        r = min(255, int(base_color[0] * lightness_factor))
        g = min(255, int(base_color[1] * lightness_factor))
        b = min(255, int(base_color[2] * lightness_factor))
        
        return (r, g, b)
    
    def _color_tile_by_biome(self, tile: Tile, world: World) -> None:
        """Color tile based on biome.
        
        Args:
            tile: Tile to color
            world: World instance
        """
        if tile.is_water:
            # Water color based on depth
            depth_ratio = tile.elevation / world.sea_level if world.sea_level > 0 else 0
            deep_color = self._hex_to_rgb('#024B86')
            shallow_color = self._hex_to_rgb('#8CF6FF')
            tile.color = self._interpolate_color(deep_color, shallow_color, depth_ratio)
        else:
            # Land color by biome
            biome_color = self._get_biome_color_from_name(tile.biome)
            tile.color = self._apply_altitude_lightness(biome_color, tile.elevation)
    
    def _color_tile_by_region(self, tile: Tile, world: World) -> None:
        """Color tile based on region.
        
        Args:
            tile: Tile to color
            world: World instance
        """
        if tile.is_water:
            # Water color based on depth (unchanged)
            depth_ratio = tile.elevation / world.sea_level if world.sea_level > 0 else 0
            deep_color = self._hex_to_rgb('#024B86')
            shallow_color = self._hex_to_rgb('#8CF6FF')
            tile.color = self._interpolate_color(deep_color, shallow_color, depth_ratio)
        else:
            # Land color by region
            if tile.region_id >= 0 and tile.region_id < len(world.regions):
                region = world.regions[tile.region_id]
                base_color = region.color
                # Apply subtle altitude lightness effect
                tile.color = self._apply_altitude_lightness(base_color, tile.elevation)
            else:
                # Unassigned region - gray
                tile.color = (128, 128, 128)
    
    def _color_tile_by_development(self, tile: Tile, world: World) -> None:
        """Color tile based on development level.
        
        Args:
            tile: Tile to color
            world: World instance
        """
        if tile.is_water:
            # Water remains blue but with development influence
            base_color = self._hex_to_rgb('#024B86')
            tile.color = base_color
        else:
            # Development gradient from brown (undeveloped) to bright yellow (highly developed)
            # Now that development is uncapped, we need to normalize it
            max_development = max((t.development for t in world.tiles if not t.is_water), default=1)
            if max_development > 0:
                dev_ratio = min(1.0, tile.development / max_development)
            else:
                dev_ratio = 0.0
            
            undeveloped_color = (101, 67, 33)  # Dark brown
            developed_color = (255, 215, 0)     # Gold
            tile.color = self._interpolate_color(undeveloped_color, developed_color, dev_ratio)
    
    def _color_tile_by_population(self, tile: Tile, world: World) -> None:
        """Color tile based on population density.
        
        Args:
            tile: Tile to color
            world: World instance
        """
        if tile.is_water:
            # Water remains blue
            base_color = self._hex_to_rgb('#024B86')
            tile.color = base_color
        else:
            # Population gradient from green (low) to red (high)
            # First, calculate max population for normalization
            max_pop = max((t.population for t in world.tiles if not t.is_water), default=1)
            if max_pop > 0:
                linear_ratio = min(1.0, tile.population / max_pop)
            else:
                linear_ratio = 0.0
            # Dual-curve mapping: gentle lift for low end, stretch for high end
            low_curve_exponent = self.config.get('rendering.population_map.low_curve_exponent', 0.45) if self.config else 0.45
            high_curve_exponent = self.config.get('rendering.population_map.high_curve_exponent', 1.25) if self.config else 1.25
            curve_midpoint = self.config.get('rendering.population_map.curve_midpoint', 0.4) if self.config else 0.4
            use_log_curve = self.config.get('rendering.population_map.use_log_curve', True) if self.config else True
            min_floor = self.config.get('rendering.population_map.low_end_floor', 0.0) if self.config else 0.0
            try:
                low_curve_exponent = max(0.05, float(low_curve_exponent))
            except (TypeError, ValueError):
                low_curve_exponent = 0.45
            try:
                high_curve_exponent = max(0.1, float(high_curve_exponent))
            except (TypeError, ValueError):
                high_curve_exponent = 1.25
            try:
                curve_midpoint = min(0.95, max(0.05, float(curve_midpoint)))
            except (TypeError, ValueError):
                curve_midpoint = 0.4
            try:
                min_floor = max(0.0, min(0.25, float(min_floor)))
            except (TypeError, ValueError):
                min_floor = 0.0

            pop_ratio = linear_ratio
            if use_log_curve and max_pop > 0:
                log_ratio = math.log1p(tile.population) / math.log1p(max_pop)
                pop_ratio = max(pop_ratio, log_ratio)

            if pop_ratio < curve_midpoint:
                pop_ratio = pow(pop_ratio, low_curve_exponent)
            else:
                # Remap upper band to [0,1] before applying high exponent to create separation at the top
                upper_span = max(1e-6, 1.0 - curve_midpoint)
                upper_ratio = (pop_ratio - curve_midpoint) / upper_span
                pop_ratio = curve_midpoint + pow(upper_ratio, high_curve_exponent) * upper_span

            pop_ratio = min(1.0, max(min_floor, pop_ratio))
            
            low_pop_color = (34, 139, 34)   # Forest green
            high_pop_color = (220, 20, 60)  # Crimson
            tile.color = self._interpolate_color(low_pop_color, high_pop_color, pop_ratio)

    def _build_elevation_palette_state(self, world: World) -> Dict[str, float]:
        """Pre-compute min/max elevation bands for the elevation map mode, with diagnostics."""
        land_values = [tile.elevation for tile in world.tiles if not tile.is_water]
        water_values = [tile.elevation for tile in world.tiles if tile.is_water]
        sea_level = getattr(world, 'sea_level', 0.0)
        land_min = min(land_values) if land_values else sea_level
        land_max = max(land_values) if land_values else sea_level + 0.01
        water_min = min(water_values) if water_values else 0.0
        water_max = max(water_values) if water_values else sea_level
        # Diagnostics
        print("[Renderer] Elevation normalization:")
        print(f"  land_min: {land_min:.4f}, land_max: {land_max:.4f}, sea_level: {sea_level:.4f}")
        if land_values:
            print(f"  Sample land elevations: {[float(f'{v:.4f}') for v in land_values[:10]]} ... (total {len(land_values)})")
        if water_values:
            print(f"  Sample water elevations: {[float(f'{v:.4f}') for v in water_values[:10]]} ... (total {len(water_values)})")
        return {
            'land_min': land_min,
            'land_max': land_max,
            'water_min': water_min,
            'water_max': water_max,
            'sea_level': sea_level,
        }

    def _color_tile_by_elevation(
        self,
        tile: Tile,
        world: World,
        state: Optional[Dict[str, float]],
    ) -> None:
        """Color tile using fixed absolute elevation bands."""
        if not state:
            state = self._build_elevation_palette_state(world)
        sea_level = state.get('sea_level', getattr(world, 'sea_level', 0.0))
        if tile.is_water:
            depth_span = max(sea_level - state.get('water_min', 0.0), 1e-5)
            depth_ratio = 0.0
            if tile.elevation < sea_level:
                depth_ratio = (sea_level - tile.elevation) / depth_span
            depth_ratio = max(0.0, min(1.0, depth_ratio))
            shallow_color = (115, 191, 211)
            deep_color = (6, 44, 92)
            tile.color = self._interpolate_color(shallow_color, deep_color, depth_ratio)
            return

        # Use absolute elevation bands for color mapping
        elevation = tile.elevation
        if elevation > 0.9:
            tile.color = (220, 220, 220)  # Mountainous
        elif elevation > 0.8:
            tile.color = (203, 203, 203)  # Alpine
        elif elevation > 0.65:
            tile.color = (176, 154, 110)  # Upland
        elif elevation > 0.4:
            tile.color = (110, 146, 96)   # Hill
        else:
            tile.color = (72, 118, 82)    # Lowland
    
    def _color_tile_by_control(self, tile: Tile, world: World) -> None:
        """Color tile based on political control level.
        
        Args:
            tile: Tile to color
            world: World instance
        """
        view_variant = self._resolve_map_mode_setting(world, 'control', 'control')
        if view_variant == 'tolerance':
            self._color_tile_by_control_tolerance(tile, world)
            return
        if view_variant == 'breakaway':
            self._color_tile_by_breakaway_risk(tile, world)
            return
        if tile.is_water:
            # Water remains blue
            base_color = self._hex_to_rgb('#024B86')
            tile.color = base_color
        else:
            if tile.polity_id >= 0 and tile.polity_id < len(world.polities):
                # Use polity color but vary intensity based on control level
                polity = world.polities[tile.polity_id]
                control_factor = tile.control_level / 100.0
                
                # Blend polity color with gray based on control level
                gray_color = (128, 128, 128)
                tile.color = self._interpolate_color(gray_color, polity.color, control_factor)
            else:
                # Uncontrolled territory - neutral gray
                tile.color = (100, 100, 100)

    def _color_tile_by_control_tolerance(self, tile: Tile, world: World) -> None:
        """Render control map using polity cultural tolerance levels."""
        if tile.is_water:
            tile.color = self._hex_to_rgb('#024B86')
            return
        polity_id = getattr(tile, 'polity_id', -1)
        if polity_id < 0 or polity_id >= len(world.polities):
            tile.color = (90, 90, 90)
            return
        tolerance_value = 0.5
        if hasattr(world, 'get_polity_cultural_tolerance'):
            reported = world.get_polity_cultural_tolerance(polity_id)
            if isinstance(reported, (float, int)):
                tolerance_value = max(0.0, min(1.0, float(reported)))
        low_color = (160, 70, 110)
        high_color = (70, 200, 205)
        base_color = self._interpolate_color(low_color, high_color, tolerance_value)
        control_factor = max(0.2, min(1.0, tile.control_level / 100.0 if tile.control_level is not None else 0.5))
        tile.color = self._interpolate_color((35, 35, 35), base_color, control_factor)

    def _color_tile_by_breakaway_risk(self, tile: Tile, world: World) -> None:
        """Render control map using estimated breakaway risk levels."""
        if tile.is_water:
            tile.color = self._hex_to_rgb('#024B86')
            return
        polity_id = getattr(tile, 'polity_id', -1)
        if polity_id < 0 or polity_id >= len(world.polities):
            tile.color = (80, 80, 80)
            return
        risk = self._calculate_breakaway_risk(tile, world)
        safe_color = (65, 140, 90)
        danger_color = (220, 95, 80)
        tile.color = self._interpolate_color(safe_color, danger_color, risk)

    def _calculate_breakaway_risk(self, tile: Tile, world: World) -> float:
        """Estimate breakaway pressure based on control thresholds."""
        control_level = getattr(tile, 'control_level', 0)
        config = getattr(world, 'config', None)
        base_threshold = 5
        margin = 40
        if config:
            base_threshold = config.get('simulation.polity.breakaway_low_control_threshold', base_threshold)
            margin = config.get('simulation.polity.breakaway_risk_margin', margin)
        burden = 0
        if hasattr(world, 'get_polity_administrative_burden'):
            burden_value = world.get_polity_administrative_burden(tile.polity_id)
            if isinstance(burden_value, int):
                burden = burden_value
        effective_threshold = max(0, min(99, int(round(base_threshold + burden))))
        if control_level <= effective_threshold:
            return 1.0
        margin = max(1.0, float(margin))
        delta = (control_level - effective_threshold) / margin
        return max(0.0, 1.0 - delta)

    def _color_tile_by_culture(self, tile: Tile, world: World) -> None:
        """Color tile based on dominant culture.
        
        Args:
            tile: Tile to color
            world: World instance
        """
        view_variant = self._resolve_map_mode_setting(world, 'cultures', 'mix')
        if view_variant == 'polity_primary':
            self._color_tile_by_polity_primary_culture(tile, world)
            return
        if view_variant == 'majority':
            self._color_tile_by_majority_culture(tile, world)
            return
        if tile.is_water:
            # Water remains blue
            base_color = self._hex_to_rgb('#024B86')
            tile.color = base_color
        else:
            if tile.cultural_makeup and len(tile.cultural_makeup) > 0:
                # Find the dominant culture
                dominant_culture = max(tile.cultural_makeup.items(), key=lambda x: x[1])
                culture_name, percentage = dominant_culture
                
                # Find the culture object to get its color
                culture_color = (128, 128, 128)  # Default gray
                for culture in world.cultures:
                    if culture.name == culture_name:
                        culture_color = culture.color
                        break
                
                # Blend culture color with gray based on cultural dominance
                gray_color = (180, 180, 180)
                tile.color = self._interpolate_color(gray_color, culture_color, percentage)
            else:
                # No cultural makeup - neutral gray
                tile.color = (120, 120, 120)

    def _color_tile_by_polity_primary_culture(self, tile: Tile, world: World) -> None:
        """Render culture view using each polity's primary culture."""
        if tile.is_water:
            tile.color = self._hex_to_rgb('#024B86')
            return
        polity_id = getattr(tile, 'polity_id', -1)
        if polity_id < 0 or polity_id >= len(world.polities):
            tile.color = (110, 110, 110)
            return
        polity = world.polities[polity_id]
        primary_culture = getattr(polity, 'primary_culture', None) if polity else None
        culture_color = self._get_culture_color(world, primary_culture)
        alignment = 0.0
        if primary_culture and tile.cultural_makeup:
            alignment = float(tile.cultural_makeup.get(primary_culture, 0.0))
        alignment = max(0.0, min(1.0, alignment))
        blend_factor = max(0.25, min(1.0, 0.3 + alignment))
        tile.color = self._interpolate_color((40, 40, 40), culture_color, blend_factor)

    def _color_tile_by_majority_culture(self, tile: Tile, world: World) -> None:
        """Render culture view by the tile's majority culture."""
        if tile.is_water:
            tile.color = self._hex_to_rgb('#024B86')
            return
        if not tile.cultural_makeup:
            tile.color = (120, 120, 120)
            return
        majority_culture, share = max(tile.cultural_makeup.items(), key=lambda item: item[1])
        culture_color = self._get_culture_color(world, majority_culture)
        share_value = max(0.0, min(1.0, float(share)))
        blend_factor = max(0.35, min(1.0, share_value))
        tile.color = self._interpolate_color((35, 35, 35), culture_color, blend_factor)

    def _build_polity_war_state(self, world: World) -> Dict[int, Dict[str, Any]]:
        """Collect per-polity war participation and exhaustion metrics."""
        state: Dict[int, Dict[str, Any]] = {}
        for polity in world.polities:
            if polity is None:
                continue
            state[polity.id] = {
                'at_war': False,
                'wars': 0,
                'max_exhaustion': 0.0,
            }
        for relationship in getattr(world, 'relationships', []):
            if relationship is None or relationship.status != "war":
                continue
            for polity_id in (relationship.polity_a, relationship.polity_b):
                entry = state.setdefault(polity_id, {
                    'at_war': False,
                    'wars': 0,
                    'max_exhaustion': 0.0,
                })
                entry['at_war'] = True
                entry['wars'] += 1
                exhaustion_map = getattr(relationship, 'war_exhaustion', {}) or {}
                entry['max_exhaustion'] = max(
                    entry['max_exhaustion'],
                    exhaustion_map.get(polity_id, 0.0)
                )
        return state

    def _color_tile_by_war_state(
        self,
        tile: Tile,
        world: World,
        war_state: Dict[int, Dict[str, Any]]
    ) -> None:
        """Color tile to reflect war participation and exhaustion."""
        if tile.is_water:
            depth_ratio = tile.elevation / world.sea_level if world.sea_level > 0 else 0
            deep_color = self._hex_to_rgb('#024B86')
            shallow_color = self._hex_to_rgb('#3CA0F0')
            tile.color = self._interpolate_color(deep_color, shallow_color, depth_ratio)
            return

        polity_id = getattr(tile, 'polity_id', -1)
        if polity_id < 0 or polity_id >= len(world.polities):
            tile.color = (70, 70, 70)
            return

        polity = world.polities[polity_id]
        base_color = getattr(polity, 'color', (110, 110, 110)) if polity else (110, 110, 110)
        state = war_state.get(polity_id)
        if state and state.get('at_war'):
            exhaustion_ratio = min(1.0, state.get('max_exhaustion', 0.0) / 100.0)
            wars = max(1, state.get('wars', 1))
            overlay = self._interpolate_color((70, 200, 120), (255, 70, 70), exhaustion_ratio)
            blend_factor = min(1.0, 0.45 + 0.15 * (wars - 1))
            tile.color = self._interpolate_color(base_color, overlay, blend_factor)
        else:
            tile.color = self._interpolate_color((55, 55, 65), base_color, 0.35)

        occupier_id = getattr(tile, 'occupied_by_polity_id', -1)
        if occupier_id is not None and occupier_id >= 0 and occupier_id < len(world.polities):
            occupier = world.polities[occupier_id]
            overlay_alpha = self.config.get('simulation.war.occupation_overlay_alpha', 0.5)
            tint_color = getattr(occupier, 'color', (255, 255, 255))
            tile.color = self._interpolate_color(tile.color, tint_color, max(0.0, min(1.0, overlay_alpha)))

    def _build_linguistics_state(self, world: World) -> Dict[str, Any]:
        """Collect reusable data for linguistics map rendering."""
        culture_lookup: Dict[str, Any] = {}
        max_depth = 0
        for culture in getattr(world, 'cultures', []):
            if not culture or not getattr(culture, 'name', None):
                continue
            culture_lookup[culture.name] = culture
            depth = getattr(culture, 'language_time_depth', 0) or 0
            if depth > max_depth:
                max_depth = depth
        if max_depth <= 0:
            max_depth = 1
        return {
            'culture_lookup': culture_lookup,
            'parent_colors': {},
            'max_depth': max_depth,
        }

    def _color_tile_by_linguistics(
        self,
        tile: Tile,
        world: World,
        linguistics_state: Optional[Dict[str, Any]],
    ) -> None:
        """Render tile colors emphasizing linguistic ancestry and depth."""
        if tile.is_water:
            depth_ratio = tile.elevation / world.sea_level if world.sea_level > 0 else 0
            deep_color = self._hex_to_rgb('#01344F')
            shallow_color = self._hex_to_rgb('#05A4C8')
            tile.color = self._interpolate_color(deep_color, shallow_color, depth_ratio)
            return

        culture_name, share = self._get_tile_primary_culture(tile, world)
        base_color = self._get_culture_color(world, culture_name)
        if base_color is None:
            base_color = self._get_polity_color(world, getattr(tile, 'polity_id', -1))
        if base_color is None:
            base_color = (120, 120, 130)

        share_factor = max(0.0, min(1.0, share))
        base_blend = self._interpolate_color((35, 35, 45), base_color, 0.5 + share_factor * 0.4)

        culture = None
        if linguistics_state and culture_name:
            culture = linguistics_state.get('culture_lookup', {}).get(culture_name)
        depth = max(0, getattr(culture, 'language_time_depth', 0) if culture else 0)
        max_depth = max(1, (linguistics_state or {}).get('max_depth', 1))
        depth_ratio = min(1.0, depth / max_depth) if max_depth else 0.0

        parent_name = getattr(culture, 'language_parent', None) if culture else None
        parent_color = self._resolve_language_parent_color(parent_name, linguistics_state)
        if parent_color is None and culture_name:
            parent_color = self._resolve_language_parent_color(culture_name, linguistics_state)
        if parent_color:
            base_blend = self._interpolate_color(parent_color, base_blend, 0.6 + depth_ratio * 0.3)

        tint_anchor = (245, 235, 210) if depth_ratio > 0.5 else (85, 125, 180)
        tint_strength = 0.2 + depth_ratio * 0.5
        tile.color = self._interpolate_color(base_blend, tint_anchor, tint_strength)

    def _get_tile_primary_culture(self, tile: Tile, world: World) -> Tuple[Optional[str], float]:
        """Return the tile's dominant culture and its share."""
        makeup = getattr(tile, 'cultural_makeup', None)
        if isinstance(makeup, dict) and makeup:
            culture_name, share = max(makeup.items(), key=lambda item: item[1])
            try:
                share_value = float(share)
            except (TypeError, ValueError):
                share_value = 0.0
            return culture_name, max(0.0, min(1.0, share_value))
        polity_id = getattr(tile, 'polity_id', -1)
        if polity_id is not None and 0 <= polity_id < len(world.polities):
            polity = world.polities[polity_id]
            if polity and getattr(polity, 'primary_culture', None):
                return polity.primary_culture, 0.0
        return None, 0.0

    def _get_polity_color(self, world: World, polity_id: Optional[int]) -> Optional[Tuple[int, int, int]]:
        """Return the color assigned to the given polity when available."""
        if polity_id is None or polity_id < 0 or polity_id >= len(world.polities):
            return None
        polity = world.polities[polity_id]
        if polity and getattr(polity, 'color', None):
            return polity.color
        return None

    def _resolve_language_parent_color(
        self,
        parent_name: Optional[str],
        linguistics_state: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[int, int, int]]:
        """Resolve a stable color for a language parent or culture name."""
        if not parent_name:
            return None
        state = linguistics_state or {'parent_colors': {}, 'culture_lookup': {}}
        cache = state.setdefault('parent_colors', {})
        if parent_name in cache:
            return cache[parent_name]
        culture_lookup = state.get('culture_lookup', {})
        culture = culture_lookup.get(parent_name)
        color = getattr(culture, 'color', None) if culture else None
        if color is None:
            color = self._color_from_string(parent_name)
        cache[parent_name] = color
        return color

    def _color_from_string(self, label: str) -> Tuple[int, int, int]:
        """Generate a deterministic pastel color from a string label."""
        digest = hashlib.sha1(label.encode('utf-8')).digest()
        return (
            80 + digest[0] % 120,
            80 + digest[1] % 120,
            80 + digest[2] % 120,
        )

    def render_tile(self, tile: Tile) -> None:
        """Render a single tile.
        
        Args:
            tile: Tile to render
        """
        if len(tile.vertices) < 3:
            return
        
        # Transform vertices using camera
        transformed_vertices = []
        for x, y in tile.vertices:
            screen_x, screen_y = self.world_to_screen(x, y)
            transformed_vertices.append((screen_x, screen_y))
        
        # Check if tile is visible on screen (simple bounds check)
        if not self._is_tile_visible(transformed_vertices):
            return
        
        if len(transformed_vertices) >= 3:
            try:
                pygame.draw.polygon(self.screen, tile.color, transformed_vertices)
                
                # Optional: draw tile outline for debugging
                if self.config.get('rendering.show_tile_borders', False):
                    pygame.draw.polygon(self.screen, (255, 255, 255), transformed_vertices, 1)
            except Exception:
                # Skip tiles that cause rendering issues
                pass
    
    def render_tile_info(self, tile: Tile, position: Tuple[int, int]) -> None:
        """Render tile information at given position.
        
        Args:
            tile: Tile to show info for
            position: Screen position to render info
        """
        font = pygame.font.Font(None, 24)
        
        info_lines = [
            f"Elevation: {tile.elevation:.2f}",
            f"Type: {'Water' if tile.is_water else 'Land'}",
            f"Center: ({tile.center[0]:.0f}, {tile.center[1]:.0f})"
        ]
        
        y_offset = 0
        for line in info_lines:
            text = font.render(line, True, (255, 255, 255))
            text_rect = text.get_rect()
            
            # Draw background for readability
            bg_rect = pygame.Rect(position[0], position[1] + y_offset, 
                                text_rect.width + 10, text_rect.height + 4)
            pygame.draw.rect(self.screen, (0, 0, 0, 128), bg_rect)
            
            # Draw text
            self.screen.blit(text, (position[0] + 5, position[1] + y_offset + 2))
            y_offset += text_rect.height + 4
    
    def render_statistics(self, world: World) -> None:
        """Render world statistics on screen.
        
        Args:
            world: World to show statistics for
        """
        stats = world.calculate_statistics()
        font = pygame.font.Font(None, 24)
        
        stat_lines = [
            f"Tiles: {stats['total_tiles']}",
            f"Water: {stats['water_percentage']:.1f}%",
            f"Elevation: {stats['min_elevation']:.2f} - {stats['max_elevation']:.2f}",
            f"Sea Level: {stats['sea_level']:.2f}",
            f"Temperature: {stats.get('avg_temperature', 0):.2f} avg" if 'avg_temperature' in stats else "Temperature: N/A",
            f"Rainfall: {stats.get('avg_rainfall', 0):.2f} avg" if 'avg_rainfall' in stats else "Rainfall: N/A",
            f"Camera: ({self.camera_x:.0f}, {self.camera_y:.0f})",
            f"Zoom: {self.zoom:.2f}x"
        ]
        
        y_offset = 10
        for line in stat_lines:
            text = font.render(line, True, (255, 255, 255))
            
            # Draw background
            bg_rect = pygame.Rect(10, y_offset, text.get_width() + 10, text.get_height() + 4)
            pygame.draw.rect(self.screen, (0, 0, 0, 128), bg_rect)
            
            # Draw text
            self.screen.blit(text, (15, y_offset + 2))
            y_offset += text.get_height() + 8
    
    def _clamp_vertices(self, vertices: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Clamp vertices to screen bounds.
        
        Args:
            vertices: List of vertex coordinates
            
        Returns:
            Clamped vertices
        """
        clamped = []
        for x, y in vertices:
            x = max(0, min(self.width - 1, x))
            y = max(0, min(self.height - 1, y))
            clamped.append((x, y))
        return clamped
    
    def _is_tile_visible(self, vertices: List[Tuple[int, int]]) -> bool:
        """Check if tile is visible on screen.
        
        Args:
            vertices: Screen coordinates of tile vertices
            
        Returns:
            True if tile might be visible
        """
        if not vertices:
            return False
        
        # Simple bounding box check
        min_x = min(x for x, y in vertices)
        max_x = max(x for x, y in vertices)
        min_y = min(y for x, y in vertices)
        max_y = max(y for x, y in vertices)
        
        # Check if bounding box intersects screen
        return (max_x >= 0 and min_x <= self.width and 
                max_y >= 0 and min_y <= self.height)
    
    def get_tile_at_position(self, world: World, position: Tuple[int, int]) -> Optional[Tile]:
        """Get tile at screen position.
        
        Args:
            world: World instance
            position: Screen position
            
        Returns:
            Tile at position, or None if not found
        """
        world_x, world_y = self.screen_to_world(position[0], position[1])
        return world.get_tile_by_position(world_x, world_y)
    
    def pan_camera(self, dx: float, dy: float) -> None:
        """Pan the camera by given amounts.
        
        Args:
            dx: Horizontal pan amount
            dy: Vertical pan amount
        """
        self.camera_x += dx * self.pan_speed / self.zoom
        self.camera_y += dy * self.pan_speed / self.zoom
    
    def zoom_camera(self, zoom_delta: float, mouse_pos: Tuple[int, int]) -> None:
        """Zoom the camera around the mouse position.
        
        Args:
            zoom_delta: Amount to zoom (positive = zoom in)
            mouse_pos: Mouse position to zoom around
        """
        old_zoom = self.zoom
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom + zoom_delta * self.zoom_speed))
        
        # Adjust camera position to zoom around mouse
        if self.zoom != old_zoom:
            zoom_factor = self.zoom / old_zoom
            mouse_x, mouse_y = mouse_pos
            
            # Convert mouse position to world coordinates
            world_x = (mouse_x - self.width / 2) / old_zoom + self.camera_x
            world_y = (mouse_y - self.height / 2) / old_zoom + self.camera_y
            
            # Adjust camera to keep mouse position fixed in world space
            self.camera_x = world_x - (mouse_x - self.width / 2) / self.zoom
            self.camera_y = world_y - (mouse_y - self.height / 2) / self.zoom
    
    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates.
        
        Args:
            world_x: World X coordinate
            world_y: World Y coordinate
            
        Returns:
            Screen coordinates
        """
        screen_x = (world_x - self.camera_x) * self.zoom + self.width / 2
        screen_y = (world_y - self.camera_y) * self.zoom + self.height / 2
        return (int(screen_x), int(screen_y))
    
    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates.
        
        Args:
            screen_x: Screen X coordinate
            screen_y: Screen Y coordinate
            
        Returns:
            World coordinates
        """
        world_x = (screen_x - self.width / 2) / self.zoom + self.camera_x
        world_y = (screen_y - self.height / 2) / self.zoom + self.camera_y
        return (world_x, world_y)
    
    def render_map_mode_status(self, world: World) -> None:
        """Render current map mode status in top-right corner.
        
        Args:
            world: World instance
        """
        font = pygame.font.Font(None, 24)
        
        # Map mode text
        mode_text = f"Map Mode: {world.current_map_mode.title()}"
        text_surface = font.render(mode_text, True, (255, 255, 255))
        
        # Background for readability
        text_rect = text_surface.get_rect()
        bg_rect = pygame.Rect(
            self.width - text_rect.width - 15,
            10,
            text_rect.width + 10,
            text_rect.height + 5
        )
        pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect)
        
        # Render text
        self.screen.blit(text_surface, (self.width - text_rect.width - 10, 12))
        
        # Additional info for regions mode
        if world.current_map_mode == "regions":
            region_count = len(world.regions)
            info_text = f"Regions: {region_count}"
            info_surface = font.render(info_text, True, (200, 200, 200))
            
            info_rect = info_surface.get_rect()
            info_bg_rect = pygame.Rect(
                self.width - info_rect.width - 15,
                40,
                info_rect.width + 10,
                info_rect.height + 5
            )
            pygame.draw.rect(self.screen, (0, 0, 0, 180), info_bg_rect)
            self.screen.blit(info_surface, (self.width - info_rect.width - 10, 42))
        
        # Show polity info on all modes
        if world.polities:
            active_polities = sum(
                1
                for polity in world.polities
                if polity and getattr(polity, 'tile_indices', None)
            )
            polity_text = f"Polities: {active_polities} active"
            polity_surface = font.render(polity_text, True, (200, 200, 200))
            
            y_offset = 70 if world.current_map_mode == "regions" else 40
            polity_rect = polity_surface.get_rect()
            polity_bg_rect = pygame.Rect(
                self.width - polity_rect.width - 15,
                y_offset,
                polity_rect.width + 10,
                polity_rect.height + 5
            )
            pygame.draw.rect(self.screen, (0, 0, 0, 180), polity_bg_rect)
            self.screen.blit(polity_surface, (self.width - polity_rect.width - 10, y_offset + 2))
    
    def render_polity_borders(self, world: World) -> None:
        """Render borders around polity territories.
        
        Args:
            world: World instance
        """
        if not world.polities:
            return
        
        def world_to_screen_float(world_x: float, world_y: float) -> Tuple[float, float]:
            return (
                (world_x - self.camera_x) * self.zoom + self.width / 2,
                (world_y - self.camera_y) * self.zoom + self.height / 2
            )

        border_inset = 2.0  # pixels to shift inward toward each polity
        # For each polity, find external edges and draw borders
        for polity in world.polities:
            if not polity.tile_indices:
                continue
            
            border_color = polity.color
            border_width = 3
            
            # Find all external edges for this polity
            external_edges = self._find_polity_external_edges(world, polity)
            
            # Draw each external edge
            for edge_start, edge_end, tile_center in external_edges:
                start_screen = world_to_screen_float(edge_start[0], edge_start[1])
                end_screen = world_to_screen_float(edge_end[0], edge_end[1])
                center_screen = world_to_screen_float(tile_center[0], tile_center[1])

                edge_vec = (end_screen[0] - start_screen[0], end_screen[1] - start_screen[1])
                length = math.hypot(edge_vec[0], edge_vec[1])
                if length == 0:
                    continue

                perp = (-edge_vec[1] / length, edge_vec[0] / length)
                midpoint = ((start_screen[0] + end_screen[0]) * 0.5,
                            (start_screen[1] + end_screen[1]) * 0.5)
                center_vec = (center_screen[0] - midpoint[0], center_screen[1] - midpoint[1])
                if center_vec[0] * perp[0] + center_vec[1] * perp[1] < 0:
                    perp = (-perp[0], -perp[1])

                offset_vec = (perp[0] * border_inset, perp[1] * border_inset)
                start_offset = (start_screen[0] + offset_vec[0], start_screen[1] + offset_vec[1])
                end_offset = (end_screen[0] + offset_vec[0], end_screen[1] + offset_vec[1])

                start_pos = (int(round(start_offset[0])), int(round(start_offset[1])))
                end_pos = (int(round(end_offset[0])), int(round(end_offset[1])))

                # Only draw if edge is visible on screen
                if (self._is_point_on_screen(start_pos) or 
                    self._is_point_on_screen(end_pos)):
                    pygame.draw.line(self.screen, border_color, start_pos, end_pos, border_width)

    def _find_polity_external_edges(self, world: World, polity) -> List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        """Find all external edges of a polity's territory.
        
        Args:
            world: World instance
            polity: Polity to find edges for
            
        Returns:
            List of edge tuples containing start, end, and tile center coordinates
        """
        external_edges = []
        polity_tile_set = set(polity.tile_indices)
        
        for tile_idx in polity.tile_indices:
            tile = world.tiles[tile_idx]
            vertices = tile.vertices
            tile_center = tile.center
            
            # Check each edge of this tile's polygon
            for i in range(len(vertices)):
                edge_start = vertices[i]
                edge_end = vertices[(i + 1) % len(vertices)]
                
                # Check if this edge is shared with another tile of the same polity
                is_external = True
                
                # Check all neighbor tiles
                for neighbor_idx in tile.neighbors:
                    if neighbor_idx in polity_tile_set:
                        neighbor_tile = world.tiles[neighbor_idx]
                        neighbor_vertices = neighbor_tile.vertices
                        
                        # Check if this edge is shared with the neighbor
                        if self._edges_match(edge_start, edge_end, neighbor_vertices):
                            is_external = False
                            break
                
                if is_external:
                    external_edges.append((edge_start, edge_end, tile_center))
        
        return external_edges
    
    def _edges_match(self, edge_start: Tuple[float, float], edge_end: Tuple[float, float], 
                    vertices: List[Tuple[float, float]]) -> bool:
        """Check if an edge matches any edge in a list of vertices.
        
        Args:
            edge_start: Start point of edge to check
            edge_end: End point of edge to check
            vertices: List of vertices to check against
            
        Returns:
            True if edge is found in vertices
        """
        tolerance = 1.0  # Allow for small floating point differences
        
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            
            # Check both directions (edge could be reversed)
            if ((abs(edge_start[0] - v1[0]) < tolerance and abs(edge_start[1] - v1[1]) < tolerance and
                 abs(edge_end[0] - v2[0]) < tolerance and abs(edge_end[1] - v2[1]) < tolerance) or
                (abs(edge_start[0] - v2[0]) < tolerance and abs(edge_start[1] - v2[1]) < tolerance and
                 abs(edge_end[0] - v1[0]) < tolerance and abs(edge_end[1] - v1[1]) < tolerance)):
                return True
        
        return False
    
    def _is_point_on_screen(self, point: Tuple[int, int]) -> bool:
        """Check if a point is visible on screen.
        
        Args:
            point: Screen coordinates
            
        Returns:
            True if point is on screen
        """
        return (0 <= point[0] <= self.width and 0 <= point[1] <= self.height)
    
    def render_polity_text(self, world: World) -> None:
        """Render polity names positioned intelligently within their territories.
        
        Args:
            world: World instance
        """
        if not world.polities:
            return
        
        # Don't show polity names in region view mode
        if world.current_map_mode == "regions":
            return
        
        # Check if text rendering is enabled
        polity_text_cfg = self.config.get('rendering.polity_text', {}) if self.config else {}
        show_names = polity_text_cfg.get('show_names', True)
        show_gloss = polity_text_cfg.get('show_gloss', False)
        gloss_font_scale = max(0.1, float(polity_text_cfg.get('gloss_font_scale', 0.7)))
        gloss_offset = float(polity_text_cfg.get('gloss_offset', 0.65))
        if not show_names:
            return
        
        font_size = polity_text_cfg.get('font_size', 24)
        text_color = tuple(polity_text_cfg.get('text_color', [255, 255, 255]))
        outline_color = tuple(polity_text_cfg.get('outline_color', [0, 0, 0]))
        hide_zoom_threshold = polity_text_cfg.get('hide_zoom_threshold', 3.0)
        if self.zoom >= hide_zoom_threshold:
            return

        zoom_scale = max(0.5, min(1.5, self.zoom))
        max_polity_tiles = max(
            (len(polity.tile_indices) for polity in world.polities if polity and polity.tile_indices),
            default=1,
        )
        min_font_size = polity_text_cfg.get('min_font_size', 10)

        min_visible_tiles = max(0, int(polity_text_cfg.get('min_visible_tiles', 1)))
        min_visible_ratio = max(0.0, float(polity_text_cfg.get('min_visible_ratio', 0.0)))
        zoom_visibility_ratio = max(0.0, float(polity_text_cfg.get('zoom_visibility_ratio', 0.05)))
        zoom_visibility_power = max(0.0, float(polity_text_cfg.get('zoom_visibility_power', 1.0)))
        reference_zoom = max(0.05, float(polity_text_cfg.get('zoom_visibility_reference_zoom', 1.0)))

        safe_zoom = max(0.01, self.zoom)
        zoom_out_ratio = max(0.0, (reference_zoom / safe_zoom) - 1.0)
        zoom_ratio_bonus = 0.0
        if zoom_out_ratio > 0.0 and zoom_visibility_ratio > 0.0:
            exponent = zoom_visibility_power if zoom_visibility_power > 0 else 1.0
            zoom_ratio_bonus = zoom_visibility_ratio * (zoom_out_ratio ** exponent)
        tile_ratio_threshold = min_visible_ratio + zoom_ratio_bonus
        ratio_tile_requirement = 0
        if max_polity_tiles > 0:
            ratio_tile_requirement = int(math.ceil(tile_ratio_threshold * max_polity_tiles))
        visibility_tile_threshold = max(min_visible_tiles, ratio_tile_requirement)
        
        for polity in world.polities:
            if polity is None or not polity.tile_indices:
                continue
            if len(polity.tile_indices) < visibility_tile_threshold:
                continue
            polity_size_ratio = len(polity.tile_indices) / max_polity_tiles if max_polity_tiles > 0 else 0
            size_factor = 0.6 + 0.6 * math.sqrt(max(0.0, polity_size_ratio))
            dynamic_font_size = max(min_font_size, int(font_size * size_factor * zoom_scale))
            dynamic_font = pygame.font.Font(None, dynamic_font_size)
            
            # Find the geometric center of the polity territory
            text_position = self._find_polity_text_position(world, polity)
            
            if text_position:
                screen_pos = self.world_to_screen(text_position[0], text_position[1])
                
                # Only render if position is on screen
                if self._is_point_on_screen(screen_pos):
                    # Create text surface
                    text_surface = dynamic_font.render(polity.name, True, text_color)
                    text_rect = text_surface.get_rect(center=screen_pos)
                    
                    # Draw outline by rendering black text at offset positions
                    outline_surface = dynamic_font.render(polity.name, True, outline_color)
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:  # Skip center position
                                outline_rect = outline_surface.get_rect(
                                    center=(screen_pos[0] + dx, screen_pos[1] + dy)
                                )
                                self.screen.blit(outline_surface, outline_rect)
                    
                    # Draw main text on top
                    self.screen.blit(text_surface, text_rect)

                    if show_gloss:
                        gloss_text = getattr(polity, 'name_gloss', None)
                        if gloss_text:
                            gloss_text = f"\"{gloss_text}\""
                            gloss_font_size = max(min_font_size, int(dynamic_font_size * gloss_font_scale))
                            gloss_font = pygame.font.Font(None, gloss_font_size)
                            gloss_surface = gloss_font.render(gloss_text, True, text_color)
                            gloss_rect = gloss_surface.get_rect(center=(
                                screen_pos[0],
                                screen_pos[1] + int(dynamic_font_size * gloss_offset)
                            ))
                            gloss_outline = gloss_font.render(gloss_text, True, outline_color)
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    if dx == 0 and dy == 0:
                                        continue
                                    outline_rect = gloss_outline.get_rect(
                                        center=(gloss_rect.centerx + dx, gloss_rect.centery + dy)
                                    )
                                    self.screen.blit(gloss_outline, outline_rect)
                            self.screen.blit(gloss_surface, gloss_rect)

    def render_region_labels(self, world: World) -> None:
        """Render overlay labels for regions while in region view mode."""
        if world.current_map_mode != "regions":
            return
        label_config = self.config.get('rendering.region_labels', {})
        if not label_config.get('enabled', True):
            return
        min_zoom = label_config.get('min_zoom', 0.4)
        max_zoom = label_config.get('max_zoom')
        if self.zoom < min_zoom:
            return
        if isinstance(max_zoom, (int, float)) and max_zoom > 0 and self.zoom > max_zoom:
            return

        labels = self._build_region_label_data(world)
        if not labels:
            return

        text_color = tuple(label_config.get('text_color', [235, 235, 235]))
        outline_color = tuple(label_config.get('outline_color', [0, 0, 0]))
        base_font_size = self.config.get('rendering.polity_text.font_size', 24)  # Use same size as polity text
        min_font_size = self.config.get('rendering.polity_text.min_font_size', 10)  # Use same min size as polity text
        max_tiles = max(label['tile_count'] for label in labels)

        for label in labels:
            tile_ratio = 0.0
            if max_tiles > 0:
                tile_ratio = label['tile_count'] / max_tiles
            size_factor = 0.7 + 0.3 * math.sqrt(max(0.0, tile_ratio))
            font_size = max(min_font_size, int(base_font_size * size_factor))
            font = pygame.font.Font(None, font_size)

            screen_pos = self.world_to_screen(label['position'][0], label['position'][1])
            if not self._is_point_on_screen(screen_pos):
                continue

            text_surface = font.render(label['name'], True, text_color)
            text_rect = text_surface.get_rect(center=screen_pos)
            outline_surface = font.render(label['name'], True, outline_color)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    outline_rect = outline_surface.get_rect(
                        center=(screen_pos[0] + dx, screen_pos[1] + dy)
                    )
                    self.screen.blit(outline_surface, outline_rect)
            self.screen.blit(text_surface, text_rect)

    def _build_region_label_data(self, world: World) -> List[Dict[str, Any]]:
        labels: List[Dict[str, Any]] = []
        if not world or not world.regions:
            return labels
        total_tiles = len(world.tiles)

        for region in world.regions:
            if not region.tile_indices:
                continue
            owner_counts: Dict[int, int] = {}
            for tile_idx in region.tile_indices:
                if tile_idx < 0 or tile_idx >= total_tiles:
                    continue
                tile = world.tiles[tile_idx]
                polity_id = getattr(tile, 'polity_id', -1)
                if polity_id >= 0:
                    owner_counts[polity_id] = owner_counts.get(polity_id, 0) + 1
            if not owner_counts:
                continue  # Skip unoccupied regions

            dominant_polity_id, _ = max(owner_counts.items(), key=lambda item: item[1])
            if dominant_polity_id < 0 or dominant_polity_id >= len(world.polities):
                continue
            polity = world.polities[dominant_polity_id]
            if polity is None or not getattr(polity, 'is_active', True):
                continue
            culture_name = getattr(polity, 'primary_culture', None)
            if not culture_name:
                continue

            label_token: Optional[str] = None
            label_token = getattr(region, 'name', None)
            if not label_token and hasattr(world, 'ensure_region_language_name'):
                label_token = world.ensure_region_language_name(region.id, culture_name)
            if not label_token:
                label_token = f"Region {region.id}"
            cleaned_label = label_token.strip()
            normalized = cleaned_label.lower()
            if normalized.endswith(" region") or normalized.startswith("region "):
                label_text = cleaned_label
            else:
                label_text = f"{cleaned_label} Region"

            position = self._get_region_label_position(world, region)
            if not position:
                continue

            labels.append({
                'region_id': region.id,
                'name': label_text,
                'position': position,
                'tile_count': len(region.tile_indices),
            })
        return labels

    def _get_region_label_position(self, world: World, region: Region) -> Optional[Tuple[float, float]]:
        """Determine a stable label anchor for a region."""
        center_idx = getattr(region, 'center_tile_index', -1)
        if 0 <= center_idx < len(world.tiles):
            center_tile = world.tiles[center_idx]
            return center_tile.center
        total_x = 0.0
        total_y = 0.0
        count = 0
        for tile_idx in region.tile_indices:
            if 0 <= tile_idx < len(world.tiles):
                center = world.tiles[tile_idx].center
                total_x += center[0]
                total_y += center[1]
                count += 1
        if count > 0:
            return (total_x / count, total_y / count)
        return None
    
    def render_population_centers(self, world: World) -> None:
        """Render population center dots and names.
        
        Args:
            world: World instance
        """
        if not world.population_centers:
            return
        
        # Check if rendering is enabled
        show_names = self.config.get('rendering.population_centers.show_names', True)
        min_zoom_for_names = self.config.get('rendering.population_centers.min_zoom_for_names', 1.0)
        dot_color = tuple(self.config.get('rendering.population_centers.dot_color', [255, 215, 0]))
        dot_size = self.config.get('rendering.population_centers.dot_size', 4)
        font_size = self.config.get('rendering.population_centers.font_size', 12)
        text_color = tuple(self.config.get('rendering.population_centers.text_color', [255, 255, 255]))
        outline_color = tuple(self.config.get('rendering.population_centers.outline_color', [0, 0, 0]))
        capital_tiles = self._collect_capital_tiles(world)
        capital_color = tuple(self.config.get('rendering.population_centers.capital_color', list(dot_color)))
        capital_size = int(self.config.get('rendering.population_centers.capital_size', max(6, int(dot_size) + 2)))
        capital_outline_color = tuple(self.config.get('rendering.population_centers.capital_outline_color', list(outline_color)))
        capital_outline_width = int(self.config.get('rendering.population_centers.capital_outline_width', 1))
        star_points = max(4, int(self.config.get('rendering.population_centers.capital_star_points', 5)))
        star_inner_ratio = float(self.config.get('rendering.population_centers.capital_star_inner_ratio', 0.5))
        
        font = pygame.font.Font(None, font_size)
        rendered_capital_tiles: set[int] = set()
        
        for center in world.population_centers:
            tile_idx = center.tile_index
            if tile_idx >= len(world.tiles):
                continue
                
            tile = world.tiles[tile_idx]
            screen_pos = self.world_to_screen(tile.center[0], tile.center[1])
            
            # Only render if position is on screen
            if self._is_point_on_screen(screen_pos):
                # Draw capital stars to distinguish them from normal settlements
                if tile_idx in capital_tiles:
                    if tile_idx in rendered_capital_tiles:
                        continue
                    rendered_capital_tiles.add(tile_idx)
                    self._draw_star(
                        screen_pos,
                        radius=capital_size,
                        color=capital_color,
                        outline_color=capital_outline_color,
                        outline_width=capital_outline_width,
                        points=star_points,
                        inner_ratio=star_inner_ratio,
                    )
                else:
                    pygame.draw.circle(self.screen, dot_color, screen_pos, dot_size)

                # Draw name if enabled and zoomed in enough
                if show_names and self.zoom >= min_zoom_for_names:
                    text_surface = font.render(center.name, True, text_color)
                    text_pos = (screen_pos[0], screen_pos[1] + dot_size + 8)
                    text_rect = text_surface.get_rect(center=text_pos)

                    outline_surface = font.render(center.name, True, outline_color)
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:  # Skip center position
                                outline_rect = outline_surface.get_rect(
                                    center=(text_pos[0] + dx, text_pos[1] + dy)
                                )
                                self.screen.blit(outline_surface, outline_rect)

                    self.screen.blit(text_surface, text_rect)

    def _collect_capital_tiles(self, world: World) -> set[int]:
        """Return tile indices for active polities' capitals."""
        capitals: set[int] = set()
        if not hasattr(world, 'polities'):
            return capitals
        total_tiles = len(getattr(world, 'tiles', []))
        for polity in world.polities:
            if polity is None or not getattr(polity, 'is_active', True):
                continue
            tile_idx = getattr(polity, 'capital_tile_index', -1)
            if isinstance(tile_idx, int) and 0 <= tile_idx < total_tiles:
                capitals.add(tile_idx)
        return capitals

    def _draw_star(
        self,
        center: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int],
        outline_color: Tuple[int, int, int],
        outline_width: int,
        points: int = 5,
        inner_ratio: float = 0.5,
    ) -> None:
        """Draw a simple star polygon centered at the given point."""
        safe_points = max(2, int(points))
        outer_r = max(1, int(radius))
        inner_r = max(1, int(outer_r * max(0.05, min(0.95, inner_ratio))))
        angle_step = math.pi / safe_points
        start_angle = -math.pi / 2
        vertices: List[Tuple[int, int]] = []
        for i in range(safe_points * 2):
            r = outer_r if i % 2 == 0 else inner_r
            angle = start_angle + i * angle_step
            vertices.append(
                (
                    int(center[0] + math.cos(angle) * r),
                    int(center[1] + math.sin(angle) * r),
                )
            )
        try:
            pygame.draw.polygon(self.screen, color, vertices)
            if outline_width > 0:
                pygame.draw.polygon(self.screen, outline_color, vertices, outline_width)
        except Exception:
            return
    
    def _find_polity_text_position(self, world: World, polity) -> Optional[Tuple[float, float]]:
        """Find the optimal position to place polity name text.
        
        Args:
            world: World instance
            polity: Polity to find text position for
            
        Returns:
            World coordinates for text placement, or None if not found
        """
        if not polity.tile_indices:
            return None
        
        # Calculate centroid of all polity tiles
        total_x = 0.0
        total_y = 0.0
        
        for tile_idx in polity.tile_indices:
            tile = world.tiles[tile_idx]
            total_x += tile.center[0]
            total_y += tile.center[1]
        
        centroid_x = total_x / len(polity.tile_indices)
        centroid_y = total_y / len(polity.tile_indices)
        
        # Find the tile closest to the centroid for better positioning
        best_tile = None
        min_distance = float('inf')
        
        for tile_idx in polity.tile_indices:
            tile = world.tiles[tile_idx]
            distance = ((tile.center[0] - centroid_x) ** 2 + 
                       (tile.center[1] - centroid_y) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                best_tile = tile
        
        if best_tile:
            return best_tile.center
        else:
            return (centroid_x, centroid_y)
    
    def render_simulation_date(self, world: World) -> None:
        """Render the current simulation date in the top-left corner.
        
        Args:
            world: World instance
        """
        font = pygame.font.Font(None, 28)
        date_text = world.get_simulation_date()
        text_surface = font.render(date_text, True, (255, 255, 255))
        
        # Background for readability
        text_rect = text_surface.get_rect()
        bg_rect = pygame.Rect(10, 10, text_rect.width + 10, text_rect.height + 5)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect)
        
        # Render text
        self.screen.blit(text_surface, (15, 12))