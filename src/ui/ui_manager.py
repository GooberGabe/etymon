"""User interface management for Etymon."""

import pygame
from typing import Dict, List, Optional, Tuple, Any

from src.core.config_manager import ConfigManager
from src.world.world_data import World, Tile


class UIManager:
    """Manages user interface elements and interactions."""
    
    def __init__(self, config: ConfigManager, screen: pygame.Surface):
        """Initialize UI manager.
        
        Args:
            config: Configuration manager
            screen: Pygame screen surface
        """
        self.config = config
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()
        
        # UI state
        self.show_statistics = True
        self.show_tile_info = False
        self.show_map_mode_panel = True
        self.tooltips_enabled = True
        self.tile_info_position: Optional[Tuple[int, int]] = None
        self.selected_tile: Optional[Tile] = None
        self.map_mode_setting_button_rects: Dict[Tuple[str, str], pygame.Rect] = {}
        self.graph_view_active = False
        self._graph_prev_auto_tick_enabled: Optional[bool] = None
        self.graph_button_rects: Dict[str, pygame.Rect] = {}
        self.graph_status_message: Optional[str] = None
        self.graph_status_timer = 0.0
        self.graph_legend_rects: Dict[int, pygame.Rect] = {}
        self.graph_isolated_polity_id: Optional[int] = None
        
        # Mouse state
        self.mouse_position = (0, 0)
        self.mouse_pressed = False
        self.right_mouse_pressed = False
        self.last_mouse_position = (0, 0)
        self.sim_control_rects: dict = {}
        self.hover_anchor_position: Optional[Tuple[int, int]] = None
        self.hover_timer = 0.0
        self.hover_delay_seconds = self.config.get('ui.tooltips.hover_delay_seconds', 1.5)
        self._last_update_ticks: Optional[int] = None
        self._tooltip_active = False
        self._hover_movement_threshold = 2  # pixels of wiggle room
        self._elevation_stats: Optional[Dict[str, float]] = None
    
    def handle_event(self, event: pygame.event.Event) -> None:
        """Handle pygame events.
        
        Args:
            event: Pygame event
        """
        if self.graph_view_active:
            self._handle_graph_event(event)
            return
        if event.type == pygame.MOUSEMOTION:
            self.last_mouse_position = self.mouse_position
            self.mouse_position = event.pos
            self._handle_hover_motion(event.pos)
            
            # Handle right-click panning
            if self.right_mouse_pressed:
                dx = event.pos[0] - self.last_mouse_position[0]
                dy = event.pos[1] - self.last_mouse_position[1]
                if hasattr(self, 'map_renderer'):
                    self.map_renderer.pan_camera(-dx, -dy)  # Invert for natural panning
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                self.mouse_pressed = True
                # Simulation controls have highest priority
                if self._handle_simulation_controls_click(event.pos):
                    return
                # Check if click is on map mode panel
                if self._handle_map_mode_panel_click(event.pos):
                    return  # Panel handled the click
                self._handle_mouse_click(event.pos)
            elif event.button == 3:  # Right click
                self.right_mouse_pressed = True
                
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left click
                self.mouse_pressed = False
            elif event.button == 3:  # Right click
                self.right_mouse_pressed = False
                
        elif event.type == pygame.MOUSEWHEEL:
            # Handle zoom with mouse wheel
            if hasattr(self, 'map_renderer'):
                zoom_delta = event.y  # Positive for zoom in, negative for zoom out
                self.map_renderer.zoom_camera(zoom_delta, self.mouse_position)
                
        elif event.type == pygame.KEYDOWN:
            self._handle_keypress(event.key)
    
    def update(self) -> None:
        """Update UI state, including hover-based tooltip timing."""
        current_ticks = pygame.time.get_ticks()
        if self._last_update_ticks is None:
            self._last_update_ticks = current_ticks
            return
        delta = (current_ticks - self._last_update_ticks) / 1000.0
        self._last_update_ticks = current_ticks
        if delta <= 0:
            return
        if self.graph_status_message:
            self.graph_status_timer = max(0.0, self.graph_status_timer - delta)
            if self.graph_status_timer <= 0:
                self.graph_status_message = None
        if (not self.tooltips_enabled or
            self.hover_anchor_position is None or
                not hasattr(self, 'world') or
                not hasattr(self, 'map_renderer') or
                self.right_mouse_pressed):
            return
        if not self._tooltip_active:
            self.hover_timer += delta
            if self.hover_timer >= self.hover_delay_seconds:
                tile = self.map_renderer.get_tile_at_position(self.world, self.hover_anchor_position)
                if tile and not tile.is_water:
                    self.selected_tile = tile
                    self.tile_info_position = self.hover_anchor_position
                    self.show_tile_info = True
                    self._tooltip_active = True
                else:
                    self._hide_tooltip()
        else:
            # Keep tooltip anchored to current hover position
            self.tile_info_position = self.hover_anchor_position
    
    def render(self) -> None:
        """Render UI elements."""
        # Render simulation controls regardless of map panel visibility
        self._render_simulation_controls()
        # Render map mode panel
        if self.show_map_mode_panel:
            self._render_map_mode_panel()
        
        # Render tile information tooltip
        if self.show_tile_info and self.selected_tile and self.tile_info_position:
            self._render_tile_tooltip()
        
        # Render help text
        self._render_help_text()

        if self.graph_view_active:
            self._render_polity_graph_overlay()
    
    def set_world_reference(self, world: World) -> None:
        """Set reference to current world for tile queries.
        
        Args:
            world: Current world instance
        """
        self.world = world
        self._compute_elevation_stats()
    
    def set_map_renderer_reference(self, map_renderer) -> None:
        """Set reference to map renderer for camera controls.
        
        Args:
            map_renderer: MapRenderer instance
        """
        self.map_renderer = map_renderer

    def _compute_elevation_stats(self) -> None:
        """Cache world elevation bounds for the elevation map mode tooltip."""
        world = getattr(self, 'world', None)
        if not world or not getattr(world, 'tiles', None):
            self._elevation_stats = None
            return
        land_values = [tile.elevation for tile in world.tiles if not tile.is_water]
        water_values = [tile.elevation for tile in world.tiles if tile.is_water]
        sea_level = getattr(world, 'sea_level', 0.0)
        land_min = min(land_values) if land_values else sea_level
        land_max = max(land_values) if land_values else sea_level
        water_min = min(water_values) if water_values else 0.0
        water_max = max(water_values) if water_values else sea_level
        self._elevation_stats = {
            'sea_level': sea_level,
            'land_min': land_min,
            'land_max': land_max,
            'water_min': water_min,
            'water_max': water_max,
        }

    def _classify_relief_band(self, ratio: float) -> str:
        """Translate a normalized relief value into a descriptive label."""
        ratio = max(0.0, min(1.0, ratio))
        if ratio < 0.2:
            return "Coastal Plain"
        if ratio < 0.4:
            return "Lowlands"
        if ratio < 0.65:
            return "Highlands"
        if ratio < 0.9:
            return "Alpine"
        return "Peaks"
    
    def _handle_mouse_click(self, position: Tuple[int, int]) -> None:
        """Handle mouse click events.
        
        Args:
            position: Click position
        """
        # Treat clicks as hover resets so tooltip timing restarts at the click location
        self._handle_hover_motion(position)
    
    def _handle_keypress(self, key: int) -> None:
        """Handle key press events.
        
        Args:
            key: Pygame key constant
        """
        if key == pygame.K_TAB:
            self.show_statistics = not self.show_statistics
        elif key == pygame.K_h:
            # Toggle help (could expand this later)
            pass
        elif key == pygame.K_t:
            self.tooltips_enabled = not self.tooltips_enabled
            self.hover_timer = 0.0
            if not self.tooltips_enabled:
                self._hide_tooltip()
        elif key == pygame.K_m:
            # Cycle through map modes
            if hasattr(self, 'world'):
                self._cycle_map_mode()
        elif key == pygame.K_g:
            self.toggle_graph_view()
        elif key == pygame.K_l:
            # Print language catalog for hovered polity's primary culture (linguistics mode only)
            if hasattr(self, 'world') and self.world.current_map_mode == "linguistics":
                self._print_language_catalog_for_hovered_polity()

    def toggle_graph_view(self) -> None:
        """Toggle the full-screen polity development graph."""
        if not hasattr(self, 'world') or self.world is None:
            return
        if self.graph_view_active:
            self._close_graph_view()
        else:
            self._open_graph_view()

    def _open_graph_view(self) -> None:
        if self.graph_view_active:
            return
        world = getattr(self, 'world', None)
        if not world:
            return
        self.graph_view_active = True
        self._graph_prev_auto_tick_enabled = getattr(world, 'auto_tick_enabled', False)
        if hasattr(world, 'set_auto_tick_enabled'):
            world.set_auto_tick_enabled(False)
        if hasattr(world, 'record_polity_development_snapshot'):
            world.record_polity_development_snapshot()
        self.graph_button_rects = {}
        self.graph_legend_rects = {}
        self.graph_isolated_polity_id = None
        self._hide_tooltip()

    def _close_graph_view(self) -> None:
        if not self.graph_view_active:
            return
        world = getattr(self, 'world', None)
        if world and self._graph_prev_auto_tick_enabled:
            if hasattr(world, 'set_auto_tick_enabled'):
                world.set_auto_tick_enabled(True)
        self.graph_view_active = False
        self._graph_prev_auto_tick_enabled = None
        self.graph_button_rects = {}
        self.graph_legend_rects = {}
        self.graph_isolated_polity_id = None

    def _handle_graph_event(self, event: pygame.event.Event) -> None:
        """Handle events while the graph overlay is visible."""
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_g, pygame.K_ESCAPE):
                self.toggle_graph_view()
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            close_rect = self.graph_button_rects.get('close')
            if close_rect and close_rect.collidepoint(event.pos):
                self.toggle_graph_view()
                return
            purge_rect = self.graph_button_rects.get('purge')
            if purge_rect and purge_rect.collidepoint(event.pos):
                self._purge_tileless_polities()
                return
            if self._handle_graph_legend_click(event.pos):
                return

    def _handle_graph_legend_click(self, position: Tuple[int, int]) -> bool:
        """Toggle isolated series selection when clicking the legend."""
        if not self.graph_legend_rects:
            return False
        for polity_id, rect in self.graph_legend_rects.items():
            if rect.collidepoint(position):
                if self.graph_isolated_polity_id == polity_id:
                    self.graph_isolated_polity_id = None
                else:
                    self.graph_isolated_polity_id = polity_id
                return True
        return False

    def _purge_tileless_polities(self) -> None:
        """Invoke the world's purge routine and show feedback."""
        world = getattr(self, 'world', None)
        if not world or not hasattr(world, 'purge_tileless_polities'):
            return
        removed = world.purge_tileless_polities()
        if removed and hasattr(world, 'record_polity_development_snapshot'):
            world.record_polity_development_snapshot()
        if removed:
            noun = "polity" if removed == 1 else "polities"
            self.graph_status_message = f"Purged {removed} tile-less {noun}."
        else:
            self.graph_status_message = "No tile-less polities found."
        self.graph_status_timer = 3.0
    
    def _render_tile_tooltip(self) -> None:
        """Render tile information tooltip with map mode specific details."""
        if not self.selected_tile:
            return
            
        font = pygame.font.Font(None, 20)
        
        current_mode = getattr(self, 'world', None) and self.world.current_map_mode or 'biomes'

        if current_mode == "war":
            info_lines = self._build_war_tooltip_info()
        else:
            info_lines = [f"Map Mode: {current_mode.title()}"]
            view_label = self._get_map_mode_setting_label(current_mode)
            if view_label:
                info_lines.append(f"View: {view_label}")
            info_lines.append(f"Coordinates: {getattr(self.selected_tile, 'center', (0, 0))}")

            if current_mode == "cultures":
                self._add_culture_tooltip_info(info_lines)
            elif current_mode == "control":
                self._add_control_tooltip_info(info_lines)
            elif current_mode == "population":
                self._add_population_tooltip_info(info_lines)
            elif current_mode == "development":
                self._add_development_tooltip_info(info_lines)
            elif current_mode == "regions":
                self._add_region_tooltip_info(info_lines)
            elif current_mode == "linguistics":
                self._add_linguistics_tooltip_info(info_lines)
            elif current_mode == "elevation":
                self._add_elevation_tooltip_info(info_lines)
            else:
                self._add_biome_tooltip_info(info_lines)

            info_lines.extend([
                f"Type: {'Water' if self.selected_tile.is_water else 'Land'}",
                f"Elevation: {self.selected_tile.elevation:.3f}",
            ])
        
        # Calculate tooltip size
        max_width = 0
        total_height = 0
        line_height = font.get_height()
        
        for line in info_lines:
            text_width = font.size(line)[0]
            max_width = max(max_width, text_width)
            total_height += line_height + 2
        
        # Position tooltip to avoid screen edges
        tooltip_width = max_width + 20
        tooltip_height = total_height + 10
        
        x = self.tile_info_position[0] + 10
        y = self.tile_info_position[1] + 10
        
        # Keep tooltip on screen
        if x + tooltip_width > self.width:
            x = self.tile_info_position[0] - tooltip_width - 10
        if y + tooltip_height > self.height:
            y = self.tile_info_position[1] - tooltip_height - 10
        
        # Draw tooltip background
        tooltip_rect = pygame.Rect(x, y, tooltip_width, tooltip_height)
        pygame.draw.rect(self.screen, (0, 0, 0, 200), tooltip_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), tooltip_rect, 1)
        
        # Draw text
        text_y = y + 5
        for line in info_lines:
            text = font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (x + 10, text_y))
            text_y += line_height + 2

    def _handle_hover_motion(self, position: Tuple[int, int]) -> None:
        """Track hover movement to control tooltip visibility."""
        if self.hover_anchor_position is None:
            self.hover_anchor_position = position
            self.hover_timer = 0.0
            return
        dx = abs(position[0] - self.hover_anchor_position[0])
        dy = abs(position[1] - self.hover_anchor_position[1])
        if dx > self._hover_movement_threshold or dy > self._hover_movement_threshold:
            self.hover_anchor_position = position
            self.hover_timer = 0.0
            self._tooltip_active = False
            self._hide_tooltip()

    def _hide_tooltip(self) -> None:
        """Clear tooltip visibility state."""
        self.show_tile_info = False
        self.selected_tile = None
        self._tooltip_active = False

    def _print_language_catalog_for_hovered_polity(self) -> None:
        """Print the complete language catalog for the hovered polity's primary culture."""
        if not self.selected_tile:
            print("No tile selected - hover over a tile first")
            return
            
        world = getattr(self, 'world', None)
        if not world:
            print("World not available")
            return
            
        polity_id = getattr(self.selected_tile, 'polity_id', -1)
        if polity_id < 0 or polity_id >= len(world.polities):
            print("Selected tile is not controlled by any polity")
            return
            
        polity = world.polities[polity_id]
        primary_culture_name = polity.primary_culture
        if not primary_culture_name:
            print(f"Polity {polity.name} (ID {polity_id}) has no primary culture assigned")
            return
            
        # Find the culture object
        culture = None
        for c in world.cultures:
            if c.name == primary_culture_name:
                culture = c
                break
                
        if not culture:
            print(f"Primary culture '{primary_culture_name}' not found in world cultures")
            return
            
        # Get the evolved catalog for this culture
        catalog = world.get_culture_language(primary_culture_name)
        if not catalog:
            print(f"Culture '{primary_culture_name}' has no associated language catalog")
            return
            
        # Print the catalog
        print(f"\n=== LANGUAGE CATALOG: {catalog.name} ===")
        print(f"Culture: {primary_culture_name}")
        print(f"Polity: {polity.name} (ID {polity_id})")
        print(f"Words: {catalog.word_count()}")
        print(f"Categories: {', '.join(catalog.categories())}")
        
        # Show generation info if available
        generation = catalog.metadata.get('generation', 1)
        if generation > 1:
            root_name = catalog.metadata.get('root_name', 'Unknown')
            print(f"Derived from: {root_name} (Generation {generation})")
        print()
        
        # Group words by category
        words_by_category = {}
        for word in catalog.words:
            category = word.category or "uncategorized"
            if category not in words_by_category:
                words_by_category[category] = []
            words_by_category[category].append(word)
        
        # Print words by category
        for category in sorted(words_by_category.keys()):
            print(f"--- {category.upper()} ---")
            for word in sorted(words_by_category[category], key=lambda w: w.gloss or ""):
                gloss = word.gloss or "(no gloss)"
                print(f"  {word.phonetic_form} - {gloss}")
            print()

    def _build_war_tooltip_info(self) -> List[str]:
        lines: List[str] = []
        tile = self.selected_tile
        world = getattr(self, 'world', None)
        if not world:
            return ["War Intel: World reference unavailable."]
        if tile.is_water:
            return ["War Intel: No combat data over water tiles."]
        polity_id = getattr(tile, 'polity_id', -1)
        if polity_id < 0:
            return ["War Intel: Uncontrolled territory."]
        polity_label = self._get_polity_label(polity_id)
        lines.append(f"Polity: {polity_label} (ID {polity_id})")
        relationships = [rel for rel in world.relationships if rel and rel.involves(polity_id) and rel.shared_border_tiles > 0]
        if not relationships:
            lines.append("No known diplomatic relationships.")
            return lines
        wars = [rel for rel in relationships if rel.status == "war"]
        if wars:
            lines.append("Active Wars:")
            for rel in wars:
                other_id = rel.other(polity_id)
                other_label = self._get_polity_label(other_id)
                exhaustion = rel.war_exhaustion.get(polity_id, 0.0)
                since = rel.war_start_year if rel.war_start_year is not None else "Unknown"
                lines.append(f"  vs {other_label}: Exhaustion {exhaustion:.1f} | Since Year {since}")
        peaceful = [rel for rel in relationships if rel.status != "war"]
        if peaceful:
            lines.append("Diplomatic Status:")
            for rel in peaceful:
                other_id = rel.other(polity_id)
                other_label = self._get_polity_label(other_id)
                status = rel.status.title()
                relation_value = None
                try:
                    relation_value = world._get_current_relation(polity_id, other_id, rel)
                except Exception:
                    relation_value = None
                line = f"  {status} with {other_label}"
                if relation_value is not None:
                    line += f" | Relation {relation_value:+.1f}"
                if rel.truce_until_year and world.current_year < rel.truce_until_year:
                    line += f" | Truce ends Year {rel.truce_until_year}"
                lines.append(line)
        if not wars and not peaceful:
            lines.append("No diplomatic data available.")
        return lines

    def _get_polity_label(self, polity_id: Optional[int]) -> str:
        if polity_id is None or polity_id < 0:
            return "Unknown Polity"
        world = getattr(self, 'world', None)
        if not world or polity_id >= len(world.polities):
            return f"Polity {polity_id}"
        polity = world.polities[polity_id]
        if polity and getattr(polity, 'name', None):
            return polity.name
        return f"Polity {polity_id}"

    def _get_map_mode_setting_label(self, mode: str) -> Optional[str]:
        world = getattr(self, 'world', None)
        if not world or not hasattr(world, 'get_map_mode_setting_label'):
            return None
        return world.get_map_mode_setting_label(mode)

    def _add_culture_tooltip_info(self, info_lines: list) -> None:
        """Add culture-specific tooltip information."""
        if self.selected_tile.is_water:
            info_lines.append("Cultural Makeup: N/A (Water)")
            return
        
        world = getattr(self, 'world', None)
        if hasattr(self.selected_tile, 'cultural_makeup') and self.selected_tile.cultural_makeup:
            info_lines.append("Cultural Makeup:")
            # Sort cultures by percentage (highest first)
            sorted_cultures = sorted(self.selected_tile.cultural_makeup.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            for culture_name, percentage in sorted_cultures:
                display_name = culture_name
                if world and hasattr(world, 'get_culture_display_label'):
                    friendly = world.get_culture_display_label(culture_name)
                    if friendly:
                        display_name = friendly
                info_lines.append(f"  {display_name}: {percentage:.1%}")
                
            # Show dominant culture heritage if available
            if world and world.cultures:
                dominant_culture_name = sorted_cultures[0][0]
                dominant_label = dominant_culture_name
                if hasattr(world, 'get_culture_display_label'):
                    friendly_dominant = world.get_culture_display_label(dominant_culture_name)
                    if friendly_dominant:
                        dominant_label = friendly_dominant
                for culture in world.cultures:
                    if culture.name == dominant_culture_name and culture.heritage:
                        info_lines.append(f"Heritage of {dominant_label}:")
                        for parent, heritage_pct in culture.heritage.items():
                            info_lines.append(f"  From {parent}: {heritage_pct:.1%}")
                        break
        else:
            info_lines.append("Cultural Makeup: None")
        
        # Add polity info for context
        if hasattr(self.selected_tile, 'polity_id') and self.selected_tile.polity_id >= 0:
            polity_name = f"Polity {self.selected_tile.polity_id}"
            if world and self.selected_tile.polity_id < len(world.polities):
                polity = world.polities[self.selected_tile.polity_id]
                if polity:
                    polity_name = polity.name
                    primary_label = polity.primary_culture or "None"
                    if hasattr(world, 'get_culture_display_label'):
                        friendly_primary = world.get_culture_display_label(polity.primary_culture)
                        if friendly_primary:
                            primary_label = friendly_primary
                else:
                    primary_label = "None"
                info_lines.append(f"Polity: {polity_name}")
                info_lines.append(f"Primary Culture of Polity: {primary_label}")
                tolerance_value = None
                if world and hasattr(world, 'get_polity_cultural_tolerance'):
                    tolerance_value = world.get_polity_cultural_tolerance(self.selected_tile.polity_id)
                if tolerance_value is not None:
                    info_lines.append(f"Cultural Tolerance: {tolerance_value:.2f}")

    def _add_control_tooltip_info(self, info_lines: list) -> None:
        """Add control-specific tooltip information."""
        if self.selected_tile.is_water:
            info_lines.append("Control: N/A (Water)")
            return
        world = getattr(self, 'world', None)
        if hasattr(self.selected_tile, 'polity_id') and self.selected_tile.polity_id >= 0:
            polity_name = f"Polity {self.selected_tile.polity_id}"
            polity = None
            if world and self.selected_tile.polity_id < len(world.polities):
                polity = world.polities[self.selected_tile.polity_id]
                polity_name = polity.name
                
            control_level = getattr(self.selected_tile, 'control_level', 50)
            info_lines.extend([
                f"Controlled by: {polity_name}",
                f"Control Level: {control_level}%",
                f"Population: {getattr(self.selected_tile, 'population', 0):,}",
                f"Development: {getattr(self.selected_tile, 'development', 0):.2f}",
            ])
            
            # Show if this affects control
            if hasattr(self.selected_tile, 'cultural_makeup') and self.selected_tile.cultural_makeup:
                polity_culture = getattr(polity, 'primary_culture', 'Unknown') if polity else 'Unknown'
                culture_alignment = self.selected_tile.cultural_makeup.get(polity_culture, 0.0)
                info_lines.append(f"Cultural Alignment: {culture_alignment:.1%}")
            control_debug = getattr(self.selected_tile, 'control_debug', None)
            if isinstance(control_debug, dict) and control_debug:
                start_val = control_debug.get('start_control')
                end_val = control_debug.get('end_control')
                tick_marker = control_debug.get('tick_marker')
                if tick_marker is not None:
                    info_lines.append(f"Control Diagnostics: tick {tick_marker}")
                if start_val is not None and end_val is not None:
                    info_lines.append(f"Control (start→end): {start_val}% → {end_val}%")
                self._append_control_modifier_section(
                    info_lines,
                    "Tick Modifiers",
                    control_debug.get('tick_modifiers'),
                )
                self._append_control_modifier_section(
                    info_lines,
                    "Flat Modifiers",
                    control_debug.get('flat_modifiers'),
                )
                self._append_control_caps_section(info_lines, control_debug.get('caps'))
            else:
                info_lines.append("Control Modifiers: No recent diagnostics")

            if world and hasattr(world, 'get_polity_administrative_burden'):
                base_threshold = 10
                config_source = getattr(world, 'config', None)
                if config_source:
                    base_threshold = config_source.get('simulation.polity.breakaway_low_control_threshold', base_threshold)
                burden = world.get_polity_administrative_burden(self.selected_tile.polity_id)
                if burden is None:
                    burden = 0
                effective_threshold = base_threshold + burden
                effective_threshold = max(0, min(99, effective_threshold))
                info_lines.append(
                    f"Administrative Burden: +{burden} pts (Low-control threshold {effective_threshold}%)"
                )
        else:
            info_lines.append("Control: Uncontrolled territory")

    def _append_control_modifier_section(self, info_lines: list, title: str, entries: Optional[list]) -> None:
        """Render a modifier section (tick or flat) in the tooltip."""
        info_lines.append(f"{title}:")
        if not entries:
            info_lines.append("  (none)")
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            label = entry.get('label', 'Unknown')
            delta = entry.get('delta')
            if delta is None:
                continue
            info_lines.append(f"  {label}: {delta:+d}")
            details = entry.get('details')
            if details:
                info_lines.append(f"    {details}")

    def _append_control_caps_section(self, info_lines: list, entries: Optional[list]) -> None:
        """Render active control caps in the tooltip."""
        info_lines.append("Caps:")
        if not entries:
            info_lines.append("  (none)")
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            label = entry.get('label', 'Cap')
            value = entry.get('value')
            if value is None:
                continue
            info_lines.append(f"  {label}: ≤{value}%")
            details = entry.get('details')
            if details:
                info_lines.append(f"    {details}")

    def _add_population_tooltip_info(self, info_lines: list) -> None:
        """Add population-specific tooltip information."""
        if self.selected_tile.is_water:
            info_lines.append("Population: N/A (Water)")
            return
            
        population = getattr(self.selected_tile, 'population', 0)
        development = getattr(self.selected_tile, 'development', 0)
        last_deaths = getattr(self.selected_tile, 'last_tick_deaths', 0)
        
        info_lines.extend([
            f"Population: {population:,}",
            f"Development: {development:.2f}",
            f"Last Tick Deaths: {last_deaths}",
        ])
        
        # Calculate development cap if world is available
        if hasattr(self, 'world') and self.world:
            dev_cap = self.world._calculate_development_cap(self.selected_tile)
            info_lines.append(f"Development Cap: {dev_cap:,.0f}")
            
            # Show if it's a population center
            for center in self.world.population_centers:
                if center.tile_index < len(self.world.tiles) and self.world.tiles[center.tile_index] == self.selected_tile:
                    info_lines.append(f"Population Center: {center.name}")
                    info_lines.append(f"Center Threshold: {center.original_threshold}")
                    age_years = max(0, self.world.current_year - center.established_year)
                    info_lines.append(f"Center Age: {age_years} years")
                    break

    def _add_development_tooltip_info(self, info_lines: list) -> None:
        """Add development-specific tooltip information."""
        if self.selected_tile.is_water:
            info_lines.append("Development: N/A (Water)")
            return
            
        development = getattr(self.selected_tile, 'development', 0)
        population = getattr(self.selected_tile, 'population', 0)
        
        info_lines.extend([
            f"Development: {development:.2f}",
            f"Population: {population:,}",
        ])
        
        if hasattr(self, 'world') and self.world:
            dev_cap = self.world._calculate_development_cap(self.selected_tile)
            info_lines.append(f"Development Cap: {dev_cap:,.0f}")
            
            # Show development efficiency
            if population > 0:
                dev_per_pop = development / population
                info_lines.append(f"Dev per Person: {dev_per_pop:.3f}")
            
            # Show climate factors affecting development
            if hasattr(self.selected_tile, 'temperature') and hasattr(self.selected_tile, 'rainfall'):
                info_lines.extend([
                    f"Temperature: {self.selected_tile.temperature:.3f}",
                    f"Rainfall: {self.selected_tile.rainfall:.3f}",
                ])

    def _add_region_tooltip_info(self, info_lines: list) -> None:
        """Add region-specific tooltip information."""
        region_id = getattr(self.selected_tile, 'region_id', -1)
        if region_id >= 0:
            region_name = f"Region {region_id}"
            if hasattr(self, 'world') and self.world and region_id < len(self.world.regions):
                region = self.world.regions[region_id]
                region_name = region.name
                info_lines.append(f"Region: {region_name}")
                info_lines.append(f"Region ID: {region_id}")
            else:
                info_lines.append(f"Region: {region_name}")
        else:
            info_lines.append("Region: Unassigned")
            
        info_lines.extend([
            f"Biome: {getattr(self.selected_tile, 'biome', 'Unknown')}",
            f"Temperature: {getattr(self.selected_tile, 'temperature', 0):.3f}",
            f"Rainfall: {getattr(self.selected_tile, 'rainfall', 0):.3f}",
        ])

    def _add_linguistics_tooltip_info(self, info_lines: list) -> None:
        """Add settlement/polity etymology and lineage details."""
        world = getattr(self, 'world', None)
        tile = self.selected_tile
        if not world or not tile:
            info_lines.append("Linguistics data unavailable.")
            return
        if tile.is_water:
            info_lines.append("Linguistics: N/A (Water)")
            return
        tile_index = self._get_tile_index(tile)
        if tile_index is None:
            info_lines.append("Linguistics: Tile lookup unavailable.")
            return

        culture_name, share = self._determine_tile_majority_culture(tile)
        if culture_name:
            label = culture_name
            if hasattr(world, 'get_culture_display_label'):
                friendly = world.get_culture_display_label(culture_name)
                if friendly:
                    label = friendly
            info_lines.append(f"Dominant Culture: {label} ({share:.1%})")
        else:
            info_lines.append("Dominant Culture: Unknown")

        language_label = None
        catalog = None
        if culture_name and hasattr(world, 'get_culture_language'):
            catalog = world.get_culture_language(culture_name)
            if catalog:
                language_label = getattr(catalog, 'name', None)
        if language_label:
            info_lines.append(f"Language Catalog: {language_label}")
            if catalog:
                # Show generation if available
                generation = catalog.metadata.get('generation', 1)
                if generation > 1:
                    root_name = catalog.metadata.get('root_name', 'Unknown')
                    info_lines.append(f"Derived from: {root_name} (Gen {generation})")
                # Show transformation if available
                transformation = catalog.metadata.get('transformation')
                if transformation:
                    info_lines.append(f"Last Transformation: {transformation.replace('_', ' ').title()}")
                # Show a few word examples
                if catalog.words:
                    examples = [f"{w.gloss}='{w.phonetic_form}'" for w in catalog.words[:3]]
                    info_lines.append(f"Sample Words: {', '.join(examples)}")

        lineage: List[Dict[str, Any]] = []
        if culture_name and hasattr(world, 'get_culture_lineage'):
            lineage = world.get_culture_lineage(culture_name)
        if lineage:
            info_lines.append("Culture Lineage:")
            for entry in lineage[:5]:
                depth = int(entry.get('depth', 0) or 0)
                indent = "  " * min(3, max(0, depth))
                label = entry.get('name') or "Unknown"
                segments: List[str] = []
                share_value = entry.get('share')
                if isinstance(share_value, (float, int)) and share_value > 0:
                    segments.append(f"{share_value:.0%}")
                language = entry.get('language')
                if language:
                    segments.append(language)
                birth_year = entry.get('birth_year')
                if isinstance(birth_year, int) and birth_year > 0:
                    segments.append(f"est. {birth_year}")
                details = f" ({', '.join(segments)})" if segments else ""
                info_lines.append(f"{indent}• {label}{details}")
        else:
            info_lines.append("Culture Lineage: No ancestry data yet")

        center = self._get_population_center_by_tile(tile_index)
        center_history = []
        if hasattr(world, 'get_population_center_history'):
            center_history = world.get_population_center_history(tile_index)
        if center or center_history:
            info_lines.append("Settlement Names:")
            if center:
                info_lines.append(
                    f"  Current: {center.name} (est. Year {center.established_year})"
                )
            for entry in reversed(center_history[-3:]):
                info_lines.append(self._format_name_history_entry(entry))
        else:
            info_lines.append("Settlement Names: None recorded")

        polity_id = getattr(tile, 'polity_id', -1)
        polity = None
        if polity_id is not None and 0 <= polity_id < len(world.polities):
            polity = world.polities[polity_id]
        if polity:
            info_lines.append(f"Polity: {polity.name} (ID {polity_id})")
            polity_history = []
            if hasattr(world, 'get_polity_name_history'):
                polity_history = world.get_polity_name_history(polity_id)
            if polity_history:
                info_lines.append("Polity Names:")
                for entry in reversed(polity_history[-3:]):
                    info_lines.append(self._format_name_history_entry(entry))
        else:
            info_lines.append("Polity: None")

    def _add_biome_tooltip_info(self, info_lines: list) -> None:
        """Add biome-specific tooltip information."""
        info_lines.extend([
            f"Biome: {getattr(self.selected_tile, 'biome', 'Unknown')}",
            f"Temperature: {getattr(self.selected_tile, 'temperature', 0):.3f}",
            f"Rainfall: {getattr(self.selected_tile, 'rainfall', 0):.3f}",
        ])
        
        if not self.selected_tile.is_water:
            info_lines.extend([
                f"Population: {getattr(self.selected_tile, 'population', 0):,}",
                f"Development: {getattr(self.selected_tile, 'development', 0):.2f}",
            ])
            
        info_lines.append(f"Neighbors: {len(self.selected_tile.neighbors)}")

    def _add_elevation_tooltip_info(self, info_lines: list) -> None:
        """Add elevation-specific tooltip context."""
        world = getattr(self, 'world', None)
        tile = self.selected_tile
        if not world or not tile:
            info_lines.append("Elevation data unavailable.")
            return
        stats = self._elevation_stats
        if not stats:
            self._compute_elevation_stats()
            stats = self._elevation_stats
        if not stats:
            info_lines.append("Elevation statistics unavailable.")
            return
        sea_level = stats.get('sea_level', getattr(world, 'sea_level', 0.0))
        elevation = getattr(tile, 'elevation', 0.0)
        delta = elevation - sea_level
        info_lines.append(f"Sea Level: {sea_level:.3f}")
        info_lines.append(f"Sea Level Δ: {delta:+.3f}")
        if tile.is_water:
            depth_span = max(sea_level - stats.get('water_min', 0.0), 1e-6)
            depth_ratio = (sea_level - elevation) / depth_span if elevation < sea_level else 0.0
            depth_ratio = max(0.0, min(1.0, depth_ratio))
            info_lines.append(f"Relative Depth: {depth_ratio:.2f}")
            return
        relief_base = max(sea_level, stats.get('land_min', sea_level))
        relief_span = max(stats.get('land_max', sea_level) - relief_base, 1e-6)
        relief_ratio = (elevation - relief_base) / relief_span
        relief_ratio = max(0.0, min(1.0, relief_ratio))
        band = self._classify_relief_band(relief_ratio)
        info_lines.append(f"Relief Band: {band}")
        info_lines.append(f"Normalized Elevation: {relief_ratio:.2f}")
        info_lines.append(f"Temperature: {getattr(tile, 'temperature', 0):.3f}")
        info_lines.append(f"Rainfall: {getattr(tile, 'rainfall', 0):.3f}")
    
    def _render_help_text(self) -> None:
        """Render help text in corner of screen."""
        font = pygame.font.Font(None, 18)
        
        help_lines = [
            "Controls:",
            "SPACE - Generate new world",
            "       (reloads config file)",
            "ENTER - Advance tick",
            "       (auto-ticks every 0.5s)",
            "Buttons: Pause / 2x",
            "       (top-right panel)",
            "Right-click + drag - Pan",
            "Mouse wheel - Zoom",
            "TAB - Toggle statistics",
            "T - Toggle tile info",
            "M - Cycle map modes",
            "    (Use panel to select mode)",
            "    Polity borders always shown",
            "G - Polity graph view",
            "    (pauses while open)",
            "L - Print language catalog",
            "    (linguistics mode only)",
            "Left-click - Select tile/mode",
            "ESC - Exit"
        ]
        
        y_offset = self.height - (len(help_lines) * 20) - 10
        
        for line in help_lines:
            text = font.render(line, True, (255, 255, 255))
            
            # Draw background for readability
            if line == "Controls:":
                bg_rect = pygame.Rect(self.width - 150, y_offset - 2, 145, 18)
                pygame.draw.rect(self.screen, (0, 0, 0, 128), bg_rect)
            else:
                bg_rect = pygame.Rect(self.width - 150, y_offset - 2, 145, 18)
                pygame.draw.rect(self.screen, (0, 0, 0, 64), bg_rect)
            
            self.screen.blit(text, (self.width - 145, y_offset))
            y_offset += 20

    def _downsample_history(self, entries: List[Dict[str, Any]], max_points: int) -> List[Dict[str, Any]]:
        """Reduce a history list to at most max_points samples."""
        if max_points <= 0 or len(entries) <= max_points:
            return list(entries)
        step = max(1, len(entries) // max_points)
        sampled = entries[::step]
        if sampled[-1] is not entries[-1]:
            sampled.append(entries[-1])
        return sampled

    def _render_polity_graph_overlay(self) -> None:
        """Render the full-screen polity development graph overlay."""
        self.graph_legend_rects = {}
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((8, 10, 25, 235))
        self.screen.blit(overlay, (0, 0))
        title_font = pygame.font.Font(None, 42)
        subtitle_font = pygame.font.Font(None, 24)
        axis_font = pygame.font.Font(None, 20)
        title_surf = title_font.render("Polity Development Graph", True, (240, 240, 240))
        self.screen.blit(title_surf, (40, 28))
        subtitle = "Press G to close • Simulation paused while open"
        subtitle_surf = subtitle_font.render(subtitle, True, (210, 210, 210))
        self.screen.blit(subtitle_surf, (40, 70))

        legend_width = 230
        margin = 60
        graph_width = max(220, self.width - legend_width - margin * 2)
        graph_height = max(220, self.height - 220)
        graph_rect = pygame.Rect(margin, 110, graph_width, graph_height)
        legend_rect = pygame.Rect(
            graph_rect.right + 20,
            graph_rect.top,
            max(140, self.width - (graph_rect.right + 40)),
            graph_height,
        )
        pygame.draw.rect(self.screen, (18, 24, 38), graph_rect)
        pygame.draw.rect(self.screen, (90, 110, 150), graph_rect, 1)
        pygame.draw.rect(self.screen, (14, 18, 30), legend_rect)
        pygame.draw.rect(self.screen, (80, 100, 140), legend_rect, 1)

        world = getattr(self, 'world', None)
        history_map = {}
        if world and hasattr(world, 'get_polity_development_history'):
            history_map = world.get_polity_development_history()

        series_list: List[Dict[str, Any]] = []
        for polity_id, entries in history_map.items() if history_map else []:
            if not entries:
                continue
            polity = None
            if 0 <= polity_id < len(world.polities):  # type: ignore[arg-type]
                polity = world.polities[polity_id]
            label = getattr(polity, 'name', f"Polity {polity_id}") if polity else f"Polity {polity_id}"
            raw_color = getattr(polity, 'color', (200, 200, 200)) if polity else (200, 200, 200)
            color = tuple(int(max(0, min(255, component))) for component in raw_color)
            active = getattr(polity, 'is_active', False) if polity else bool(entries[-1].get('is_active', False))
            series_list.append({
                'id': polity_id,
                'label': label,
                'color': color,
                'entries': list(entries),
                'active': active,
            })

        all_ticks = [entry.get('tick', 0) for series in series_list for entry in series['entries']]
        all_dev = [entry.get('development', 0.0) for series in series_list for entry in series['entries']]
        min_tick = min(all_ticks) if all_ticks else 0
        max_tick = max(all_ticks) if all_ticks else 1
        tick_range = max(1, max_tick - min_tick)
        max_dev_value = max(all_dev) if all_dev else 0.0
        dev_scale = max(1.0, max_dev_value)

        axis_color = (130, 150, 190)
        pygame.draw.line(self.screen, axis_color, graph_rect.bottomleft, graph_rect.bottomright, 2)
        pygame.draw.line(self.screen, axis_color, graph_rect.bottomleft, graph_rect.topleft, 2)

        if series_list:
            min_year = min((entry.get('year', 0) for series in series_list for entry in series['entries']), default=0)
            max_year = max((entry.get('year', 0) for series in series_list for entry in series['entries']), default=0)
        else:
            min_year = max_year = 0
        year_left = axis_font.render(f"Year {min_year}", True, axis_color)
        year_right = axis_font.render(f"Year {max_year}", True, axis_color)
        self.screen.blit(year_left, (graph_rect.left, graph_rect.bottom + 8))
        right_pos = (graph_rect.right - year_right.get_width(), graph_rect.bottom + 8)
        self.screen.blit(year_right, right_pos)
        dev_label = axis_font.render(f"Max Dev: {max_dev_value:.0f}", True, axis_color)
        self.screen.blit(dev_label, (graph_rect.left, graph_rect.top - 28))

        max_points = max(80, graph_rect.width // 2)
        isolated_polity_id = self.graph_isolated_polity_id
        for series in series_list:
            sampled = self._downsample_history(series['entries'], max_points)
            points: List[Tuple[int, int]] = []
            for entry in sampled:
                tick = entry.get('tick', 0)
                development = max(0.0, float(entry.get('development', 0.0)))
                x_ratio = (tick - min_tick) / tick_range if tick_range else 0.0
                y_ratio = development / dev_scale if dev_scale else 0.0
                x = graph_rect.left + x_ratio * graph_rect.width
                y = graph_rect.bottom - y_ratio * graph_rect.height
                points.append((int(x), int(y)))
            if not points:
                continue
            color = series['color']
            if not series['active']:
                color = tuple(int(component * 0.5) for component in color)
            if isolated_polity_id is not None and series['id'] != isolated_polity_id:
                color = tuple(max(20, int(component * 0.2)) for component in color)
            width = 3 if isolated_polity_id is None or series['id'] == isolated_polity_id else 1
            if len(points) == 1:
                pygame.draw.circle(self.screen, color, points[0], max(2, width))
            else:
                pygame.draw.lines(self.screen, color, False, points, width)

        legend_font = pygame.font.Font(None, 20)
        series_sorted = sorted(
            series_list,
            key=lambda s: (
                -int(s['active']),
                -float(s['entries'][-1].get('development', 0.0)),
                s['label'],
            )
        )
        legend_y = legend_rect.top + 10
        hint_height = 0
        if series_sorted:
            hint_text = legend_font.render("Click a color to isolate", True, (190, 200, 230))
            self.screen.blit(hint_text, (legend_rect.left + 10, legend_y))
            hint_height = hint_text.get_height() + 6
        legend_y += hint_height
        remaining_height = max(22, legend_rect.height - (legend_y - legend_rect.top) - 10)
        max_rows = max(1, remaining_height // 22)
        for idx, series in enumerate(series_sorted[:max_rows]):
            entry_rect = pygame.Rect(legend_rect.left + 6, legend_y - 2, legend_rect.width - 12, 20)
            if isolated_polity_id == series['id']:
                pygame.draw.rect(self.screen, (60, 85, 130), entry_rect)
            elif isolated_polity_id is not None:
                pygame.draw.rect(self.screen, (25, 30, 45), entry_rect)
            color_box = pygame.Rect(entry_rect.left + 4, legend_y, 14, 14)
            base_color = series['color'] if series['active'] else tuple(int(c * 0.5) for c in series['color'])
            if isolated_polity_id is not None and series['id'] != isolated_polity_id:
                base_color = tuple(max(20, int(component * 0.2)) for component in base_color)
            color = base_color
            pygame.draw.rect(self.screen, color, color_box)
            pygame.draw.rect(self.screen, (20, 20, 20), color_box, 1)
            latest_dev = series['entries'][-1].get('development', 0.0)
            label_text = series['label'] if series['active'] else f"{series['label']} (inactive)"
            label_color = (230, 230, 230)
            if isolated_polity_id is not None and series['id'] != isolated_polity_id:
                label_color = (150, 150, 165)
            label = legend_font.render(f"{label_text} · {latest_dev:.0f}", True, label_color)
            self.screen.blit(label, (color_box.right + 8, legend_y - 2))
            self.graph_legend_rects[series['id']] = entry_rect
            legend_y += 22
        if len(series_sorted) > max_rows:
            more_text = legend_font.render("…", True, (230, 230, 230))
            self.screen.blit(more_text, (legend_rect.left + 10, legend_y))

        if not series_list:
            empty_text = subtitle_font.render(
                "No historical data yet. Let the simulation run to build it.",
                True,
                (235, 235, 235),
            )
            empty_rect = empty_text.get_rect(center=graph_rect.center)
            self.screen.blit(empty_text, empty_rect)

        button_font = pygame.font.Font(None, 26)

        def draw_button(rect: pygame.Rect, label: str) -> None:
            pygame.draw.rect(self.screen, (45, 60, 90), rect)
            pygame.draw.rect(self.screen, (200, 210, 235), rect, 1)
            text = button_font.render(label, True, (255, 255, 255))
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

        close_rect = pygame.Rect(self.width - 170, 30, 130, 40)
        purge_rect = pygame.Rect(self.width - 230, self.height - 80, 200, 44)
        draw_button(close_rect, "Close (G)")
        draw_button(purge_rect, "Purge Tile-less")
        self.graph_button_rects = {
            'close': close_rect,
            'purge': purge_rect,
        }

        if self.graph_status_message:
            status_font = pygame.font.Font(None, 24)
            status_surf = status_font.render(self.graph_status_message, True, (255, 235, 205))
            status_rect = status_surf.get_rect(center=(self.width // 2, self.height - 40))
            self.screen.blit(status_surf, status_rect)
    
    def _cycle_map_mode(self) -> None:
        """Cycle through available map modes."""
        if not hasattr(self, 'world'):
            return
        
        current_mode = self.world.current_map_mode
        modes = self.world.available_map_modes
        
        try:
            current_index = modes.index(current_mode)
            next_index = (current_index + 1) % len(modes)
            self._set_map_mode(modes[next_index])
        except ValueError:
            # Current mode not in list, default to first mode
            self._set_map_mode(modes[0])
    
    def _set_map_mode(self, mode: str) -> None:
        """Set the current map mode.
        
        Args:
            mode: Map mode to set
        """
        if not hasattr(self, 'world') or mode not in self.world.available_map_modes:
            return
        
        self.world.current_map_mode = mode
        
        # Trigger re-coloring in the renderer
        if hasattr(self, 'map_renderer'):
            self.map_renderer.update_tile_colors(self.world)
    
    def _set_map_mode_setting(self, mode: str, option: str) -> None:
        """Set the active sub-setting for the given map mode."""
        world = getattr(self, 'world', None)
        if not world or not hasattr(world, 'set_map_mode_setting'):
            return
        try:
            changed = world.set_map_mode_setting(mode, option)
        except Exception:
            changed = False
        if changed and hasattr(self, 'map_renderer'):
            self.map_renderer.update_tile_colors(world)
    
    def get_mouse_world_position(self) -> Tuple[int, int]:
        """Get current mouse position in world coordinates.
        
        Returns:
            Mouse position in world space
        """
        return self.mouse_position

    def _handle_simulation_controls_click(self, position: Tuple[int, int]) -> bool:
        """Handle clicks on the simulation control buttons."""
        if not hasattr(self, 'world'):
            return False
        if not self.sim_control_rects:
            return False
        pause_rect = self.sim_control_rects.get('pause')
        speed_rect = self.sim_control_rects.get('speed')
        if pause_rect and pause_rect.collidepoint(position):
            self.world.toggle_auto_tick()
            return True
        if speed_rect and speed_rect.collidepoint(position):
            self.world.toggle_fast_tick_speed()
            return True
        return False

    def _render_simulation_controls(self) -> None:
        """Render pause and speed controls for the simulation."""
        if not hasattr(self, 'world'):
            self.sim_control_rects = {}
            return
        panel_width = 230
        panel_height = 80
        margin = 10
        panel_x = self.width - panel_width - margin
        panel_y = margin
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, (10, 10, 15), panel_rect)
        pygame.draw.rect(self.screen, (200, 200, 200), panel_rect, 1)
        title_font = pygame.font.Font(None, 22)
        font = pygame.font.Font(None, 20)
        title = title_font.render("Simulation", True, (255, 255, 255))
        self.screen.blit(title, (panel_x + 10, panel_y + 6))
        running = getattr(self.world, 'auto_tick_enabled', False)
        status_text = "Running" if running else "Paused"
        status_color = (120, 220, 140) if running else (220, 140, 120)
        status_surface = font.render(f"Status: {status_text}", True, status_color)
        self.screen.blit(status_surface, (panel_x + 10, panel_y + 30))
        button_width = (panel_width - 30) // 2
        button_height = 28
        button_y = panel_y + panel_height - button_height - 10
        pause_rect = pygame.Rect(panel_x + 10, button_y, button_width, button_height)
        speed_rect = pygame.Rect(pause_rect.right + 10, button_y, button_width, button_height)

        def draw_button(rect: pygame.Rect, label: str, active: bool = False) -> None:
            base_color = (70, 70, 80)
            active_color = (100, 140, 220)
            fill_color = active_color if active else base_color
            pygame.draw.rect(self.screen, fill_color, rect)
            pygame.draw.rect(self.screen, (220, 220, 220), rect, 1)
            text_surface = font.render(label, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=rect.center)
            self.screen.blit(text_surface, text_rect)

        pause_label = "Pause" if running else "Resume"
        draw_button(pause_rect, pause_label, active=not running)
        fast_active = self.world.is_fast_tick_enabled() if hasattr(self.world, 'is_fast_tick_enabled') else False
        draw_button(speed_rect, "2x Speed", active=fast_active)
        self.sim_control_rects = {
            'panel': panel_rect,
            'pause': pause_rect,
            'speed': speed_rect,
        }
    
    def _handle_map_mode_panel_click(self, position: Tuple[int, int]) -> bool:
        """Handle clicks on the map mode panel.
        
        Args:
            position: Click position
            
        Returns:
            True if click was handled by panel
        """
        if not self.show_map_mode_panel or not hasattr(self, 'world'):
            return False
        if self._handle_map_mode_settings_click(position):
            return True
        
        # Get panel configuration
        panel_config = self.config.get('ui.map_mode_panel', {})
        panel_x = panel_config.get('position', {}).get('x', 10)
        panel_y_config = panel_config.get('position', {}).get('y', 50)
        panel_width = panel_config.get('width', 140)
        panel_height = panel_config.get('height', 180)
        
        # Handle negative y for bottom positioning
        if panel_y_config < 0:
            panel_y = self.height + panel_y_config
        else:
            panel_y = panel_y_config
        
        # Check if click is within panel bounds
        if (panel_x <= position[0] <= panel_x + panel_width and 
            panel_y <= position[1] <= panel_y + panel_height):
            
            # Determine which mode was clicked
            modes = self.world.available_map_modes
            mode_height = (panel_height - 30) // len(modes)  # Account for title space
            click_y_in_panel = position[1] - panel_y - 25  # Account for title
            
            if click_y_in_panel >= 0:
                mode_index = click_y_in_panel // mode_height
                if 0 <= mode_index < len(modes):
                    self._set_map_mode(modes[mode_index])
            
            return True
        
        return False

    def _handle_map_mode_settings_click(self, position: Tuple[int, int]) -> bool:
        """Handle clicks on dynamically rendered map-mode setting buttons."""
        if not self.map_mode_setting_button_rects:
            return False
        for (mode, option), rect in self.map_mode_setting_button_rects.items():
            if rect.collidepoint(position):
                self._set_map_mode_setting(mode, option)
                return True
        return False
    
    def _render_map_mode_panel(self) -> None:
        """Render the map mode selection panel."""
        if not hasattr(self, 'world'):
            return
        self.map_mode_setting_button_rects = {}
        
        # Get panel configuration
        panel_config = self.config.get('ui.map_mode_panel', {})
        panel_x = panel_config.get('position', {}).get('x', 10)
        panel_y_config = panel_config.get('position', {}).get('y', 50)
        panel_width = panel_config.get('width', 140)
        panel_height = panel_config.get('height', 180)
        bg_color = tuple(panel_config.get('background_color', [0, 0, 0, 180]))
        border_color = tuple(panel_config.get('border_color', [255, 255, 255]))
        title_color = tuple(panel_config.get('title_color', [255, 255, 255]))
        text_color = tuple(panel_config.get('text_color', [200, 200, 200]))
        selected_color = tuple(panel_config.get('selected_color', [100, 150, 255]))
        font_size = panel_config.get('font_size', 18)
        
        # Handle negative y for bottom positioning
        if panel_y_config < 0:
            panel_y = self.height + panel_y_config
        else:
            panel_y = panel_y_config
        
        # Create font
        font = pygame.font.Font(None, font_size)
        title_font = pygame.font.Font(None, font_size + 2)
        
        # Draw panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, bg_color[:3], panel_rect)
        pygame.draw.rect(self.screen, border_color, panel_rect, 1)
        
        # Draw title
        title_text = title_font.render("Map Mode", True, title_color)
        title_rect = title_text.get_rect()
        title_x = panel_x + (panel_width - title_rect.width) // 2
        self.screen.blit(title_text, (title_x, panel_y + 5))
        
        # Draw mode options
        modes = self.world.available_map_modes
        mode_height = (panel_height - 30) // len(modes)  # Account for title space
        
        for i, mode in enumerate(modes):
            mode_y = panel_y + 25 + (i * mode_height)
            mode_rect = pygame.Rect(panel_x + 5, mode_y, panel_width - 10, mode_height - 2)
            
            # Highlight selected mode
            if mode == self.world.current_map_mode:
                pygame.draw.rect(self.screen, selected_color, mode_rect)
                current_text_color = (255, 255, 255)
            else:
                current_text_color = text_color
            
            # Draw mode text
            mode_display = mode.replace('_', ' ').title()
            mode_text = font.render(mode_display, True, current_text_color)
            text_x = panel_x + 10
            text_y = mode_y + (mode_height - font.get_height()) // 2
            self.screen.blit(mode_text, (text_x, text_y))
        self._render_map_mode_settings_panel(panel_rect, panel_config)

    def _render_map_mode_settings_panel(self, base_rect: pygame.Rect, panel_config: dict) -> None:
        """Render contextual buttons for map modes that expose variant settings."""
        world = getattr(self, 'world', None)
        if not world or not hasattr(world, 'get_map_mode_setting_options'):
            return
        mode = world.current_map_mode
        options = world.get_map_mode_setting_options(mode)
        if not options:
            return
        settings_margin = panel_config.get('settings_margin', 10)
        settings_width = panel_config.get('settings_width', base_rect.width)
        settings_x = base_rect.right + settings_margin
        settings_y = base_rect.top
        if settings_x + settings_width > self.width:
            settings_x = base_rect.left - settings_margin - settings_width
        settings_x = max(0, settings_x)
        button_height = panel_config.get('settings_button_height', 28)
        button_spacing = panel_config.get('settings_button_spacing', 6)
        padding = 10
        total_button_height = len(options) * button_height + max(0, len(options) - 1) * button_spacing
        bg_color = tuple(panel_config.get('settings_background_color', [8, 8, 12]))
        border_color = tuple(panel_config.get('settings_border_color', [200, 200, 200]))
        selected_color = tuple(panel_config.get('settings_selected_color', [100, 150, 255]))
        button_color = tuple(panel_config.get('settings_button_color', [50, 60, 75]))
        text_color = tuple(panel_config.get('settings_text_color', [230, 230, 230]))
        title_font_size = panel_config.get('settings_title_font_size', 20)
        button_font_size = panel_config.get('settings_font_size', 18)
        title_font = pygame.font.Font(None, title_font_size)
        button_font = pygame.font.Font(None, button_font_size)
        title = world.get_map_mode_setting_title(mode) if hasattr(world, 'get_map_mode_setting_title') else None
        if not title:
            title = f"{mode.title()} Options"
        title_surface = title_font.render(title, True, text_color)
        title_height = title_surface.get_height()
        settings_height = padding * 2 + title_height + total_button_height
        if settings_y + settings_height > self.height - 10:
            settings_y = max(0, self.height - 10 - settings_height)
        settings_rect = pygame.Rect(settings_x, settings_y, settings_width, settings_height)
        pygame.draw.rect(self.screen, bg_color, settings_rect)
        pygame.draw.rect(self.screen, border_color, settings_rect, 1)
        title_x = settings_rect.x + (settings_width - title_surface.get_width()) // 2
        title_y = settings_rect.y + padding // 2
        self.screen.blit(title_surface, (title_x, title_y))
        current_value = world.get_map_mode_setting(mode) if hasattr(world, 'get_map_mode_setting') else None
        button_top = title_y + title_surface.get_height() + padding // 2
        for value, label in options:
            btn_rect = pygame.Rect(
                settings_rect.x + padding,
                button_top,
                settings_width - padding * 2,
                button_height,
            )
            is_active = current_value == value
            fill_color = selected_color if is_active else button_color
            pygame.draw.rect(self.screen, fill_color, btn_rect)
            pygame.draw.rect(self.screen, border_color, btn_rect, 1)
            text_surface = button_font.render(label, True, (0, 0, 0) if is_active else text_color)
            text_pos = text_surface.get_rect(center=btn_rect.center)
            self.screen.blit(text_surface, text_pos)
            self.map_mode_setting_button_rects[(mode, value)] = btn_rect
            button_top += button_height + button_spacing

    def _get_tile_index(self, tile: Optional[Tile]) -> Optional[int]:
        """Return the index of the provided tile within the world list."""
        world = getattr(self, 'world', None)
        if not world or tile is None:
            return None
        try:
            return world.tiles.index(tile)
        except ValueError:
            return None

    def _determine_tile_majority_culture(self, tile: Tile) -> Tuple[Optional[str], float]:
        """Return the dominant culture name and share for a tile."""
        makeup = getattr(tile, 'cultural_makeup', None)
        if isinstance(makeup, dict) and makeup:
            culture_name, share = max(makeup.items(), key=lambda item: item[1])
            try:
                share_val = float(share)
            except (TypeError, ValueError):
                share_val = 0.0
            return culture_name, max(0.0, min(1.0, share_val))
        world = getattr(self, 'world', None)
        polity_id = getattr(tile, 'polity_id', -1)
        if (
            world
            and polity_id is not None
            and 0 <= polity_id < len(world.polities)
        ):
            polity = world.polities[polity_id]
            if polity and getattr(polity, 'primary_culture', None):
                return polity.primary_culture, 0.0
        return None, 0.0

    def _get_population_center_by_tile(self, tile_index: Optional[int]):
        """Find the population center bound to a tile index."""
        world = getattr(self, 'world', None)
        if world is None or tile_index is None:
            return None
        for center in world.population_centers:
            if center.tile_index == tile_index:
                return center
        return None

    def _format_name_history_entry(self, entry: Dict[str, Any]) -> str:
        """Convert a name-history dictionary into a readable line."""
        name = entry.get('name') or 'Unnamed'
        segments: List[str] = []
        year = entry.get('year')
        if isinstance(year, (int, float)):
            segments.append(f"Year {int(year)}")
        elif isinstance(year, str) and year:
            segments.append(f"Year {year}")
        reason = entry.get('reason')
        if reason:
            segments.append(reason.replace('_', ' '))
        language = entry.get('language')
        culture = entry.get('culture')
        if language:
            segments.append(language)
        elif culture:
            segments.append(culture)
        note = entry.get('note')
        if note:
            segments.append(note)
        previous = entry.get('previous')
        if previous and previous != name:
            segments.append(f"from {previous}")
        summary = " · ".join(segments)
        if summary:
            return f"  - {name} ({summary})"
        return f"  - {name}"