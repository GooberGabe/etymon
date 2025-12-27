"""Main application class for Etymon."""

import pygame
import sys
from typing import Optional

from src.core.config_manager import ConfigManager
from src.world.world_generator import WorldGenerator
from src.world.world_save_manager import WorldSaveManager
from src.rendering.map_renderer import MapRenderer
from src.ui.ui_manager import UIManager


class Application:
    """Main application class that orchestrates all systems."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the application.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.running = True
        
        # Initialize pygame
        pygame.init()
        
        # Get configuration
        self.width = config.get('world.width', 800)
        self.height = config.get('world.height', 600)
        self.fps = config.get('rendering.fps', 60)
        self.title = config.get('rendering.window_title', 'Etymon')
        
        # Initialize display
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)
        self.clock = pygame.time.Clock()
        
        # Initialize systems using dependency injection pattern
        self.world_generator = WorldGenerator(config)
        self.save_manager = WorldSaveManager(config)
        self.map_renderer = MapRenderer(config, self.screen)
        self.ui_manager = UIManager(config, self.screen)
        
        # Set up cross-references for camera controls
        self.ui_manager.set_map_renderer_reference(self.map_renderer)
        
        # Generate initial world
        self.current_world = None
        self.generate_world()
    
    def generate_world(self) -> None:
        """Generate a new world or load from disk when configured."""
        self.config.reload_config()
        loaded_save = self.config.get('simulation.loaded_save')
        if loaded_save:
            try:
                self.current_world = self.save_manager.load_world(loaded_save)
            except Exception as exc:
                print(f"[save] Failed to load '{loaded_save}': {exc}. Generating a new world instead.")
                self.current_world = self.world_generator.generate_world()
        else:
            self.current_world = self.world_generator.generate_world()

        # Pass world reference to UI manager
        self.ui_manager.set_world_reference(self.current_world)

    def save_current_world(self, save_name: Optional[str] = None) -> None:
        """Persist the currently running world to disk."""
        if not self.current_world:
            print("[save] No active world to persist.")
            return
        try:
            self.save_manager.save_world(self.current_world, save_name)
        except Exception as exc:
            print(f"[save] Failed to save world: {exc}")
    
    def handle_events(self) -> None:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.generate_world()
                elif event.key == pygame.K_RETURN:  # Enter key to advance tick
                    if self.current_world:
                        self.current_world.advance_tick()
                elif event.key == pygame.K_F5:
                    self.save_current_world()
            
            # Pass events to UI manager
            self.ui_manager.handle_event(event)
    
    def update(self) -> None:
        """Update application state."""
        self.ui_manager.update()
        
        # Update auto-tick system
        if self.current_world:
            import time
            current_time = time.time()
            if not hasattr(self.current_world, 'last_tick_time'):
                self.current_world.last_tick_time = current_time
            self.current_world.update_auto_tick(current_time)
    
    def render(self) -> None:
        """Render the current frame."""
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Render world if available
        if self.current_world:
            self.map_renderer.render_world(self.current_world)
            
            # Render statistics if UI says to show them
            if self.ui_manager.show_statistics:
                self.map_renderer.render_statistics(self.current_world)
            
            # Render current map mode status
            self.map_renderer.render_map_mode_status(self.current_world)
            
            # Render simulation date
            self.map_renderer.render_simulation_date(self.current_world)
        
        # Render UI
        self.ui_manager.render()
        
        # Update display
        pygame.display.flip()
    
    def run(self) -> None:
        """Main application loop."""
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(self.fps)
        pygame.quit()
