#!/usr/bin/env python3
"""
Etymon - A map simulation game focused on world generation and visualization.

Implementation Phase I: Basic world generation with Voronoi tiles and Perlin noise.
"""

import sys
import pygame
from src.core.application import Application
from src.core.config_manager import ConfigManager


def main():
    """Main entry point for the application."""
    try:
        # Initialize configuration
        config = ConfigManager()
        
        # Initialize and run the application
        app = Application(config)
        app.run()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)
    finally:
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
