# Heightmaps Directory

This directory contains optional heightmap images for world generation.

## Usage

- Place grayscale PNG images in this directory
- Black pixels (0) represent minimum elevation
- White pixels (255) represent maximum elevation
- If this directory is empty, the system will generate terrain using Perlin noise

## File Format

- **Format**: PNG
- **Color Mode**: Grayscale
- **Dimensions**: Should match or be scalable to world dimensions in config

## Naming Convention

- `heightmap.png` - Default heightmap file
- Multiple files can be added for variety