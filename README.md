# Etymon

Etymon is a map simulation game focused on world generation and visualization.

## Current Implementation: Phase I + II (Partial)

### Features
- Voronoi-based polygonal tile generation
- Perlin noise elevation mapping with configurable height curves
- **Temperature/rainfall climate system** (Phase II)
- **Biome-based terrain visualization** (Phase II)
- Sea level application for geography
- Real-time configuration reloading
- Camera controls (pan/zoom)
- Interactive tile selection with climate data

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

```bash
python main.py
```

### Controls
- **Space**: Generate new world (reloads config file for real-time tweaking)
- **Right-click + drag**: Pan camera around the world
- **Mouse wheel**: Zoom in/out (zoom around mouse cursor)
- **Left-click**: Select tile for information
- **TAB**: Toggle statistics display  
- **T**: Toggle tile info mode
- **ESC**: Exit application

### Real-time Configuration

You can modify `config/world_generation.json` while the application is running. Press **SPACE** to regenerate the world with your new settings instantly! This allows for:
- Live tweaking of world dimensions, tile count, and sea level
- Real-time adjustment of Perlin noise parameters
- **Climate system tuning**: Adjust equator position, temperature ranges, and rainfall patterns
- **Height curve modification**: Control terrain distribution with `height_curve`:
  - Formula: `coefficient * x^exponent`
  - `1.0 * x^1.0` = Linear distribution (default)
  - `1.5 * x^2.0` = Amplified squared (more contrast, flatter areas + higher peaks)
  - `0.8 * x^0.5` = Dampened square root (gentler mountains)
  - `2.0 * x^1.0` = Amplified linear (stretches elevation range)
  - Coefficient > 1 amplifies heights, < 1 dampens them
  - Exponent > 1 creates flatter worlds, < 1 creates mountainous terrain
- Color scheme changes that apply immediately
- Experimentation with different noise seeds
- **Random seed generation**: Set `"seed": -1` to generate a random seed each time
- Camera control customization (pan speed, zoom speed, zoom limits)

### Configuration

World generation settings can be modified in `config/world_generation.json`.

## Architecture

The project follows a modular design with dependency injection patterns:

- `src/core/`: Core application logic
- `src/world/`: World generation systems
- `src/rendering/`: Visualization and rendering
- `src/ui/`: User interface components
- `config/`: Configuration files
- `assets/`: Static assets (heightmaps, etc.)

## Future Phases

See `DESIGN_DOCUMENT.md` for planned features in subsequent implementation phases.
