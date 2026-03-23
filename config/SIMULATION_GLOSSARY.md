# Simulation Config Glossary

Configuration is now split across three files and optionally aggregated by `config/world_generation.json`:
- `config/simulation.json`: World generation, simulation, population, culture, war, diplomacy, linguistics.
- `config/rendering.json`: Colors, camera, and render-layer settings.
- `config/ui.json`: UI controls and map mode panel settings.
- `config/world_generation.json`: Legacy shim that includes the three files above.

Keep this document in sync whenever fields are added, removed, or renamed.

## simulation.json
- **world**: Dimensions, noise (scale/octaves/persistence/lacunarity/seed), height curve, climate bands (temperature/rainfall), regions, and river carving parameters.
- **generation.verbose_logging**: Enables extra worldgen console output (seed, tectonics, etc.).
- **linguistics**: Catalog directory/seed/time-shift cadence and naming controls for settlements/polities plus word evolution tracking toggles.
	- **polity_naming.pattern_overrides**: Optional per-scenario templates (rank1/2/3 strings) and base-type overrides for the organic naming system; keys correspond to tanistry/primogeniture/election branches described in the design.
	- **polity_naming.special_title_suffixes**: Ordered list of suffixes (e.g., `ate`, `dom`, `host`) used when forging tanistry breakaway titles from dynastic names.
- **simulation.tick_system**: Auto-tick toggle and tick interval (seconds between ticks).
- **simulation.debug**: Profilers for ticks and render, print toggles, and history depth.
- **simulation.logging**: Allowed categories plus per-category flags (e.g., culture_debug, diplomacy_debug, polity_status) to gate console logging noise.
- **simulation.population**: Birth/death rates, climate/rainfall penalties, development death reduction, overcrowding/overcap penalties, thresholds.
- **simulation.development**: Growth/decay rates, population-center thresholds, hysteresis/demotion rules, climate/river bonuses, development cap scaling.
- **simulation.migration**: Destination potential weighting, overcrowding/death penalties, baseline migration, elevation penalties, exodus/major migration, polity expansion triggers, cultural bias.
- **simulation.control**: Assimilation penalty, migration influence, projection rate, border decay, overdevelopment decay, bleed chance.
- **simulation.polity**: Breakaway toggles/thresholds, spawn delays, administrative burden scaling, tanistry fracture odds, rank percentiles.
- **simulation.culture**: Culture spawn cadence, cooldowns, growth, assimilation rates, syncretism thresholds/duration, alignment/tolerance tuning, color variation.
- **simulation.leaders**: Trait system enablement, age/tenure ranges, trait distributions, and per-trait effects (battle, control, diplomacy, assimilation).
- **simulation.war**: Declaration odds growth, supply income/decay, combat strength factors, capture/occupation thresholds, exhaustion math, annexation ratios, victory boosts.
- **simulation.relationships**: Border/culture/region modifiers, dominance thresholds, grievance/warmongering penalties.
- **simulation.diaspora**: Delay before centers form and migration multiplier during the delay window.
- **simulation.initial_generation**: Population/development seeding ranges and elevation thresholds at world creation.

## rendering.json
- **colors**: Water palette and biome color table with temperature/rainfall conditions.
- **camera**: Pan/zoom speeds and zoom bounds.
- **rendering**: FPS cap, window title, label styles for polities and population centers, and river stroke sizing.

## ui.json
- **ui.map_mode_panel**: Toggle, position, size, palette, font size, and available map modes.
- **ui.tooltips**: Hover delay before showing tooltip text.

*Update this glossary whenever configuration schema changes or defaults move between files.*
