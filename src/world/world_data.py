from typing import List, Tuple, Optional, Dict, Set, FrozenSet, Any, Sequence, Union, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter, deque, defaultdict
import numpy as np
import random
import time
import math
import numpy as np
import itertools
import traceback
import unicodedata
import hashlib
import colorsys

from src.linguistics import (
    LanguageCatalog,
    Typology,
    Word,
    load_all_catalogs,
    enforce_phonotactics_form,
    simplify_orthography,
    transform_language,
    build_compound,
    build_compound_with_gloss,
)
IPA_TO_ASCII: Dict[str, str] = {
    "æ": "ae",
    "ɐ": "a",
    "ə": "e",
    "ɚ": "er",
    "ɝ": "er",
    "ɛ": "e",
    "ɜ": "e",
    "ɞ": "oe",
    "ɤ": "o",
    "ɯ": "u",
    "ɪ": "i",
    "ɨ": "i",
    "ʏ": "y",
    "ʊ": "u",
    "ɔ": "o",
    "ʌ": "u",
    "θ": "th",
    "ð": "th",
    "ŋ": "ng",
    "ɲ": "ny",
    "ɳ": "n",
    "ɱ": "m",
    "ɴ": "n",
    "ʃ": "sh",
    "ʒ": "zh",
    "ʂ": "sh",
    "ʐ": "zh",
    "ɕ": "sh",
    "ʑ": "zh",
    "ç": "ch",
    "ʝ": "y",
    "ʎ": "y",
    "ɟ": "j",
    "ɡ": "g",
    "ɢ": "g",
    "ɠ": "g",
    "ʛ": "g",
    "ɓ": "b",
    "ɗ": "d",
    "ʄ": "j",
    "ɸ": "f",
    "β": "b",
    "ʋ": "v",
    "ɣ": "g",
    "χ": "kh",
    "ɾ": "r",
    "ɽ": "r",
    "ɹ": "r",
    "ɻ": "r",
    "ɭ": "l",
    "ʔ": "",
    "ʰ": "h",
    "ʲ": "y",
    "ʷ": "w",
    "ˠ": "g"
}

IPA_SILENT_MARKERS: Set[str] = {
    "ː",
    "ˑ",
    "ˈ",
    "ˌ",
    "˞",
    "ˤ",
    "ˀ",
    "ˁ",
    "˳",
    "‿"
}

REGION_NAME_CATEGORIES: Tuple[str, ...] = (
    "geographic",
    "natural_world",
    "settlement",
    "military",
    "numerical",
)

REGION_GLOSS_PREFIX = "region:"

SETTLEMENT_NAME_CATEGORIES: Tuple[str, ...] = (
    "settlement",
    "geographic",
    "natural_world",
    "military",
)

# Enhanced settlement naming patterns for realism
SETTLEMENT_TYPE_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "river": ("settlement", "geographic", "natural_world"),  # River settlements often named after water features
    "coastal": ("settlement", "geographic", "natural_world"),  # Coastal settlements often named after harbors/bays
    "mountain": ("geographic", "natural_world", "settlement"),  # Mountain settlements often named after peaks/valleys
    "hill": ("geographic", "natural_world", "settlement"),  # Hill settlements often named after elevations
    "plain": ("settlement", "military", "geographic"),  # Plain settlements often named after people/events
    "forest": ("natural_world", "geographic", "settlement"),  # Forest settlements often named after trees/woods
    "desert": ("geographic", "natural_world", "settlement"),  # Desert settlements often named after oases/water
    "capital": ("settlement", "military", "geographic"),  # Capitals often named after founders/rulers
}

SETTLEMENT_COMPOUND_PATTERNS: Tuple[str, ...] = (
    "adjective_noun",  # e.g., "New London", "High Castle"
    "noun_adjective",  # e.g., "London New", "Castle High"
    "noun_genitive",   # e.g., "King's Landing", "River's Bend"
    "compound_noun",   # e.g., "Waterford", "Springfield"
)

# Enhanced polity naming patterns for more variation
POLITY_TYPE_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "duchy": ("geographic", "settlement", "military"),     # Duchies often named after territories/cities
    "kingdom": ("geographic", "military", "settlement"),   # Kingdoms often named after territories/cities
    "empire": ("military", "geographic", "natural_world"), # Empires often named after conquests/lands
}

CARDINAL_NAME_DESCRIPTORS: Tuple[str, ...] = (
    "North",
    "South",
    "East",
    "West",
    "Upper",
    "Lower",
    "High",
    "Low",
    "Central",
)

DEFAULT_POLITY_NAMING_PATTERNS: Dict[str, Dict[str, Any]] = {
    "itinerant_pluralist": {
        "base_type": "cultural",
        "templates": ("{base_adjective} Tribes", "{base_adjective} Tribes", "{base_adjective} Confederation"),
    },
    "itinerant_orthodox": {
        "base_type": "cultural",
        "templates": ("{base} Clan", "{base_adjective} Horde", "{base_adjective} Empire"),
    },
    "tanistry_pluralist": {
        "base_type": "dynastic",
        "templates": ("{base} Clan", "House {base}", "House {base}"),
    },
    "tanistry_orthodox": {
        "base_type": "cultural",
        "templates": ("{base} Clan", "{base_adjective} Kingdom", "Crown of {base}"),
    },
    "successor_state": {
        "base_type": "cultural",
        "templates": ("{base} {title}", "{base_adjective} {title}", "{base_adjective} Empire"),
        "requires_title": True,
        "title_type": "dynastic_loan",
    },
    "primogeniture_pluralist": {
        "base_type": "cultural",
        "templates": ("Duchy of {base}", "Kingdom of {base}", "{base_adjective} Empire"),
    },
    "primogeniture_divine_right": {
        "base_type": "cultural",
        "templates": ("Duchy of {base}", "Kingdom of {base}", "{base_adjective} Empire"),
    },
    "primogeniture_mandate": {
        "base_type": "dynastic",
        "templates": ("{base} Dynasty", "{base} Dynasty", "{base} Dynasty"),
    },
    "primogeniture_military": {
        "base_type": "cultural",
        "templates": ("{base}", "{base}", "{base_adjective} Empire"),
    },
    "primogeniture_other": {
        "base_type": "cultural",
        "templates": ("State of {base}", "{base_adjective} Federation", "{base_adjective} Empire"),
    },
    "election_popular": {
        "base_type": "regional",
        "templates": ("Republic of {base}", "Republic of {base}", "Grand Republic of {base}"),
    },
    "election_other": {
        "base_type": "dynastic",
        "templates": ("House {base}", "{base} Dynasty", "{base} Empire"),
    },
    "vassal_imperial": {
        "base_type": "regional",
        "templates": (
            "{overlord_adjective} {base}",
            "{overlord_adjective} {base}",
            "{overlord_adjective} {base}",
        ),
    },
    "fallback": {
        "base_type": "cultural",
        "templates": ("Duchy of {base}", "Kingdom of {base}", "{base_adjective} Empire"),
    },
}

COUNTABLE_VOWELS = set("aeiouy")

CULTURE_NAME_CATEGORIES: Tuple[str, ...] = (
    "kinship",
    "personal_name",
    "military",
    "settlement",
)

CULTURE_NAME_DESCRIPTORS: Tuple[str, ...] = (
    "Tribe",
    "Tribes",
    "Folk",
    "Kindred",
    "People",
    "League",
    "Confederacy",
    "Circle",
    "Lineage",
)

CULTURE_NAME_SUFFIXES: Tuple[str, ...] = (
    "ian",
    "i",
    "ite",
    "ean",
    "an",
    "en",
    "ish",
)

# Cultural concern schema describing idea spectra (0.0=leftmost idea, 1.0=rightmost idea)

CULTURAL_CONCERNS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "succession",
        "label": "Succession Patterns",
        "ideas": ("Election", "Primogeniture", "Tanistry"),
        "initial_range": (0.5, 1.0),
    },
    {
        "key": "legitimacy",
        "label": "Source of Legitimacy",
        "ideas": ("Popular Consent", "Mandate of Heaven", "Divine Right", "Military Acclamation"),
        "initial_range": (0.25, 1.0),
    },
    {
        "key": "universalism",
        "label": "Universalism vs Plurality",
        "ideas": ("Universalist", "Hybrid", "Pluralist"),
        "initial_range": (0, 1.0),
    },
    {
        "key": "kinship_state",
        "label": "Kinship and the State",
        "ideas": ("Meritocratic", "Oligarchic", "Aristocratic"),
        "initial_range": (0.0, 1.0),
    },
    {
        "key": "itinerance",
        "label": "Itinerance (Sedentary vs Migratory)",
        "ideas": ("Sedentary", "Mixed", "Migratory"),
        "initial_range": (0.0, 1.0),
    },
)

POLITY_COMPONENT_CATEGORIES: Tuple[str, ...] = (
    "geographic",
    "natural_world",
    "settlement",
    "military",
)

DYNASTY_TITLE_CATEGORIES: Tuple[str, ...] = (
    "dynasty_title",
    "noble_title",
)

TOLERANCE_BASE_RATE = 0.005
TOLERANCE_MAX_RATE = 0.0125
TOLERANCE_DIVERSITY_REFERENCE = 6
TOLERANCE_DIVERSITY_NEUTRAL = 0.4
TOLERANCE_WAR_PENALTY_STEP = 0.4
TOLERANCE_VARIANCE = 0.15

MAP_MODE_SETTING_DEFINITIONS: Dict[str, List[Tuple[str, str]]] = {
    "cultures": [
        ("mix", "Culture Mix"),
        ("polity_primary", "Polity Primary"),
        ("majority", "Tile Majority"),
    ],
    "control": [
        ("control", "Control Level"),
        ("tolerance", "Tolerance Level"),
        ("breakaway", "Breakaway Risk"),
    ],
}

MAP_MODE_SETTING_TITLES: Dict[str, str] = {
    "cultures": "Culture View",
    "control": "Control Overlay",
}

FALLBACK_CONSONANTS: Tuple[str, ...] = (
    "p",
    "t",
    "k",
    "m",
    "n",
    "s",
    "l",
    "r",
)

FALLBACK_VOWELS: Tuple[str, ...] = (
    "a",
    "e",
    "i",
    "o",
    "u",
)

# Leader trait system defaults
LEADER_TRAIT_POOL: Tuple[str, ...] = (
    "CHARISMATIC",
    "TACTICAL_GENIUS",
    "WARMONGER",
    "PEACEMAKER",
    "DIPLOMAT",
    "TOLERANT",
    "TRADITIONAL",
    "ARCHITECT",
    "JUST",
    "FRUGAL",
    "MACHIAVELLIAN",
)

LEADER_TRAIT_CONFLICTS: Dict[str, Set[str]] = {
    "WARMONGER": {"PEACEMAKER"},
    "PEACEMAKER": {"WARMONGER"},
}


@dataclass
class Leader:
    """Represents a leader of a polity."""
    
    name: str                           # Leader's name
    age: int                            # Leader's age
    culture: str                        # Leader's culture (placeholder for now)
    dynasty: Optional[str] = None       # Dynastic/house label
    traits: List[str] = field(default_factory=list)  # Optional leader traits
    accession_year: int = 0             # Year leader assumed power
    term_years: int = 0                 # Planned tenure length
    

@dataclass
class Polity:
    """Represents a political entity that controls tiles."""
    
    id: int                             # Unique polity identifier
    name: str                           # Polity name
    color: Tuple[int, int, int]         # Primary polity color for borders
    name_gloss: Optional[str] = None    # Optional semantic gloss for the polity name
    leader: Optional[Leader] = None     # Current leader
    primary_culture: Optional[str] = None  # Dominant culture (assigned later when unknown)
    tile_indices: List[int] = field(default_factory=list)  # Indices of controlled tiles
    suzerain_id: int = -1               # ID of parent polity (-1 = independent)
    vassal_ids: List[int] = None        # IDs of vassal polities
    integration_level: int = 100        # Integration level if vassal (1-100)
    is_active: bool = True              # False when polity is eradicated/removed
    capital_tile_index: int = -1        # Tile index of the polity's capital
    leader_generation: int = 0          # Number of leaders appointed so far
    title_rank: Optional[str] = None    # Rank descriptor derived from development percentile
    language_name_component: Optional[str] = None  # Stable lexeme from culture catalog
    name_from_language: bool = False    # True once culture catalog determined the polity name
    naming_bases: Dict[str, str] = field(default_factory=dict)  # Cached base names per scheme
    naming_pattern_key: Optional[str] = None  # Last applied naming pattern identifier
    cultural_tolerance: float = 0.5     # 0=exclusive/xenophobic, 1=highly tolerant
    name_history: List[Dict[str, Any]] = field(default_factory=list)
    dynasty_title_token: Optional[str] = None  # Active dynastic-loan title when applicable
    dynasty_title_source: Optional[str] = None  # catalog vs minted
    dynasty_title_culture: Optional[str] = None  # Culture that supplied the current title
    spawned_from_explosion: bool = False  # True when polity emerged from a polity explosion
    spawned_from_succession_crisis: bool = False  # True when polity emerged from a succession crisis
    
    def __post_init__(self):
        """Initialize derived properties."""
        if self.tile_indices is None:
            self.tile_indices = []
        if self.vassal_ids is None:
            self.vassal_ids = []
        if self.naming_bases is None:
            self.naming_bases = {}
        try:
            tolerance = float(self.cultural_tolerance)
        except (TypeError, ValueError):
            tolerance = 0.5
        self.cultural_tolerance = max(0.0, min(1.0, tolerance))


@dataclass
class Relationship:
    """Represents the diplomatic state between two polities."""

    polity_a: int
    polity_b: int
    status: str = "peace"
    war_start_year: Optional[int] = None
    last_war_end_year: Optional[int] = None
    last_status_change_tick: int = 0
    met: bool = False
    shared_border_tiles: int = 0
    ticking_modifiers: Dict[int, float] = field(default_factory=dict)
    war_exhaustion: Dict[int, float] = field(default_factory=dict)
    war_exhaustion_anchor_values: Dict[int, float] = field(default_factory=dict)
    war_exhaustion_anchor_years: Dict[int, int] = field(default_factory=dict)
    truce_until_year: Optional[int] = None
    occupied_tiles: Dict[int, Set[int]] = field(default_factory=dict)

    def involves(self, polity_id: int) -> bool:
        return polity_id == self.polity_a or polity_id == self.polity_b

    def other(self, polity_id: int) -> Optional[int]:
        if polity_id == self.polity_a:
            return self.polity_b
        if polity_id == self.polity_b:
            return self.polity_a
        return None


@dataclass
class Culture:
    """Represents a culture with heritage tracking."""
    
    name: str                           # Culture name
    color: Tuple[int, int, int]         # Display color for culture view
    heritage: Dict[str, float]          # Cultural heritage composition {parent_culture: percentage}
    origin_tile_index: int              # Tile index where this culture originated
    birth_year: int                     # Year when this culture was created
    home_region_id: int = -1            # Region identifier where the culture originated
    immunity_end_year: Optional[int] = None  # Year when assimilation immunity ends
    is_initial: bool = False            # True for starting scenario cultures
    language_name: Optional[str] = None
    language_parent: Optional[str] = None
    language_last_update_year: Optional[int] = None
    language_time_depth: int = 0
    language_transformation: Optional[str] = None
    ideas: Dict[str, float] = field(default_factory=dict)  # Cultural concern spectra (0-1)
    
    def __post_init__(self):
        """Initialize derived properties."""
        if not self.heritage:
            self.heritage = {}
        if self.ideas is None:
            self.ideas = {}


@dataclass
class PopulationCenter:
    """Represents a persistent population center."""

    tile_index: int
    name: str
    original_threshold: int
    established_year: int
    established_tick: int
    phonetic_name: Optional[str] = None  # Phonetic form for language transformation
    low_control_ticks: int = 0
    name_history: List[Dict[str, Any]] = field(default_factory=list)
    demotion_grace_ticks: int = 0
    demotion_buffer_ticks: int = 0


@dataclass
class Tile:
    """Represents a single polygonal tile in the world."""
    
    # Geometric properties
    vertices: List[Tuple[float, float]]  # Polygon vertices
    center: Tuple[float, float]          # Center point
    
    # Physical properties
    elevation: float                     # Height value (0.0 = sea level)
    is_water: bool                       # Whether tile is below sea level
    
    # Visual properties
    color: Tuple[int, int, int]         # RGB color for rendering
    
    # Connectivity
    neighbors: List[int]                 # Indices of neighboring tiles
    
    # Climate properties (Phase II) - must have defaults since they come after required fields
    temperature: float = 0.0             # Temperature value (0.0 = coldest, 1.0 = hottest)
    rainfall: float = 0.0                # Rainfall value (0.0 = driest, 1.0 = wettest)
    biome: str = "unknown"               # Biome name
    region_id: int = -1                  # Region ID this tile belongs to (-1 = unassigned)
    
    # Political properties (Phase III)
    polity_id: int = -1                  # Polity ID that controls this tile (-1 = uncontrolled)
    control_level: int = 50              # Local loyalty/control level (1-100)
    population: int = 0                  # Population in this tile
    development: float = 0.0             # Development level (0.0 = undeveloped, 1.0 = highly developed)
    cultural_makeup: Dict[str, float] = field(default_factory=dict)  # Cultural composition {culture_name: percentage}
    control_debug: Dict[str, Any] = field(default_factory=dict)  # Per-tile control modifier diagnostics
    
    # Simulation properties
    last_tick_deaths: int = 0            # Number of deaths in the last tick (for migration calculations)
    last_culture_spawn_year: Optional[int] = None  # Most recent year a culture spawned here
    last_war_supply_tick: int = -1       # Simulation tick when war supply last reinforced this tile
    occupied_by_polity_id: int = -1      # Current occupying polity (-1 = none)
    occupation_since_tick: int = -1      # Simulation tick when occupation began
    occupation_relation: Optional[Tuple[int, int]] = None  # Relationship key responsible for occupation
    temporary_control_bonus: float = 0.0  # Temporary control bonus from war victories
    major_migration_cooldown: int = 0    # Ticks to wait before rerunning major migration checks
    river_ids: List[int] = field(default_factory=list)  # Rivers that traverse this tile
    river_flux: float = 0.0              # Accumulated river flow strength through the tile
    river_neighbors: Dict[int, float] = field(default_factory=dict)  # Neighbor tile flux along shared river edges
    is_river_lake: bool = False          # True when the tile is a terminal lake for local drainage
    
    def __post_init__(self):
        """Initialize derived properties."""
        if not self.neighbors:
            self.neighbors = []
        if self.cultural_makeup is None:
            self.cultural_makeup = {}


@dataclass
class River:
    """Represents a generated river polyline across multiple tiles."""

    id: int
    tile_indices: List[int]
    points: List[Tuple[float, float]]
    flux: float
    terminates_in_sea: bool = False
    terminates_in_lake: bool = False

    def __post_init__(self) -> None:
        if not self.tile_indices:
            self.tile_indices = []
        if not self.points:
            self.points = []


@dataclass
class Region:
    """Represents a geographical region grouping multiple tiles."""

    id: int
    name: str
    color: Tuple[int, int, int]
    tile_indices: List[int]
    center_tile_index: int = -1

    def __post_init__(self) -> None:
        if not self.tile_indices:
            self.tile_indices = []


class World:
    """Represents the complete generated world."""
    
    def __init__(self, width: int, height: int, config=None):
        """Initialize world.
        
        Args:
            width: World width in pixels
            height: World height in pixels
            config: Configuration manager for simulation settings
        """
        self.width = width
        self.height = height
        self.tiles: List[Tile] = []
        self.sea_level = 0.0
        self.config = config
        self.rivers: List[River] = []
        self.river_lakes: Set[int] = set()
        self._culture_concern_map: Dict[str, Dict[str, Any]] = {entry["key"]: entry for entry in CULTURAL_CONCERNS}
        
        # Map modes and regions
        self.regions: List['Region'] = []   # List of regions
        self.polities: List['Polity'] = []  # List of polities
        self.cultures: List['Culture'] = [] # List of cultures
        self._culture_name_registry: Set[str] = set()
        self._pending_culture_name_reservations: Set[str] = set()
        self._region_seed_words: Dict[int, Word] = {}
        self.current_map_mode = "biomes"    # Current map view mode (biomes/regions/development/population/control)
        self.available_map_modes = [
            "biomes",
            "elevation",
            "regions",
            "development",
            "population",
            "control",
            "cultures",
            "linguistics",
            "war",
        ]
        self._map_mode_settings: Dict[str, str] = {}
        for mode, options in MAP_MODE_SETTING_DEFINITIONS.items():
            default_value = self._resolve_map_mode_default_setting(mode, options)
            if default_value:
                self._map_mode_settings[mode] = default_value
        
        # Simulation state
        self.current_year = 1
        self.current_season = 0  # 0=Spring, 1=Summer, 2=Fall, 3=Winter
        self.season_names = ["Spring", "Summer", "Fall", "Winter"]
        self.ticks_per_year = len(self.season_names)
        self.total_ticks = 0
        self.population_centers: List[PopulationCenter] = []
        self.world_seed = config.get('world.noise.seed') if config else None
        
        # Auto-tick system (2 ticks per second)
        self.auto_tick_enabled = config.get('simulation.tick_system.auto_tick_enabled', True) if config else True
        self.tick_interval = config.get('simulation.tick_system.tick_interval', 0.5) if config else 0.5
        self.last_tick_time = 0.0
        self.default_tick_interval = self.tick_interval
        self.tick_speed_multiplier = 1.0
        self.syncretism_tracker: Dict[Tuple[int, Tuple[str, str]], int] = {}
        self.syncretic_cultures: Dict[FrozenSet[str], str] = {}
        self._syncretic_parent_lookup: Dict[str, FrozenSet[str]] = {}
        self.region_name_tokens: Dict[int, str] = {}
        self._region_placeholder_names: Dict[int, str] = {}
        self.relationships: List[Relationship] = []
        self.relationship_lookup: Dict[Tuple[int, int], Relationship] = {}
        self.relationship_tick_interval = 1
        raw_relationship_buckets = config.get('simulation.relationships.yearly_check_buckets', self.ticks_per_year) if config else self.ticks_per_year
        try:
            relationship_buckets = int(raw_relationship_buckets)
        except (TypeError, ValueError):
            relationship_buckets = self.ticks_per_year
        self.relationship_yearly_check_buckets = max(1, relationship_buckets)
        raw_relationship_year_cadence = config.get('simulation.relationships.yearly_check_interval_years', 1) if config else 1
        try:
            relationship_year_cadence = int(raw_relationship_year_cadence)
        except (TypeError, ValueError):
            relationship_year_cadence = 1
        self.relationship_yearly_check_interval_years = max(1, relationship_year_cadence)
        raw_cold_war_stride = config.get('simulation.war.cold_war_stride_ticks', self.ticks_per_year) if config else self.ticks_per_year
        try:
            cold_war_stride = int(raw_cold_war_stride)
        except (TypeError, ValueError):
            cold_war_stride = self.ticks_per_year
        self.cold_war_stride_ticks = max(1, cold_war_stride)
        raw_hot_war_window = config.get('simulation.war.hot_war_recent_years', 2) if config else 2
        try:
            hot_war_window = int(raw_hot_war_window)
        except (TypeError, ValueError):
            hot_war_window = 2
        self.hot_war_recent_years = max(1, hot_war_window)
        self._relationship_borders_initialized = False
        self.polity_war_supply: Dict[int, float] = {}
        self.polity_administrative_burden: Dict[int, int] = {}
        self._mandate_region_event_years: Dict[int, int] = {}
        self.frontline_supply: Dict[Tuple[int, int], float] = {}
        self.capture_cooldowns: Dict[int, int] = {}
        self.capture_progress: Dict[Tuple[int, int], float] = {}
        self.language_catalog_dir: Optional[Path] = None
        self._language_rng = random.Random()
        self._language_rng_seed: Optional[int] = None
        self._language_pool: Dict[str, LanguageCatalog] = {}
        self._unused_language_names: List[str] = []
        self.culture_languages: Dict[str, LanguageCatalog] = {}
        self._language_initialized = False
        self._culture_dynasty_style_cache: Dict[str, str] = {}
        self._culture_dynasty_title_reservations: Dict[str, Dict[str, Any]] = {}
        self._culture_adjective_cache: Dict[str, str] = {}
        raw_interval = self.config.get('linguistics.time_shift_years', 200) if self.config else 200
        try:
            interval_val = int(raw_interval)
        except (TypeError, ValueError):
            interval_val = 200
        self.language_time_shift_years = max(1, interval_val)
        self.culture_population_threshold = (
            self.config.get('simulation.culture.min_population_for_culture', 150)
            if self.config else 150
        )

        default_log_categories = ["tick", "polity", "culture", "war", "battle"]
        allowed_categories = None
        if config:
            allowed_categories = config.get('simulation.logging.allowed_categories')
        if not allowed_categories:
            normalized_categories = set(default_log_categories)
        elif isinstance(allowed_categories, str):
            normalized_categories = {allowed_categories}
        else:
            normalized_categories = set(allowed_categories)
        if config:
            category_flags = config.get('simulation.logging.category_flags')
            if isinstance(category_flags, dict):
                for cat, enabled in category_flags.items():
                    if not cat:
                        continue
                    if bool(enabled):
                        normalized_categories.add(cat)
                    else:
                        normalized_categories.discard(cat)
        self._log_category_defaults = set(normalized_categories)
        self._log_categories = set(normalized_categories)
        self.polity_development_history: Dict[int, List[Dict[str, Any]]] = {}
        profiling_default = config.get('simulation.debug.profile_ticks', False) if config else False
        history_size = config.get('simulation.debug.profile_tick_history', 60) if config else 60
        try:
            history_size = max(1, int(history_size))
        except (TypeError, ValueError):
            history_size = 60
        self.tick_profiling_enabled = bool(profiling_default)
        self.tick_profile_print = bool(config.get('simulation.debug.profile_ticks_print', True)) if config else True
        summary_interval = config.get('simulation.debug.profile_tick_summary_year_interval', 25) if config else 25
        try:
            summary_interval = max(1, int(summary_interval))
        except (TypeError, ValueError):
            summary_interval = 25
        self.tick_profile_summary_interval_years = summary_interval
        summary_top_n = config.get('simulation.debug.profile_tick_summary_top_n', 3) if config else 3
        try:
            summary_top_n = max(1, int(summary_top_n))
        except (TypeError, ValueError):
            summary_top_n = 3
        self.tick_profile_summary_top_n = summary_top_n
        self.tick_profile_summary_print = bool(config.get('simulation.debug.profile_tick_summary_print', True)) if config else True
        self._tick_profile_last_summary_year = 0
        self._tick_profile_history: deque = deque(maxlen=history_size)
        self._tick_profile_sections: Dict[str, float] = {}
        self._tick_profile_start_time = 0.0
        self._tick_workload_counters: Dict[str, int] = {}
        self._relation_eval_cache: Optional[Dict[str, Dict[Any, Any]]] = None
        self._war_frontline_pair_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._war_frontline_polity_versions: Dict[int, int] = {}
        self._focus_filter_polity_id: Optional[int] = None
        self._focus_filter_leader_name: Optional[str] = None
        self._focus_filter_terms: Set[str] = set()
        
        # Statistics
        self.water_tiles = 0
        self.land_tiles = 0
        self._coastal_tile_cache: List[int] = []
        self._coastal_tile_set: Set[int] = set()
        self._coastal_cache_ready = False

    # ------------------------------------------------------------------
    # Cultural ideas: initialization, inheritance, and paired language shifts
    # ------------------------------------------------------------------
    def _culture_concern_defs(self) -> Dict[str, Dict[str, Any]]:
        return self._culture_concern_map

    def _idea_label_for_value(self, concern_key: str, value: float) -> Optional[str]:
        concern = self._culture_concern_map.get(concern_key)
        if not concern:
            return None
        ideas = list(concern.get('ideas') or [])
        if not ideas:
            return None
        clamped = max(0.0, min(1.0, float(value)))
        # Bias toward middle ideas when only three options exist so edges occupy less space
        if len(ideas) == 3:
            if clamped < 0.20:
                return ideas[0]
            if clamped >= 0.75:
                return ideas[2]
            return ideas[1]
        if len(ideas) == 4:
            if clamped < 0.2:
                return ideas[0]
            if clamped > 0.8:
                return ideas[3]
            if clamped < 0.5:
                return ideas[1]
            return ideas[2]
        bucket = min(len(ideas) - 1, int(clamped * len(ideas)))
        return ideas[bucket]

    def _random_initial_idea_value(self, concern_key: str) -> float:
        concern = self._culture_concern_map.get(concern_key, {})
        low, high = concern.get('initial_range', (0.0, 1.0))
        try:
            low_f = float(low)
            high_f = float(high)
        except (TypeError, ValueError):
            low_f, high_f = 0.5, 1.0
        if high_f < low_f:
            high_f = low_f
        return max(0.0, min(1.0, random.uniform(low_f, high_f)))

    def _ensure_culture_ideas(
        self,
                culture: Optional[Culture],
                heritage_override: Optional[Dict[str, float]] = None,
                *,
                jitter: float = 0.0,
            ) -> None:
        if culture is None:
            return
        if getattr(culture, 'ideas', None) is None:
            culture.ideas = {}
        concern_defs = self._culture_concern_defs()
        heritage = heritage_override if heritage_override is not None else getattr(culture, 'heritage', {}) or {}
        parent_mix: Dict[str, float] = dict(heritage)
        if not parent_mix:
            parents = self._get_syncretic_parents_for(getattr(culture, 'name', None))
            if parents:
                share = 1.0 / max(1, len(parents))
                parent_mix = {parent: share for parent in parents}
        for key, concern in concern_defs.items():
            if key in culture.ideas:
                continue
            weighted_value = 0.0
            total_weight = 0.0
            for parent_name, share in parent_mix.items():
                parent = self._find_culture_by_name(parent_name)
                if parent is None:
                    continue
                self._ensure_culture_ideas(parent)
                if parent.ideas and key in parent.ideas:
                    weight = max(0.0, float(share))
                    weighted_value += parent.ideas[key] * weight
                    total_weight += weight
            if total_weight > 0:
                base_value = weighted_value / total_weight
            else:
                base_value = self._random_initial_idea_value(key)
            if jitter > 0.0:
                base_value += random.uniform(-jitter, jitter)
            culture.ideas[key] = max(0.0, min(1.0, base_value))

    def _get_culture_idea_label(self, culture_name: Optional[str], concern_key: str) -> Optional[str]:
        if not culture_name:
            return None
        culture = self._find_culture_by_name(culture_name)
        if culture is None:
            return None
        self._ensure_culture_ideas(culture)
        value = culture.ideas.get(concern_key)
        if value is None:
            return None
        return self._idea_label_for_value(concern_key, value)

    def _get_polity_idea_label(self, polity: Optional['Polity'], concern_key: str) -> Optional[str]:
        """Convenience wrapper to fetch a polity's primary culture idea label."""
        if polity is None:
            return None
        return self._get_culture_idea_label(getattr(polity, 'primary_culture', None), concern_key)

    def _get_polity_idea_label_lower(self, polity: Optional['Polity'], concern_key: str) -> str:
        """Return a normalized lowercase idea label for the given concern."""
        return (self._get_polity_idea_label(polity, concern_key) or '').lower()

    def _polity_has_idea_label(self, polity: Optional['Polity'], concern_key: str, label_fragment: str) -> bool:
        """Return True when the polity's idea label for a concern contains the target fragment."""
        fragment = (label_fragment or '').strip().lower()
        if not fragment:
            return False
        return fragment in self._get_polity_idea_label_lower(polity, concern_key)

    def _backfill_culture_ideas(self) -> None:
        for culture in self.cultures:
            self._ensure_culture_ideas(culture)

    def _shift_culture_ideas(
        self,
        culture_name: Optional[str],
        reason: str,
        intensity: float = 0.1,
    ) -> None:
        """Nudge cultural ideas in small steps, typically alongside language shifts."""
        if not culture_name:
            return
        culture = self._find_culture_by_name(culture_name)
        if culture is None:
            return
        self._ensure_culture_ideas(culture)
        concern_keys = list(self._culture_concern_map.keys())
        if not concern_keys:
            return
        magnitude = max(0.0, min(0.1, float(intensity)))
        shifts = 1  # keep language/idea shifts paired and modest
        naming_sensitive_keys = {"succession", "legitimacy", "universalism"}
        naming_shift_detected = False
        for _ in range(shifts):
            concern_key = random.choice(concern_keys)
            current = culture.ideas.get(concern_key, self._random_initial_idea_value(concern_key))
            delta = random.uniform(-magnitude, magnitude)
            new_value = max(0.0, min(1.0, current + delta))
            # Apply gentle elastic recentering so extreme idea values drift back toward the midpoint over time.
            distance_from_center = abs(new_value - 0.5)
            if distance_from_center > 0.35:
                pull = (0.5 - new_value) * 0.08
                new_value = max(0.0, min(1.0, new_value + pull))
            culture.ideas[concern_key] = new_value
            old_label = self._idea_label_for_value(concern_key, current)
            new_label = self._idea_label_for_value(concern_key, new_value)
            if concern_key in naming_sensitive_keys and old_label != new_label:
                naming_shift_detected = True
            old_display = old_label if old_label is not None else f"{current:.2f}"
            new_display = new_label if new_label is not None else f"{new_value:.2f}"
            self._log_event(
                "culture",
                f"[ideas] {culture.name} shifted {concern_key} from {old_display} to {new_display} ({reason}; Δ≤{magnitude:.2f})",
            )
        if naming_shift_detected:
            self._refresh_polity_naming_for_culture(culture.name)

    # ------------------------------------------------------------------
    # Polity naming style helpers
    # ------------------------------------------------------------------
    def _stable_weighted_choice(self, key: str, options: Sequence[str], weights: Optional[Sequence[float]] = None) -> str:
        """Deterministically select an option using a stable hash-based RNG."""
        if not options:
            return ""
        digest = hashlib.sha256(key.encode('utf-8')).hexdigest()
        seed = int(digest[:16], 16)
        rng = random.Random(seed)
        if weights and len(weights) == len(options):
            total = sum(max(0.0, float(w)) for w in weights)
            if total <= 0:
                return rng.choice(list(options))
            normalized = []
            cumulative = 0.0
            for w in weights:
                cumulative += max(0.0, float(w)) / total
                normalized.append(cumulative)
            roll = rng.random()
            for option, threshold in zip(options, normalized):
                if roll <= threshold:
                    return option
            return options[-1]
        return rng.choice(list(options))

    def _get_dynastic_style_for_culture(self, culture_name: Optional[str]) -> str:
        """Return a deterministic dynastic naming flavor for the culture."""
        if not culture_name:
            return "house_prefix"
        cache = self._culture_dynasty_style_cache
        if culture_name in cache:
            return cache[culture_name]
        options = ("house_prefix", "dynasty_suffix", "rank_suffix")
        weights = (0.4, 0.35, 0.25)
        style = self._stable_weighted_choice(f"dynasty_style:{culture_name}:{self.world_seed}", options, weights)
        cache[culture_name] = style
        return style

    def _format_dynastic_polity_name(self, polity: Polity, token: str, style: str) -> str:
        """Format dynastic polity names with cultural flavor (house, dynasty, or titled)."""
        base = token.strip()
        lower = base.lower()
        if lower.startswith("house "):
            base = base.split(" ", 1)[1].strip() or base
            lower = base.lower()

        def _compose_ranked_title(rank_label: str, core: str) -> str:
            """Return rank-aware display, preferring 'Rank of Core' for duchies/kingdoms."""
            rank_lower = (rank_label or "").lower()
            if "empire" in rank_lower:
                return f"{core} {rank_label}"
            if "kingdom" in rank_lower or "duchy" in rank_lower:
                return f"{rank_label} of {core}"
            return f"{rank_label} of {core}"

        if style == "dynasty_suffix":
            suffix = "Dynasty"
            if lower.endswith(" dynasty"):
                return token
            return f"{base} {suffix}"
        if style == "rank_suffix":
            rank = polity.title_rank or "Empire"
            rank_label = rank if rank else "Empire"
            return _compose_ranked_title(rank_label, base)
        # Default to house prefix
        if base.lower().startswith("house "):
            return base
        return f"House {base}"

    def _release_dynasty_title(self, polity: Optional[Polity]) -> None:
        """Unreserve and clear any active dynastic-loan title attached to a polity."""
        if polity is None:
            return
        token = getattr(polity, 'dynasty_title_token', None)
        if not token:
            return
        source = getattr(polity, 'dynasty_title_source', None)
        culture_key = getattr(polity, 'dynasty_title_culture', None)
        if source == 'catalog' and token and culture_key:
            reservations = self._culture_dynasty_title_reservations.get(culture_key)
            lowered = token.lower()
            if reservations:
                entry = reservations.get(lowered)
                owner_id = entry
                if isinstance(entry, dict):
                    owner_id = entry.get('owner')
                if owner_id == polity.id:
                    reservations.pop(lowered, None)
                    if not reservations:
                        self._culture_dynasty_title_reservations.pop(culture_key, None)
        polity.dynasty_title_token = None
        polity.dynasty_title_source = None
        polity.dynasty_title_culture = None

    def _set_culture_dynasty_reservation(self, culture_name: Optional[str], token: Optional[str], polity_id: int) -> None:
        """Record the canonical dynasty loan token for a culture."""
        if not culture_name or not token:
            return
        reservations = self._culture_dynasty_title_reservations.setdefault(culture_name, {})
        reservations.clear()
        reservations[token.lower()] = {'owner': polity_id, 'token': token}

    def _reserve_catalog_dynasty_title(
        self,
        polity: Optional[Polity],
        culture_name: Optional[str],
        *,
        avoid_root: Optional[str] = None,
        force_new: bool = False,
    ) -> Optional[str]:
        """Claim or create a dynasty loan token rooted in a culture's language."""
        if polity is None or not culture_name:
            return None
        reservations = self._culture_dynasty_title_reservations.setdefault(culture_name, {})

        def _extract_reserved_token() -> Optional[str]:
            if not reservations:
                return None
            token_lower, entry = next(iter(reservations.items()))
            if isinstance(entry, dict):
                return entry.get('token') or token_lower
            return token_lower

        if not force_new:
            existing_token = _extract_reserved_token()
            if existing_token:
                self._set_culture_dynasty_reservation(culture_name, existing_token, polity.id)
                return existing_token

        disallow: Set[str] = set()
        normalized_avoid = self._normalize_dynasty_token(avoid_root, assume_formatted=True) if avoid_root else None
        if normalized_avoid:
            disallow.add(normalized_avoid.lower())

        token = self._mint_dynasty_title_token(polity, avoid_root=avoid_root)
        if not token:
            token = self._select_culture_language_token(
                culture_name,
                DYNASTY_TITLE_CATEGORIES,
                allow_fallback=False,
                disallow=disallow or None,
            )
            if token:
                token = self._format_polity_name_component(token)
                token = self._ensure_distinct_dynasty_root(polity, token, avoid_root)
        if not token:
            fallback = self._select_culture_language_token(
                culture_name,
                POLITY_COMPONENT_CATEGORIES,
                allow_fallback=False,
                disallow=disallow or None,
            )
            if fallback:
                token = self._format_polity_name_component(fallback)
                token = self._ensure_distinct_dynasty_root(polity, token, avoid_root)
        if not token:
            return None
        self._set_culture_dynasty_reservation(culture_name, token, polity.id)
        return token

    def _get_culture_dynasty_loan_token(self, culture_name: Optional[str]) -> Optional[str]:
        """Return the dynastic loan token currently associated with a culture, if any."""
        if not culture_name:
            return None
        reservations = self._culture_dynasty_title_reservations.get(culture_name, {})
        if reservations:
            token_lower, entry = next(iter(reservations.items()))
            if isinstance(entry, dict):
                stored = entry.get('token')
                if stored:
                    return stored
            if token_lower:
                return token_lower
        best_token: Optional[str] = None
        best_rank = -1
        for polity in self.polities:
            if not polity or not getattr(polity, 'is_active', True):
                continue
            if getattr(polity, 'dynasty_title_source', None) != 'catalog':
                continue
            if getattr(polity, 'dynasty_title_culture', None) != culture_name:
                continue
            token = getattr(polity, 'dynasty_title_token', None)
            if not token:
                continue
            rank_value = self._rank_value(getattr(polity, 'title_rank', None))
            if rank_value > best_rank:
                best_rank = rank_value
                best_token = token
        return best_token

    def _ensure_distinct_dynasty_root(
        self,
        polity: Optional[Polity],
        root: Optional[str],
        avoid_root: Optional[str],
    ) -> Optional[str]:
        """Return a dynasty root that does not duplicate the provided avoid token."""
        if not root:
            return None
        normalized_root = self._normalize_dynasty_token(root, assume_formatted=True) or root
        normalized_root = normalized_root.strip()
        if not normalized_root:
            return None
        if not avoid_root:
            return normalized_root
        normalized_avoid = self._normalize_dynasty_token(avoid_root, assume_formatted=True) or avoid_root
        normalized_avoid = (normalized_avoid or "").strip()
        if normalized_avoid and normalized_root.lower() == normalized_avoid.lower():
            disallow = {normalized_root.lower()}
            culture_name = getattr(polity, 'primary_culture', None)
            alternate = self._select_culture_language_token(
                culture_name,
                POLITY_COMPONENT_CATEGORIES,
                allow_fallback=True,
                disallow=disallow,
            )
            if alternate:
                formatted = self._format_polity_name_component(alternate)
                normalized_alt = self._normalize_dynasty_token(formatted, assume_formatted=True) or formatted
                normalized_alt = (normalized_alt or "").strip()
                if normalized_alt and normalized_alt.lower() not in disallow:
                    return normalized_alt
            return f"{normalized_root}an"
        return normalized_root

    def _mint_dynasty_title_token(self, polity: Optional[Polity], *, avoid_root: Optional[str] = None) -> Optional[str]:
        """Synthesize a dynasty-style title when catalogs lack explicit entries."""
        if polity is None:
            return None
        suzerain_id = getattr(polity, 'suzerain_id', -1)
        source_polity = self._get_polity(suzerain_id) if suzerain_id >= 0 else polity
        leader = getattr(source_polity, 'leader', None) if source_polity else None
        dynasty = getattr(leader, 'dynasty', None)
        if not dynasty and leader and getattr(leader, 'name', None):
            parts = [token for token in leader.name.split() if token.lower() not in {"of", "the"}]
            dynasty = parts[-1] if parts else leader.name
        if not dynasty:
            dynasty = self._resolve_dynastic_base_name(polity)
        base = self._format_polity_name_component(dynasty)
        base = self._ensure_distinct_dynasty_root(polity, base, avoid_root)
        if not base:
            return None
        suffixes = None
        if self.config:
            suffixes = self.config.get('linguistics.polity_naming.special_title_suffixes')
        if not isinstance(suffixes, (list, tuple)) or not suffixes:
            suffixes = ["ate", "dom"]
        chooser_seed = f"title:{base}:{getattr(polity, 'id', 0)}"
        suffix = self._stable_weighted_choice(chooser_seed, tuple(str(s) for s in suffixes)) or "ate"
        if base.lower().endswith(suffix[:1].lower()):
            title = base + suffix[1:]
        else:
            title = f"{base}{suffix}"
        catalog = self.get_culture_language(getattr(polity, 'primary_culture', None))
        if catalog and getattr(catalog, 'typology', None):
            try:
                enforced = enforce_phonotactics_form(title, catalog.typology)
                if enforced:
                    title = enforced
            except Exception:
                pass
        return self._format_polity_name_component(title) or title

    def _get_or_create_dynasty_title(
        self,
        polity: Optional[Polity],
        *,
        avoid_root: Optional[str] = None,
    ) -> Optional[str]:
        """Fetch an existing dynasty title or synthesize a temporary one."""
        if polity is None:
            return None
        culture_name = getattr(polity, 'primary_culture', None)
        existing_token = getattr(polity, 'dynasty_title_token', None)
        if existing_token:
            if getattr(polity, 'dynasty_title_culture', None) == culture_name:
                return existing_token
            self._release_dynasty_title(polity)
        token: Optional[str] = None
        source = 'catalog'
        force_new = self._is_polity_empire_rank(polity)
        if not force_new:
            inherited_token = self._get_culture_dynasty_loan_token(culture_name)
            if inherited_token:
                self._set_culture_dynasty_reservation(culture_name, inherited_token, polity.id)
                token = inherited_token
        if not token:
            token = self._reserve_catalog_dynasty_title(
                polity,
                culture_name,
                avoid_root=avoid_root,
                force_new=force_new,
            )
        if not token:
            token = self._mint_dynasty_title_token(polity, avoid_root=avoid_root)
            source = 'minted'
        if not token:
            return None
        polity.dynasty_title_token = token
        polity.dynasty_title_source = source
        polity.dynasty_title_culture = culture_name
        return token

    def _leader_first_name(self, leader: Optional[Leader]) -> Optional[str]:
        """Extract a leader's first usable name token for naming schemes."""
        if not leader or not getattr(leader, 'name', None):
            return None
        tokens = [token for token in leader.name.split() if token.lower() not in {"of", "the"}]
        if not tokens:
            return None
        if tokens[0].lower() == "house" and len(tokens) > 1:
            return tokens[1]
        return tokens[0]

    def _refresh_polity_naming_for_culture(self, culture_name: Optional[str]) -> None:
        if not culture_name:
            return
        for polity in self.polities:
            if polity and getattr(polity, 'primary_culture', None) == culture_name:
                self._refresh_polity_naming(polity, reason="idea_change")

    def _get_polity_rank_tier(self, polity: Optional[Polity]) -> int:
        if polity is None:
            return 0
        rank_label = (polity.title_rank or "").lower()
        if "empire" in rank_label:
            return 2
        if "king" in rank_label:
            return 1
        return 0

    def _determine_polity_naming_plan(self, polity: Optional[Polity]) -> Dict[str, Any]:
        key = "fallback"
        if polity is not None:
            overlord = self._get_polity_suzerain(polity)
            if overlord:
                overlord_universalism = (self._get_polity_idea_label(overlord, 'universalism') or '').lower()
                if 'universal' in overlord_universalism and self._get_polity_rank_tier(overlord) >= 2:
                    return self._get_configured_naming_pattern('vassal_imperial')
            # --- Itinerant/Pluralist/Orthodox pattern selection ---
            itinerance = (self._get_polity_idea_label(polity, 'itinerance') or '').lower()
            universalism = (self._get_polity_idea_label(polity, 'universalism') or '').lower()
            # Only apply itinerant patterns to fully migratory polities
            if 'migratory' in itinerance:
                if 'plural' in universalism:
                    return self._get_configured_naming_pattern('itinerant_pluralist')
                else:
                    return self._get_configured_naming_pattern('itinerant_orthodox')
            succession = (self._get_polity_idea_label(polity, 'succession') or '').lower()
            legitimacy = (self._get_polity_idea_label(polity, 'legitimacy') or '').lower()
            successor_spawn = bool(getattr(polity, 'spawned_from_succession_crisis', False))
            if successor_spawn and 'plural' not in universalism:
                key = 'successor_state'
            elif 'tanistry' in succession:
                if 'plural' in universalism:
                    key = 'tanistry_pluralist'
                else:
                    explosion_spawn = bool(getattr(polity, 'spawned_from_explosion', False))
                    key = 'tanistry_orthodox' if explosion_spawn else 'tanistry_pluralist'
            elif 'primogeniture' in succession:
                if 'plural' in universalism:
                    key = 'primogeniture_pluralist'
                elif 'divine right' in legitimacy:
                    key = 'primogeniture_divine_right'
                elif 'mandate' in legitimacy:
                    key = 'primogeniture_mandate'
                elif 'military' in legitimacy or 'acclam' in legitimacy:
                    key = 'primogeniture_military'
                else:
                    key = 'primogeniture_other'
            elif 'elect' in succession:
                key = 'election_popular' if 'popular' in legitimacy else 'election_other'
        return self._get_configured_naming_pattern(key)

    def _get_configured_naming_pattern(self, key: str) -> Dict[str, Any]:
        base = DEFAULT_POLITY_NAMING_PATTERNS.get(key) or DEFAULT_POLITY_NAMING_PATTERNS['fallback']
        pattern: Dict[str, Any] = dict(base)
        pattern.setdefault('requires_title', False)
        pattern.setdefault('title_type', None)
        override: Optional[Dict[str, Any]] = None
        if self.config:
            override = self.config.get(f"linguistics.polity_naming.pattern_overrides.{key}")
        if isinstance(override, dict):
            base_type = override.get('base_type')
            if isinstance(base_type, str):
                pattern['base_type'] = base_type.lower()
            templates = override.get('templates')
            if isinstance(templates, (list, tuple)) and len(templates) >= 1:
                normalized = tuple(str(t) for t in templates)
                if len(normalized) == 1:
                    pattern['templates'] = normalized * 3
                elif len(normalized) == 2:
                    pattern['templates'] = normalized + (normalized[-1],)
                else:
                    pattern['templates'] = normalized[:3]
            requires_title = override.get('requires_title')
            if requires_title is not None:
                pattern['requires_title'] = bool(requires_title)
            title_type = override.get('title_type')
            if isinstance(title_type, str):
                pattern['title_type'] = title_type
        templates = pattern.get('templates')
        if not isinstance(templates, (list, tuple)) or len(templates) < 3:
            fallback_templates = DEFAULT_POLITY_NAMING_PATTERNS['fallback']['templates']
            pattern['templates'] = fallback_templates
        else:
            pattern['templates'] = tuple(str(t) for t in templates)
        pattern['base_type'] = pattern.get('base_type', 'cultural')
        pattern['key'] = key if key in DEFAULT_POLITY_NAMING_PATTERNS else 'fallback'
        return pattern

    def _resolve_polity_naming_base(self, polity: Optional[Polity], base_type: str) -> str:
        if base_type == 'dynastic':
            return self._resolve_dynastic_base_name(polity)
        if base_type == 'regional':
            return self._resolve_regional_base_name(polity)
        return self._resolve_cultural_base_name(polity)

    def _resolve_cultural_base_name(self, polity: Optional[Polity]) -> str:
        if polity is None:
            return "Realm"
        cached = polity.naming_bases.get('cultural') if hasattr(polity, 'naming_bases') else None
        if cached:
            return cached
        self._assign_polity_language_name(polity)
        token = polity.language_name_component
        if not token:
            capital_idx = self._ensure_polity_capital(polity)
            capital_name = self._get_population_center_name(capital_idx)
            token = capital_name or polity.name
        formatted = self._format_polity_name_component(token)
        if not formatted:
            formatted = f"Realm {polity.id}" if polity is not None else "Realm"
        polity.naming_bases['cultural'] = formatted
        self._ensure_polity_cultural_adjective(polity, formatted)
        return formatted

    def _ensure_polity_cultural_adjective(self, polity: Optional[Polity], base_name: Optional[str]) -> Optional[str]:
        if polity is None or not base_name:
            return None
        cached = polity.naming_bases.get('cultural_adjective') if hasattr(polity, 'naming_bases') else None
        if cached:
            return cached
        culture_name = getattr(polity, 'primary_culture', None)
        adjective = None
        if culture_name:
            adjective = self._culture_adjective_cache.get(culture_name)
        if not adjective:
            adjective = self._derive_culture_adjective(base_name, culture_name)
            if adjective and culture_name and getattr(polity, 'name_from_language', False):
                self._culture_adjective_cache[culture_name] = adjective
        if adjective:
            polity.naming_bases['cultural_adjective'] = adjective
        return adjective

    def _resolve_dynastic_base_name(self, polity: Optional[Polity]) -> str:
        if polity is None:
            return "Dynasty"
        leader = getattr(polity, 'leader', None)
        dynasty = getattr(leader, 'dynasty', None)
        if not dynasty and leader and getattr(leader, 'name', None):
            parts = [token for token in leader.name.split() if token.lower() not in {"of", "the"}]
            dynasty = parts[-1] if parts else leader.name
        if not dynasty:
            dynasty = polity.naming_bases.get('dynastic') or self._resolve_cultural_base_name(polity)
        formatted = self._format_polity_name_component(dynasty)
        formatted = self._normalize_dynasty_token(formatted, assume_formatted=True) or formatted
        if not formatted:
            formatted = self._resolve_cultural_base_name(polity)
        previous = polity.naming_bases.get('dynastic')
        if formatted != previous:
            polity.naming_bases['dynastic'] = formatted
        return formatted

    def _resolve_regional_base_name(self, polity: Optional[Polity]) -> str:
        if polity is None:
            return "Province"
        cached = polity.naming_bases.get('regional') if hasattr(polity, 'naming_bases') else None
        if cached:
            return cached
        capital_idx = self._ensure_polity_capital(polity)
        region_label: Optional[str] = None
        if capital_idx is not None and 0 <= capital_idx < len(self.tiles):
            region_id = getattr(self.tiles[capital_idx], 'region_id', -1)
            if region_id >= 0:
                if not self.region_name_tokens or region_id not in self.region_name_tokens:
                    self.preseed_region_names()
                token = self.region_name_tokens.get(region_id)
                if token:
                    region_label = self._format_polity_name_component(token)
        if not region_label:
            region_label = self._resolve_cultural_base_name(polity)
        polity.naming_bases['regional'] = region_label
        return region_label

    def _generate_special_dynastic_title(
        self,
        polity: Optional[Polity],
        title_type: Optional[str],
        *,
        avoid_root: Optional[str] = None,
    ) -> str:
        if title_type != 'dynastic_loan' or polity is None:
            return "Realm"
        token = self._get_or_create_dynasty_title(polity, avoid_root=avoid_root)
        if token:
            return token
        return "Realm"

    def _compose_polity_name_from_pattern(
        self,
        polity: Optional[Polity],
        template: str,
        base_name: str,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not template:
            return base_name
        extras = dict(extras or {})
        extras['title'] = str(extras.get('title', '')).strip()
        extras.setdefault('base_adjective', base_name)
        existing: Set[str] = {
            getattr(p, 'name', '')
            for p in self.polities
            if p is not None and p is not polity and getattr(p, 'name', None)
        }
        if polity and getattr(polity, 'name', None):
            existing.discard(polity.name)
        culture_name = getattr(polity, 'primary_culture', None) if polity else None
        descriptor_pool = self._culture_direction_descriptors(culture_name) or list(CARDINAL_NAME_DESCRIPTORS)
        variants = [base_name]
        seen = {base_name.lower()}
        for descriptor in descriptor_pool:
            variant = f"{descriptor} {base_name}"
            lowered = variant.lower()
            if lowered not in seen:
                variants.append(variant)
                seen.add(lowered)
        uses_adjective = 'base_adjective' in template
        for variant in variants:
            variant_extras = extras
            if uses_adjective:
                variant_extras = dict(extras)
                variant_adj = variant_extras.get('base_adjective')
                if polity is not None:
                    variant_adj = self._derive_culture_adjective(variant, culture_name) or variant_adj
                else:
                    variant_adj = variant_adj or variant
                variant_extras['base_adjective'] = variant_adj
            candidate = template.format(base=variant, **variant_extras).strip()
            if candidate and candidate not in existing:
                return candidate
        fallback_extras = extras
        if uses_adjective:
            fallback_extras = dict(extras)
            variant_adj = fallback_extras.get('base_adjective')
            if polity is not None:
                variant_adj = self._derive_culture_adjective(base_name, culture_name) or variant_adj
            else:
                variant_adj = variant_adj or base_name
            fallback_extras['base_adjective'] = variant_adj
        fallback = template.format(base=base_name, **fallback_extras).strip()
        if polity is None:
            return fallback
        return self._ensure_unique_polity_name(
            fallback,
            exclude=polity.name,
            existing=existing,
            culture_name=culture_name,
        )

    def _is_polity_vassal(self, polity: Optional[Polity]) -> bool:
        """Return True when the polity is subordinate to a suzerain."""
        if polity is None:
            return False
        try:
            return int(getattr(polity, 'suzerain_id', -1)) >= 0
        except (TypeError, ValueError):
            return False

    def _get_polity_suzerain(self, polity: Optional[Polity]) -> Optional[Polity]:
        """Return the active overlord polity when one exists."""
        if polity is None:
            return None
        try:
            suzerain_id = int(getattr(polity, 'suzerain_id', -1))
        except (TypeError, ValueError):
            return None
        if suzerain_id < 0:
            return None
        return self._get_polity(suzerain_id)

    def _get_overlord_adjective(self, polity: Optional[Polity]) -> str:
        """Derive an adjective from the overlord's cultural base."""
        overlord = self._get_polity_suzerain(polity)
        if overlord is None:
            return ""
        base = self._resolve_polity_naming_base(overlord, 'cultural')
        adjective = self._ensure_polity_cultural_adjective(overlord, base) or base or getattr(overlord, 'name', '')
        return (adjective or '').strip()

    def _refresh_polity_naming(self, polity: Optional[Polity], reason: str = "idea_change") -> None:
        if polity is None or not getattr(polity, 'is_active', True):
            return
        self._assign_polity_language_name(polity)
        plan = self._determine_polity_naming_plan(polity)
        needs_dynastic_loan = bool(plan.get('requires_title')) and (plan.get('title_type') == 'dynastic_loan')
        if not needs_dynastic_loan:
            self._release_dynasty_title(polity)
        base_type = str(plan.get('base_type', 'cultural')).lower()
        base_name = self._resolve_polity_naming_base(polity, base_type)
        templates_value = plan.get('templates')
        if not isinstance(templates_value, (list, tuple)):
            templates_value = DEFAULT_POLITY_NAMING_PATTERNS['fallback']['templates']
        templates: Tuple[str, ...] = tuple(templates_value)
        if self._is_polity_vassal(polity):
            rank_index = 0
        else:
            rank_index = min(len(templates) - 1, max(0, self._get_polity_rank_tier(polity)))
        template = templates[rank_index]
        extras: Dict[str, Any] = {}
        avoid_root: Optional[str] = None
        if base_name:
            normalized = self._normalize_dynasty_token(base_name, assume_formatted=True)
            avoid_root = normalized or base_name
        base_adjective = base_name
        if base_type == 'cultural':
            base_adjective = self._ensure_polity_cultural_adjective(polity, base_name) or base_name
        if plan.get('requires_title'):
            extras['title'] = self._generate_special_dynastic_title(
                polity,
                plan.get('title_type'),
                avoid_root=avoid_root,
            )
        else:
            extras['title'] = plan.get('title_fallback', '')
        extras['base_adjective'] = base_adjective
        extras['overlord_adjective'] = self._get_overlord_adjective(polity) if self._is_polity_vassal(polity) else ''
        new_name = self._compose_polity_name_from_pattern(polity, template, base_name, extras)
        if not new_name or new_name == polity.name:
            polity.naming_pattern_key = plan.get('key')
            return
        self._set_polity_name(
            polity,
            new_name,
            "naming_pattern",
            culture=getattr(polity, 'primary_culture', None),
            note=plan.get('key'),
            force=False,
        )
        polity.naming_pattern_key = plan.get('key')
    
    def add_tile(self, tile: Tile) -> int:
        """Add a tile to the world.
        
        Args:
            tile: Tile to add
            
        Returns:
            Index of the added tile
        """
        self.tiles.append(tile)
        if tile.is_water:
            self.water_tiles += 1
        else:
            self.land_tiles += 1
        return len(self.tiles) - 1
    
    def get_tile_by_position(self, x: float, y: float) -> Optional[Tile]:
        """Get tile containing the given position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Tile that contains the point when found, otherwise None
        """
        point = (x, y)
        for tile in self.tiles:
            if not tile.vertices:
                continue
            if self._point_in_polygon(point, tile.vertices):
                return tile
        return None

    def _point_in_polygon(self, point: Tuple[float, float], vertices: List[Tuple[float, float]]) -> bool:
        """Return True when a point lies inside the polygon defined by vertices."""
        x, y = point
        inside = False
        if not vertices:
            return inside
        j = len(vertices) - 1
        for i, (xi, yi) in enumerate(vertices):
            xj, yj = vertices[j]
            if (yi > y) != (yj > y):
                denom = yj - yi
                if abs(denom) < 1e-9:
                    j = i
                    continue
                intersect_x = (xj - xi) * (y - yi) / denom + xi
                if x < intersect_x:
                    inside = not inside
            j = i
        return inside

    def get_population_center_history(self, tile_index: int) -> List[Dict[str, Any]]:
        center = self._get_population_center_for_tile(tile_index)
        if not center:
            return []
        history = getattr(center, 'name_history', None)
        if not history:
            culture = None
            if 0 <= tile_index < len(self.tiles):
                culture = self._get_tile_majority_culture(self.tiles[tile_index])
            self._initialize_population_center_history(
                center,
                culture=culture,
                reason="legacy_init",
                note="reconstructed",
            )
        return list(center.name_history)

    def get_polity_name_history(self, polity_or_id: Optional[Union[int, 'Polity']]) -> List[Dict[str, Any]]:
        polity: Optional[Polity]
        if isinstance(polity_or_id, Polity):
            polity = polity_or_id
        elif isinstance(polity_or_id, int):
            polity = self._get_polity(polity_or_id)
        else:
            polity = None
        if polity is None:
            return []
        if not getattr(polity, 'name_history', None):
            self._initialize_polity_name_history(
                polity,
                reason="legacy_init",
                note="reconstructed",
            )
        return list(polity.name_history)

    def get_culture_lineage(self, culture_name: Optional[str]) -> List[Dict[str, Any]]:
        lineage: List[Dict[str, Any]] = []
        if not culture_name:
            return lineage
        visited: Set[str] = set()

        def _walk(name: str, depth: int, share: float) -> None:
            if name in visited:
                return
            visited.add(name)
            culture = self._find_culture_by_name(name)
            entry: Dict[str, Any] = {
                'name': name,
                'depth': depth,
                'share': share,
                'language': self._describe_language_origin(name),
            }
            if culture is not None:
                entry['birth_year'] = getattr(culture, 'birth_year', None)
            lineage.append(entry)
            if not culture or not getattr(culture, 'heritage', None):
                return
            parents = sorted(culture.heritage.items(), key=lambda item: item[1], reverse=True)
            for parent_name, pct in parents:
                normalized = max(0.0, min(1.0, float(pct)))
                if normalized <= 0.0:
                    continue
                _walk(parent_name, depth + 1, share * normalized)

        _walk(culture_name, 0, 1.0)
        return lineage

    def get_tile_majority_culture(self, tile: Tile) -> Optional[str]:
        """Public wrapper for majority culture lookup."""
        return self._get_tile_majority_culture(tile)

    def _describe_language_origin(self, culture_name: Optional[str]) -> Optional[str]:
        if not culture_name:
            return None
        catalog = self.get_culture_language(culture_name)
        if catalog and getattr(catalog, 'name', None):
            return f"{culture_name} · {catalog.name}"
        return culture_name

    def _set_population_center_name(
        self,
        center: Optional[PopulationCenter],
        new_name: Optional[str],
        reason: str,
        *,
        phonetic_name: Optional[str] = None,
        culture: Optional[str] = None,
        note: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        if center is None or not new_name:
            return False
        if not force and center.name == new_name:
            return False
        previous = getattr(center, 'name', None)
        center.name = new_name
        if phonetic_name is not None:
            center.phonetic_name = phonetic_name
        entry: Dict[str, Any] = {
            'name': new_name,
            'reason': reason,
            'year': self.current_year,
            'tick': self.total_ticks,
        }
        if phonetic_name is not None:
            entry['phonetic_name'] = phonetic_name
        if previous and previous != new_name:
            entry['previous'] = previous
        if culture:
            entry['culture'] = culture
        language_label = self._describe_language_origin(culture)
        if language_label:
            entry['language'] = language_label
        if note:
            entry['note'] = note
        center.name_history.append(entry)
        return True

    def _set_polity_name(
        self,
        polity: Optional[Polity],
        new_name: Optional[str],
        reason: str,
        *,
        culture: Optional[str] = None,
        language_component: Optional[str] = None,
        note: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        if polity is None or not new_name:
            return False
        if not force and polity.name == new_name:
            return False
        previous = getattr(polity, 'name', None)
        polity.name = new_name
        entry: Dict[str, Any] = {
            'name': new_name,
            'reason': reason,
            'year': self.current_year,
            'tick': self.total_ticks,
        }
        if previous and previous != new_name:
            entry['previous'] = previous
        if culture:
            entry['culture'] = culture
        language_label = language_component or self._describe_language_origin(culture)
        if language_label:
            entry['language'] = language_label
        if note:
            entry['note'] = note
        polity.name_history.append(entry)
        return True

    def _update_polity_suzerain(
        self,
        polity: Optional[Polity],
        new_suzerain_id: Optional[int],
        *,
        refresh_name: bool = True,
    ) -> bool:
        """Assign or clear a suzerain for a polity and refresh dependent state."""
        if polity is None:
            return False
        current_id = getattr(polity, 'suzerain_id', -1)
        try:
            normalized_id = -1 if new_suzerain_id is None else int(new_suzerain_id)
        except (TypeError, ValueError):
            normalized_id = -1
        if normalized_id == current_id:
            return False

        old_parent_label: Optional[str] = None
        if isinstance(current_id, int) and current_id >= 0:
            old_parent = self._get_polity(current_id)
            if old_parent and getattr(old_parent, 'vassal_ids', None) is not None:
                old_parent.vassal_ids = [vid for vid in old_parent.vassal_ids if vid != polity.id]
            old_parent_label = getattr(old_parent, 'name', f"Polity {current_id}") if old_parent else f"Polity {current_id}"

        new_parent = self._get_polity(normalized_id) if normalized_id >= 0 else None
        if new_parent:
            if getattr(new_parent, 'vassal_ids', None) is None:
                new_parent.vassal_ids = []
            if polity.id not in new_parent.vassal_ids:
                new_parent.vassal_ids.append(polity.id)
            try:
                polity.color = self._generate_vassal_color(getattr(new_parent, 'color', None), polity.id)
            except Exception:
                pass
            polity.integration_level = 0.0

        polity.suzerain_id = normalized_id
        if normalized_id < 0:
            polity.integration_level = 100.0
        if normalized_id >= 0:
            ended_wars = self._white_peace_polity_wars(polity.id, reason="vassalization")
            if ended_wars > 0:
                self._log_event(
                    "war_end",
                    f"Auto-white-peace ended {ended_wars} war(s) for vassal {getattr(polity, 'name', f'Polity {polity.id}')}"
                )
            self._sync_overlord_relationships_for_vassal(polity.id)
        subject_label = getattr(polity, 'name', f"Polity {polity.id}")
        if normalized_id >= 0:
            overlord_label = getattr(new_parent, 'name', f"Polity {normalized_id}") if new_parent else f"Polity {normalized_id}"
            self._log_polity_event(
                'vassalization',
                f"[vassal] {subject_label} became a vassal of {overlord_label}",
                polity,
                new_parent,
            )
        elif old_parent_label:
            self._log_polity_event(
                'vassalization',
                f"[vassal] {subject_label} is no longer a vassal of {old_parent_label}",
                polity,
            )
        if refresh_name:
            self._refresh_polity_naming(polity, reason="suzerain_change")
        return True

    def _initialize_population_center_history(
        self,
        center: PopulationCenter,
        *,
        culture: Optional[str],
        reason: str,
        note: Optional[str] = None,
    ) -> None:
        self._set_population_center_name(
            center,
            center.name,
            reason,
            culture=culture,
            note=note,
            force=True,
        )

    def _initialize_polity_name_history(
        self,
        polity: Polity,
        *,
        reason: str = "founding",
        note: Optional[str] = None,
    ) -> None:
        self._set_polity_name(
            polity,
            polity.name,
            reason,
            culture=getattr(polity, 'primary_culture', None),
            language_component=polity.language_name_component,
            note=note,
            force=True,
        )

    def get_map_mode_setting_options(self, mode: str) -> List[Tuple[str, str]]:
        """Return (value, label) pairs for a mode's settings."""
        return list(MAP_MODE_SETTING_DEFINITIONS.get(mode, []))

    def get_map_mode_setting_title(self, mode: str) -> Optional[str]:
        """Return the human-readable title for a mode's settings panel."""
        return MAP_MODE_SETTING_TITLES.get(mode)

    def get_map_mode_setting(self, mode: str) -> Optional[str]:
        """Return the active setting value for a mode when defined."""
        return self._map_mode_settings.get(mode)

    def get_map_mode_setting_label(self, mode: str, option: Optional[str] = None) -> Optional[str]:
        """Return the label for the active or specified setting option."""
        options = MAP_MODE_SETTING_DEFINITIONS.get(mode)
        if not options:
            return None
        target = option or self._map_mode_settings.get(mode)
        if target is None:
            return None
        for value, label in options:
            if value == target:
                return label
        return None

    def set_map_mode_setting(self, mode: str, option: str) -> bool:
        """Update a mode's setting. Returns True when the value changes."""
        options = MAP_MODE_SETTING_DEFINITIONS.get(mode)
        if not options:
            return False
        valid_values = {value for value, _ in options}
        if option not in valid_values:
            return False
        previous = self._map_mode_settings.get(mode)
        if previous == option:
            return False
        self._map_mode_settings[mode] = option
        return True

    def _resolve_map_mode_default_setting(self, mode: str, options: List[Tuple[str, str]]) -> Optional[str]:
        """Select the default view for a map mode, honoring config overrides."""
        if not options:
            return None
        config_value = None
        if self.config:
            config_value = self.config.get(f"ui.map_modes.{mode}.default_setting")
        valid_values = [value for value, _ in options]
        if isinstance(config_value, str) and config_value in valid_values:
            return config_value
        return valid_values[0]
    
    def calculate_statistics(self) -> dict:
        """Calculate world statistics.
        
        Returns:
            Dictionary of world statistics
        """
        elevations = [tile.elevation for tile in self.tiles]
        land_tiles = [tile for tile in self.tiles if not tile.is_water]
        
        stats = {
            'total_tiles': len(self.tiles),
            'water_tiles': self.water_tiles,
            'land_tiles': self.land_tiles,
            'water_percentage': (self.water_tiles / len(self.tiles)) * 100,
            'min_elevation': min(elevations) if elevations else 0,
            'max_elevation': max(elevations) if elevations else 0,
            'avg_elevation': np.mean(elevations) if elevations else 0,
            'sea_level': self.sea_level,
        }
        
        # Add climate statistics for land tiles
        if land_tiles:
            temperatures = [tile.temperature for tile in land_tiles]
            rainfalls = [tile.rainfall for tile in land_tiles]
            
            stats.update({
                'avg_temperature': np.mean(temperatures),
                'min_temperature': min(temperatures),
                'max_temperature': max(temperatures),
                'avg_rainfall': np.mean(rainfalls),
                'min_rainfall': min(rainfalls),
                'max_rainfall': max(rainfalls)
            })
        
        return stats

    def set_world_seed(self, seed: Optional[int]) -> None:
        """Store the authoritative world seed for deterministic systems."""
        self.world_seed = seed

    def initialize_language_system(self, catalog_dir: str | Path, seed: Optional[int] = None) -> None:
        """Load language catalogs and seed deterministic RNG for linguistic hooks."""

        path = Path(catalog_dir)
        self.language_catalog_dir = path
        rng_seed = seed if seed is not None else (self.world_seed if self.world_seed is not None else random.randint(1, 1_000_000))
        self._language_rng_seed = rng_seed
        self._language_rng = random.Random(rng_seed)

        if not path.exists():
            self._language_pool = {}
            self._unused_language_names = []
            self._language_initialized = False
            self._log_event(
                "culture",
                f"[language] Catalog directory '{path}' not found; linguistic integration disabled",
            )
            return

        try:
            self._language_pool = load_all_catalogs(path)
        except Exception as exc:  # pragma: no cover - defensive logging path
            self._language_pool = {}
            self._unused_language_names = []
            self._language_initialized = False
            self._log_event(
                "culture",
                f"[language] Failed to load language catalogs from {path}: {exc}",
            )
            return

        self._unused_language_names = list(self._language_pool.keys())
        if hasattr(self._language_rng, 'shuffle'):
            self._language_rng.shuffle(self._unused_language_names)
        else:  # pragma: no cover - legacy fallback
            random.shuffle(self._unused_language_names)
        used_language_names = {
            culture.language_name
            for culture in self.cultures
            if getattr(culture, "language_name", None)
        }
        if used_language_names:
            self._unused_language_names = [
                name for name in self._unused_language_names if name not in used_language_names
            ]
        self._language_initialized = True
        self._log_event(
            "culture",
            f"[language] Loaded {len(self._language_pool)} catalogs from {path}",
        )
        for polity in self.polities:
            self._assign_polity_language_name(polity)
        self.preseed_region_names()

    def _clone_catalog(self, catalog: LanguageCatalog) -> LanguageCatalog:
        return LanguageCatalog.from_dict(catalog.to_dict())

    def _pop_base_catalog(self, preferred_catalog: Optional[str]) -> Optional[LanguageCatalog]:
        if preferred_catalog and preferred_catalog in self._language_pool:
            catalog = self._language_pool[preferred_catalog]
            try:
                self._unused_language_names.remove(preferred_catalog)
            except ValueError:
                pass
            return self._clone_catalog(catalog)
        while self._unused_language_names:
            candidate = self._unused_language_names.pop(0)
            catalog = self._language_pool.get(candidate)
            if catalog:
                return self._clone_catalog(catalog)
        return None

    def _synthesize_language_from_existing(self, time_depth: int = 1) -> Optional[Tuple[LanguageCatalog, str, int]]:
        if not self.culture_languages:
            return None
        parent_name = self._language_rng.choice(list(self.culture_languages.keys()))
        parent_catalog = self.culture_languages[parent_name]
        derived = transform_language(
            parent_catalog,
            transformation_type="time_evolution",
            time_depth=time_depth,
            rng=self._language_rng,
        )
        return derived, parent_name, time_depth

    def _record_culture_language(
        self,
        culture: Culture,
        catalog: LanguageCatalog,
        *,
        parent: Optional[str],
        transformation: str,
        time_depth: int,
    ) -> None:
        if culture is None or catalog is None:
            return
        catalog.metadata = dict(catalog.metadata)
        catalog.metadata["assigned_culture"] = culture.name
        catalog.metadata["assigned_year"] = self.current_year
        if parent:
            catalog.metadata["parent_language"] = parent
        self.culture_languages[culture.name] = catalog
        culture.language_name = catalog.name
        culture.language_parent = parent
        culture.language_last_update_year = self.current_year
        base_depth = getattr(culture, "language_time_depth", 0)
        culture.language_time_depth = max(0, base_depth) + max(0, time_depth)
        culture.language_transformation = transformation
        self._refresh_population_center_names_for_primary_culture(culture.name)

    def _serialize_culture_language(self, culture_name: str) -> Optional[Dict[str, object]]:
        catalog = self.culture_languages.get(culture_name)
        if not catalog:
            return None
        return catalog.to_dict()

    def _deserialize_language_catalog(self, payload: Optional[Dict[str, object]]) -> Optional[LanguageCatalog]:
        if not payload:
            return None
        try:
            return LanguageCatalog.from_dict(payload)
        except Exception:
            return None

    def get_culture_language(self, culture_name: Optional[str]) -> Optional[LanguageCatalog]:
        if not culture_name:
            return None
        return self.culture_languages.get(culture_name)

    def assign_base_language(
        self,
        culture: Optional[Culture],
        preferred_catalog: Optional[str] = None,
    ) -> Optional[LanguageCatalog]:
        if culture is None:
            return None
        catalog = self._pop_base_catalog(preferred_catalog)
        parent_label: Optional[str] = None
        transformation = "base"
        depth = 0
        if catalog is None:
            synthesized = self._synthesize_language_from_existing()
            if synthesized:
                catalog, parent_label, depth = synthesized
                transformation = "time_evolution"
        if catalog is None:
            status = "language system not initialized" if not self._language_initialized else "no catalogs available"
            self._log_event(
                "culture",
                f"[language] Unable to assign language to culture {culture.name} ({status})",
            )
            return None
        self._record_culture_language(
            culture,
            catalog,
            parent=parent_label,
            transformation=transformation,
            time_depth=depth,
        )
        self._log_event(
            "culture",
            f"[language] Assigned language '{catalog.name}' to culture {culture.name}",
        )
        return catalog

    def assign_derivative_language(
        self,
        culture: Optional[Culture],
        parent_culture_name: Optional[str],
        *,
        time_depth: int = 1,
    ) -> Optional[LanguageCatalog]:
        if culture is None:
            return None
        if not parent_culture_name:
            return self.assign_base_language(culture)
        parent_catalog = self.get_culture_language(parent_culture_name)
        if parent_catalog is None:
            return self.assign_base_language(culture)
        if time_depth <= 0:
            clone = self._clone_catalog(parent_catalog)
            self._record_culture_language(
                culture,
                clone,
                parent=parent_culture_name,
                transformation="clone",
                time_depth=0,
            )
            self._log_event(
                "culture",
                f"[language] Cloned language '{clone.name}' for {culture.name} from {parent_culture_name}",
            )
            return clone
        derived = transform_language(
            parent_catalog,
            transformation_type="time_evolution",
            time_depth=time_depth,
            rng=self._language_rng,
        )
        self._record_culture_language(
            culture,
            derived,
            parent=parent_culture_name,
            transformation="time_evolution",
            time_depth=time_depth,
        )
        self._log_event(
            "culture",
            f"[language] Derived language '{derived.name}' for {culture.name} from {parent_culture_name} (+{time_depth})",
        )
        return derived

    def assign_language_clone(self, culture: Optional[Culture], parent_culture_name: Optional[str]) -> Optional[LanguageCatalog]:
        if culture is None or not parent_culture_name:
            return None
        parent_catalog = self.get_culture_language(parent_culture_name)
        if parent_catalog is None:
            return self.assign_base_language(culture)
        clone = self._clone_catalog(parent_catalog)
        self._record_culture_language(
            culture,
            clone,
            parent=parent_culture_name,
            transformation="clone",
            time_depth=0,
        )
        self._log_event(
            "culture",
            f"[language] Cloned language '{clone.name}' for {culture.name} from {parent_culture_name}",
        )
        return clone

    def assign_syncretic_language(
        self,
        culture: Optional[Culture],
        parent_a: str,
        parent_b: str,
        share_a: float,
        share_b: float,
    ) -> Optional[LanguageCatalog]:
        if culture is None:
            return None
        lang_a = self.get_culture_language(parent_a)
        lang_b = self.get_culture_language(parent_b)
        if lang_a and lang_b:
            total = max(share_a + share_b, 1e-6)
            if share_a >= share_b:
                primary_name, substrate_name = parent_a, parent_b
                primary_catalog, substrate_catalog = lang_a, lang_b
                primary_share, substrate_share = share_a, share_b
            else:
                primary_name, substrate_name = parent_b, parent_a
                primary_catalog, substrate_catalog = lang_b, lang_a
                primary_share, substrate_share = share_b, share_a
            # Calculate substrate influence - more aggressive than simple proportional weighting
            # Substrate languages should have significant phonological influence even with minority status
            substrate_ratio = substrate_share / total
            # Boost substrate influence: minimum 0.3, scales up to 0.8 for equal shares
            influence = max(0.3, min(0.8, 0.3 + substrate_ratio * 0.5))
            contact_context = {
                "primary_share": primary_share,
                "substrate_share": substrate_share,
                "dominance_ratio": primary_share / total,
                "primary_generation": primary_catalog.metadata.get("generation", 1),
                "substrate_generation": substrate_catalog.metadata.get("generation", 1),
            }
            blended = transform_language(
                primary_catalog,
                transformation_type="substrate",
                substrate_typology=substrate_catalog.typology,
                substrate_catalog=substrate_catalog,
                contact_context=contact_context,
                influence_strength=influence,
                rng=self._language_rng,
            )
            self._record_culture_language(
                culture,
                blended,
                parent=f"{parent_a}+{parent_b}",
                transformation="substrate",
                time_depth=0,
            )
            self._log_event(
                "culture",
                f"[language] Syncretic language '{blended.name}' for {culture.name} from {primary_name}/{substrate_name} (w={influence:.2f})",
            )
            return blended
        if lang_a:
            return self.assign_language_clone(culture, parent_a)
        if lang_b:
            return self.assign_language_clone(culture, parent_b)
        return self.assign_base_language(culture)

    # ------------------------------------------------------------------
    # Linguistic polity naming and maintenance
    # ------------------------------------------------------------------

    def assign_polity_language_name(self, polity: Optional[Polity]) -> None:
        """Public wrapper so external systems can trigger naming."""

        self._assign_polity_language_name(polity)

    def _assign_polity_language_name(self, polity: Optional[Polity]) -> None:
        if polity is None:
            return
        component = polity.language_name_component
        component_was_new = False
        polity.name_gloss = getattr(polity, 'name_gloss', None)
        culture = self._find_culture_by_name(polity.primary_culture) if polity and polity.primary_culture else None
        if not component and culture is not None:
            catalog = self.get_culture_language(culture.name)
            if catalog is None:
                catalog = self.assign_base_language(culture)

            # Determine polity type for better naming
            polity_type = self._determine_polity_type(polity)

            # Check if compound naming is enabled
            enable_compound = self.config.get('linguistics.polity_naming.enable_compound_names', True) if self.config else True
            
            # Try compound naming first (like "Deutschland")
            if enable_compound:
                compound_probability = self.config.get('linguistics.polity_naming.compound_name_probability', 0.7) if self.config else 0.7
                if self._language_rng.random() < compound_probability:
                    compound = self._generate_compound_polity_name(culture.name, polity_type, polity.title_rank or "Kingdom")
                    if compound:
                        component, gloss = compound
                        polity.language_name_component = component
                        polity.name_gloss = gloss
                        component_was_new = True
                        polity.name_from_language = True

            # Fallback to traditional single token
            if not component:
                enable_type_based = self.config.get('linguistics.polity_naming.enable_type_based_categories', True) if self.config else True
                if enable_type_based:
                    type_categories = POLITY_TYPE_PATTERNS.get(polity_type, POLITY_COMPONENT_CATEGORIES)
                else:
                    type_categories = POLITY_COMPONENT_CATEGORIES
                token = self._select_language_token(catalog, type_categories) if catalog else None
                if token:
                    component = token
                    polity.language_name_component = component
                    polity.name_gloss = None
                    component_was_new = True
                    polity.name_from_language = True
        if not component:
            capital_idx = self._ensure_polity_capital(polity)
            capital_name = self._get_population_center_name(capital_idx)
            if capital_name and capital_name.endswith(" Capital"):
                capital_name = capital_name[:-8].strip()
            if capital_name and capital_name.startswith("Tribal Settlement"):
                capital_name = None
            if not capital_name and capital_idx is not None and 0 <= capital_idx < len(self.tiles):
                # Prefer settlement-derived name over region for cultureless starts
                generated_name, _ = self._generate_settlement_name(capital_idx)
                capital_name = generated_name
            if capital_name:
                component = capital_name
                polity.language_name_component = component
                polity.name_gloss = None
                polity.name_from_language = False
                component_was_new = True
        if not component:
            polity.language_name_component = None
            polity.name_gloss = None
            polity.name_from_language = False
            polity.name = "Tribe"
            polity.title_rank = "Tribe"
            return
        thresholds = self._compute_polity_rank_thresholds()
        dev_value = self._calculate_polity_development_value(polity.id)
        desired_rank = self._rank_for_development(dev_value, thresholds)
        self._apply_polity_rank(polity, desired_rank, allow_downgrade=False)
        if component_was_new:
            if polity.name_from_language and culture:
                detail = f"culture {culture.name}"
            elif polity.name_from_language:
                detail = "primary culture"
            else:
                detail = "capital"
            self._log_polity_event(
                "polity",
                f"{polity.name} now draws its name from {detail}",
                polity,
            )

    def _select_polity_name_component(self, catalog: LanguageCatalog) -> Optional[str]:
        preferred_categories = ["geographic", "natural_world", "settlement", "military"]
        candidates: List[str] = []
        for category in preferred_categories:
            category_terms = self._collect_language_tokens(catalog, [category])
            candidates.extend(category_terms)
        if not candidates:
            fallback_terms = self._collect_language_tokens(catalog)
            candidates = fallback_terms
        if not candidates:
            return None
        return self._language_rng.choice(candidates)

    def _romanize_polity_token(self, token: str) -> str:
        normalized = unicodedata.normalize('NFKD', token)
        buffer: List[str] = []
        for char in normalized:
            if char in IPA_SILENT_MARKERS:
                continue
            replacement = IPA_TO_ASCII.get(char)
            if replacement is None:
                lower = char.lower()
                replacement = IPA_TO_ASCII.get(lower)
            if replacement is not None:
                buffer.append(replacement)
                continue
            if unicodedata.category(char) == 'Mn':
                continue
            codepoint = ord(char)
            if 32 <= codepoint < 127:
                buffer.append(char)
        collapsed = ''.join(buffer)
        return ' '.join(collapsed.split())

    def _format_polity_name_component(self, token: Optional[str]) -> str:
        if not token:
            return ""

        def _sanitize_segment(segment: str) -> str:
            if not segment:
                return ""
            vowels = set("aeiouyAEIOUY")
            buffer: List[str] = []
            run_char = ""
            run_len = 0
            for ch in segment:
                if ch in {"-", "'"}:
                    buffer.append(ch)
                    run_char = ""
                    run_len = 0
                    continue
                if not ch.isalpha():
                    continue
                if ch == run_char:
                    run_len += 1
                else:
                    run_char = ch
                    run_len = 1
                max_run = 1 if ch in vowels else 2
                if run_len <= max_run:
                    buffer.append(ch)
            cleaned_segment = ''.join(buffer).strip("-'")
            if not cleaned_segment:
                return ""
            lowered = cleaned_segment.lower()
            if lowered.startswith('q') and len(lowered) >= 2 and lowered[1] not in {'u', 'w'}:
                cleaned_segment = ('k' if cleaned_segment[0].islower() else 'K') + cleaned_segment[1:]
            return cleaned_segment

        romanized = self._romanize_polity_token(token)
        cleaned = simplify_orthography(romanized.strip())
        if not cleaned:
            # Fallback to ASCII-safe characters from the original token
            fallback = ''.join(ch for ch in token if ch.isascii() and (ch.isalnum() or ch in {"-", "'", " "}))
            cleaned = simplify_orthography(fallback.strip())
        if not cleaned:
            return ""
        parts: List[str] = []
        for segment in cleaned.split():
            safe_segment = ''.join(ch for ch in segment if ch.isalpha() or ch in {"-", "'"})
            safe_segment = _sanitize_segment(safe_segment)
            if safe_segment:
                parts.append(safe_segment)
        if not parts:
            return ""
        # For compound names (single segment), keep all lowercase except first letter
        if len(parts) == 1:
            return parts[0][0].upper() + parts[0][1:].lower()
        # For multi-segment names, capitalize each segment
        return " ".join(part[0].upper() + part[1:] if len(part) > 1 else part.upper() for part in parts)

    def _derive_culture_adjective(self, base_label: Optional[str], culture_name: Optional[str]) -> Optional[str]:
        if not base_label:
            return None
        segments = [segment for segment in base_label.split() if segment]
        if not segments:
            return None
        prefix = segments[:-1]
        root = segments[-1]
        adjective_root = self._apply_adjective_suffix(root)
        if not adjective_root:
            return None
        combined = " ".join(prefix + [adjective_root]) if prefix else adjective_root
        catalog = self.get_culture_language(culture_name) if culture_name else None
        typology = getattr(catalog, 'typology', None) if catalog else None
        if typology:
            enforced = enforce_phonotactics_form(combined, typology)
            if enforced:
                combined = enforced
        return self._format_polity_name_component(combined)

    def _apply_adjective_suffix(self, root: str) -> str:
        token = root.strip()
        if not token:
            return ""
        lower = token.lower()
        preserved_suffixes = ("ian", "ean", "ish", "ese", "ic", "i", "an", "ite")
        if any(lower.endswith(suffix) for suffix in preserved_suffixes):
            return token
        if lower.endswith(("land", "mark")):
            return f"{token}ish"
        if lower.endswith("stan"):
            return f"{token}i"
        if lower.endswith(("ia", "ea", "oa", "ya", "ua")) and len(token) > 2:
            return f"{token[:-1]}n"
        if lower.endswith("a"):
            return f"{token}n"
        if lower.endswith(("us", "um")) and len(token) > 2:
            return f"{token[:-2]}an"
        if lower.endswith("y") and len(token) > 1:
            return f"{token[:-1]}ian"
        if lower.endswith(("e", "i", "o", "u")):
            return f"{token}an"
        if lower.endswith(("th", "ph")):
            return f"{token}ic"
        if lower.endswith("n"):
            return f"{token}ic"
        if lower.endswith(("r", "l")):
            return f"{token}ian"
        if lower.endswith("m"):
            return f"{token}ian"
        if lower.endswith(("d", "t", "g")):
            return f"{token}ic"
        return f"{token}ian"

    def _normalize_dynasty_token(self, label: Optional[str], *, assume_formatted: bool = False) -> str:
        """Strip common dynasty affixes (House, Clan, Dynasty) while keeping the core token."""
        if not label:
            return ""
        token = label.strip()
        if not token:
            return ""
        if not assume_formatted:
            token = self._format_polity_name_component(token)
        if not token:
            return ""
        words = token.split()
        if not words:
            return ""
        leading = {"house", "clan", "dynasty", "lineage", "tribe", "people", "folk", "family"}
        trailing = {"dynasty", "clan", "house", "lineage", "realm", "host"}
        while len(words) > 1 and words[0].lower() in leading:
            words = words[1:]
        while len(words) > 1 and words[-1].lower() in trailing:
            words = words[:-1]
        candidate = " ".join(words).strip()
        return candidate or token

    def _format_settlement_display_name(self, token: Optional[str]) -> str:
        if not token:
            return ""
        romanized = self._romanize_polity_token(token)
        cleaned = simplify_orthography(romanized.strip())
        if cleaned:
            return cleaned
        fallback = token.strip()
        return simplify_orthography(fallback) if fallback else ""

    def _region_gloss_key(self, region_id: int) -> str:
        return f"{REGION_GLOSS_PREFIX}{max(0, region_id)}"

    def _assign_region_token(self, region_id: int, token: str, *, seed_word: Optional[Word] = None) -> None:
        if region_id < 0 or region_id >= len(self.regions):
            return
        normalized_token = token.strip() or f"Region {region_id + 1}"
        gloss = self._region_gloss_key(region_id)
        if seed_word is None:
            seed_word = Word(phonetic_form=normalized_token, gloss=gloss, category='geographic')
        elif seed_word.gloss != gloss:
            seed_word = seed_word.clone(gloss=gloss)
        self.region_name_tokens[region_id] = normalized_token
        self._region_seed_words[region_id] = seed_word
        self._apply_region_display_name(region_id, normalized_token)

    def _initialize_region_seed_words_from_existing(self) -> None:
        if not self.region_name_tokens:
            return
        if self._region_seed_words and len(self._region_seed_words) == len(self.region_name_tokens):
            return
        for region_id, token in self.region_name_tokens.items():
            gloss = self._region_gloss_key(region_id)
            self._region_seed_words[region_id] = Word(
                phonetic_form=token,
                gloss=gloss,
                category='geographic',
            )

    def preseed_region_names(self) -> None:
        if not self.regions:
            return
        if self.region_name_tokens:
            self._initialize_region_seed_words_from_existing()
            return
        candidate_words: List[Tuple[str, Word]] = []
        if self._language_pool:
            for catalog in self._language_pool.values():
                tokens = self._collect_language_tokens(catalog, REGION_NAME_CATEGORIES)
                for token in tokens:
                    # Find the corresponding Word object
                    for category in REGION_NAME_CATEGORIES:
                        for word in catalog.iter_words_by_category(category):
                            source = word.phonetic_form or word.gloss or ""
                            formatted = self._format_polity_name_component(source)
                            if formatted and formatted == token:
                                candidate_words.append((formatted, word))
                                break
                        else:
                            continue
                        break
        rng = self._language_rng or random.Random()
        rng.shuffle(candidate_words)
        disallow: Set[str] = set()
        word_index = 0
        for region in self.regions:
            token: Optional[str] = None
            seed_word: Optional[Word] = None
            while word_index < len(candidate_words):
                formatted, source_word = candidate_words[word_index]
                word_index += 1
                lowered = formatted.lower()
                if lowered in disallow:
                    continue
                token = formatted
                seed_word = Word(
                    phonetic_form=source_word.phonetic_form,
                    gloss=self._region_gloss_key(region.id),
                    category=source_word.category or 'geographic',
                )
                break
            if not token:
                token = f"Region {region.id + 1}"
                seed_word = Word(
                    phonetic_form=token,
                    gloss=self._region_gloss_key(region.id),
                    category='geographic',
                )
            self._assign_region_token(region.id, token, seed_word=seed_word)
            disallow.add(token.lower())

    def _ensure_region_word_for_culture(self, culture: Optional[Culture], region_id: int) -> Optional[Word]:
        if culture is None or region_id < 0 or region_id >= len(self.regions):
            return None
        catalog = self.get_culture_language(culture.name)
        if catalog is None:
            return None
        gloss = self._region_gloss_key(region_id)
        for word in catalog.words:
            if (word.gloss or "").lower() == gloss.lower():
                return word
        
        # Generate a name from geographic words in the culture's language
        geographic_tokens = [word.phonetic_form for word in catalog.words if word.category == 'geographic' and not self._is_english_like_token(word.phonetic_form)]
        if geographic_tokens:
            index = region_id % len(geographic_tokens)
            base_token = geographic_tokens[index]  # Cycle through geographic tokens
            seed_word = Word(phonetic_form=base_token, gloss=gloss, category='geographic')
        else:
            # Fallback to a generic linguistic name
            base_token = f"Terra{region_id}"
            seed_word = Word(phonetic_form=base_token, gloss=gloss, category='geographic')
        
        new_word = Word(
            phonetic_form=seed_word.phonetic_form,
            gloss=gloss,
            category=seed_word.category or 'geographic',
        )
        catalog.words.append(new_word)
        return new_word

    def _catalog_metadata_tokens(
        self,
        catalog: Optional[LanguageCatalog],
        category: str,
    ) -> List[str]:
        """Return supplemental tokens stored in catalog metadata for specialized categories."""
        tokens: List[str] = []
        if catalog is None:
            return tokens
        metadata = getattr(catalog, 'metadata', None) or {}
        if not metadata:
            return tokens
        normalized = category.lower()
        key_map: Dict[str, Tuple[str, ...]] = {
            'dynasty_title': ('dynasty_titles', 'dynasty_title', 'royal_titles'),
            'noble_title': ('noble_titles', 'aristocratic_titles'),
        }
        for meta_key in key_map.get(normalized, ()):  # Only known keys
            values = metadata.get(meta_key)
            if isinstance(values, (list, tuple)):
                for entry in values:
                    if isinstance(entry, str) and entry.strip():
                        tokens.append(entry)
            elif isinstance(values, str) and values.strip():
                tokens.append(values)
        return tokens

    def _is_english_like_token(self, word: str) -> bool:
        """Heuristic filter to avoid overusing plain English words in generated names."""
        if not word:
            return True
        english_words = {
            'hill', 'plain', 'glacier', 'desert', 'mountain', 'river', 'lake', 'sea',
            'forest', 'jungle', 'temple', 'city', 'town', 'village', 'castle', 'fort',
            'island', 'gulf', 'ocean', 'valley', 'cliff', 'cave', 'spring', 'waterfall',
            'bridge', 'road', 'path', 'gate', 'wall', 'fjord', 'peak', 'victory', 'battle',
            'war', 'peace', 'love', 'hate', 'joy', 'sad', 'anger', 'fear', 'hope', 'dream',
            'night', 'day', 'sun', 'moon', 'star', 'sky', 'earth', 'fire', 'water', 'air',
            'wind', 'rain', 'snow', 'ice', 'heat', 'cold', 'light', 'dark', 'good', 'bad',
            'big', 'small', 'fast', 'slow', 'high', 'low', 'long', 'short', 'wide',
            'narrow', 'thick', 'thin', 'strong', 'weak', 'hard', 'soft', 'hot', 'cool',
            'wet', 'dry', 'full', 'empty', 'new', 'old', 'young', 'ancient', 'rich', 'poor',
            'happy', 'sad', 'region'
        }
        return word.lower() in english_words

    def _collect_language_tokens(
        self,
        catalog: Optional[LanguageCatalog],
        categories: Optional[Sequence[str]] = None
    ) -> List[str]:
        tokens: List[str] = []
        if catalog is None:
            return tokens
        
        if categories:
            for category in categories:
                for word in catalog.iter_words_by_category(category):
                    source = word.phonetic_form or word.gloss or ""
                    if source and not self._is_english_like_token(source):
                        formatted = self._format_polity_name_component(source)
                        if formatted:
                            tokens.append(formatted)
                metadata_tokens = self._catalog_metadata_tokens(catalog, category)
                for token in metadata_tokens:
                    if token and not self._is_english_like_token(token):
                        formatted = self._format_polity_name_component(token)
                        if formatted:
                            tokens.append(formatted)
        else:
            for word in catalog.words:
                source = word.phonetic_form or word.gloss or ""
                if source and not self._is_english_like_token(source):
                    formatted = self._format_polity_name_component(source)
                    if formatted:
                        tokens.append(formatted)
            # Include metadata-backed tokens when gathering broadly
            for category in ('dynasty_title', 'noble_title'):
                for token in self._catalog_metadata_tokens(catalog, category):
                    if token and not self._is_english_like_token(token):
                        formatted = self._format_polity_name_component(token)
                        if formatted:
                            tokens.append(formatted)
        return tokens

    def _select_language_token(
        self,
        catalog: Optional[LanguageCatalog],
        categories: Sequence[str],
        *,
        allow_fallback: bool = True,
        disallow: Optional[Set[str]] = None
    ) -> Optional[str]:
        if not hasattr(self, '_language_rng'):
            self._language_rng = random.Random(self._config.world_seed + 1000)
        if catalog is None:
            return None

        def _filter(tokens: List[str]) -> List[str]:
            if not tokens or not disallow:
                return tokens
            filtered = [token for token in tokens if token.lower() not in disallow]
            return filtered or tokens  # Fall back to originals when all are filtered out

        for category in categories:
            words = self._collect_language_tokens(catalog, [category])
            words = _filter(words)
            if words:
                return self._language_rng.choice(words)
        if not allow_fallback:
            return None
        fallback_tokens = self._collect_language_tokens(catalog)
        fallback_tokens = _filter(fallback_tokens)
        if not fallback_tokens:
            return None
        return self._language_rng.choice(fallback_tokens)

    def _select_language_word(
        self,
        catalog: Optional[LanguageCatalog],
        categories: Sequence[str],
        *,
        allow_fallback: bool = True,
    ) -> Optional[Word]:
        if not hasattr(self, '_language_rng'):
            self._language_rng = random.Random(self._config.world_seed + 1000)
        if catalog is None:
            return None

        for category in categories:
            candidates = [
                word for word in catalog.iter_words_by_category(category)
                if word and not self._is_english_like_token(word.phonetic_form or word.gloss or "")
            ]
            if candidates:
                return self._language_rng.choice(candidates)

        if not allow_fallback:
            return None

        fallback_pool = [
            word for word in catalog.words
            if word and not self._is_english_like_token(word.phonetic_form or word.gloss or "")
        ]
        if fallback_pool:
            return self._language_rng.choice(fallback_pool)
        return None

    def _select_culture_language_token(
        self,
        culture_name: Optional[str],
        categories: Sequence[str],
        *,
        allow_fallback: bool = True,
        disallow: Optional[Set[str]] = None
    ) -> Optional[str]:
        if not culture_name:
            return None
        catalog = self.get_culture_language(culture_name)
        if catalog is None:
            culture = self._find_culture_by_name(culture_name)
            if culture is not None:
                catalog = self.assign_base_language(culture)
        return self._select_language_token(
            catalog,
            categories,
            allow_fallback=allow_fallback,
            disallow=disallow
        )

    def _select_culture_language_word(
        self,
        culture_name: Optional[str],
        categories: Sequence[str],
        *,
        allow_fallback: bool = True,
    ) -> Optional[Word]:
        if not culture_name:
            return None
        catalog = self.get_culture_language(culture_name)
        if catalog is None:
            culture = self._find_culture_by_name(culture_name)
            if culture is not None:
                catalog = self.assign_base_language(culture)
        return self._select_language_word(
            catalog,
            categories,
            allow_fallback=allow_fallback,
        )

    def _apply_region_display_name(self, region_id: int, token: str) -> None:
        if region_id < 0 or region_id >= len(self.regions):
            return
        label = simplify_orthography(token.strip())
        region = self.regions[region_id]
        default_like = not region.name or region.name.startswith("Region ")
        if default_like and label:
            region.name = f"{label} Region"

    def _ensure_region_language_name(self, region_id: int, culture_name: Optional[str]) -> Optional[str]:
        if region_id < 0 or region_id >= len(self.regions):
            return None
        if self.region_name_tokens and region_id not in self._region_seed_words:
            self._initialize_region_seed_words_from_existing()
        disallow = {
            token.lower()
            for rid, token in self.region_name_tokens.items()
            if isinstance(token, str) and rid != region_id
        }
        candidate: Optional[str] = None
        seed_word: Optional[Word] = None
        culture = self._find_culture_by_name(culture_name) if culture_name else None
        if culture:
            region_word = self._ensure_region_word_for_culture(culture, region_id)
            if region_word:
                phonetic = region_word.phonetic_form or ""
                formatted = self._format_polity_name_component(phonetic)
                if not formatted:
                    gloss = region_word.gloss or ""
                    formatted = self._format_polity_name_component(gloss)
                if not formatted:
                    formatted = phonetic or gloss or ""
                if formatted:
                    candidate = formatted
                    seed_word = region_word.clone(gloss=self._region_gloss_key(region_id))
        if candidate is None:
            seed_word = seed_word or self._region_seed_words.get(region_id)
            if seed_word:
                candidate = self._format_polity_name_component(seed_word.phonetic_form or seed_word.gloss or "")
        if candidate is None and culture:
            catalog = self.get_culture_language(culture.name)
            if catalog:
                token = self._select_language_token(catalog, ('geographic', 'natural_world', 'settlement'))
                if token:
                    candidate = token
        if candidate is None:
            existing = self.region_name_tokens.get(region_id)
            if existing and isinstance(existing, str) and existing.strip().startswith("Region "):
                existing = None
            candidate = (existing.strip() if isinstance(existing, str) else None) or f"Region {region_id + 1}"
        base_candidate = candidate.strip() or f"Region {region_id + 1}"
        adjusted = base_candidate
        suffix = " 2"
        while adjusted.lower() in disallow:
            adjusted = f"{base_candidate}{suffix}"
            suffix = " " + str(int(suffix.strip()) + 1)
        current = self.region_name_tokens.get(region_id)
        if current is None or adjusted.lower() != current.lower():
            self._assign_region_token(region_id, adjusted, seed_word=seed_word)
            self._log_event(
                "culture",
                f"[region] Region {region_id} named '{self.regions[region_id].name}' via {culture_name or 'unknown culture'}"
            )
        return adjusted

    def ensure_region_name_for_culture(self, culture: Optional[Culture]) -> Optional[str]:
        """Public helper to bind a culture's home region to a lexical name."""
        if culture is None:
            return None
        return self._ensure_region_language_name(culture.home_region_id, culture.name)

    def ensure_region_language_name(self, region_id: int, culture_name: Optional[str] = None) -> Optional[str]:
        """Ensure a region has a language-derived name seeded from the provided culture."""
        return self._ensure_region_language_name(region_id, culture_name)

    def _ensure_region_word_for_culture(self, culture, region_id):
        if not culture or not hasattr(culture, 'language') or not culture.language:
            return None
        def _is_english_like(text):
            english_words = {'glacier', 'desert', 'mountain', 'river', 'lake', 'sea', 'forest', 'jungle', 'temple', 'city', 'town', 'village', 'castle', 'fort', 'island', 'gulf', 'ocean', 'valley', 'cliff', 'cave', 'spring', 'waterfall', 'bridge', 'road', 'path', 'gate', 'wall', 'fjord', 'peak', 'victory', 'battle', 'war', 'peace', 'love', 'hate', 'joy', 'sad', 'anger', 'fear', 'hope', 'dream', 'night', 'day', 'sun', 'moon', 'star', 'sky', 'earth', 'fire', 'water', 'air', 'wind', 'rain', 'snow', 'ice', 'heat', 'cold', 'light', 'dark', 'good', 'bad', 'big', 'small', 'fast', 'slow', 'high', 'low', 'long', 'short', 'wide', 'narrow', 'thick', 'thin', 'strong', 'weak', 'hard', 'soft', 'hot', 'cool', 'wet', 'dry', 'full', 'empty', 'new', 'old', 'young', 'ancient', 'rich', 'poor', 'happy', 'sad', 'hill', 'plain', 'region'}
            return text.lower() in english_words
        categories = ['geographic', 'natural_world', 'settlement']  # Try these categories in order
        for category in categories:
            if category in culture.language:
                tokens = culture.language[category]
                filtered_tokens = [token for token in tokens if not (_is_english_like(token['phonetic_form']) or _is_english_like(token.get('gloss', '')))]
                if not filtered_tokens:
                    filtered_tokens = tokens  # Fallback to unfiltered if all filtered
                if filtered_tokens:
                    import random
                    token = random.choice(filtered_tokens)
                    return token
        return None

    def _token_contains_english_words(self, token: str) -> bool:
        """Check if a region token contains English words."""
        if not token:
            return False
        words = token.lower().split()
        english_words = {
            'hill', 'plain', 'glacier', 'desert', 'mountain', 'river', 'lake', 'sea',
            'forest', 'jungle', 'temple', 'city', 'town', 'village', 'castle', 'fort',
            'island', 'gulf', 'ocean', 'valley', 'cliff',
            'cave', 'spring', 'waterfall', 'bridge', 'road', 'path', 'gate', 'wall',
            'fjord', 'peak', 'victory', 'battle', 'war', 'peace', 'love', 'hate', 'joy', 'sad',
            'anger', 'fear', 'hope', 'dream', 'night', 'day', 'sun', 'moon', 'star',
            'sky', 'earth', 'fire', 'water', 'air', 'wind', 'rain', 'snow', 'ice',
            'heat', 'cold', 'light', 'dark', 'good', 'bad', 'big', 'small', 'fast',
            'slow', 'high', 'low', 'long', 'short', 'wide', 'narrow', 'thick', 'thin',
            'strong', 'weak', 'hard', 'soft', 'hot', 'cool', 'wet', 'dry', 'full',
            'empty', 'new', 'old', 'young', 'ancient', 'rich', 'poor', 'happy', 'sad', 'region'
        }
        return any(word in english_words for word in words)

    def _generate_non_english_region_token(self, region_id: int) -> str:
        """Generate a new region token that doesn't contain English words."""
        # Try to generate a new token from language catalogs
        culture_name = None  # Use default culture selection
        token = self._ensure_region_language_name(region_id, culture_name)
        if token and not self._token_contains_english_words(token):
            return token
        
        # Fallback to a generic name
        return f"Region {region_id + 1}"

    def get_culture_display_label(self, culture_name: Optional[str]) -> Optional[str]:
        """Return a player-facing culture label, falling back to linguistic tokens when needed."""
        if not culture_name:
            return None
        culture = self._find_culture_by_name(culture_name)
        if not culture:
            return culture_name
        base_name = culture.name or culture_name
        if base_name and not base_name.startswith("Culture_"):
            return base_name
        region_token: Optional[str] = None
        if culture.home_region_id >= 0:
            region_token = self._ensure_region_language_name(culture.home_region_id, culture.name)
        label = (region_token or culture.language_name or base_name or culture_name).strip()
        normalized = label.lower()
        if normalized.endswith(" culture"):
            return label
        return f"{label} Culture"

    def _compute_relative_descriptor(
        self,
        origin_tile_index: Optional[int],
        reference_tile_index: Optional[int]
    ) -> Optional[str]:
        if origin_tile_index is None or reference_tile_index is None:
            return None
        if not (0 <= origin_tile_index < len(self.tiles)):
            return None
        if not (0 <= reference_tile_index < len(self.tiles)):
            return None
        origin_tile = self.tiles[origin_tile_index]
        reference_tile = self.tiles[reference_tile_index]
        dx = origin_tile.center[0] - reference_tile.center[0]
        dy = origin_tile.center[1] - reference_tile.center[1]
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        descriptor: Optional[str] = None
        # Prefer horizontal or vertical descriptors when there is a clear bias
        if abs_dx >= abs_dy * 1.25 and abs_dx > 0.01:
            descriptor = "East" if dx > 0 else "West"
        elif abs_dy > abs_dx * 1.25 and abs_dy > 0.01:
            # Screen coordinates grow downward; positive dy means southward shift
            descriptor = "South" if dy > 0 else "North"
        else:
            elevation_delta = origin_tile.elevation - reference_tile.elevation
            if abs(elevation_delta) > 0.01:
                descriptor = "High" if elevation_delta > 0 else "Low"
        return descriptor

    def _make_unique_culture_name(
        self,
        base_name: str,
        origin_tile_index: Optional[int],
        *,
        forbidden: Optional[Set[str]] = None
    ) -> str:
        base = base_name or self._next_culture_name('R')
        normalized_base = self._normalize_culture_name_token(base)
        existing_lower: Set[str] = set(self._culture_name_registry)
        existing_lower.update(self._pending_culture_name_reservations)
        for culture in self.cultures:
            normalized = self._normalize_culture_name_token(culture.name)
            if normalized:
                existing_lower.add(normalized)
        def _is_allowed(normalized: Optional[str]) -> bool:
            if not normalized:
                return False
            if normalized in existing_lower:
                return False
            if forbidden and normalized in forbidden:
                return False
            return True

        if _is_allowed(normalized_base):
            return base

        # Second attempt: use the settlement/city label for the origin tile
        city_label = self._derive_city_label(origin_tile_index)
        normalized_city = self._normalize_culture_name_token(city_label)
        if _is_allowed(normalized_city):
            return city_label

        # Prepare descriptor-based variations anchored to the most contextually relevant name
        anchor = city_label or base
        duplicates: List[Culture] = []
        if normalized_base:
            duplicates = [
                culture for culture in self.cultures
                if self._normalize_culture_name_token(culture.name) == normalized_base
            ]
        descriptor_order: List[str] = []
        if duplicates:
            reference_tile = duplicates[0].origin_tile_index
            primary_descriptor = self._compute_relative_descriptor(origin_tile_index, reference_tile)
            if primary_descriptor:
                descriptor_order.append(primary_descriptor)

        # Cardinal descriptors take precedence, followed by High/Low as a fallback
        for descriptor in ["North", "South", "East", "West", "High", "Low"]:
            if descriptor not in descriptor_order:
                descriptor_order.append(descriptor)

        for descriptor in descriptor_order:
            candidate = f"{descriptor} {anchor}"
            normalized_candidate = self._normalize_culture_name_token(candidate)
            if _is_allowed(normalized_candidate):
                return candidate

        suffix = 2
        while True:
            candidate = f"{anchor} {suffix}"
            normalized_candidate = self._normalize_culture_name_token(candidate)
            if _is_allowed(normalized_candidate):
                return candidate
            suffix += 1

    def _generate_regional_culture_name(
        self,
        region_id: int,
        source_culture: Optional[str],
        origin_tile_index: Optional[int]
    ) -> Optional[str]:
        if region_id < 0:
            return None
        token = self.region_name_tokens.get(region_id)
        if not token:
            token = self._ensure_region_language_name(region_id, source_culture)
        if not token:
            return None

        base_label = self._format_polity_name_component(token) or token.strip()
        if not base_label:
            return None
        normalized_region = self._normalize_culture_name_token(base_label)
        forbidden_tokens: Set[str] = set()
        if normalized_region:
            forbidden_tokens.add(normalized_region)
        base_culture_label = self._normalize_culture_name_token(f"{base_label} Culture")
        if base_culture_label:
            forbidden_tokens.add(base_culture_label)
        forbidden_ref = forbidden_tokens or None

        candidates: List[str] = []

        def add_candidate(raw: Optional[str]) -> None:
            candidate = (raw or "").strip()
            if not candidate:
                return
            normalized = self._normalize_culture_name_token(candidate)
            if not normalized:
                return
            if normalized_region and normalized == normalized_region:
                return
            if candidate not in candidates:
                candidates.append(candidate)

        def build_demonym(stem: str, suffix: str) -> Optional[str]:
            collapsed = stem.replace(" ", "")
            if not collapsed:
                return None
            trimmed = collapsed
            if trimmed and trimmed[-1].lower() in COUNTABLE_VOWELS and suffix.startswith(("i", "e")):
                trimmed = trimmed[:-1]
            if trimmed.endswith('y') and suffix.startswith(("i", "e")):
                trimmed = trimmed[:-1] + "i"
            return f"{trimmed}{suffix}"

        suffixes = list(CULTURE_NAME_SUFFIXES)
        self._language_rng.shuffle(suffixes)
        for suffix in suffixes:
            demonym = build_demonym(base_label, suffix)
            if demonym:
                formatted = self._format_polity_name_component(demonym)
                add_candidate(formatted)

        lexeme: Optional[str] = None
        if source_culture:
            disallow = {normalized_region} if normalized_region else None
            lexeme = self._select_culture_language_token(
                source_culture,
                CULTURE_NAME_CATEGORIES,
                disallow=disallow
            )
        lexeme_label = self._format_polity_name_component(lexeme) if lexeme else ""
        if lexeme_label:
            add_candidate(f"{lexeme_label} of {base_label}")
            add_candidate(f"{base_label} {lexeme_label}")
            add_candidate(f"{lexeme_label} {base_label}")

        descriptors = list(CULTURE_NAME_DESCRIPTORS)
        self._language_rng.shuffle(descriptors)
        for descriptor in descriptors:
            add_candidate(f"{base_label} {descriptor}")
        for descriptor in descriptors[:2]:
            add_candidate(f"{descriptor} of {base_label}")

        for candidate in candidates:
            unique = self._make_unique_culture_name(
                candidate,
                origin_tile_index,
                forbidden=forbidden_ref
            )
            normalized_unique = self._normalize_culture_name_token(unique)
            if normalized_unique and normalized_unique != normalized_region:
                return unique

        fallback = f"{base_label} Culture"
        fallback_unique = self._make_unique_culture_name(
            fallback,
            origin_tile_index,
            forbidden=forbidden_ref
        )
        normalized_fallback = self._normalize_culture_name_token(fallback_unique)
        if forbidden_ref and normalized_fallback in forbidden_ref:
            # As a last resort, fall back to generic series tokens
            return self._make_unique_culture_name(
                self._next_culture_name('R'),
                origin_tile_index,
                forbidden=forbidden_ref
            )
        return fallback_unique

    def _generate_culture_demonym_from_region(self, region_name: str, origin_tile_index: Optional[int]) -> Optional[str]:
        """Generate a proper culture demonym name from a region name."""
        if not region_name:
            return None
        
        base_label = self._format_polity_name_component(region_name) or region_name.strip()
        if not base_label:
            return None

        # Prefer directional variants before suffix demonyms to reduce near-duplicates
        cardinal_first = self._make_unique_culture_name(base_label, origin_tile_index)
        if cardinal_first:
            return cardinal_first

        # Use the same demonym generation logic as regional culture names
        suffixes = list(CULTURE_NAME_SUFFIXES)
        self._language_rng.shuffle(suffixes)

        def build_demonym(stem: str, suffix: str) -> Optional[str]:
            collapsed = stem.replace(" ", "")
            if not collapsed:
                return None
            trimmed = collapsed
            if trimmed and trimmed[-1].lower() in COUNTABLE_VOWELS and suffix.startswith(("i", "e")):
                trimmed = trimmed[:-1]
            if trimmed.endswith('y') and suffix.startswith(("i", "e")):
                trimmed = trimmed[:-1] + "i"
            return f"{trimmed}{suffix}"
        
        for suffix in suffixes[:5]:  # Try first 5 suffixes
            demonym = build_demonym(base_label, suffix)
            if demonym:
                formatted = self._format_polity_name_component(demonym)
                if formatted and len(formatted) > 3:  # Ensure reasonable length
                    unique = self._make_unique_culture_name(
                        formatted,
                        origin_tile_index
                    )
                    if unique:
                        return unique
        
        # Fallback
        fallback = f"{base_label} Culture"
        return self._make_unique_culture_name(fallback, origin_tile_index)

    def _get_population_center_for_tile(self, tile_index: int) -> Optional[PopulationCenter]:
        if tile_index < 0:
            return None
        for center in self.population_centers:
            if center.tile_index == tile_index:
                return center
        return None

    def _get_population_center_name(self, tile_index: Optional[int]) -> Optional[str]:
        if tile_index is None:
            return None
        center = self._get_population_center_for_tile(tile_index)
        return center.name if center else None

    def _derive_city_label(self, tile_index: Optional[int]) -> Optional[str]:
        name = self._get_population_center_name(tile_index)
        if not name:
            return None
        label = name.replace('_', ' ').strip()
        if label.endswith(" Capital"):
            label = label[:-8].strip()
        return label or None

    def _normalize_culture_name_token(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        normalized = name.strip().lower()
        return normalized or None

    def _register_culture_name(self, name: Optional[str]) -> None:
        normalized = self._normalize_culture_name_token(name)
        if normalized:
            self._culture_name_registry.add(normalized)

    def _reserve_culture_name(self, name: Optional[str]) -> Optional[str]:
        normalized = self._normalize_culture_name_token(name)
        if not normalized:
            return None
        if normalized in self._culture_name_registry:
            return None
        if normalized in self._pending_culture_name_reservations:
            return None
        self._pending_culture_name_reservations.add(normalized)
        return normalized

    def _release_reserved_culture_name(self, name: Optional[str]) -> None:
        normalized = self._normalize_culture_name_token(name)
        if normalized:
            self._pending_culture_name_reservations.discard(normalized)

    def _commit_reserved_culture_name(self, name: Optional[str]) -> None:
        normalized = self._normalize_culture_name_token(name)
        if not normalized:
            return
        self._pending_culture_name_reservations.discard(normalized)
        self._culture_name_registry.add(normalized)

    def _is_culture_name_in_use(self, name: Optional[str]) -> bool:
        normalized = self._normalize_culture_name_token(name)
        if not normalized:
            return False
        if normalized in self._culture_name_registry:
            return True
        if normalized in self._pending_culture_name_reservations:
            return True
        for culture in self.cultures:
            existing = self._normalize_culture_name_token(culture.name)
            if existing and existing == normalized:
                return True
        return False

    def _reserve_unique_culture_name(self, base_name: Optional[str], origin_tile_index: Optional[int]) -> Tuple[str, Optional[str]]:
        candidate = base_name or self._next_culture_name('R')
        reservation = self._reserve_culture_name(candidate)
        if reservation is None:
            candidate = self._make_unique_culture_name(candidate, origin_tile_index)
            reservation = self._reserve_culture_name(candidate)
        return candidate, reservation

    def _rebuild_culture_name_registry(self) -> None:
        self._culture_name_registry = set()
        self._pending_culture_name_reservations = set()
        for culture in self.cultures:
            # Check if culture name contains English words and rename if needed
            if self._culture_name_contains_english_words(culture.name):
                old_name = culture.name
                new_name = self._generate_non_english_culture_name(culture)
                culture.name = new_name
                self._log_event(
                    "culture_debug",
                    f"[culture] Renamed culture from '{old_name}' to '{new_name}' due to English word filtering"
                )
            self._register_culture_name(culture.name)

    def _culture_name_contains_english_words(self, name: str) -> bool:
        """Check if a culture name contains English words."""
        if not name:
            return False
        words = name.lower().split()
        english_words = {
            'hill', 'plain', 'glacier', 'desert', 'mountain', 'river', 'lake', 'sea',
            'forest', 'jungle', 'temple', 'city', 'town', 'village', 'castle', 'fort',
            'island', 'bay', 'gulf', 'ocean', 'valley', 'canyon', 'plateau', 'cliff',
            'cave', 'spring', 'waterfall', 'bridge', 'road', 'path', 'gate', 'wall',
            'fjord', 'peak', 'victory', 'battle', 'war', 'peace', 'love', 'hate', 'joy', 'sad',
            'anger', 'fear', 'hope', 'dream', 'night', 'day', 'sun', 'moon', 'star',
            'sky', 'earth', 'fire', 'water', 'air', 'wind', 'rain', 'snow', 'ice',
            'heat', 'cold', 'light', 'dark', 'good', 'bad', 'big', 'small', 'fast',
            'slow', 'high', 'low', 'long', 'short', 'wide', 'narrow', 'thick', 'thin',
            'strong', 'weak', 'hard', 'soft', 'hot', 'cool', 'wet', 'dry', 'full',
            'empty', 'new', 'old', 'young', 'ancient', 'rich', 'poor', 'happy', 'sad'
        }
        return any(word in english_words for word in words)

    def _generate_non_english_culture_name(self, culture: Culture) -> str:
        """Generate a new culture name that doesn't contain English words."""
        # Try to generate a regional name first
        if culture.home_region_id is not None and culture.home_region_id >= 0:
            regional_name = self._generate_regional_culture_name(
                culture.home_region_id,
                None,  # Don't use parent culture to avoid inheritance of English words
                culture.origin_tile_index
            )
            if regional_name and not self._culture_name_contains_english_words(regional_name):
                return regional_name
        
        # Fallback to a generic name
        return self._make_unique_culture_name(
            self._next_culture_name('R'),
            culture.origin_tile_index
        )

    def _ensure_unique_settlement_name(self, base_name: str, *, exclude: Optional[str] = None) -> str:
        candidate = base_name or "Tribal Settlement"
        existing = {center.name for center in self.population_centers if center.name != exclude}
        if candidate not in existing:
            return candidate
        suffix = 2
        while f"{candidate} {suffix}" in existing:
            suffix += 1
        return f"{candidate} {suffix}"

    def _collect_settlement_name_disallow(self, *, exclude: Optional[str] = None) -> Set[str]:
        disallow: Set[str] = set()
        for center in self.population_centers:
            label = center.name
            if not label or (exclude and label == exclude):
                continue
            normalized = self._format_polity_name_component(label)
            if normalized:
                disallow.add(normalized.lower())
        return disallow

    def _apply_historical_sound_changes_to_name(self, center: PopulationCenter, culture_name: str) -> Optional[str]:
        """Apply sound changes from culture's language evolution to settlement names."""
        if not center.phonetic_name:
            return None

        catalog = self.get_culture_language(culture_name)
        if not catalog or not catalog.typology.sound_changes:
            return None

        # Use the current phonetic name as the base for evolution
        original_phonetic = center.phonetic_name

        # Apply sound changes cumulatively
        evolved_name = original_phonetic
        for change in catalog.typology.sound_changes:
            # Create a temporary word to apply the change
            temp_word = Word(phonetic_form=evolved_name, gloss="settlement")
            evolved_word = change.apply(temp_word, self._language_rng)
            evolved_name = evolved_word.phonetic_form

        # Only return if significantly different
        if evolved_name != original_phonetic and len(evolved_name) > 2:
            return evolved_name

        return None

    def _apply_language_evolution_to_settlement_names(self, culture_name: str) -> None:
        """Apply language evolution sound changes to settlement names for the given culture."""
        if not culture_name:
            return

        updated_count = 0
        for center in self.population_centers:
            if not center or center.tile_index >= len(self.tiles):
                continue

            tile = self.tiles[center.tile_index]
            polity = self._get_polity(tile.polity_id)
            if not polity or polity.primary_culture != culture_name:
                continue

            # Apply sound changes to the phonetic name
            evolved_phonetic = self._apply_historical_sound_changes_to_name(center, culture_name)
            if evolved_phonetic:
                # Generate new display name from evolved phonetic form
                display_name = self._romanize_polity_token(evolved_phonetic)
                if display_name and display_name != center.name:
                    self._set_population_center_name(
                        center,
                        display_name,
                        "language_evolution",
                        phonetic_name=evolved_phonetic,
                        culture=culture_name,
                        note="sound_changes_applied",
                    )
                    updated_count += 1

        if updated_count:
            self._log_event(
                "culture",
                f"[language] Applied sound changes to {updated_count} settlement name(s) for culture '{culture_name}'"
            )

    def _preserve_historical_name_layers(self, center: PopulationCenter, new_name: str, reason: str) -> None:
        """Record name changes in the settlement's historical record."""
        enable_preservation = self.config.get('linguistics.settlement_naming.enable_historical_preservation', True) if self.config else True
        if not enable_preservation:
            return
            
        if not center.name_history:
            center.name_history = []

        # Avoid duplicate consecutive entries
        if center.name_history and center.name_history[-1].get("name") == center.name:
            return

        historical_entry = {
            "name": center.name,
            "year": getattr(self, 'current_year', 0),
            "reason": reason,
            "culture_context": self._get_tile_majority_culture(self.tiles[center.tile_index]) if center.tile_index < len(self.tiles) else None
        }

        center.name_history.append(historical_entry)

        # Limit history to prevent unbounded growth
        max_history = self.config.get('linguistics.settlement_naming.max_name_history_length', 10) if self.config else 10
        if len(center.name_history) > max_history:
            center.name_history = center.name_history[-max_history:]

    def _administrative_rename_settlement(self, center: PopulationCenter, new_name: str, polity_id: int, reason: str = "administrative") -> bool:
        """Handle official name changes by polities (e.g., renaming for political reasons)."""
        if not center or not new_name or polity_id < 0:
            return False

        # Some names resist administrative changes (cultural significance)
        resistance_chance = self.config.get('linguistics.settlement_naming.administrative_rename_resistance', 0.3) if self.config else 0.3
        if center.name_history and len(center.name_history) > 3:
            # Ancient settlements more resistant to renaming
            resistance_chance += 0.2

        if self._language_rng.random() < resistance_chance:
            # Name persists despite administrative attempt
            return False

        # Preserve the old name in history
        self._preserve_historical_name_layers(center, new_name, f"{reason}_rename")

        polity = self._get_polity(polity_id)
        culture = polity.primary_culture if polity else None

        return self._set_population_center_name(
            center,
            new_name,
            reason,
            culture=culture,
            note=f"polity_{polity_id}",
            force=True  # Administrative changes can override existing names
        )

    def _determine_polity_type(self, polity: Polity) -> str:
        """Determine polity type based on rank."""
        if not polity or not polity.title_rank:
            return "duchy"

        rank = polity.title_rank.lower()
        if "duchy" in rank:
            return "duchy"
        elif "kingdom" in rank:
            return "kingdom"
        elif "empire" in rank:
            return "empire"
        else:
            return "duchy"

    def _generate_compound_polity_name(self, culture_name: str, polity_type: str, rank: str) -> Optional[Tuple[str, Optional[str]]]:
        """Generate short, semantically-bound compounds using typology-aware repairs.

        Returns (name, gloss) when successful.
        """
        catalog = self.get_culture_language(culture_name)
        if not catalog or not catalog.typology:
            return None

        cfg = self.config or {}
        naming_cfg = cfg.get('linguistics', {}).get('polity_naming', {}) if cfg else {}
        max_chars = naming_cfg.get('compound_max_chars', 14)
        max_syllables = naming_cfg.get('compound_max_syllables', 4)
        linkers = naming_cfg.get('compound_linkers', ["a", "e", "i", "o"])
        require_gloss = naming_cfg.get('compound_require_gloss', False)

        # Get polity type-specific categories (fallback if pairs not provided)
        type_categories = POLITY_TYPE_PATTERNS.get(polity_type, POLITY_COMPONENT_CATEGORIES)

        # Parse allowed pairs from config
        configured_pairs = naming_cfg.get('compound_allowed_pairs', [])
        allowed_pairs = [
            (pair[0], pair[1])
            for pair in configured_pairs
            if isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[0] and pair[1]
        ]

        # If none configured, use a sensible default set
        if not allowed_pairs:
            allowed_pairs = [
                ("geographic", "natural_world"),
                ("geographic", "settlement"),
                ("settlement", "geographic"),
                ("military", "geographic"),
                ("geographic", "military"),
                ("natural_world", "geographic"),
            ]

        compound_attempts = 5
        for _ in range(compound_attempts):
            # Choose a pair; if config empty, pair list already defaulted
            left_cat, right_cat = self._language_rng.choice(allowed_pairs)

            left_word = self._select_culture_language_word(
                culture_name, (left_cat,), allow_fallback=False
            )
            right_word = self._select_culture_language_word(
                culture_name, (right_cat,), allow_fallback=False
            )

            # Fallback to type-based selection if configured pair not available
            if not left_word:
                fallback_left = self._select_culture_language_word(
                    culture_name, type_categories, allow_fallback=True
                )
                left_word = fallback_left
            if not right_word:
                fallback_right = self._select_culture_language_word(
                    culture_name, ("geographic", "natural_world"), allow_fallback=True
                )
                right_word = fallback_right

            if not left_word or not right_word:
                continue

            left_text = left_word.phonetic_form or left_word.gloss or ""
            right_text = right_word.phonetic_form or right_word.gloss or ""
            compound_with_gloss = build_compound_with_gloss(
                left_text,
                left_word.gloss,
                right_text,
                right_word.gloss,
                catalog.typology,
                max_chars=max_chars,
                max_syllables=max_syllables,
                linkers=linkers,
                rng=self._language_rng,
            )

            if not compound_with_gloss:
                continue

            compound, gloss = compound_with_gloss
            if require_gloss and not gloss:
                continue

            formatted = self._format_polity_name_component(compound)
            max_length = self.config.get('linguistics.polity_naming.max_name_length', 25) if self.config else 25
            if formatted and len(formatted) > 3 and len(formatted) <= max_length:
                return formatted, gloss

        return None

    def _generate_descriptive_polity_name(self, culture_name: str, polity_type: str, rank: str) -> Optional[str]:
        """Generate descriptive polity names like 'Kingdom of the Isles'."""
        catalog = self.get_culture_language(culture_name)
        if not catalog:
            return None

        # Get length limits from config
        max_length = self.config.get('linguistics.polity_naming.max_descriptive_name_length', 35) if self.config else 35

        # Select descriptor (adjective-like)
        descriptor = self._select_culture_language_token(
            culture_name, ("geographic", "natural_world", "military"), allow_fallback=True
        )

        # Select territory/base name
        territory = self._select_culture_language_token(
            culture_name, ("settlement", "geographic"), allow_fallback=True
        )

        if descriptor and territory:
            # Try full descriptive name like "Kingdom of the Golden Plain"
            full_name = f"{rank} of the {descriptor} {territory}"
            if len(full_name) <= max_length:
                return full_name
            
            # Fall back to shorter form: "Kingdom of the Plain"
            short_name = f"{rank} of {territory}"
            if len(short_name) <= max_length:
                return short_name
            
            # Final fallback: just the territory name
            return territory

        return None

    def _culture_direction_descriptors(self, culture_name: Optional[str]) -> List[str]:
        """Return unique, title-cased direction tokens from the culture's catalog."""
        if not culture_name:
            return []

        catalog = self.get_culture_language(culture_name)
        if catalog is None:
            culture = self._find_culture_by_name(culture_name)
            if culture is not None:
                catalog = self.assign_base_language(culture)

        if not catalog or not getattr(catalog, 'words', None):
            return []

        descriptors: List[str] = []
        seen: Set[str] = set()
        for word in catalog.words:
            if getattr(word, 'category', None) != 'direction':
                continue
            token = word.phonetic_form or word.gloss or ""
            formatted = self._format_polity_name_component(token).strip()
            if not formatted:
                continue
            lowered = formatted.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            if len(formatted) > 1:
                formatted = formatted[0].upper() + formatted[1:]
            else:
                formatted = formatted.upper()
            descriptors.append(formatted)
        return descriptors

    def _ensure_unique_polity_name(
        self,
        base_name: str,
        *,
        exclude: Optional[str] = None,
        existing: Optional[Set[str]] = None,
        culture_name: Optional[str] = None,
    ) -> str:
        """Ensure polity names are unique across the world."""
        candidate = base_name or "Unnamed Polity"
        if existing is None:
            existing = {polity.name for polity in self.polities if getattr(polity, 'name', None)}
        existing = {name for name in existing if name and name != exclude}
        if candidate not in existing:
            return candidate

        # For polities, try variations before adding numbers
        descriptors = self._culture_direction_descriptors(culture_name)
        fallback_descriptors = [
            descriptor for descriptor in CARDINAL_NAME_DESCRIPTORS
            if descriptor.lower() not in {d.lower() for d in descriptors}
        ]
        descriptor_pool = descriptors + list(fallback_descriptors)

        variations = [f"{descriptor} {candidate}" for descriptor in descriptor_pool]
        variations.extend([
            f"Great {candidate}",
            f"New {candidate}",
        ])

        for variation in variations:
            if variation not in existing:
                return variation

        # Fallback to numbered versions
        suffix = 2
        while f"{candidate} {suffix}" in existing:
            suffix += 1
        return f"{candidate} {suffix}"

    def _sample_typology_consonant(self, typology: Typology) -> str:
        pool = [phoneme for phoneme in typology.consonants if phoneme]
        if not pool:
            pool = list(FALLBACK_CONSONANTS)
        return self._language_rng.choice(pool)

    def _sample_typology_vowel(self, typology: Typology) -> str:
        pool = [phoneme for phoneme in typology.vowels if phoneme]
        if not pool:
            pool = list(FALLBACK_VOWELS)
        return self._language_rng.choice(pool)

    def _realize_typology_pattern(self, pattern: str, typology: Typology) -> str:
        buffer: List[str] = []
        for symbol in pattern:
            upper = symbol.upper()
            if upper == 'C':
                buffer.append(self._sample_typology_consonant(typology))
            elif upper == 'V':
                buffer.append(self._sample_typology_vowel(typology))
            else:
                buffer.append(symbol)
        return ''.join(buffer)

    def _synthesize_settlement_token_for_culture(
        self,
        culture_name: Optional[str],
        tile_index: int,
        *,
        disallow: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """Build a phonotactically valid placeholder when catalogs lack toponyms."""
        if not culture_name:
            return None
        catalog = self.get_culture_language(culture_name)
        if catalog is None:
            culture = self._find_culture_by_name(culture_name)
            if culture is not None:
                catalog = self.assign_base_language(culture)
        if catalog is None:
            return None
        typology = catalog.typology
        patterns = [pattern for pattern in typology.syllable_patterns if pattern]
        if not patterns:
            patterns = ["CV", "CVC"]
        syllable_count = self._language_rng.randint(2, 3)
        pieces: List[str] = []
        for _ in range(syllable_count):
            pattern = self._language_rng.choice(patterns)
            pieces.append(self._realize_typology_pattern(pattern, typology))
        prototype = ''.join(pieces).strip()
        if not prototype:
            parts = culture_name.split()
            stem = parts[0] if parts else culture_name or "Settlement"
            prototype = f"{stem}{tile_index + 1}"
        candidate = enforce_phonotactics_form(prototype, typology) or prototype
        normalized = self._format_polity_name_component(candidate).lower()
        base_candidate = candidate
        suffix = 2
        while disallow and normalized in disallow:
            candidate = f"{base_candidate} {suffix}"
            normalized = self._format_polity_name_component(candidate).lower()
            suffix += 1
        return candidate

    def _generate_region_placeholder_name(
        self,
        tile_index: int,
        region_label: Optional[str],
        *,
        disallow: Optional[Set[str]] = None,
        exclude: Optional[str] = None,
    ) -> Optional[str]:
        cached = self._region_placeholder_names.get(tile_index)
        if cached:
            return cached
        base = (region_label or "").strip()
        attempts = 0
        while attempts < 8:
            syllable_count = self._language_rng.randint(1, 2)
            segments: List[str] = []
            for _ in range(syllable_count):
                onset = self._language_rng.choice(FALLBACK_CONSONANTS)
                vowel = self._language_rng.choice(FALLBACK_VOWELS)
                coda = self._language_rng.choice(FALLBACK_CONSONANTS) if self._language_rng.random() < 0.35 else ""
                segments.append(f"{onset}{vowel}{coda}")
            suffix = ''.join(segments).capitalize()
            candidate = f"{base} {suffix}".strip() or f"Region {tile_index + 1}"
            formatted = self._format_settlement_display_name(candidate)
            normalized = self._format_polity_name_component(formatted or candidate)
            lowered = normalized.lower() if normalized else None
            if lowered and disallow and lowered in disallow:
                attempts += 1
                continue
            unique = self._ensure_unique_settlement_name(formatted or candidate, exclude=exclude)
            normalized_unique = self._format_polity_name_component(unique)
            lowered_unique = normalized_unique.lower() if normalized_unique else None
            if lowered_unique and disallow and lowered_unique in disallow:
                attempts += 1
                continue
            self._region_placeholder_names[tile_index] = unique
            return unique
        return None

    def _determine_settlement_type(self, tile_index: int) -> str:
        """Determine settlement type based on terrain and geography."""
        if tile_index < 0 or tile_index >= len(self.tiles):
            return "plain"

        tile = self.tiles[tile_index]
        elevation = tile.elevation
        biome = tile.biome.lower()

        # Determine type based on elevation and biome
        if elevation > 0.7:
            return "mountain"
        elif elevation > 0.4:
            return "hill"
        elif biome in ["desert", "savanna", "steppe"]:
            return "desert"
        elif biome in ["forest", "taiga", "jungle"]:
            return "forest"
        elif biome in ["grassland", "prairie", "tundra"]:
            return "plain"
        elif self._is_near_water(tile_index):
            if elevation < 0.1:  # Near sea level
                return "coastal"
            else:
                return "river"
        else:
            return "plain"

    def _is_near_water(self, tile_index: int) -> bool:
        """Check if tile is adjacent to water."""
        if tile_index < 0 or tile_index >= len(self.tiles):
            return False

        tile = self.tiles[tile_index]
        for neighbor_idx in tile.neighbors:
            if neighbor_idx >= 0 and neighbor_idx < len(self.tiles):
                neighbor = self.tiles[neighbor_idx]
                if neighbor.is_water:
                    return True
        return tile.is_water

    def _generate_settlement_name(
        self,
        tile_index: int,
        *,
        prefer_capital: bool = False,
        fallback: str = "Tribal Settlement",
        exclude_name: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        if tile_index < 0 or tile_index >= len(self.tiles):
            return fallback, None
        tile = self.tiles[tile_index]
        name = None
        phonetic_name = None
        disallow_tokens = self._collect_settlement_name_disallow(exclude=exclude_name)
        majority_culture = self._get_tile_majority_culture(tile)

        # Determine settlement type for more realistic naming
        enable_terrain = self.config.get('linguistics.settlement_naming.enable_terrain_based_categories', True) if self.config else True
        if enable_terrain:
            settlement_type = self._determine_settlement_type(tile_index)
            type_categories = SETTLEMENT_TYPE_PATTERNS.get(settlement_type, SETTLEMENT_NAME_CATEGORIES)
        else:
            type_categories = SETTLEMENT_NAME_CATEGORIES

        if majority_culture:
            existing_lexeme = self._get_settlement_word_form(majority_culture, tile_index)
            if existing_lexeme:
                formatted = self._format_settlement_display_name(existing_lexeme)
                if formatted:
                    self._region_placeholder_names.pop(tile_index, None)
                    name = formatted
                    phonetic_name = existing_lexeme
            if not name:
                token = self._select_culture_language_token(
                    majority_culture,
                    type_categories,  # Use type-specific categories
                    disallow=disallow_tokens or None,
                )
                if not token:
                    token = self._synthesize_settlement_token_for_culture(
                        majority_culture,
                        tile_index,
                        disallow=disallow_tokens or None,
                    )
                if token:
                    display = self._format_settlement_display_name(token)
                    if display:
                        self._upsert_settlement_word_for_culture(majority_culture, tile_index, display)
                        normalized = self._format_polity_name_component(display)
                        if normalized:
                            disallow_tokens.add(normalized.lower())
                        self._region_placeholder_names.pop(tile_index, None)
                        name = display
                        phonetic_name = token
        region_id = tile.region_id
        region_token = self.region_name_tokens.get(region_id) if region_id is not None else None
        if not name:
            placeholder = self._generate_region_placeholder_name(
                tile_index,
                region_token,
                disallow=disallow_tokens or None,
                exclude=exclude_name,
            )
            if placeholder:
                name = placeholder
        if not name and region_token:
            name = region_token
        if not name:
            name = fallback
        if prefer_capital and not name.endswith("Capital"):
            name = f"{name} Capital"
        
        # Ensure we have a phonetic form - if not from language, use display name
        if not phonetic_name:
            phonetic_name = name  # Fallback: use display name as phonetic
        
        return self._ensure_unique_settlement_name(name, exclude=exclude_name), phonetic_name

    def _rename_population_center_for_tile(self, tile_index: int) -> None:
        center = self._get_population_center_for_tile(tile_index)
        if not center:
            return
        if self._update_population_center_name_for_center(center):
            return
        name = center.name or ""
        default_like = (
            not name or
            name.startswith("Settlement_") or
            name.startswith("Tribal Settlement")
        )
        tile = self.tiles[tile_index]
        region_like = False
        region_id = getattr(tile, 'region_id', None)
        if region_id is not None:
            region_token = self.region_name_tokens.get(region_id)
            if region_token:
                normalized_region = region_token.strip()
                if normalized_region:
                    region_like = (
                        name == normalized_region or
                        name.startswith(f"{normalized_region} ") or
                        name.startswith(f"{normalized_region} Capital")
                    )
        default_like = default_like or region_like
        if not default_like:
            return
        prefer_capital = name.endswith("Capital")
        new_display_name, new_phonetic_name = self._generate_settlement_name(
            tile_index,
            prefer_capital=prefer_capital,
            exclude_name=center.name
        )
        if new_display_name != center.name:
            culture = self._get_tile_majority_culture(tile)
            # Preserve historical name before changing
            self._preserve_historical_name_layers(center, new_display_name, "culture_shift")
            self._set_population_center_name(
                center,
                new_display_name,
                "placeholder_refresh",
                phonetic_name=new_phonetic_name,
                culture=culture,
                note="auto_regen",
            )

    def _update_population_center_name_for_center(
        self,
        center: PopulationCenter,
        *,
        preferred_culture: Optional[str] = None
    ) -> bool:
        if not center or center.tile_index >= len(self.tiles):
            return False
        tile_idx = center.tile_index
        tile = self.tiles[tile_idx]
        culture_name = preferred_culture
        if not culture_name:
            polity = self._get_polity(tile.polity_id)
            culture_name = getattr(polity, 'primary_culture', None)
        if not culture_name:
            return False
        lexeme = self._get_settlement_word_form(culture_name, tile_idx)
        if not lexeme:
            return False
        display = self._format_settlement_display_name(lexeme)
        if not display:
            return False
        changed = self._set_population_center_name(
            center,
            display,
            "language_refresh",
            culture=culture_name,
            note=f"lexeme:{tile_idx}",
        )
        if changed:
            self._region_placeholder_names.pop(center.tile_index, None)
        return changed

    def _get_settlement_word_form(self, culture_name: Optional[str], tile_index: int) -> Optional[str]:
        if not culture_name:
            return None
        catalog = self.get_culture_language(culture_name)
        if catalog is None:
            return None
        gloss = str(tile_index)
        for word in catalog.words:
            category = (word.category or "").lower()
            if category == 'settlement' and str(word.gloss) == gloss:
                return word.phonetic_form
        return None

    def _get_tile_cultures_above_threshold(self, tile: Tile, threshold: float = 0.01) -> Set[str]:
        cultures: Set[str] = set()
        if not tile.cultural_makeup:
            return cultures
        total = sum(max(0.0, share) for share in tile.cultural_makeup.values())
        if total <= 0:
            for name, share in tile.cultural_makeup.items():
                if share > 0:
                    cultures.add(name)
            return cultures
        for name, share in tile.cultural_makeup.items():
            if share / total >= threshold:
                cultures.add(name)
        return cultures

    def _upsert_settlement_word_for_culture(self, culture_name: str, tile_index: int, label: str) -> None:
        if not culture_name:
            return
        catalog = self.get_culture_language(culture_name)
        if catalog is None:
            culture = self._find_culture_by_name(culture_name)
            if culture is None:
                return
            catalog = self.assign_base_language(culture)
        if catalog is None:
            return
        gloss = str(tile_index)
        for word in catalog.words:
            category = (word.category or "").lower()
            if category == 'settlement' and str(word.gloss) == gloss:
                word.phonetic_form = label
                return
        catalog.words.append(
            Word(
                phonetic_form=label,
                gloss=gloss,
                category='settlement'
            )
        )

    def _handle_population_center_polity_claim(self, tile_index: int, new_polity_id: int) -> None:
        if tile_index < 0 or tile_index >= len(self.tiles) or new_polity_id < 0:
            return
        center = self._get_population_center_for_tile(tile_index)
        if not center:
            return
        tile = self.tiles[tile_index]
        polity = self._get_polity(new_polity_id)
        base_display_name, base_phonetic_name = center.name, center.phonetic_name
        if not base_display_name:
            base_display_name, base_phonetic_name = self._generate_settlement_name(tile_index)
        base_label = base_phonetic_name or base_display_name
        eligible_cultures = self._get_tile_cultures_above_threshold(tile, 0.01)
        if polity and polity.primary_culture:
            eligible_cultures.add(polity.primary_culture)
        for culture_name in eligible_cultures:
            self._upsert_settlement_word_for_culture(culture_name, tile_index, base_label)
        preferred_culture = polity.primary_culture if polity else None
        if preferred_culture:
            self._update_population_center_name_for_center(center, preferred_culture=preferred_culture)

    def _refresh_population_center_names_for_primary_culture(self, culture_name: Optional[str]) -> None:
        """Refresh names for centers governed by polities whose primary culture matches culture_name.

        Args:
            culture_name: Name of the culture serving (or about to serve) as a polity's primary culture.
        """
        if not culture_name:
            return
        for center in self.population_centers:
            tile_idx = center.tile_index
            if tile_idx < 0 or tile_idx >= len(self.tiles):
                continue
            tile = self.tiles[tile_idx]
            polity = self._get_polity(tile.polity_id)
            if polity and polity.primary_culture == culture_name:
                self._update_population_center_name_for_center(center, preferred_culture=culture_name)

    def _refresh_population_center_names_for_polity(self, polity_id: int) -> None:
        polity = self._get_polity(polity_id)
        if not polity or not polity.primary_culture:
            return
        for center in self.population_centers:
            tile_idx = center.tile_index
            if tile_idx < 0 or tile_idx >= len(self.tiles):
                continue
            if self.tiles[tile_idx].polity_id == polity_id:
                self._update_population_center_name_for_center(center, preferred_culture=polity.primary_culture)

    def _get_polity_rank_percentiles(self) -> Tuple[float, float]:
        def _clamp_percentile(value: Any, default: float) -> float:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = default
            return max(0.0, min(1.0, numeric))

        kingdom_default = 0.3
        empire_default = 0.8
        if self.config:
            kingdom_pct = _clamp_percentile(
                self.config.get('simulation.polity.rank_percentiles.kingdom'),
                kingdom_default,
            )
            empire_pct = _clamp_percentile(
                self.config.get('simulation.polity.rank_percentiles.empire'),
                empire_default,
            )
        else:
            kingdom_pct = kingdom_default
            empire_pct = empire_default
        if empire_pct < kingdom_pct:
            empire_pct = kingdom_pct
        return kingdom_pct, empire_pct

    def _percentile_value(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate the value at a given percentile from a sorted list."""
        if not sorted_values:
            return 0.0
        
        if percentile <= 0.0:
            return sorted_values[0]
        if percentile >= 1.0:
            return sorted_values[-1]
        
        # Calculate the index
        index = percentile * (len(sorted_values) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        
        # Linear interpolation
        if lower_index == upper_index:
            return sorted_values[lower_index]
        
        fraction = index - lower_index
        return sorted_values[lower_index] + fraction * (sorted_values[upper_index] - sorted_values[lower_index])

    def _compute_polity_rank_thresholds(self) -> Tuple[float, ...]:
        values = sorted(
            self._calculate_polity_development_value(polity.id)
            for polity in self.polities
            if polity and polity.is_active
        )
        if not values:
            return (0.0, 0.0)  # 2 thresholds for 3 ranks

        # Get percentiles from config or use defaults
        config_thresholds = self.config.get('linguistics.polity_naming.rank_thresholds', {}) if self.config else {}
        duchy_pct = config_thresholds.get('duchy_percentile', 0.3)
        kingdom_pct = config_thresholds.get('kingdom_percentile', 0.7)

        percentiles = [duchy_pct, kingdom_pct]
        thresholds = tuple(self._percentile_value(values, pct) for pct in percentiles)
        return thresholds

    def _rank_for_development(self, dev_value: float, thresholds: Tuple[float, ...]) -> str:
        duchy_threshold, kingdom_threshold = thresholds

        if dev_value >= kingdom_threshold:
            return "Empire"
        elif dev_value >= duchy_threshold:
            return "Kingdom"
        else:
            return "Duchy"

    def _rank_value(self, rank: Optional[str]) -> int:
        ordering = {
            "Duchy": 0,
            "Kingdom": 1,
            "Empire": 2
        }
        return ordering.get(rank or "", -1)

    def _is_polity_empire_rank(self, polity: Optional[Polity]) -> bool:
        """Return True when a polity qualifies for the configured empire tier."""
        if polity is None:
            return False
        return self._rank_value(getattr(polity, 'title_rank', None)) >= self._rank_value("Empire")

    def _apply_polity_rank(self, polity: Polity, desired_rank: str, *, allow_downgrade: bool) -> bool:
        current_value = self._rank_value(polity.title_rank)
        desired_value = self._rank_value(desired_rank)
        if current_value >= 0 and desired_value < current_value and not allow_downgrade:
            desired_rank = polity.title_rank or desired_rank
            desired_value = current_value
        if desired_value < 0:
            desired_rank = polity.title_rank or "Tribe"
            desired_value = self._rank_value(desired_rank)
        polity.title_rank = desired_rank
        # Keep existing polity names stable; rank changes no longer rename the polity
        return desired_value > current_value

    def _process_polity_rank_updates(self) -> None:
        named_polities = [
            polity for polity in self.polities
            if polity.language_name_component
        ]
        if not named_polities:
            return
        thresholds = self._compute_polity_rank_thresholds()
        for polity in named_polities:
            dev_value = self._calculate_polity_development_value(polity.id)
            desired_rank = self._rank_for_development(dev_value, thresholds)
            upgraded = self._apply_polity_rank(polity, desired_rank, allow_downgrade=False)
            if upgraded:
                self._log_event(
                    "polity",
                    f"{polity.name} elevated to {desired_rank} via development growth",
                )

    def _process_language_time_shifts(self) -> None:
        interval = getattr(self, 'language_time_shift_years', 200)
        if interval <= 0:
            return
        updated_count = 0
        for culture in self.cultures:
            if not culture.language_name:
                continue
            last_year = culture.language_last_update_year
            if last_year is None:
                last_year = culture.birth_year
            if last_year is None:
                last_year = self.current_year
            if self.current_year - last_year < interval:
                continue
            catalog = self.culture_languages.get(culture.name)
            if catalog is None:
                continue
            updated = transform_language(
                catalog,
                transformation_type="time_evolution",
                time_depth=1,
                rng=self._language_rng,
            )
            self._record_culture_language(
                culture,
                updated,
                parent=catalog.name,
                transformation="time_evolution",
                time_depth=1,
            )
            self._shift_culture_ideas(culture.name, "language_shift", intensity=0.1)
            # Apply sound changes to settlement names for this culture
            self._apply_language_evolution_to_settlement_names(culture.name)
            updated_count += 1
        if updated_count:
            self._log_event(
                "culture",
                f"[language] Time-shifted {updated_count} culture language(s) in Year {self.current_year}",
            )

    def _serialize_rng_state(self, state: Optional[Tuple[Any, ...]]) -> Optional[Dict[str, Any]]:
        """Convert a Random.getstate() tuple into JSON-friendly primitives."""

        if not state or len(state) < 3:
            return None
        version, sequence, gaussian = state
        try:
            version_int = int(version)
            sequence_list = [int(value) for value in sequence]
        except (TypeError, ValueError):
            return None
        payload: Dict[str, Any] = {
            'version': version_int,
            'state': sequence_list,
            'gauss': None
        }
        if gaussian is not None:
            try:
                payload['gauss'] = float(gaussian)
            except (TypeError, ValueError):
                payload['gauss'] = None
        return payload

    def _deserialize_rng_state(self, payload: Optional[Dict[str, Any]]) -> Optional[Tuple[Any, ...]]:
        """Recreate a Random state tuple from serialized data."""

        if not payload:
            return None
        version = payload.get('version')
        sequence = payload.get('state')
        if version is None or sequence is None:
            return None
        try:
            version_int = int(version)
            sequence_tuple = tuple(int(value) for value in sequence)
        except (TypeError, ValueError):
            return None
        gaussian_value = payload.get('gauss')
        if gaussian_value is None:
            gaussian = None
        else:
            try:
                gaussian = float(gaussian_value)
            except (TypeError, ValueError):
                gaussian = None
        return (version_int, sequence_tuple, gaussian)

    def _serialize_language_system_state(self) -> Dict[str, Any]:
        """Capture catalog metadata so saves can restore linguistic hooks."""

        state: Dict[str, Any] = {
            'catalog_dir': str(self.language_catalog_dir) if self.language_catalog_dir else None,
            'rng_seed': self._language_rng_seed,
            'initialized': self._language_initialized,
            'time_shift_years': self.language_time_shift_years,
        }
        if self._language_initialized and self._language_rng:
            rng_state = self._serialize_rng_state(self._language_rng.getstate())
            if rng_state:
                state['rng_state'] = rng_state
        return state

    def _restore_language_system_state(self, payload: Optional[Dict[str, Any]]) -> None:
        """Rebuild the language system according to serialized metadata."""

        if not payload:
            return
        catalog_dir = payload.get('catalog_dir')
        rng_seed = payload.get('rng_seed')
        initialized = bool(payload.get('initialized', False))
        if catalog_dir:
            try:
                self.language_catalog_dir = Path(catalog_dir)
            except (TypeError, ValueError):  # pragma: no cover - best effort
                self.language_catalog_dir = None
        else:
            self.language_catalog_dir = None
        if rng_seed is not None:
            try:
                self._language_rng_seed = int(rng_seed)
            except (TypeError, ValueError):
                self._language_rng_seed = None
        else:
            self._language_rng_seed = None
        rng_state = self._deserialize_rng_state(payload.get('rng_state'))
        if initialized and self.language_catalog_dir:
            self.initialize_language_system(self.language_catalog_dir, seed=self._language_rng_seed)
            if rng_state and self._language_rng:
                try:
                    self._language_rng.setstate(rng_state)
                except Exception:  # pragma: no cover - defensive
                    pass
        else:
            self._language_initialized = initialized
            if rng_state and self._language_rng:
                try:
                    self._language_rng.setstate(rng_state)
                except Exception:  # pragma: no cover - defensive
                    pass

    def _leader_settings(self) -> Dict[str, Any]:
        if not self.config:
            return {}
        settings = self.config.get('simulation.leaders')
        return settings or {}

    def _leader_traits_enabled(self) -> bool:
        settings = self._leader_settings()
        return bool(settings.get('enable_traits', True))

    def _get_trait_settings(self, trait: str) -> Dict[str, Any]:
        settings = self._leader_settings()
        trait_map = settings.get('traits') or {}
        return trait_map.get((trait or '').lower(), {})

    def _polity_has_trait(self, polity: Optional[Polity], trait: str) -> bool:
        if polity is None or not self._leader_traits_enabled():
            return False
        leader = getattr(polity, 'leader', None)
        if leader is None:
            return False
        trait_upper = (trait or '').upper()
        return any((t or '').upper() == trait_upper for t in getattr(leader, 'traits', []) or [])

    def _get_trait_value(
        self,
        polity_or_id: Optional[Any],
        trait: str,
        key: str,
        default: float = 1.0
    ) -> float:
        polity = None
        if isinstance(polity_or_id, int):
            polity = self._get_polity(polity_or_id)
        else:
            polity = polity_or_id
        if not self._polity_has_trait(polity, trait):
            return default
        trait_settings = self._get_trait_settings(trait)
        return trait_settings.get(key, default)

    def _get_leader_rng(self, polity_id: int, generation_index: int, salt: str) -> random.Random:
        base_seed = self.world_seed if self.world_seed is not None else 0
        safe_generation = max(1, generation_index)
        payload = f"{base_seed}:{polity_id}:{safe_generation}:{salt}"
        digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        seed_int = int(digest[:16], 16)
        return random.Random(seed_int)

    def _roll_leader_age(
        self,
        rng: random.Random,
        *,
        base_age: Optional[int] = None,
        max_offset: int = 20,
        outlier_chance: float = 0.12
    ) -> int:
        """Roll a leader age, optionally biasing around a provided baseline."""
        settings = self._leader_settings()
        min_age = int(settings.get('min_age', 28))
        max_age = int(settings.get('max_age', 75))
        if max_age < min_age:
            max_age = min_age

        if base_age is None:
            return rng.randint(min_age, max_age)

        offset = rng.randint(-max_offset, max_offset)
        if rng.random() < max(0.0, min(1.0, outlier_chance)):
            # Rare outliers can stretch the range modestly
            outlier_span = int(max_offset * 1.5)
            offset = rng.randint(-outlier_span, outlier_span)

        target_age = base_age + offset
        return max(min_age, min(max_age, target_age))

    def _roll_leader_tenure(self, polity: Optional[Polity], generation_index: int) -> int:
        settings = self._leader_settings()
        min_years = max(1, int(settings.get('min_tenure_years', 10)))
        max_years = max(min_years, int(settings.get('max_tenure_years', 80)))
        if max_years == min_years:
            return max_years
        alpha = max(0.1, float(settings.get('tenure_beta_alpha', 2.0)))
        beta = max(0.1, float(settings.get('tenure_beta_beta', 3.0)))
        polity_id = polity.id if polity is not None else -1
        rng = self._get_leader_rng(polity_id, generation_index, 'tenure')
        sample = rng.betavariate(alpha, beta)
        tenure = int(round(min_years + (max_years - min_years) * sample))
        return max(min_years, min(max_years, tenure))

    def _roll_trait_slot_count(self, rng: random.Random) -> int:
        weights = self._leader_settings().get('trait_count_weights', {"0": 0.35, "1": 0.45, "2": 0.2})
        parsed: List[Tuple[int, float]] = []
        for key, value in weights.items():
            try:
                parsed.append((max(0, min(2, int(key))), max(0.0, float(value))))
            except (TypeError, ValueError):
                continue
        if not parsed:
            return 0
        total_weight = sum(entry[1] for entry in parsed)
        if total_weight <= 0:
            return 0
        roll = rng.random() * total_weight
        cumulative = 0.0
        for count, weight in sorted(parsed, key=lambda item: item[0]):
            cumulative += weight
            if roll <= cumulative:
                return count
        return parsed[-1][0]

    def _generate_leader_traits(
        self,
        polity: Optional[Polity],
        previous_leader: Optional[Leader],
        generation_index: int
    ) -> List[str]:
        if not self._leader_traits_enabled():
            return []
        polity_id = polity.id if polity is not None else -1
        rng = self._get_leader_rng(polity_id, generation_index, 'traits')
        slot_count = self._roll_trait_slot_count(rng)
        if slot_count <= 0:
            return []
        traits: List[str] = []
        prev_traits = []
        if previous_leader and getattr(previous_leader, 'traits', None):
            prev_traits = [(t or '').upper() for t in previous_leader.traits]
        inherit_chance = float(self._leader_settings().get('trait_inheritance_chance', 0.25))
        if prev_traits and rng.random() < inherit_chance:
            inherited = rng.choice(prev_traits)
            traits.append(inherited)
        available = [trait for trait in LEADER_TRAIT_POOL if trait not in traits]
        while len(traits) < slot_count and available:
            pick = rng.choice(available)
            traits.append(pick)
            conflicts = LEADER_TRAIT_CONFLICTS.get(pick, set())
            available = [trait for trait in available if trait != pick and trait not in conflicts]
        return traits

    def _generate_leader_name(
        self,
        polity: Optional[Polity],
        culture: Optional[str],
        generation_index: int,
        *,
        inherited_dynasty: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        polity_id = polity.id if polity is not None else -1
        rng = self._get_leader_rng(polity_id, generation_index, 'name')

        def _pick_personal_name(culture_name: Optional[str]) -> Optional[str]:
            token = self._select_culture_language_token(culture_name, ["personal_name"], allow_fallback=False) if culture_name else None
            if not token:
                token = self._select_culture_language_token(culture_name, ["kinship", "settlement"], allow_fallback=True) if culture_name else None
            if token:
                formatted = self._format_polity_name_component(token)
                if formatted:
                    return formatted
            return None

        def _pick_dynasty_name(culture_name: Optional[str]) -> Optional[str]:
            token = self._select_culture_language_token(culture_name, ["kinship", "settlement", "geographic", "military"], allow_fallback=True) if culture_name else None
            if token:
                formatted = self._format_polity_name_component(token)
                if formatted:
                    return f"House {formatted}"
            return None

        personal = _pick_personal_name(culture)
        dynasty = inherited_dynasty or _pick_dynasty_name(culture)

        if personal and dynasty:
            return f"{personal} of {dynasty}", dynasty
        if personal:
            return personal, dynasty
        # Fallback to legacy title/epithet when catalogs lack names
        titles = [
            "Chancellor",
            "Marshal",
            "Dynast",
            "Archon",
            "Prefect",
            "Strategos",
            "Warden",
            "Patriarch",
        ]
        epithets = [
            "Resolute",
            "Cunning",
            "Bold",
            "Patient",
            "Vigilant",
            "Visionary",
            "Stoic",
            "Sage",
        ]
        base_name = culture if culture and culture != 'Unknown' else (polity.name.split()[0] if polity and polity.name else 'Council')
        title = rng.choice(titles)
        epithet = rng.choice(epithets)
        return f"{base_name} {title} {epithet}", None

    def _determine_primary_culture_for_leader(
        self,
        polity: Optional[Polity],
        capital_tile: Optional[Tile],
        generation_index: int
    ) -> Optional[str]:
        if polity is None or capital_tile is None or not capital_tile.cultural_makeup:
            return polity.primary_culture if polity else None
        weights: Dict[str, float] = {name: max(0.0, value) for name, value in capital_tile.cultural_makeup.items()}
        if not weights:
            return polity.primary_culture
        bonus = float(self._leader_settings().get('status_quo_bonus', 0.25))
        if polity.primary_culture:
            weights[polity.primary_culture] = weights.get(polity.primary_culture, 0.0) + max(0.0, bonus)
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return polity.primary_culture
        polity_id = polity.id if polity is not None else -1
        rng = self._get_leader_rng(polity_id, generation_index, 'culture_shift')
        roll = rng.random() * total_weight
        cumulative = 0.0
        for culture_name, weight in sorted(weights.items()):
            cumulative += weight
            if roll <= cumulative:
                return culture_name
        return polity.primary_culture

    def _appoint_new_leader(
        self,
        polity: Optional[Polity],
        inherit_from: Optional[Leader] = None,
        reason: str = "succession",
        *,
        preserve_dynasty: bool = True,
    ) -> None:
        if polity is None or not getattr(polity, 'is_active', True):
            return
        next_generation = max(0, getattr(polity, 'leader_generation', 0)) + 1
        capital_idx = self._ensure_polity_capital(polity)
        capital_tile = self.tiles[capital_idx] if capital_idx is not None and 0 <= capital_idx < len(self.tiles) else None
        new_primary = self._determine_primary_culture_for_leader(polity, capital_tile, next_generation)
        if new_primary and new_primary != polity.primary_culture:
            old_culture = polity.primary_culture or 'Unassigned'
            self._log_event(
                'culture',
                f"[culture] {polity.name} shifted primary culture from {old_culture} to {new_primary} under new leadership"
            )
            self._apply_polity_primary_culture(polity, new_primary)

        culture_label = polity.primary_culture or new_primary or 'Unknown'
        rng_age = self._get_leader_rng(polity.id, next_generation, 'age')
        fracture_contexts = {'tanistry_fracture', 'succession_crisis'}
        base_age = inherit_from.age if reason in fracture_contexts and inherit_from and getattr(inherit_from, 'age', None) else None
        leader_age = self._roll_leader_age(rng_age, base_age=base_age)
        term_years = self._roll_leader_tenure(polity, next_generation)
        traits = self._generate_leader_traits(polity, inherit_from, next_generation)
        inherited_dynasty = getattr(inherit_from, 'dynasty', None) if (preserve_dynasty and inherit_from) else None
        leader_name = self._generate_leader_name(
            polity,
            culture_label,
            next_generation,
            inherited_dynasty=inherited_dynasty
        )
        if isinstance(leader_name, tuple):
            display_name, dynasty_name = leader_name
        else:
            display_name, dynasty_name = leader_name, None
        polity.leader = Leader(
            name=display_name,
            dynasty=dynasty_name,
            age=leader_age,
            culture=culture_label,
            traits=traits,
            accession_year=self.current_year,
            term_years=term_years
        )
        polity.leader_generation = next_generation
        self._refresh_polity_naming(polity, reason="leadership_change")
        trait_note = f" traits={','.join(traits)}" if traits else ""
        self._log_polity_event(
            'polity',
            f"{polity.name} appointed {display_name} in Year {self.current_year} ({reason}).{trait_note}",
            polity,
        )
        if reason != 'initialization':
            self._apply_leadership_control_penalty(polity)

    def _ensure_polity_leader_state(self, polity: Optional[Polity], refresh_existing: bool = False) -> None:
        if polity is None or not getattr(polity, 'is_active', True):
            return
        if polity.leader_generation <= 0:
            polity.leader_generation = 1
        leader = getattr(polity, 'leader', None)
        if leader is None or refresh_existing:
            self._appoint_new_leader(polity, inherit_from=leader if refresh_existing else None, reason='initialization')
            return
        if leader.traits is None:
            leader.traits = []
        if leader.accession_year <= 0:
            leader.accession_year = self.current_year
        if leader.term_years <= 0:
            leader.term_years = self._roll_leader_tenure(polity, polity.leader_generation)
        if not leader.traits and self._leader_traits_enabled():
            leader.traits = self._generate_leader_traits(polity, None, polity.leader_generation)

    def initialize_leader_system(self, refresh_existing: bool = False) -> None:
        for polity in self.polities:
            self._ensure_polity_leader_state(polity, refresh_existing=refresh_existing)

    def _succession_crisis_settings(self) -> Dict[str, float]:
        defaults = {
            'base': 0.1,
            'control_anchor': 90.0,
            'per_point': 0.01,
            'cap': 0.95,
        }
        if not self.config:
            return defaults
        def _safe_float(value: Any, fallback: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback
        base = _safe_float(self.config.get('simulation.polity.succession_crisis_base_chance'), defaults['base'])
        base = max(0.0, min(1.0, base))
        anchor = _safe_float(self.config.get('simulation.polity.succession_crisis_control_anchor'), defaults['control_anchor'])
        anchor = max(0.0, min(100.0, anchor))
        per_point = _safe_float(self.config.get('simulation.polity.succession_crisis_chance_per_control_point'), defaults['per_point'])
        per_point = max(0.0, per_point)
        cap = _safe_float(self.config.get('simulation.polity.succession_crisis_chance_cap'), defaults['cap'])
        cap = max(0.0, min(1.0, cap))
        return {
            'base': base,
            'control_anchor': anchor,
            'per_point': per_point,
            'cap': cap,
        }

    def _tanistry_fracture_settings(self) -> Dict[str, float]:
        defaults = {
            'base': 0.25,
            'admin_per_point': 0.0,
            'cap': 0.8,
        }
        if not self.config:
            return defaults
        def _safe_float(value: Any, fallback: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback
        base = _safe_float(self.config.get('simulation.polity.tanistry_fracture_base_chance'), defaults['base'])
        base = max(0.0, min(1.0, base))
        admin_per = _safe_float(self.config.get('simulation.polity.tanistry_fracture_admin_chance_per_point'), defaults['admin_per_point'])
        admin_per = max(0.0, admin_per)
        cap = _safe_float(self.config.get('simulation.polity.tanistry_fracture_chance_cap'), defaults['cap'])
        cap = max(0.0, min(1.0, cap))
        return {
            'base': base,
            'admin_per_point': admin_per,
            'cap': cap,
        }

    def _compute_tanistry_fracture_chance(self, polity: Optional[Polity]) -> float:
        """Derive the probability that a tanistry succession triggers a fracture."""
        settings = self._tanistry_fracture_settings()
        chance = settings['base']
        if polity is not None:
            burden_raw = self.get_polity_administrative_burden(polity)
            burden_val = max(0.0, float(burden_raw or 0.0))
            chance += burden_val * settings['admin_per_point']
        chance = min(settings['cap'], chance)
        return max(0.0, min(0.95, chance))

    def _calculate_polity_average_control(self, polity: Optional[Polity]) -> float:
        if polity is None:
            return 100.0
        total = 0.0
        count = 0
        for tile_idx in getattr(polity, 'tile_indices', []) or []:
            if tile_idx < 0 or tile_idx >= len(self.tiles):
                continue
            tile = self.tiles[tile_idx]
            if tile.is_water or tile.polity_id != polity.id:
                continue
            total += tile.control_level
            count += 1
        if count == 0:
            return 100.0
        return total / count

    def _succession_crisis_chance_from_control(self, control_average: float) -> float:
        settings = self._succession_crisis_settings()
        clamped = max(0.0, min(100.0, float(control_average)))
        delta = settings['control_anchor'] - clamped
        chance = settings['base'] + delta * settings['per_point']
        return max(0.0, min(settings['cap'], chance))

    def _apply_succession_trait_modifier(self, chance: float, previous_leader: Optional[Leader]) -> float:
        if previous_leader is None:
            return max(0.0, min(0.95, chance))
        traits = {(trait or '').upper() for trait in getattr(previous_leader, 'traits', []) or []}
        reduction = 0.0
        for trait in ('DIPLOMAT', 'MACHIAVELLIAN', 'TRADITIONAL'):
            if trait in traits:
                reduction += 0.15
        adjusted = chance - reduction
        return max(0.0, min(0.95, adjusted))

    def _prepare_crisis_dynasty_title(self, polity: Optional[Polity], mode: str) -> None:
        if polity is None:
            return
        culture_name = getattr(polity, 'primary_culture', None)
        avoid_root = self._resolve_cultural_base_name(polity)
        mode_lower = (mode or 'minted').lower()
        if mode_lower == 'catalog':
            token = self._reserve_catalog_dynasty_title(
                polity,
                culture_name,
                avoid_root=avoid_root,
                force_new=False,
            )
            if token:
                polity.dynasty_title_token = token
                polity.dynasty_title_source = 'catalog'
                polity.dynasty_title_culture = culture_name
            return
        minted_title = self._mint_dynasty_title_token(polity, avoid_root=avoid_root)
        if minted_title:
            self._release_dynasty_title(polity)
            polity.dynasty_title_token = minted_title
            polity.dynasty_title_source = 'minted'
            polity.dynasty_title_culture = culture_name

    def _execute_succession_crisis(
        self,
        polity: Optional[Polity],
        previous_leader: Optional[Leader],
        universalism_label: str,
    ) -> None:
        if polity is None:
            return
        label = (universalism_label or '').lower()
        if 'universal' in label:
            crisis_mode = 'minted'
        elif 'hybrid' in label:
            crisis_mode = 'catalog'
        else:
            crisis_mode = 'minted'
        polity_label = getattr(polity, 'name', f"Polity {polity.id}")
        self._log_polity_event(
            'succession_crisis',
            f"[succession] {polity_label} triggered a succession crisis ({crisis_mode})",
            polity,
        )
        self._prepare_crisis_dynasty_title(polity, crisis_mode)
        self._execute_tanistry_fracture(
            polity,
            previous_leader,
            share_dynasty=True,
            spawn_flag=False,
            fracture_label='succession crisis',
            naming_reason='succession_crisis',
        )

    def _maybe_trigger_succession_crisis(
        self,
        polity: Optional[Polity],
        previous_leader: Optional[Leader],
        next_generation: int,
    ) -> bool:
        if polity is None or previous_leader is None:
            return False
        avg_control = self._calculate_polity_average_control(polity)
        chance = self._succession_crisis_chance_from_control(avg_control)
        chance = self._apply_succession_trait_modifier(chance, previous_leader)
        if chance <= 0.0:
            return False
        crisis_rng = self._get_leader_rng(polity.id, next_generation, 'succession_crisis')
        if crisis_rng.random() >= chance:
            return False
        universalism_label = (self._get_polity_idea_label(polity, 'universalism') or '').lower()
        self._execute_succession_crisis(polity, previous_leader, universalism_label)
        return True

    def _process_leadership_cycle(self) -> None:
        for polity in self.polities:
            if polity is None or not getattr(polity, 'is_active', True):
                continue
            self._ensure_polity_leader_state(polity)
            leader = getattr(polity, 'leader', None)
            if leader is None:
                continue
            leader.age += 1
            if leader.term_years <= 0:
                leader.term_years = self._roll_leader_tenure(polity, polity.leader_generation)
            years_served = max(0, self.current_year - leader.accession_year)
            if years_served >= leader.term_years:
                previous_leader = leader
                succession_label = (self._get_polity_idea_label(polity, 'succession') or '').lower()
                next_generation = max(0, getattr(polity, 'leader_generation', 0)) + 1
                if 'tanistry' in succession_label:
                    tanistry_rng = self._get_leader_rng(polity.id, next_generation, 'tanistry_roll')
                    share_dynasty = False
                    fracture_chance = self._compute_tanistry_fracture_chance(polity)
                    if tanistry_rng.random() < fracture_chance:
                        self._execute_tanistry_fracture(polity, previous_leader, share_dynasty=share_dynasty)
                    else:
                        self._appoint_new_leader(
                            polity,
                            inherit_from=previous_leader,
                            reason='tanistry_passed',
                            preserve_dynasty=share_dynasty,
                        )
                else:
                    crisis_triggered = False
                    if 'primogeniture' in succession_label:
                        crisis_triggered = self._maybe_trigger_succession_crisis(polity, previous_leader, next_generation)
                    if not crisis_triggered:
                        self._appoint_new_leader(polity, inherit_from=previous_leader, reason='term_expired')

    def _execute_tanistry_fracture(
        self,
        polity: Optional[Polity],
        previous_leader: Optional[Leader],
        *,
        share_dynasty: bool = True,
        spawn_flag: bool = True,
        fracture_label: str = 'tanistry',
        naming_reason: str = 'tanistry_fracture',
    ) -> None:
        """Split a polity into rival claimant shards and start an immediate civil war."""
        if polity is None or not getattr(polity, 'tile_indices', None):
            return

        active_tiles = [idx for idx in polity.tile_indices if 0 <= idx < len(self.tiles) and not self.tiles[idx].is_water]
        if len(active_tiles) < 6:
            # Too small to fracture meaningfully; fall back to a normal succession
            self._appoint_new_leader(polity, inherit_from=previous_leader, reason='term_expired')
            return

        # Scale shards with polity size: 2 minimum, up to 5 for very large realms
        tile_count = len(active_tiles)
        if tile_count >= 30:
            target_shards = 5
        elif tile_count >= 24:
            target_shards = 4
        elif tile_count >= 12:
            target_shards = 3
        else:
            target_shards = 2
        centers = [
            center.tile_index
            for center in self.population_centers
            if 0 <= center.tile_index < len(self.tiles)
            and self.tiles[center.tile_index].polity_id == polity.id
            and not self.tiles[center.tile_index].is_water
        ]
        centers = sorted(set(centers), key=lambda idx: (self.tiles[idx].population, self.tiles[idx].development), reverse=True)

        seeds: List[int] = []
        for center_idx in centers:
            if len(seeds) >= target_shards:
                break
            seeds.append(center_idx)

        rng = self._get_leader_rng(polity.id, getattr(polity, 'leader_generation', 0) or 1, 'tanistry')
        while len(seeds) < target_shards and active_tiles:
            candidate = rng.choice(active_tiles)
            if candidate not in seeds:
                seeds.append(candidate)

        seeds = [seed for seed in seeds if seed in active_tiles]
        if len(seeds) < 2:
            self._appoint_new_leader(polity, inherit_from=previous_leader, reason='term_expired')
            return

        allowed: Set[int] = set(active_tiles)
        cluster_map: Dict[int, int] = {}
        queue: deque = deque()
        for cluster_id, seed_idx in enumerate(seeds):
            cluster_map[seed_idx] = cluster_id
            queue.append(seed_idx)

        while queue:
            current = queue.popleft()
            cluster_id = cluster_map[current]
            for neighbor_idx in self.tiles[current].neighbors:
                if neighbor_idx not in allowed or neighbor_idx in cluster_map:
                    continue
                neighbor = self.tiles[neighbor_idx]
                if neighbor.is_water or neighbor.polity_id != polity.id:
                    continue
                cluster_map[neighbor_idx] = cluster_id
                queue.append(neighbor_idx)

        if len(cluster_map) < len(active_tiles):
            default_cluster = 0
            for tile_idx in active_tiles:
                if tile_idx not in cluster_map:
                    cluster_map[tile_idx] = default_cluster

        clusters: Dict[int, List[int]] = defaultdict(list)
        for tile_idx, cluster_id in cluster_map.items():
            clusters[cluster_id].append(tile_idx)

        if len(clusters) < 2:
            self._appoint_new_leader(polity, inherit_from=previous_leader, reason='term_expired')
            return

        parent_cluster_id = max(
            clusters.items(),
            key=lambda item: (polity.capital_tile_index in item[1], len(item[1]))
        )[0]
        parent_seed = seeds[parent_cluster_id] if parent_cluster_id < len(seeds) else clusters[parent_cluster_id][0]

        cluster_to_polity: Dict[int, int] = {parent_cluster_id: polity.id}
        shard_ids: List[int] = [polity.id]
        new_polities: List[Polity] = []
        crisis_spawn = naming_reason == 'succession_crisis'
        dynasty_token = ""
        if share_dynasty and previous_leader:
            dynasty_token = self._normalize_dynasty_token(getattr(previous_leader, 'dynasty', None))
        shared_dynasty_title: Optional[str] = None
        shared_dynasty_culture: Optional[str] = None
        if share_dynasty:
            shared_dynasty_culture = getattr(polity, 'dynasty_title_culture', None) or getattr(polity, 'primary_culture', None)
            existing_token = getattr(polity, 'dynasty_title_token', None)
            existing_source = getattr(polity, 'dynasty_title_source', None)
            if existing_token:
                shared_dynasty_title = getattr(polity, 'dynasty_title_token', None) or existing_token
                shared_dynasty_culture = getattr(polity, 'dynasty_title_culture', None) or shared_dynasty_culture
                if existing_source == 'catalog' and shared_dynasty_culture:
                    self._set_culture_dynasty_reservation(shared_dynasty_culture, existing_token, polity.id)
            if not shared_dynasty_title:
                avoid_root = self._resolve_cultural_base_name(polity)
                minted_title = self._mint_dynasty_title_token(polity, avoid_root=avoid_root)
                if minted_title:
                    self._release_dynasty_title(polity)
                    polity.dynasty_title_token = minted_title
                    polity.dynasty_title_source = 'minted'
                    polity.dynasty_title_culture = getattr(polity, 'primary_culture', None)
                    shared_dynasty_title = minted_title
                    shared_dynasty_culture = polity.dynasty_title_culture

        for cluster_id, tile_list in clusters.items():
            if cluster_id == parent_cluster_id:
                continue
            seed_idx = seeds[cluster_id] if cluster_id < len(seeds) else tile_list[0]
            new_id = len(self.polities)
            color = self._generate_polity_color(new_id)
            name_hint = self._derive_city_label(seed_idx) or polity.name
            polity_name = self._generate_breakaway_polity_name(name_hint, polity.primary_culture or name_hint)
            if dynasty_token:
                style = self._get_dynastic_style_for_culture(polity.primary_culture)
                polity_name = self._format_dynastic_polity_name(polity, dynasty_token, style)

            new_polity = Polity(
                id=new_id,
                name=polity_name,
                color=color,
                leader=None,
                primary_culture=polity.primary_culture,
                tile_indices=[],
                capital_tile_index=seed_idx,
                spawned_from_explosion=spawn_flag,
                spawned_from_succession_crisis=crisis_spawn,
            )
            if shared_dynasty_title:
                new_polity.dynasty_title_token = shared_dynasty_title
                new_polity.dynasty_title_source = 'minted'
                new_polity.dynasty_title_culture = shared_dynasty_culture
            self.polities.append(new_polity)
            cluster_to_polity[cluster_id] = new_id
            shard_ids.append(new_id)
            new_polities.append(new_polity)

        def control_target(tile_idx: int, is_capital: bool) -> int:
            tile = self.tiles[tile_idx]
            base = 80 if is_capital else 60
            return max(25, min(90, int(min(tile.control_level, base))))

        for cluster_id, tile_list in clusters.items():
            if cluster_id == parent_cluster_id:
                continue
            target_polity_id = cluster_to_polity[cluster_id]
            seed_idx = seeds[cluster_id] if cluster_id < len(seeds) else tile_list[0]
            for tile_idx in tile_list:
                control_value = control_target(tile_idx, tile_idx == seed_idx)
                self._transfer_tile_to_polity(tile_idx, target_polity_id, control_value)

        parent_tiles = sorted(clusters[parent_cluster_id])
        polity.tile_indices = parent_tiles
        polity.capital_tile_index = parent_seed
        for tile_idx in parent_tiles:
            tile = self.tiles[tile_idx]
            target_value = control_target(tile_idx, tile_idx == parent_seed)
            if tile.control_level != target_value:
                before = tile.control_level
                tile.control_level = target_value
                self._record_control_change(tile, before, target_value, label="Tanistry Fracture", category="flat")

        self._ensure_polity_capital(polity)
        self._assign_polity_language_name(polity)
        self._refresh_polity_naming(polity, reason=naming_reason)
        self._appoint_new_leader(
            polity,
            inherit_from=previous_leader,
            reason=naming_reason,
            preserve_dynasty=share_dynasty,
        )

        for shard in new_polities:
            self._ensure_polity_capital(shard)
            self._assign_polity_language_name(shard)
            self._refresh_polity_naming(shard, reason=naming_reason)
            self._appoint_new_leader(
                shard,
                inherit_from=previous_leader,
                reason=naming_reason,
                preserve_dynasty=share_dynasty,
            )

        self._relationship_borders_initialized = False
        for shard_id in shard_ids:
            self._ensure_relationships_for_polity(shard_id)
        for i in range(len(shard_ids)):
            for j in range(i + 1, len(shard_ids)):
                relationship = self._get_relationship(shard_ids[i], shard_ids[j], create=True)
                if relationship:
                    relationship.met = True
                    self._begin_war(relationship, initiator_id=shard_ids[i], target_id=shard_ids[j])

        self._log_event(
            'polity',
            f"{polity.name} fractured into {len(shard_ids)} {fracture_label} claimant(s) in Year {self.current_year}"
        )

    def _apply_leader_tile_bonuses(self) -> None:
        if not self._leader_traits_enabled():
            return
        self._apply_charismatic_control_bonus()
        self._apply_military_acclamation_bonus()

    def _apply_charismatic_control_bonus(self) -> None:
        trait_settings = self._get_trait_settings('CHARISMATIC')
        pull_ratio = float(trait_settings.get('control_pull_ratio', 0.04))
        max_bonus = int(trait_settings.get('max_bonus_per_tick', 4))
        min_bonus = int(trait_settings.get('min_bonus_per_tick', 1))
        if pull_ratio <= 0:
            return
        for polity in self.polities:
            if not self._polity_has_trait(polity, 'CHARISMATIC'):
                continue
            for tile_idx in getattr(polity, 'tile_indices', []) or []:
                if tile_idx < 0 or tile_idx >= len(self.tiles):
                    continue
                tile = self.tiles[tile_idx]
                if tile.is_water or tile.polity_id != polity.id:
                    continue
                deficit = 100 - tile.control_level
                if deficit <= 0:
                    continue
                bonus = max(min_bonus, int(math.ceil(deficit * pull_ratio)))
                if max_bonus > 0:
                    bonus = min(max_bonus, bonus)
                before = tile.control_level
                tile.control_level = min(100, tile.control_level + bonus)
                self._record_control_change(
                    tile,
                    before,
                    tile.control_level,
                    label="Charismatic Leader",
                    details=f"bonus {bonus}",
                )

    def _apply_military_acclamation_bonus(self) -> None:
        """Boost control globally when legitimacy hinges on military acclamation and the leader fits the mold."""
        bonus = 8  # Large, flat control pull applied across owned tiles
        valid_traits = {"CHARISMATIC", "DIPLOMAT", "TACTICAL_GENIUS"}
        for polity in self.polities:
            if polity is None or not getattr(polity, 'tile_indices', None):
                continue
            legitimacy_label = (self._get_polity_idea_label(polity, 'legitimacy') or '').lower()
            if 'acclam' not in legitimacy_label:
                continue
            leader = getattr(polity, 'leader', None)
            if leader is None or not getattr(leader, 'traits', None):
                continue
            if not any((t or '').upper() in valid_traits for t in leader.traits):
                continue
            for tile_idx in getattr(polity, 'tile_indices', []) or []:
                if tile_idx < 0 or tile_idx >= len(self.tiles):
                    continue
                tile = self.tiles[tile_idx]
                if tile.is_water or tile.polity_id != polity.id:
                    continue
                if tile.control_level >= 100:
                    continue
                before = tile.control_level
                tile.control_level = min(100, tile.control_level + bonus)
                if tile.control_level != before:
                    self._record_control_change(
                        tile,
                        before,
                        tile.control_level,
                        label="Military Acclamation",
                        category="tick",
                        details=f"bonus {bonus}",
                    )

    def _apply_leadership_control_penalty(self, polity: Optional[Polity]) -> None:
        if polity is None or not getattr(polity, 'tile_indices', None):
            return
        if self._polity_has_trait(polity, 'MACHIAVELLIAN'):
            return
        succession_label = (self._get_polity_idea_label(polity, 'succession') or '').lower()
        legitimacy_label = (self._get_polity_idea_label(polity, 'legitimacy') or '').lower()
        penalty_factor = 0.6
        if 'elect' in succession_label:
            penalty_factor = 0.75  # Elections retain most authority through formal transitions
        if 'acclam' in legitimacy_label:
            penalty_factor = 0.5  # Military Acclamation still struggles without its champion
        for tile_idx in list(polity.tile_indices):
            if tile_idx < 0 or tile_idx >= len(self.tiles):
                continue
            tile = self.tiles[tile_idx]
            if tile.is_water or tile.polity_id != polity.id:
                continue
            if tile.control_level >= 100:
                continue
            penalized = int(math.floor(tile.control_level * penalty_factor))
            before = tile.control_level
            tile.control_level = max(0, penalized)
            self._record_control_change(
                tile,
                before,
                tile.control_level,
                label="Leadership Transition Penalty",
                category="flat",
            )

    def _get_alignment_threshold(self, polity: Optional[Polity], base_threshold: float) -> float:
        threshold = base_threshold
        if self._polity_has_trait(polity, 'TOLERANT'):
            trait_settings = self._get_trait_settings('TOLERANT')
            threshold *= max(0.1, float(trait_settings.get('alignment_threshold_multiplier', 0.7)))
        tolerance_span = self.config.get('simulation.culture.tolerance_alignment_threshold_span', 0.12) if self.config else 0.12
        threshold -= self._get_tolerance_bias(polity) * tolerance_span
        return max(0.01, min(0.99, threshold))

    def _get_alignment_penalty_multiplier(self, polity: Optional[Polity], base_multiplier: float) -> float:
        multiplier = base_multiplier
        if self._polity_has_trait(polity, 'TOLERANT'):
            trait_settings = self._get_trait_settings('TOLERANT')
            multiplier *= max(0.1, float(trait_settings.get('penalty_multiplier', 0.5)))
        return max(0.0, multiplier)

    def _get_tolerance_cap_bonus(self, polity: Optional[Polity]) -> int:
        if not self._polity_has_trait(polity, 'TOLERANT'):
            return 0
        trait_settings = self._get_trait_settings('TOLERANT')
        return int(trait_settings.get('control_cap_bonus', 10))

    def _get_polity_tolerance(self, polity: Optional[Polity]) -> float:
        if polity is None:
            return 0.5
        try:
            value = float(getattr(polity, 'cultural_tolerance', 0.5))
        except (TypeError, ValueError):
            value = 0.5
        clamped = max(0.0, min(1.0, value))
        polity.cultural_tolerance = clamped
        return clamped

    def _get_tolerance_bias(self, polity: Optional[Polity]) -> float:
        tolerance = self._get_polity_tolerance(polity)
        return max(-1.0, min(1.0, (tolerance - 0.5) * 2.0))

    def _get_misalignment_tolerance_factor(self, polity: Optional[Polity]) -> float:
        tolerance = self._get_polity_tolerance(polity)
        return max(0.1, 1.75 - 1.5 * tolerance)

    def _get_regional_culture_spawn_threshold(self, base_threshold: int, polity: Optional[Polity]) -> int:
        tolerance = self._get_polity_tolerance(polity)
        factor = 1.0
        if base_threshold > 0:
            factor = 1.0 + TOLERANCE_VARIANCE * (0.5 - tolerance) * 2.0
        adjusted = int(round(base_threshold * factor))
        return max(1, adjusted)

    def _get_syncretism_threshold(self, base_threshold: float, polity: Optional[Polity]) -> float:
        tolerance = self._get_polity_tolerance(polity)
        factor = 1.0 + TOLERANCE_VARIANCE * (0.5 - tolerance) * 2.0
        return max(0.05, base_threshold * factor)

    def _apply_tolerance_control_bias(
        self,
        polity: Optional[Polity],
        control_ratio: float,
        *,
        is_capital: bool = False
    ) -> float:
        if is_capital:
            return 1.0
        tolerance = self._get_polity_tolerance(polity)
        if tolerance >= 0.5:
            return control_ratio
        bias = min(1.0, max(0.0, (0.5 - tolerance) * 2.0))
        adjusted = control_ratio + bias * (1.0 - control_ratio)
        return max(control_ratio, min(1.0, adjusted))

    def _get_assimilation_rate_multiplier(self, polity: Optional[Polity]) -> float:
        tolerance = self._get_polity_tolerance(polity)
        if tolerance >= 0.5:
            return 1.0
        bias = min(1.0, max(0.0, (0.5 - tolerance) * 2.0))
        return 1.0 + 0.5 * bias

    def _apply_polity_tolerance_dp_modifier(self, tile: Optional[Tile], potential: float) -> float:
        if tile is None or getattr(tile, 'polity_id', -1) < 0:
            return potential
        polity = self._get_polity(tile.polity_id)
        if not polity:
            return potential
        dp_span = self.config.get('simulation.culture.tolerance_dp_span', 25) if self.config else 25
        return potential + self._get_tolerance_bias(polity) * dp_span

    def _get_syncretic_parents_for(self, culture_name: Optional[str]) -> Optional[FrozenSet[str]]:
        if not culture_name:
            return None
        parents = self._syncretic_parent_lookup.get(culture_name)
        if parents:
            return parents
        for parent_set, syncretic_name in self.syncretic_cultures.items():
            if syncretic_name == culture_name:
                self._syncretic_parent_lookup[syncretic_name] = parent_set
                return parent_set
        return None

    def _compute_primary_alignment_share(self, tile: Tile, polity: Optional[Polity]) -> float:
        if tile is None or polity is None or not polity.primary_culture:
            return 0.0
        if not tile.cultural_makeup:
            return 0.0
        alignment = tile.cultural_makeup.get(polity.primary_culture, 0.0)
        for culture_name, share in tile.cultural_makeup.items():
            if share <= 0.0 or culture_name == polity.primary_culture:
                continue
            parents = self._get_syncretic_parents_for(culture_name)
            if parents and polity.primary_culture in parents:
                alignment += share
        return min(1.0, alignment)

    def _collect_polity_center_cultures(self, polity: Optional[Polity]) -> Set[str]:
        if polity is None or not getattr(polity, 'is_active', True):
            return set()
        cultures: Set[str] = set()
        for center in self.population_centers:
            idx = center.tile_index
            if idx < 0 or idx >= len(self.tiles):
                continue
            tile = self.tiles[idx]
            if tile.polity_id != polity.id or not tile.cultural_makeup:
                continue
            cultures.update(name for name, share in tile.cultural_makeup.items() if share > 0.0)
        return cultures

    def _count_active_wars_for_polity(self, polity_id: Optional[int]) -> int:
        if polity_id is None or polity_id < 0:
            return 0
        wars = 0
        for relationship in self.relationships:
            if relationship.status != "war":
                continue
            if relationship.polity_a == polity_id or relationship.polity_b == polity_id:
                wars += 1
        return wars

    def _process_polity_cultural_tolerance(self) -> None:
        base_rate = self.config.get('simulation.culture.tolerance_base_rate', TOLERANCE_BASE_RATE) if self.config else TOLERANCE_BASE_RATE
        max_rate = self.config.get('simulation.culture.tolerance_max_rate', TOLERANCE_MAX_RATE) if self.config else TOLERANCE_MAX_RATE
        diversity_reference = max(1, int(self.config.get('simulation.culture.tolerance_diversity_reference', TOLERANCE_DIVERSITY_REFERENCE))) if self.config else TOLERANCE_DIVERSITY_REFERENCE
        neutral_ratio = self.config.get('simulation.culture.tolerance_diversity_neutral', TOLERANCE_DIVERSITY_NEUTRAL) if self.config else TOLERANCE_DIVERSITY_NEUTRAL
        war_penalty_step = self.config.get('simulation.culture.tolerance_war_penalty_step', TOLERANCE_WAR_PENALTY_STEP) if self.config else TOLERANCE_WAR_PENALTY_STEP
        positive_multiplier = self.config.get('simulation.culture.tolerance_diversity_positive_multiplier', 1.5) if self.config else 1.5
        negative_multiplier = self.config.get('simulation.culture.tolerance_diversity_negative_multiplier', 1.0) if self.config else 1.0
        extreme_damping = self.config.get('simulation.culture.tolerance_extreme_damping', 0.6) if self.config else 0.6
        downward_bias = self.config.get('simulation.culture.tolerance_downward_bias', 0.05) if self.config else 0.05
        for polity in self.polities:
            if polity is None or not getattr(polity, 'is_active', True):
                continue
            tolerance = self._get_polity_tolerance(polity)
            cultures = self._collect_polity_center_cultures(polity)
            diversity_ratio = 0.0
            if cultures:
                diversity_ratio = min(1.0, len(cultures) / diversity_reference)
            positive_diversity = 0.0
            negative_diversity = 0.0
            if diversity_ratio >= neutral_ratio:
                denom = max(1e-6, 1.0 - neutral_ratio)
                positive_diversity = (diversity_ratio - neutral_ratio) / denom
            else:
                denom = max(1e-6, neutral_ratio)
                negative_diversity = (neutral_ratio - diversity_ratio) / denom
            trait_bonus = 0.0
            if self._polity_has_trait(polity, 'TOLERANT'):
                trait_bonus = min(1.0, positive_diversity + 0.35) - positive_diversity
            wars = self._count_active_wars_for_polity(polity.id)
            war_penalty = min(1.0, wars * war_penalty_step)
            positive_pressure = min(1.0, (positive_diversity + trait_bonus) * positive_multiplier)
            negative_pressure = min(1.0, (negative_diversity + war_penalty) * negative_multiplier)
            net_pressure = max(-1.0, min(1.0, positive_pressure - negative_pressure - downward_bias))
            intensity = min(1.0, max(positive_pressure, negative_pressure))
            rate = base_rate + (max_rate - base_rate) * intensity
            distance_from_center = min(1.0, abs(tolerance - 0.5) * 2.0)
            damping = max(0.05, 1.0 - extreme_damping * distance_from_center)
            delta = net_pressure * rate * damping
            if self._polity_has_idea_label(polity, 'kinship_state', 'oligarchic'):
                raw_gravity_pull = self.config.get('simulation.culture.oligarchic_tolerance_gravity_pull', 0.02) if self.config else 0.02
                try:
                    gravity_pull = max(0.0, float(raw_gravity_pull))
                except (TypeError, ValueError):
                    gravity_pull = 0.02
                delta += (0.5 - tolerance) * gravity_pull
            polity.cultural_tolerance = max(0.0, min(1.0, tolerance + delta))

    def _process_mandate_of_heaven_events(self) -> None:
        """Trigger periodic regional mandate events that strongly swing control."""
        region_to_polities: Dict[int, List[Polity]] = {}
        for polity in self.polities:
            if polity is None or not getattr(polity, 'is_active', True):
                continue
            if not self._polity_has_idea_label(polity, 'legitimacy', 'mandate'):
                continue
            region_id = self._get_capital_region_id(polity)
            if region_id is None or region_id < 0:
                continue
            region_to_polities.setdefault(region_id, []).append(polity)

        tracked_regions = set(region_to_polities.keys())
        for stale_region in [rid for rid in self._mandate_region_event_years.keys() if rid not in tracked_regions]:
            self._mandate_region_event_years.pop(stale_region, None)

        for region_id, polities in region_to_polities.items():
            next_year = self._mandate_region_event_years.get(region_id)
            if next_year is None:
                self._mandate_region_event_years[region_id] = self.current_year + random.randint(10, 100)
                continue
            if self.current_year < next_year:
                continue

            positive = random.random() < 0.5
            multiplier = 1.5 if positive else 0.5
            descriptor = 'Mandate Blessed' if positive else 'Mandate Shaken'
            direction = '+50%' if positive else '-50%'

            for polity in polities:
                impacted_tiles = 0
                for tile_idx in getattr(polity, 'tile_indices', []) or []:
                    if tile_idx < 0 or tile_idx >= len(self.tiles):
                        continue
                    tile = self.tiles[tile_idx]
                    if tile.is_water or tile.polity_id != polity.id:
                        continue
                    before = tile.control_level
                    after = int(round(before * multiplier))
                    after = max(1, min(100, after))
                    if after == before:
                        continue
                    tile.control_level = after
                    impacted_tiles += 1
                    self._record_control_change(
                        tile,
                        before,
                        after,
                        label=descriptor,
                        category='flat',
                        details=f"regional mandate event ({direction})",
                    )
                self._log_polity_event(
                    'polity',
                    f"[mandate] {polity.name} in region {region_id} was {descriptor.lower()} ({direction} control across state; tiles={impacted_tiles})",
                    polity,
                )

            self._mandate_region_event_years[region_id] = self.current_year + random.randint(10, 100)

    def _get_war_chance_trait_multiplier(self, polity_id: Optional[int]) -> float:
        if polity_id is None:
            return 1.0
        polity = self._get_polity(polity_id)
        multiplier = 1.0
        if self._polity_has_trait(polity, 'WARMONGER'):
            settings = self._get_trait_settings('WARMONGER')
            multiplier *= float(settings.get('war_chance_multiplier', 1.3))
        if self._polity_has_trait(polity, 'PEACEMAKER'):
            settings = self._get_trait_settings('PEACEMAKER')
            multiplier *= float(settings.get('war_chance_multiplier', 0.65))
        return max(0.0, multiplier)

    def set_tick_profiling_enabled(self, enabled: bool) -> None:
        """Turn tick profiling on or off at runtime."""
        self.tick_profiling_enabled = bool(enabled)

    def get_recent_tick_profiles(self, count: int = 5) -> List[Dict[str, Any]]:
        """Return the most recent tick profiling entries."""
        if count <= 0:
            return []
        return list(self._tick_profile_history)[-count:]

    def _begin_tick_profile(self) -> None:
        if not self.tick_profiling_enabled:
            return
        self._tick_profile_sections = {}
        self._tick_workload_counters = {}
        self._tick_profile_start_time = time.perf_counter()

    def _increment_workload_counter(self, name: str, delta: int = 1) -> None:
        if delta == 0:
            return
        if not isinstance(name, str) or not name:
            return
        self._tick_workload_counters[name] = self._tick_workload_counters.get(name, 0) + int(delta)

    def _build_relation_eval_cache(self) -> Dict[str, Dict[Any, Any]]:
        """Precompute expensive per-polity relation inputs used across many pair checks."""
        cache: Dict[str, Dict[Any, Any]] = {
            'development': {},
            'centers': {},
            'tile_counts': {},
            'regions': {},
            'culture_parents': {},
        }
        for polity in self.polities:
            if polity is None or not getattr(polity, 'is_active', True):
                continue
            polity_id = polity.id
            cache['development'][polity_id] = self._calculate_polity_development_value(polity_id)
            cache['centers'][polity_id] = self._get_polity_population_center_count(polity_id)
            cache['tile_counts'][polity_id] = len(getattr(polity, 'tile_indices', []) or [])
            cache['regions'][polity_id] = self._get_capital_region_id(polity)
            culture_name = getattr(polity, 'primary_culture', None)
            if culture_name and culture_name not in cache['culture_parents']:
                cache['culture_parents'][culture_name] = self._get_primary_culture_parent(culture_name)
        return cache

    def _profiled_call(self, label: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        if not self.tick_profiling_enabled:
            return func(*args, **kwargs)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            self._tick_profile_sections[label] = self._tick_profile_sections.get(label, 0.0) + elapsed

    def _complete_tick_profile(self) -> None:
        if not self.tick_profiling_enabled:
            return
        total_duration = time.perf_counter() - self._tick_profile_start_time
        season_label = self.season_names[self.current_season] if self.season_names else str(self.current_season)
        entry = {
            'tick': self.total_ticks,
            'year': self.current_year,
            'season_index': self.current_season,
            'season': season_label,
            'total_seconds': total_duration,
            'sections': dict(self._tick_profile_sections),
            'workload_counters': dict(self._tick_workload_counters),
        }
        self._tick_profile_history.append(entry)
        if self.tick_profile_print:
            sections_sorted = sorted(
                self._tick_profile_sections.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if sections_sorted:
                section_bits = ", ".join(
                    f"{name}={duration * 1000:.1f}ms" for name, duration in sections_sorted
                )
                print(
                    f"[Profiler] Tick {entry['tick']} total={total_duration * 1000:.1f}ms :: {section_bits}"
                )
            else:
                print(f"[Profiler] Tick {entry['tick']} total={total_duration * 1000:.1f}ms")
        self._maybe_print_tick_profile_summary()
        self._tick_profile_sections = {}
        self._tick_workload_counters = {}

    def _maybe_print_tick_profile_summary(self) -> None:
        """Emit a lightweight periodic profile summary to highlight persistent hotspots."""
        if not self.tick_profiling_enabled:
            return
        if not getattr(self, 'tick_profile_summary_print', True):
            return
        if self.current_season != 0:
            return
        interval = max(1, int(getattr(self, 'tick_profile_summary_interval_years', 25) or 25))
        last_year = int(getattr(self, '_tick_profile_last_summary_year', 0) or 0)
        if self.current_year - last_year < interval:
            return

        window_start = self.current_year - interval + 1
        window_entries = [
            item for item in self._tick_profile_history
            if int(item.get('year', 0) or 0) >= window_start
        ]
        if not window_entries:
            self._tick_profile_last_summary_year = self.current_year
            return

        section_totals: Dict[str, float] = {}
        total_seconds = 0.0
        for item in window_entries:
            total_seconds += float(item.get('total_seconds', 0.0) or 0.0)
            for name, duration in (item.get('sections') or {}).items():
                section_totals[name] = section_totals.get(name, 0.0) + float(duration or 0.0)

        if not section_totals:
            self._tick_profile_last_summary_year = self.current_year
            return

        top_n = max(1, int(getattr(self, 'tick_profile_summary_top_n', 3) or 3))
        top_sections = sorted(section_totals.items(), key=lambda item: item[1], reverse=True)[:top_n]
        bits = []
        for name, duration in top_sections:
            share = (duration / total_seconds * 100.0) if total_seconds > 1e-9 else 0.0
            bits.append(f"{name}={duration * 1000:.1f}ms ({share:.1f}%)")
        print(
            f"[Profiler] Years {window_start}-{self.current_year} hotspot summary :: " + ", ".join(bits)
        )
        self._tick_profile_last_summary_year = self.current_year

    def _accumulate_tick_profile_section(self, label: str, duration: float) -> None:
        """Add duration to a named subsection when profiling is active."""
        if not self.tick_profiling_enabled or duration <= 0.0:
            return
        self._tick_profile_sections[label] = self._tick_profile_sections.get(label, 0.0) + duration
    
    def advance_tick(self) -> None:
        """Advance the simulation by one tick (3 months/1 season)."""
        self.total_ticks += 1
        self.current_season += 1
        new_year = False
        self._prepare_control_debug_logging()
        self._begin_tick_profile()
        
        if self.current_season >= len(self.season_names):
            self.current_season = 0
            self.current_year += 1
            new_year = True
        
        # Process population growth and development
        self._profiled_call('population_growth', self._process_population_growth)
        self._profiled_call('culture_thresholds', self._enforce_culture_population_thresholds)
        self._profiled_call('development', self._process_development_changes)
        self._profiled_call('migration', self._process_migration)
        self._profiled_call('culture_thresholds', self._enforce_culture_population_thresholds)
        if new_year:
            self._profiled_call('leadership_cycle', self._process_leadership_cycle)
            self._profiled_call('language_shifts', self._process_language_time_shifts)
            self._profiled_call('polity_rank_updates', self._process_polity_rank_updates)
            self._profiled_call('polity_tolerance', self._process_polity_cultural_tolerance)
            self._profiled_call('mandate_events', self._process_mandate_of_heaven_events)

        self._profiled_call('relationships', self._process_relationships)
        self._profiled_call('culture_thresholds', self._enforce_culture_population_thresholds)

        self._profiled_call('control', self._process_control)
        self._profiled_call('population_centers', self._update_population_centers)
        self._profiled_call('culture_thresholds', self._enforce_culture_population_thresholds)
        self._finalize_control_debug_logging()
        self._profiled_call('polity_development_history', self._record_polity_development_history)
        self._complete_tick_profile()
        
        self._log_event(
            "tick",
            f"Advanced tick: Year {self.current_year} ({self.season_names[self.current_season]})"
        )

    def get_simulation_date(self) -> str:
        """Get formatted simulation date string."""
        return f"Year {self.current_year} ({self.season_names[self.current_season]})"
    
    def update_auto_tick(self, current_time: float) -> bool:
        """Update auto-tick system. Returns True if a tick occurred."""
        if not self.auto_tick_enabled:
            return False
            
        if current_time - self.last_tick_time >= self.tick_interval:
            self.advance_tick()
            self.last_tick_time = current_time
            return True
        return False

    def reset_tick_timer(self, current_time: Optional[float] = None) -> None:
        """Reset auto-tick timer to avoid large jumps after toggles."""
        self.last_tick_time = current_time if current_time is not None else time.time()

    def set_auto_tick_enabled(self, enabled: bool) -> None:
        """Enable or disable automatic ticking with timer reset on resume."""
        if self.auto_tick_enabled == enabled:
            return
        self.auto_tick_enabled = enabled
        if enabled:
            self.reset_tick_timer()

    def toggle_auto_tick(self) -> None:
        """Toggle automatic ticking state."""
        self.set_auto_tick_enabled(not self.auto_tick_enabled)

    def set_tick_speed_multiplier(self, multiplier: float) -> None:
        """Set tick speed multiplier (affects interval)."""
        if multiplier <= 0:
            return
        self.tick_speed_multiplier = multiplier
        # Prevent extremely small intervals that could freeze rendering
        self.tick_interval = max(0.01, self.default_tick_interval / multiplier)
        self.reset_tick_timer()

    def toggle_fast_tick_speed(self, multiplier: float = 2.0) -> None:
        """Toggle between normal speed and a faster multiplier."""
        if self.tick_speed_multiplier >= multiplier - 1e-6:
            self.set_tick_speed_multiplier(1.0)
        else:
            self.set_tick_speed_multiplier(multiplier)

    def is_fast_tick_enabled(self, multiplier: float = 2.0) -> bool:
        """Return True if the fast speed multiplier is active."""
        return self.tick_speed_multiplier >= multiplier - 1e-6

    def _log_event(
        self,
        category: str,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a log line when the category and focus filter (if any) allow it."""
        if not getattr(self, '_log_categories', None):
            return
        if category not in self._log_categories:
            return
        if self._focus_filter_polity_id is not None and not self._event_matches_focus_filter(message, context):
            return
        print(message)

    def _build_polity_log_context(
        self,
        *polities: Optional[Polity],
        base_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a normalized focus-filter context for one or more related polities."""
        context: Dict[str, Any] = dict(base_context or {})

        polity_ids: Set[int] = set()
        leader_names: Set[str] = set()
        culture_names: Set[str] = set()

        def _add_polity(polity: Optional[Polity]) -> None:
            if polity is None:
                return
            polity_ids.add(polity.id)
            leader = getattr(polity, 'leader', None)
            leader_name = getattr(leader, 'name', None)
            if leader_name:
                leader_names.add(str(leader_name))
            culture_name = getattr(polity, 'primary_culture', None)
            if culture_name:
                culture_names.add(str(culture_name))

        for polity in polities:
            _add_polity(polity)

        for key, sink in (
            ('polity_ids', polity_ids),
            ('leader_names', leader_names),
            ('culture_names', culture_names),
        ):
            value = context.get(key)
            if value is None:
                continue
            if isinstance(value, (list, tuple, set, frozenset)):
                for item in value:
                    if item is not None:
                        sink.add(item)
            else:
                sink.add(value)

        if polity_ids:
            context['polity_ids'] = sorted(int(pid) for pid in polity_ids)
        if leader_names:
            context['leader_names'] = sorted(str(name) for name in leader_names)
        if culture_names:
            context['culture_names'] = sorted(str(name) for name in culture_names)
        return context

    def _log_polity_event(
        self,
        category: str,
        message: str,
        *polities: Optional[Polity],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a log line with normalized polity context for focus filtering."""
        self._log_event(
            category,
            message,
            context=self._build_polity_log_context(*polities, base_context=context),
        )

    def get_log_categories(self) -> Set[str]:
        """Return the currently enabled logging categories."""
        return set(getattr(self, '_log_categories', set()))

    def set_log_category(self, category: str, enabled: bool = True) -> Set[str]:
        """Enable or disable a specific logging category at runtime."""
        if not category:
            return self.get_log_categories()
        if not hasattr(self, '_log_categories') or self._log_categories is None:
            self._log_categories = set()
        if enabled:
            self._log_categories.add(category)
        else:
            self._log_categories.discard(category)
        return self.get_log_categories()

    def toggle_log_category(self, category: str) -> bool:
        """Flip a logging category and return True when it becomes enabled."""
        if not category:
            return False
        new_state = category not in getattr(self, '_log_categories', set())
        self.set_log_category(category, new_state)
        return new_state

    def reset_log_categories(self, categories: Optional[Sequence[str]] = None) -> Set[str]:
        """Reset logging categories to defaults or a provided list."""
        if categories is None:
            baseline = getattr(self, '_log_category_defaults', set())
        elif isinstance(categories, str):
            baseline = {categories}
        else:
            baseline = set(categories)
        self._log_categories = set(baseline)
        return self.get_log_categories()

    def is_focus_filter_active(self) -> bool:
        """Return True when console output is restricted to a focused polity."""
        return self._focus_filter_polity_id is not None

    def get_focus_filter_polity_id(self) -> Optional[int]:
        """Return the polity ID currently used for console focus filtering."""
        return self._focus_filter_polity_id

    def clear_focus_filter(self) -> None:
        """Remove any active focus filter and show all enabled categories again."""
        self._focus_filter_polity_id = None
        self._focus_filter_leader_name = None
        self._focus_filter_terms.clear()

    def set_focus_filter_polity(self, polity_id: Optional[int]) -> bool:
        """Focus console logging on a single polity and directly linked context."""
        if polity_id is None:
            self.clear_focus_filter()
            return True
        polity = self._get_polity(polity_id)
        if not polity or not getattr(polity, 'is_active', False):
            self.clear_focus_filter()
            return False
        self._focus_filter_polity_id = polity_id
        leader = getattr(polity, 'leader', None)
        self._focus_filter_leader_name = getattr(leader, 'name', None)
        self._focus_filter_terms = self._build_focus_filter_terms(polity)
        return True

    def _normalize_focus_term(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        normalized = str(value).strip().lower()
        return normalized or None

    def _build_focus_filter_terms(self, polity: Optional[Polity]) -> Set[str]:
        terms: Set[str] = set()
        if polity is None:
            return terms

        def _add(term: Optional[str]) -> None:
            normalized = self._normalize_focus_term(term)
            if normalized:
                terms.add(normalized)

        _add(getattr(polity, 'name', None))
        _add(f"polity {polity.id}")
        _add(f"polity_{polity.id}")
        leader = getattr(polity, 'leader', None)
        _add(getattr(leader, 'name', None))
        _add(getattr(leader, 'dynasty', None))
        culture_name = getattr(polity, 'primary_culture', None)
        _add(culture_name)
        friendly_label = self.get_culture_display_label(culture_name)
        _add(friendly_label)
        naming_bases = getattr(polity, 'naming_bases', {}) or {}
        _add(naming_bases.get('cultural_adjective'))
        _add(naming_bases.get('cultural'))
        return terms

    def _refresh_focus_filter_terms(self) -> None:
        if self._focus_filter_polity_id is None:
            self._focus_filter_terms.clear()
            self._focus_filter_leader_name = None
            return
        polity = self._get_polity(self._focus_filter_polity_id)
        if not polity or not getattr(polity, 'is_active', False):
            self.clear_focus_filter()
            return
        leader = getattr(polity, 'leader', None)
        self._focus_filter_leader_name = getattr(leader, 'name', None)
        self._focus_filter_terms = self._build_focus_filter_terms(polity)

    def _context_targets_focus(self, context: Optional[Dict[str, Any]]) -> bool:
        if not context or self._focus_filter_polity_id is None:
            return False

        def _iter_values(value: Any) -> Iterable[Any]:
            if value is None:
                return ()
            if isinstance(value, (list, tuple, set, frozenset)):
                return value
            return (value,)

        focus_id = self._focus_filter_polity_id
        polity_candidates = context.get('polity_ids') or context.get('polities')
        for candidate in _iter_values(polity_candidates):
            try:
                if int(candidate) == focus_id:
                    return True
            except (TypeError, ValueError):
                continue

        focus_leader = self._normalize_focus_term(self._focus_filter_leader_name)
        leader_candidates = context.get('leader_names') or context.get('leaders')
        if focus_leader:
            for candidate in _iter_values(leader_candidates):
                normalized = self._normalize_focus_term(candidate)
                if normalized and normalized == focus_leader:
                    return True

        return False

    def _message_matches_focus_terms(self, message: Optional[str]) -> bool:
        if not message or not self._focus_filter_terms:
            return False
        lowered = message.lower()
        for term in self._focus_filter_terms:
            if term and term in lowered:
                return True
        return False

    def _event_matches_focus_filter(self, message: Optional[str], context: Optional[Dict[str, Any]]) -> bool:
        if self._focus_filter_polity_id is None:
            return True
        self._refresh_focus_filter_terms()
        if self._focus_filter_polity_id is None:
            return True
        return self._context_targets_focus(context)

    def _prepare_control_debug_logging(self) -> None:
        """Reset per-tile control modifier diagnostics for the upcoming tick."""
        if not self.tiles:
            return
        for tile in self.tiles:
            if tile.is_water:
                tile.control_debug = {}
                continue
            tile.control_debug = {
                'tick_marker': self.total_ticks,
                'start_control': tile.control_level,
                'tick_modifiers': [],
                'flat_modifiers': [],
                'caps': [],
                'end_control': tile.control_level,
            }

    def _get_control_debug_entry(self, tile: Optional[Tile]) -> Optional[Dict[str, Any]]:
        """Ensure a tile has a control debug entry for the current tick."""
        if tile is None or tile.is_water:
            return None
        entry = getattr(tile, 'control_debug', None)
        if not isinstance(entry, dict) or entry.get('tick_marker') != self.total_ticks:
            tile.control_debug = {
                'tick_marker': self.total_ticks,
                'start_control': tile.control_level,
                'tick_modifiers': [],
                'flat_modifiers': [],
                'caps': [],
                'end_control': tile.control_level,
            }
            entry = tile.control_debug
        return entry

    def _log_control_modifier(
        self,
        tile: Optional[Tile],
        label: str,
        delta: int,
        category: str = 'tick',
        details: Optional[str] = None,
    ) -> None:
        """Record a control change attributable to a specific modifier."""
        if delta == 0:
            return
        entry = self._get_control_debug_entry(tile)
        if not entry:
            return
        bucket = entry['tick_modifiers'] if category == 'tick' else entry['flat_modifiers']
        bucket.append({'label': label, 'delta': int(delta), 'details': details})
        entry['end_control'] = getattr(tile, 'control_level', entry['end_control'])

    def _log_control_cap(self, tile: Optional[Tile], label: str, value: int, details: Optional[str] = None) -> None:
        """Record an enforced cap applied to a tile's control value."""
        entry = self._get_control_debug_entry(tile)
        if not entry:
            return
        entry['caps'].append({'label': label, 'value': int(value), 'details': details})

    def _record_control_change(
        self,
        tile: Optional[Tile],
        before: int,
        after: int,
        label: str,
        category: str = 'tick',
        details: Optional[str] = None,
    ) -> None:
        """Helper to log a modifier when a change actually occurs."""
        delta = after - before
        if delta == 0:
            return
        self._log_control_modifier(tile, label, delta, category=category, details=details)

    def _finalize_control_debug_logging(self) -> None:
        """Capture final control levels after all tick operations complete."""
        if not self.tiles:
            return
        for tile in self.tiles:
            if tile.is_water:
                continue
            entry = getattr(tile, 'control_debug', None)
            if isinstance(entry, dict):
                entry['end_control'] = tile.control_level
    
    def _process_population_growth(self) -> None:
        """Process population growth using births vs deaths simulation."""
        import random
        
        # Get config values
        base_birth_rate = self.config.get('simulation.population.base_birth_rate', 0.05) if self.config else 0.05
        base_death_rate = self.config.get('simulation.population.base_death_rate', 0.01) if self.config else 0.01
        climate_death_max = self.config.get('simulation.population.climate_death_penalty_max', 0.04) if self.config else 0.04
        rainfall_death_max = self.config.get('simulation.population.rainfall_death_penalty_max', 0.03) if self.config else 0.03
        rainfall_min = self.config.get('simulation.population.rainfall_optimal_min', 0.3) if self.config else 0.3
        rainfall_max = self.config.get('simulation.population.rainfall_optimal_max', 0.7) if self.config else 0.7
        dev_death_reduction = self.config.get('simulation.population.development_death_reduction', 0.5) if self.config else 0.5
        dev_threshold_ratio = self.config.get('simulation.population.development_threshold_ratio', 1.25) if self.config else 1.25
        
        total_population_before = sum(tile.population for tile in self.tiles if not tile.is_water)
        total_growth = 0
        
        for tile in self.tiles:
            if tile.is_water or tile.population <= 0:
                continue
            polity = self._get_polity(tile.polity_id) if tile.polity_id != -1 else None

            # Calculate birth rate (always at least 1%)
            birth_rate = max(0.01, base_birth_rate)

            # Heavy penalty for tiles exceeding development cap (overdevelopment crisis)
            overcap_birth_penalty = self.config.get('simulation.population.overcap_birth_penalty', 0.8) if self.config else 0.8
            dev_cap = self._calculate_development_cap(tile)
            if tile.development > dev_cap and tile.population > 50:
                overcap_ratio = tile.development / dev_cap  # How much over cap
                # Severe birth rate reduction for overdeveloped areas (infrastructure strain, pollution, etc.)
                birth_reduction = min(overcap_birth_penalty, (overcap_ratio - 1.0) * overcap_birth_penalty)
                birth_rate = birth_rate * (1.0 - birth_reduction)

            # Overcrowding birth penalty - when population exceeds development capacity
            overcrowding_birth_penalty = self.config.get('simulation.population.overcrowding_birth_penalty', 0.8) if self.config else 0.8
            if tile.population > 0:
                if tile.development <= 0:
                    # Extreme overcrowding when development is zero or negative
                    overcrowding_birth_reduction = overcrowding_birth_penalty
                else:
                    overcrowding_ratio = tile.population / tile.development
                    if overcrowding_ratio > 1.0:
                        # Exponential penalty that gets severe quickly
                        overcrowding_severity = overcrowding_ratio - 1.0
                        overcrowding_birth_reduction = min(overcrowding_birth_penalty, overcrowding_severity * overcrowding_severity * 0.1)
                    else:
                        overcrowding_birth_reduction = 0.0
                birth_rate = birth_rate * (1.0 - overcrowding_birth_reduction)

            # Calculate death rate
            death_rate = base_death_rate

            # Climate death penalty (temperature extremes)
            temp_deviation = abs(tile.temperature - 0.6)  # Optimal temp around 0.6
            # Scale up penalty for extreme temps (closer to 0 or 1)
            extreme_temp_penalty = 0.0
            if tile.temperature < 0.15 or tile.temperature > 0.95:
                extreme_temp_penalty = climate_death_max * 2.5  # Very high penalty for uninhabitable extremes
            elif tile.temperature < 0.25 or tile.temperature > 0.85:
                extreme_temp_penalty = climate_death_max * 1.5
            climate_death_penalty = temp_deviation * climate_death_max + extreme_temp_penalty
            death_rate += climate_death_penalty

            # Rainfall death penalty (drought or flood)
            rainfall_death_penalty = 0.0
            extreme_rain_penalty = 0.0
            if tile.rainfall < 0.15 or tile.rainfall > 0.95:
                extreme_rain_penalty = rainfall_death_max * 2.5
            elif tile.rainfall < 0.25 or tile.rainfall > 0.85:
                extreme_rain_penalty = rainfall_death_max * 1.5
            if tile.rainfall < rainfall_min:
                # Drought penalty
                drought_severity = (rainfall_min - tile.rainfall) / rainfall_min
                rainfall_death_penalty = drought_severity * rainfall_death_max
            elif tile.rainfall > rainfall_max:
                # Flood penalty (less severe than drought)
                flood_severity = (tile.rainfall - rainfall_max) / (1.0 - rainfall_max)
                rainfall_death_penalty = flood_severity * rainfall_death_max * 0.5
            death_rate += rainfall_death_penalty + extreme_rain_penalty

            # Additional hard penalty for tiles that are both extremely cold/hot and extremely dry/wet
            if (tile.temperature < 0.15 or tile.temperature > 0.95) and (tile.rainfall < 0.15 or tile.rainfall > 0.95):
                death_rate += 0.10  # Additive hard penalty for truly uninhabitable tiles

            # Overcrowding death penalty - when population exceeds development
            if tile.population > 0 and tile.development > 0:
                overcrowding_ratio = tile.population / tile.development
                if overcrowding_ratio > 1.0:
                    # Exponential penalty that gets severe quickly
                    overcrowding_severity = overcrowding_ratio - 1.0
                    overcrowding_death_penalty = min(0.05, overcrowding_severity * overcrowding_severity * 0.05)
                    death_rate += overcrowding_death_penalty

            # Development bonus - reduce death rate for well-developed areas
            if tile.population > 0 and tile.development > tile.population * dev_threshold_ratio:
                # Calculate development ratio above threshold (25% = 1.25)
                dev_excess_ratio = (tile.development / tile.population) / dev_threshold_ratio
                # Apply death rate reduction (capped at configured maximum of 50%)
                death_reduction = min(dev_death_reduction, (dev_excess_ratio - 1.0) * dev_death_reduction)
                death_rate = death_rate * (1.0 - death_reduction)
            death_rate *= self._get_trait_value(polity, 'JUST', 'death_rate_multiplier', 1.0)

            # If death rate is 0, still have a 1-in-3 chance of losing 1 population
            if death_rate == 0.0 and tile.population > 0:
                import random
                if random.random() < (1.0 / 3.0):
                    # Set death rate to cause exactly 1 population loss
                    death_rate = birth_rate + (1.0 / tile.population)

            # Calculate net growth rate
            net_growth_rate = birth_rate - death_rate

            # Calculate population change with fractional handling for small populations
            raw_population_change = tile.population * net_growth_rate

            # For small populations, use probabilistic rounding to avoid stagnation
            if abs(raw_population_change) < 1.0 and tile.population > 0:
                import random
                # Probability of change based on fractional part
                probability = abs(raw_population_change - int(raw_population_change))
                if random.random() < probability:
                    population_change = int(raw_population_change) + (1 if raw_population_change > 0 else -1)
                else:
                    population_change = int(raw_population_change)
            else:
                population_change = int(raw_population_change)

            # Store death count for migration calculations
            tile.last_tick_deaths = max(0, int(tile.population * death_rate))

            # Apply population change
            old_pop = tile.population
            tile.population = max(0, tile.population + population_change)
            total_growth += (tile.population - old_pop)
        
        # Population tracking (debug removed for now)
    
    def _process_development_changes(self) -> None:
        """Process development level changes for all tiles."""
        # Get config values
        rapid_increase_rate = self.config.get('simulation.development.rapid_increase_rate', 0.1) if self.config else 0.1
        decay_rate = self.config.get('simulation.development.decay_rate', 0.001) if self.config else 0.001
        decay_threshold_ratio = self.config.get('simulation.development.decay_threshold_ratio', 0.5) if self.config else 0.5
        coastal_bonus = self.config.get('simulation.development.coastal_bonus', 0.1) if self.config else 0.1
        temp_penalty_severity = self.config.get('simulation.development.temperature_penalty_severity', 0.8) if self.config else 0.8
        rain_penalty_severity = self.config.get('simulation.development.rainfall_penalty_severity', 0.6) if self.config else 0.6
        rain_optimal_min = self.config.get('simulation.development.rainfall_optimal_min', 0.3) if self.config else 0.3
        rain_optimal_max = self.config.get('simulation.development.rainfall_optimal_max', 0.7) if self.config else 0.7
        river_flux_threshold = self.config.get('world.rivers.min_flux', 0.12) if self.config else 0.12
        river_bonus = self.config.get('simulation.development.river_bonus', 0.08) if self.config else 0.08
        river_dryness_multiplier = self.config.get('simulation.development.river_dryness_multiplier', 0.6) if self.config else 0.6
        river_cap_bonus = self.config.get('simulation.development.river_cap_bonus', 350.0) if self.config else 350.0
        
        # Precompute neighbor population means for land tiles
        src_edges, dst_edges = self._build_neighbor_edges()
        n_tiles = len(self.tiles)
        neighbor_mean_pop = np.zeros(n_tiles, dtype=float)
        if src_edges.size > 0:
            # Exclude water neighbors from the mean
            neighbor_pop = np.fromiter((self.tiles[i].population for i in range(n_tiles)), dtype=float, count=n_tiles)
            neighbor_is_land = np.fromiter((not self.tiles[i].is_water for i in range(n_tiles)), dtype=bool, count=n_tiles)
            valid = neighbor_is_land[dst_edges]
            if np.any(valid):
                src_valid = src_edges[valid]
                dst_valid = dst_edges[valid]
                sums = np.zeros(n_tiles, dtype=float)
                counts = np.zeros(n_tiles, dtype=float)
                np.add.at(sums, src_valid, neighbor_pop[dst_valid])
                np.add.at(counts, src_valid, 1.0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    neighbor_mean_pop = np.where(counts > 0, sums / counts, 0.0)

        for idx, tile in enumerate(self.tiles):
            if tile.is_water:
                continue
            polity = self._get_polity(tile.polity_id) if tile.polity_id != -1 else None
            kinship_label = self._get_polity_idea_label_lower(polity, 'kinship_state') if polity else ''

            base_dev_cap = 1000.0 + (tile.rainfall * 500.0)
            if getattr(tile, 'river_flux', 0.0) >= river_flux_threshold:
                base_dev_cap += tile.river_flux * river_cap_bonus

            if tile.population > 0:
                avg_neighbor_pop = neighbor_mean_pop[idx]
                if avg_neighbor_pop > 0:
                    pop_factor = tile.population / avg_neighbor_pop
                else:
                    pop_factor = 10.0
                dynamic_dev_cap = max(base_dev_cap, base_dev_cap * min(10.0, pop_factor))
            else:
                dynamic_dev_cap = base_dev_cap

            frugal_multiplier = self._get_trait_value(polity, 'FRUGAL', 'development_cap_multiplier', 1.0)
            dynamic_dev_cap *= max(0.5, frugal_multiplier)
            if 'meritocratic' in kinship_label:
                dynamic_dev_cap *= 1.6
            tile.current_dev_cap = dynamic_dev_cap
            
            # Calculate climate modifier for development rate
            climate_modifier = 1.0
            
            # Temperature penalty for extremes (penalty grows exponentially from optimal 0.6)
            temp_deviation = abs(tile.temperature - 0.6)
            temp_penalty = temp_deviation * temp_penalty_severity
            climate_modifier *= (1.0 - temp_penalty)
            
            # Stronger penalty for low rainfall (<0.25), mild penalty for high rainfall (>0.85)
            if tile.rainfall < 0.25:
                # Steep penalty as rainfall approaches 0
                rain_penalty = ((0.25 - tile.rainfall) / 0.25) ** 1.5 * rain_penalty_severity * 2.0
                climate_modifier *= (1.0 - min(0.95, rain_penalty))
            elif tile.rainfall > 0.85:
                # Mild penalty for very high rainfall
                rain_penalty = ((tile.rainfall - 0.85) / 0.15) * rain_penalty_severity * 0.5
                climate_modifier *= (1.0 - min(0.5, rain_penalty))
            elif tile.rainfall < rain_optimal_min:
                rain_deficit = rain_optimal_min - tile.rainfall
                rain_penalty = rain_deficit * rain_penalty_severity
                climate_modifier *= (1.0 - rain_penalty)
            elif tile.rainfall > rain_optimal_max:
                rain_excess = tile.rainfall - rain_optimal_max
                rain_penalty = rain_excess * rain_penalty_severity * 0.5
                climate_modifier *= (1.0 - rain_penalty)

            if getattr(tile, 'river_flux', 0.0) >= river_flux_threshold:
                drought_factor = max(0.0, rain_optimal_min - tile.rainfall)
                climate_modifier *= (1.0 + river_bonus + drought_factor * river_dryness_multiplier)
            
            # Coastal access bonus (check neighbors for water)
            has_coastal_access = any(self.tiles[neighbor_idx].is_water 
                                   for neighbor_idx in tile.neighbors 
                                   if neighbor_idx < len(self.tiles))
            if has_coastal_access:
                climate_modifier *= (1.0 + coastal_bonus)
            
            # Ensure climate modifier doesn't go negative
            climate_modifier = max(0.1, climate_modifier)
            
            # Control bonus for development growth (50% bonus at 100% control)
            control_bonus = 1.0
            if tile.control_level == 100:
                control_bonus = 1.5  # 50% bonus for perfect control
            
            # Check for diaspora population center delay (now reduces development growth instead of blocking it)
            diaspora_delay_years = self.config.get('simulation.diaspora.population_center_delay_years', 0) if self.config else 0
            is_diaspora_delay_active = (
                diaspora_delay_years > 0 and 
                self.current_year < diaspora_delay_years
            )
            
            # Development follows population, but with curved decay
            if tile.development < tile.population:
                # Rapidly increase development when it's lower than population
                increase_rate = rapid_increase_rate * (tile.population - tile.development) / max(1, tile.population)
                architect_multiplier = self._get_trait_value(polity, 'ARCHITECT', 'growth_multiplier', 1.0)
                development_increase = max(1, tile.population * increase_rate * climate_modifier * control_bonus * architect_multiplier)

                if 'aristocratic' in kinship_label:
                    development_increase *= 0.55

                if is_diaspora_delay_active:
                    development_increase *= 0.25  # 75% reduction during diaspora delay

                # Only apply increase if it won't exceed the development cap
                if tile.development + development_increase <= dynamic_dev_cap:
                    tile.development += development_increase
                else:
                    # Cap reached - only increase up to the cap, no further
                    if tile.development < dynamic_dev_cap:
                        tile.development = dynamic_dev_cap
            else:
                # Curved decay when population is below threshold
                decay_threshold_population = tile.development * decay_threshold_ratio
                if tile.population < decay_threshold_population:
                    # Calculate decay based on how far below threshold we are
                    population_deficit_ratio = (decay_threshold_population - tile.population) / max(1, decay_threshold_population)
                    # Exponential curve for decay intensity
                    import math
                    decay_intensity = 1.0 - math.exp(-3.0 * population_deficit_ratio)  # Steeper curve
                    decay_amount = tile.development * decay_rate * decay_intensity
                    tile.development = max(0.0, tile.development - decay_amount)
            
            # Ensure minimum is 0 (no hard cap - development can exceed cap, just can't increase further)
            tile.development = max(0.0, tile.development)

    def _calculate_environmental_suitability(self, tile) -> float:
        """Calculate environmental suitability for a single tile (0.0-1.0)."""
        # Config values
        temp_penalty_severity = self.config.get('simulation.development.temperature_penalty_severity', 0.9) if self.config else 0.9
        rain_penalty_severity = self.config.get('simulation.development.rainfall_penalty_severity', 0.7) if self.config else 0.7
        rainfall_optimal_min = self.config.get('simulation.development.rainfall_optimal_min', 0.3) if self.config else 0.3
        rainfall_optimal_max = self.config.get('simulation.development.rainfall_optimal_max', 0.7) if self.config else 0.7
        river_bonus = self.config.get('simulation.development.river_bonus', 0.1) if self.config else 0.1
        river_dryness_multiplier = self.config.get('simulation.development.river_dryness_multiplier', 0.6) if self.config else 0.6
        coastal_bonus = self.config.get('simulation.development.coastal_bonus', 0.1) if self.config else 0.1
        river_flux_threshold = self.config.get('world.rivers.min_flux', 0.12) if self.config else 0.12

        # Temperature suitability
        temp = getattr(tile, 'temperature', 0.5)
        temp_dist = abs(temp - 0.5) * 2
        temp_suit = 1.0 - (temp_dist ** 2) * temp_penalty_severity

        # Rainfall suitability
        rain = getattr(tile, 'rainfall', 0.5)
        if rain < 0.25:
            rain_suit = 1.0 - min(0.95, ((0.25 - rain) / 0.25) ** 1.5 * rain_penalty_severity * 2.0)
        elif rain > 0.85:
            rain_suit = 1.0 - min(0.5, ((rain - 0.85) / 0.15) * rain_penalty_severity * 0.5)
        elif rain < rainfall_optimal_min:
            rain_suit = 1.0 - ((rainfall_optimal_min - rain) / rainfall_optimal_min) * rain_penalty_severity
        elif rain > rainfall_optimal_max:
            rain_suit = 1.0 - ((rain - rainfall_optimal_max) / (1.0 - rainfall_optimal_max + 1e-6)) * rain_penalty_severity
        else:
            rain_suit = 1.0

        # River bonus (for dry tiles)
        river_flux = getattr(tile, 'river_flux', 0.0)
        river_suit = 1.0
        if river_flux >= river_flux_threshold:
            drought_bonus = max(0.0, rainfall_optimal_min - rain)
            river_suit += river_bonus + drought_bonus * river_dryness_multiplier

        # Coast bonus
        coast_suit = 1.0
        if hasattr(self, '_is_coastal_tile') and self._is_coastal_tile(getattr(tile, 'index', -1)):
            coast_suit += coastal_bonus

        # Combine
        env_suit = temp_suit * rain_suit * river_suit * coast_suit
        return max(0.0, min(1.0, env_suit))
    
    def _calculate_destination_potential(
        self,
        tile: Tile,
        tile_idx: Optional[int] = None,
        *,
        population_center_lookup: Optional[Set[int]] = None,
        origin_population: Optional[float] = None,
    ) -> float:
        """Calculate destination potential based on development ratio and recent deaths."""
        if tile.is_water:
            return 0.0
        
        # --- Itinerance-based DP weighting ---
        dev_ratio_factor = self.config.get('simulation.migration.development_ratio_factor', 100) if self.config else 100
        dev_migration_bonus = self.config.get('simulation.migration.development_migration_bonus', 50) if self.config else 50
        river_dp_bonus = self.config.get('simulation.migration.river_dp_bonus', 12) if self.config else 12
        coastal_dp_bonus = self.config.get('simulation.migration.coastal_dp_bonus', 15) if self.config else 15
        dominant_culture_name = self._get_tile_majority_culture(tile)
        itinerance = 0.25
        if dominant_culture_name:
            culture_obj = self._find_culture_by_name(dominant_culture_name)
            if culture_obj:
                self._ensure_culture_ideas(culture_obj)
                itinerance = culture_obj.ideas.get("itinerance", 0.25)
                label = self._idea_label_for_value("itinerance", itinerance)
                if label:
                    itinerance_label = label
                else:
                    itinerance_label = "Sedentary"
            else:
                itinerance_label = "Migratory"
        else:
            itinerance_label = "Migratory"
        if itinerance_label == "Mixed":
            dev_ratio_factor *= 0.5
        elif itinerance_label == "Migratory":
            dev_ratio_factor *= 0.2

        death_penalty_factor = self.config.get('simulation.migration.death_penalty_factor', 100) if self.config else 100
        overcrowding_penalty_factor = self.config.get('simulation.migration.overcrowding_penalty_factor', 200) if self.config else 200
        dev_threshold_ratio = self.config.get('simulation.population.development_threshold_ratio', 1.25) if self.config else 1.25
        rainfall_optimal_min = self.config.get('simulation.development.rainfall_optimal_min', 0.3) if self.config else 0.3
        river_flux_threshold = self.config.get('world.rivers.min_flux', 0.12) if self.config else 0.12
        pop_center_overcrowding_multiplier = self.config.get('simulation.migration.population_center_overcrowding_penalty_multiplier', 0.6) if self.config else 0.6
        pop_center_dp_scaling = self.config.get('simulation.migration.population_center_dp_bonus_scaling', 0.8) if self.config else 0.8
        pop_center_dp_max_extra = self.config.get('simulation.migration.population_center_dp_bonus_max_multiplier', 1.5) if self.config else 1.5

        potential = 50.0
        # Less populated DP bonus/malus for Mixed/Migratory, now scaled by environmental suitability and population differential
        less_populated_bonus = 0.0
        env_suit = self._calculate_environmental_suitability(tile)
        if itinerance_label in ("Mixed", "Migratory") and origin_population is not None:
            candidate_pop = tile.population if tile.population > 0 else 1
            origin_pop = origin_population if origin_population > 0 else 1
            pop_diff = candidate_pop - origin_pop
            pop_diff_ratio = pop_diff / max(candidate_pop, origin_pop, 1)
            # Strong malus for higher population, bonus for lower
            if itinerance_label == "Mixed":
                # Malus up to -48, bonus up to +24, scaled by env_suit
                if pop_diff > 0:
                    less_populated_bonus = -48.0 * min(1.0, pop_diff_ratio) * env_suit
                else:
                    less_populated_bonus = 24.0 * min(1.0, -pop_diff_ratio) * env_suit
            else:
                # Malus up to -108, bonus up to +54, scaled by env_suit
                if pop_diff > 0:
                    less_populated_bonus = -108.0 * min(1.0, pop_diff_ratio) * env_suit
                else:
                    less_populated_bonus = 54.0 * min(1.0, -pop_diff_ratio) * env_suit
        elif itinerance_label in ("Mixed", "Migratory"):
            # Fallback: old logic if origin_population not provided
            pop = tile.population if tile.population > 0 else 1
            pop_scale = 100.0
            pop_factor = max(0.0, 1.0 - min(1.0, pop / pop_scale))
            if itinerance_label == "Mixed":
                less_populated_bonus = 24.0 * pop_factor * env_suit
            else:
                less_populated_bonus = 54.0 * pop_factor * env_suit

        if tile_idx is None:
            tile_idx = self._find_tile_index_by_identity(tile)
        is_population_center = False
        if tile_idx is not None and tile_idx >= 0:
            if population_center_lookup is not None:
                is_population_center = tile_idx in population_center_lookup
            else:
                is_population_center = any(center.tile_index == tile_idx for center in self.population_centers)
        
        # Development to population ratio (higher is better)
        if tile.population > 0:
            dev_ratio = tile.development / tile.population
            if dev_ratio >= 1.0:
                # Good infrastructure, linear bonus
                potential += dev_ratio * dev_ratio_factor
            else:
                # Overcrowding penalty - exponential punishment that gets very severe
                overcrowding_severity = (1.0 - dev_ratio)  # 0.0 = no overcrowding, 1.0 = no infrastructure
                import math
                if overcrowding_severity > 0.5:
                    overcrowding_penalty = math.pow(overcrowding_severity, 3) * overcrowding_penalty_factor * 1.5
                else:
                    overcrowding_penalty = math.pow(overcrowding_severity, 2) * overcrowding_penalty_factor
                # No special multiplier for Mixed/Migratory
                if is_population_center:
                    overcrowding_penalty *= pop_center_overcrowding_multiplier
                potential -= overcrowding_penalty
        else:
            # Empty tiles have good appeal for overcrowded populations, now scaled by environmental suitability
            potential += dev_ratio_factor * 0.5 * env_suit
        
        # Penalty for recent deaths (people avoid places where many died)
        if hasattr(tile, 'last_tick_deaths') and tile.population > 0:
            death_penalty = (tile.last_tick_deaths / tile.population) * death_penalty_factor
            potential -= death_penalty
        
        # Development migration bonus - well-developed areas are more attractive
        if tile.population > 0 and tile.development > tile.population * dev_threshold_ratio:
            # Calculate development ratio above threshold (same as death rate reduction)
            dev_excess_ratio = (tile.development / tile.population) / dev_threshold_ratio
            # Apply migration bonus (proportional to excess development)
            migration_bonus = min(dev_migration_bonus, (dev_excess_ratio - 1.0) * dev_migration_bonus)
            potential += migration_bonus
        
        # Population center bonus - established settlements are more attractive
        population_center_bonus = self.config.get('simulation.migration.population_center_dp_bonus', 25) if self.config else 25
        # Adjust population center DP bonus by itinerance label
        if is_population_center:
            dev_ratio = tile.development / max(1, tile.population)
            bonus_increase = max(0.0, dev_ratio - 1.0) * pop_center_dp_scaling
            bonus_increase = min(bonus_increase, pop_center_dp_max_extra)
            if itinerance_label == "Mixed":
                # Apply only 40% of the bonus for Mixed
                potential += population_center_bonus * 0.4 * (1.0 + bonus_increase)
            elif itinerance_label == "Migratory":
                # No bonus for Migratory
                pass
            else:
                # Full bonus for Sedentary
                potential += population_center_bonus * (1.0 + bonus_increase)
        
        # Heavy penalty for tiles exceeding development cap (overdevelopment)
        overcap_penalty = self.config.get('simulation.migration.overcap_dp_penalty', 150) if self.config else 150
        dev_cap = self._calculate_development_cap(tile)
        if tile.development > dev_cap:
            overcap_ratio = tile.development / dev_cap  # How much over cap (1.5 = 50% over cap)
            # Exponential penalty that gets very severe for extreme overdevelopment
            import math
            penalty_severity = math.pow(overcap_ratio - 1.0, 2)  # Quadratic growth
            potential -= penalty_severity * overcap_penalty

        # River and coast bonuses are now handled in environmental suitability
        
        potential = self._apply_polity_tolerance_dp_modifier(tile, potential)
        potential += less_populated_bonus
        return min(100.0, max(-50.0, potential))

    def _calculate_destination_potential_array(self, population_center_lookup: Set[int]) -> np.ndarray:
        """Vectorized destination potential for all tiles (read-only, no side effects)."""
        n = len(self.tiles)
        if n == 0:
            return np.zeros(0, dtype=float)

        pop = np.fromiter((t.population for t in self.tiles), dtype=float, count=n)
        dev = np.fromiter((t.development for t in self.tiles), dtype=float, count=n)
        rainfall = np.fromiter((getattr(t, 'rainfall', 0.0) for t in self.tiles), dtype=float, count=n)
        river_flux = np.fromiter((getattr(t, 'river_flux', 0.0) for t in self.tiles), dtype=float, count=n)
        last_deaths = np.fromiter((getattr(t, 'last_tick_deaths', 0) for t in self.tiles), dtype=float, count=n)
        is_water = np.fromiter((t.is_water for t in self.tiles), dtype=bool, count=n)
        is_pop_center = np.fromiter(((i in population_center_lookup) for i in range(n)), dtype=bool, count=n)

        potential = np.full(n, 50.0, dtype=float)

        dev_ratio_factor = self.config.get('simulation.migration.development_ratio_factor', 100) if self.config else 100
        dev_migration_bonus = self.config.get('simulation.migration.development_migration_bonus', 50) if self.config else 50
        river_dp_bonus = self.config.get('simulation.migration.river_dp_bonus', 12) if self.config else 12
        coastal_dp_bonus = self.config.get('simulation.migration.coastal_dp_bonus', 15) if self.config else 15
        # For each tile, get dominant culture itinerance label
        itinerance_labels = []
        for t in self.tiles:
            dominant_culture_name = self._get_tile_majority_culture(t)
            if dominant_culture_name:
                culture_obj = self._find_culture_by_name(dominant_culture_name)
                if culture_obj:
                    self._ensure_culture_ideas(culture_obj)
                    itinerance = culture_obj.ideas.get("itinerance", 0.25)
                    l = self._idea_label_for_value("itinerance", itinerance)
                    if l:
                        label = l
                    else:
                        label = "Sedentary"
                else:
                    label = "Migratory"
            else:
                label = "Migratory"
            itinerance_labels.append(label)
        # Adjust weights by itinerance label (vectorized)
        dev_ratio_factors = np.array([
            dev_ratio_factor * (0.2 if l == "Migratory" else 0.5 if l == "Mixed" else 1.0)
            for l in itinerance_labels
        ])
        dev_migration_bonuses = np.array([
            dev_migration_bonus * (0.2 if l == "Migratory" else 0.5 if l == "Mixed" else 1.0)
            for l in itinerance_labels
        ])
        river_dp_bonuses = np.array([
            river_dp_bonus * (2.5 if l == "Migratory" else 1.5 if l == "Mixed" else 1.0)
            for l in itinerance_labels
        ])
        coastal_dp_bonuses = np.array([
            coastal_dp_bonus * (2.5 if l == "Migratory" else 1.5 if l == "Mixed" else 1.0)
            for l in itinerance_labels
        ])
        death_penalty_factor = self.config.get('simulation.migration.death_penalty_factor', 100) if self.config else 100
        overcrowding_penalty_factor = self.config.get('simulation.migration.overcrowding_penalty_factor', 200) if self.config else 200
        dev_threshold_ratio = self.config.get('simulation.population.development_threshold_ratio', 1.25) if self.config else 1.25
        population_center_bonus = self.config.get('simulation.migration.population_center_dp_bonus', 25) if self.config else 25
        pop_center_overcrowding_multiplier = self.config.get('simulation.migration.population_center_overcrowding_penalty_multiplier', 0.6) if self.config else 0.6
        pop_center_dp_scaling = self.config.get('simulation.migration.population_center_dp_bonus_scaling', 0.8) if self.config else 0.8
        pop_center_dp_max_extra = self.config.get('simulation.migration.population_center_dp_bonus_max_multiplier', 1.5) if self.config else 1.5
        overcap_penalty = self.config.get('simulation.migration.overcap_dp_penalty', 150) if self.config else 150
        rainfall_optimal_min = self.config.get('simulation.development.rainfall_optimal_min', 0.3) if self.config else 0.3
        river_flux_threshold = self.config.get('world.rivers.min_flux', 0.12) if self.config else 0.12


        safe_pop = np.maximum(pop, 1.0)
        dev_ratio = dev / safe_pop

        overcrowded = (pop > 0) & (dev_ratio < 1.0)
        moderate_mask = overcrowded & (dev_ratio >= 0.5)
        severe_mask = overcrowded & (dev_ratio < 0.5)
        overcrowding_severity = 1.0 - dev_ratio
        penalty = np.zeros(n, dtype=float)
        penalty[moderate_mask] = (overcrowding_severity[moderate_mask] ** 2) * overcrowding_penalty_factor
        penalty[severe_mask] = (overcrowding_severity[severe_mask] ** 3) * overcrowding_penalty_factor * 1.5
        # No special multiplier for Mixed/Migratory
        penalty[is_pop_center] *= pop_center_overcrowding_multiplier
        potential -= penalty

        well_dev = (pop > 0) & (dev_ratio >= 1.0)
        potential[well_dev] += dev_ratio_factors[well_dev] * dev_ratio[well_dev]

        with np.errstate(divide='ignore', invalid='ignore'):
            death_penalty = (last_deaths / safe_pop) * death_penalty_factor
        death_penalty[pop <= 0] = 0.0
        potential -= death_penalty

        dev_excess_ratio = np.where(pop > 0, dev_ratio / dev_threshold_ratio, 0.0)
        excess = np.maximum(0.0, dev_excess_ratio - 1.0)
        migration_bonus = np.minimum(excess * dev_migration_bonuses, dev_migration_bonuses)
        potential += migration_bonus

        if population_center_bonus != 0:
            dev_ratio_pc = dev_ratio
            bonus_increase = np.clip(np.maximum(0.0, dev_ratio_pc - 1.0) * pop_center_dp_scaling, 0.0, pop_center_dp_max_extra)
            # Adjust population center DP bonus by itinerance label (vectorized)
            for i in range(n):
                if is_pop_center[i]:
                    label = itinerance_labels[i]
                    if label == "Mixed":
                        potential[i] += population_center_bonus * 0.4 * (1.0 + bonus_increase[i])
                    elif label == "Migratory":
                        pass  # No bonus
                    else:
                        potential[i] += population_center_bonus * (1.0 + bonus_increase[i])

        dev_caps = np.fromiter((self._calculate_development_cap(t) for t in self.tiles), dtype=float, count=n)
        overcap_mask = dev > dev_caps
        if np.any(overcap_mask):
            overcap_ratio = dev[overcap_mask] / np.maximum(dev_caps[overcap_mask], 1e-6)
            penalty_severity = np.power(overcap_ratio - 1.0, 2)
            potential[overcap_mask] -= penalty_severity * overcap_penalty

        river_mask = river_flux >= river_flux_threshold
        if np.any(river_mask):
            drought_bonus = np.maximum(0.0, rainfall_optimal_min - rainfall[river_mask])
            potential[river_mask] += river_dp_bonuses[river_mask] * (1.0 + drought_bonus * 2.0)

        coastal_mask = np.fromiter((self._is_coastal_tile(i) for i in range(n)), dtype=bool, count=n)
        if np.any(coastal_mask):
            potential[coastal_mask] += coastal_dp_bonuses[coastal_mask]
        # Environmental suitability (vectorized)
        temp_penalty_severity = self.config.get('simulation.development.temperature_penalty_severity', 0.9) if self.config else 0.9
        rain_penalty_severity = self.config.get('simulation.development.rainfall_penalty_severity', 0.7) if self.config else 0.7
        rainfall_optimal_min = self.config.get('simulation.development.rainfall_optimal_min', 0.3) if self.config else 0.3
        rainfall_optimal_max = self.config.get('simulation.development.rainfall_optimal_max', 0.7) if self.config else 0.7
        river_bonus = self.config.get('simulation.development.river_bonus', 0.1) if self.config else 0.1
        river_dryness_multiplier = self.config.get('simulation.development.river_dryness_multiplier', 0.6) if self.config else 0.6
        coastal_bonus = self.config.get('simulation.development.coastal_bonus', 0.1) if self.config else 0.1
        # Temperature suitability
        temp = np.array([getattr(t, 'temperature', 0.5) for t in self.tiles], dtype=float)
        temp_dist = np.abs(temp - 0.5) * 2
        temp_suit = 1.0 - (temp_dist ** 2) * temp_penalty_severity
        # Rainfall suitability (strong penalty for <0.25, mild for >0.85)
        rain = rainfall
        rain_suit = np.ones(n, dtype=float)
        rain_very_low = rain < 0.25
        rain_very_high = rain > 0.85
        rain_low = (rain >= 0.25) & (rain < rainfall_optimal_min)
        rain_high = (rain <= 0.85) & (rain > rainfall_optimal_max)
        # Steep penalty for very low rainfall
        rain_suit[rain_very_low] = 1.0 - np.minimum(0.95, ((0.25 - rain[rain_very_low]) / 0.25) ** 1.5 * rain_penalty_severity * 2.0)
        # Mild penalty for very high rainfall
        rain_suit[rain_very_high] = 1.0 - np.minimum(0.5, ((rain[rain_very_high] - 0.85) / 0.15) * rain_penalty_severity * 0.5)
        # Standard penalties for moderate deviations
        rain_suit[rain_low] = 1.0 - ((rainfall_optimal_min - rain[rain_low]) / rainfall_optimal_min) * rain_penalty_severity
        rain_suit[rain_high] = 1.0 - ((rain[rain_high] - rainfall_optimal_max) / (1.0 - rainfall_optimal_max + 1e-6)) * rain_penalty_severity
        # River bonus (for dry tiles)
        river_suit = np.ones(n, dtype=float)
        river_mask = river_flux >= river_flux_threshold
        if np.any(river_mask):
            drought_bonus = np.maximum(0.0, rainfall_optimal_min - rainfall[river_mask])
            river_suit[river_mask] += river_bonus + drought_bonus * river_dryness_multiplier
        # Coast bonus
        coast_suit = np.ones(n, dtype=float)
        if np.any(coastal_mask):
            coast_suit[coastal_mask] += coastal_bonus
        # Combine
        env_suit = np.clip(temp_suit * rain_suit * river_suit * coast_suit, 0.0, 1.0)

        # Less populated DP bonus for Mixed/Migratory (vectorized, now scaled by env_suit)
        pop_scale = 100.0
        pop_factor = np.maximum(0.0, 1.0 - np.minimum(1.0, pop / pop_scale))
        less_populated_bonuses = np.zeros(n, dtype=float)
        for i, l in enumerate(itinerance_labels):
            if l == "Mixed":
                less_populated_bonuses[i] = 24.0 * pop_factor[i] * env_suit[i]  # Was 8.0
            elif l == "Migratory":
                less_populated_bonuses[i] = 54.0 * pop_factor[i] * env_suit[i]  # Was 18.0
        potential += less_populated_bonuses

        # Empty tile bonus (for unpopulated tiles), now scaled by environmental suitability
        empty_mask = pop <= 0
        potential[empty_mask] += dev_ratio_factors[empty_mask] * 0.5 * env_suit[empty_mask]

        potential[is_water] = 0.0

        for idx, tile in enumerate(self.tiles):
            potential[idx] = self._apply_polity_tolerance_dp_modifier(tile, potential[idx])

        return np.clip(potential, -50.0, 100.0)

    def _build_neighbor_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return directed neighbor edges as (src, dst) arrays for vectorized ops."""
        n = len(self.tiles)
        if n == 0:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        srcs: List[int] = []
        dsts: List[int] = []
        for idx, tile in enumerate(self.tiles):
            if not tile.neighbors:
                continue
            for nb in tile.neighbors:
                if 0 <= nb < n:
                    srcs.append(idx)
                    dsts.append(nb)
        if not srcs:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        return np.asarray(srcs, dtype=int), np.asarray(dsts, dtype=int)
    
    def rebuild_coastal_tile_cache(self) -> None:
        """Recompute cached coastal land tiles for rapid lookups."""
        self._coastal_tile_cache = []
        self._coastal_tile_set = set()
        for idx in range(len(self.tiles)):
            if self._compute_tile_is_coastal(idx):
                self._coastal_tile_cache.append(idx)
                self._coastal_tile_set.add(idx)
        self._coastal_cache_ready = True

    def _compute_tile_is_coastal(self, tile_index: int) -> bool:
        if tile_index < 0 or tile_index >= len(self.tiles):
            return False
        tile = self.tiles[tile_index]
        if tile.is_water:
            return False
        for neighbor_idx in tile.neighbors:
            if 0 <= neighbor_idx < len(self.tiles) and self.tiles[neighbor_idx].is_water:
                return True
        return False

    def _tiles_divided_by_river(self, tile_a_idx: int, tile_b_idx: int, flux_threshold: float) -> bool:
        """Return True when a river of at least the given flux separates two tiles."""
        if tile_a_idx < 0 or tile_b_idx < 0:
            return False
        flux = 0.0
        if tile_a_idx < len(self.tiles):
            flux = self.tiles[tile_a_idx].river_neighbors.get(tile_b_idx, 0.0)
        if flux < flux_threshold and tile_b_idx < len(self.tiles):
            flux = self.tiles[tile_b_idx].river_neighbors.get(tile_a_idx, 0.0)
        return flux >= flux_threshold

    def _rebuild_river_neighbors_from_paths(self) -> None:
        """Ensure per-tile river metadata matches the stored river polylines."""
        for tile in self.tiles:
            if tile.river_neighbors is None:
                tile.river_neighbors = {}
            if tile.river_ids is None:
                tile.river_ids = []
        for river in getattr(self, 'rivers', []) or []:
            for tile_idx in river.tile_indices:
                if 0 <= tile_idx < len(self.tiles):
                    tile = self.tiles[tile_idx]
                    if tile.is_water:
                        continue
                    if river.id not in tile.river_ids:
                        tile.river_ids.append(river.id)
            for a, b in zip(river.tile_indices, river.tile_indices[1:]):
                if a < 0 or b < 0 or a >= len(self.tiles) or b >= len(self.tiles):
                    continue
                tile_a = self.tiles[a]
                tile_b = self.tiles[b]
                if tile_a.is_water or tile_b.is_water:
                    continue
                flux_value = max(tile_a.river_flux, tile_b.river_flux, river.flux)
                tile_a.river_neighbors[b] = max(tile_a.river_neighbors.get(b, 0.0), flux_value)
                tile_b.river_neighbors[a] = max(tile_b.river_neighbors.get(a, 0.0), flux_value)

    def _is_coastal_tile(self, tile_index: int) -> bool:
        """Return True if the tile borders water."""
        if tile_index < 0 or tile_index >= len(self.tiles):
            return False
        if self._coastal_cache_ready:
            return tile_index in self._coastal_tile_set
        return self._compute_tile_is_coastal(tile_index)

    def _gather_tiles_within_distance(
        self,
        origin_idx: int,
        max_distance: int,
        predicate: Optional[Callable[[int], bool]] = None,
    ) -> List[int]:
        """Breadth-first search to collect tiles within a hop distance."""
        if origin_idx < 0 or origin_idx >= len(self.tiles):
            return []
        try:
            distance_cap = max(1, int(max_distance))
        except (TypeError, ValueError):
            distance_cap = 1
        visited: Set[int] = {origin_idx}
        queue: deque[Tuple[int, int]] = deque([(origin_idx, 0)])
        results: List[int] = []
        while queue:
            current, dist = queue.popleft()
            if current != origin_idx:
                if predicate is None or predicate(current):
                    results.append(current)
            if dist >= distance_cap:
                continue
            current_tile = self.tiles[current]
            for neighbor_idx in current_tile.neighbors:
                if neighbor_idx < 0 or neighbor_idx >= len(self.tiles):
                    continue
                if neighbor_idx in visited:
                    continue
                visited.add(neighbor_idx)
                queue.append((neighbor_idx, dist + 1))
        return results

    def _get_coastal_tiles_within_distance(self, origin_idx: int, max_distance: int) -> List[int]:
        """Return coastal land tiles within a hop distance of the origin tile."""
        return self._gather_tiles_within_distance(
            origin_idx,
            max_distance,
            predicate=lambda idx: (not self.tiles[idx].is_water) and self._is_coastal_tile(idx),
        )

    def _get_owned_capital_tiles(self) -> Dict[int, Polity]:
        """Return mapping of tile index to polity for capitals currently under owner control."""
        capitals: Dict[int, Polity] = {}
        for polity in self.polities:
            if polity is None or not getattr(polity, 'is_active', True):
                continue
            capital_idx = self._ensure_polity_capital(polity)
            if capital_idx is None:
                continue
            tile = self.tiles[capital_idx]
            if tile.polity_id == polity.id and not tile.is_water:
                capitals[capital_idx] = polity
        return capitals

    def _is_capital_tile(self, tile_index: int) -> bool:
        """Return True if the tile is the active capital for its polity."""
        if tile_index < 0 or tile_index >= len(self.tiles):
            return False
        tile = self.tiles[tile_index]
        if tile.polity_id == -1:
            return False
        polity = self._get_polity(tile.polity_id)
        return bool(polity and getattr(polity, 'capital_tile_index', -1) == tile_index and polity.is_active)

    def _determine_home_region_id(self, tile_index: int) -> int:
        """Return the region identifier associated with the given tile index."""
        if 0 <= tile_index < len(self.tiles):
            region_id = self.tiles[tile_index].region_id
            return region_id if region_id is not None else -1
        return -1

    def _calculate_development_cap(self, tile: Tile) -> float:
        """Calculate the current development cap for a tile based on neighbor populations and rainfall.
        
        Args:
            tile: Tile to calculate development cap for
            
        Returns:
            Current development cap for the tile
        """
        # Use stored cap if available (updated during development processing)
        if hasattr(tile, 'current_dev_cap'):
            return tile.current_dev_cap
            
        # Otherwise calculate it
        base_dev_cap = 1000.0 + (tile.rainfall * 500.0)  # Base cap increases linearly with rainfall
        
        if tile.neighbors and tile.population > 0:
            # Calculate average population of adjacent tiles (excluding water)
            neighbor_populations = []
            for neighbor_idx in tile.neighbors:
                if neighbor_idx < len(self.tiles) and not self.tiles[neighbor_idx].is_water:
                    neighbor_populations.append(self.tiles[neighbor_idx].population)
            
            if neighbor_populations:
                avg_neighbor_pop = sum(neighbor_populations) / len(neighbor_populations)
                
                # Calculate population difference factor
                if avg_neighbor_pop > 0:
                    pop_factor = tile.population / avg_neighbor_pop
                else:
                    # If neighbors have no population, we have access to all their "resources"
                    pop_factor = 10.0  # Cap at 10x for empty neighbors
                
                # Development cap scales with population advantage
                pop_ratio_max = self.config.get('simulation.development.development_cap_population_ratio_max', 8.0)
                return base_dev_cap * min(pop_ratio_max, pop_factor)
            else:
                return base_dev_cap
        else:
            return base_dev_cap
    
    def _process_migration(self) -> None:
        """Process population migration between tiles."""
        import random
        
        # Get config values
        dp_threshold = self.config.get('simulation.migration.destination_potential_threshold', 90) if self.config else 90
        migration_threshold = self.config.get('simulation.migration.migration_threshold', 10) if self.config else 10
        migration_rate_max = self.config.get('simulation.migration.migration_rate_max', 0.1) if self.config else 0.1
        baseline_chance = self.config.get('simulation.migration.baseline_migration_chance', 0.1) if self.config else 0.1
        baseline_min = self.config.get('simulation.migration.baseline_migration_min', 2) if self.config else 2
        baseline_max = self.config.get('simulation.migration.baseline_migration_max', 6) if self.config else 6
        cross_polity_dp_penalty = self.config.get('simulation.migration.cross_polity_reluctance', 15) if self.config else 15
        cross_polity_baseline_multiplier = self.config.get('simulation.migration.cross_polity_baseline_multiplier', 0.35) if self.config else 0.35
        major_min_population = self.config.get('simulation.migration.major_migration_min_population', 15) if self.config else 15
        major_pressure_threshold = self.config.get('simulation.migration.major_migration_pressure_threshold', 1.15) if self.config else 1.15
        major_cooldown_ticks = self.config.get('simulation.migration.major_migration_cooldown_ticks', 3) if self.config else 3
        major_loss_ratio_trigger = self.config.get('simulation.migration.major_migration_loss_ratio_trigger', 0.2) if self.config else 0.2

        # Check for diaspora population center delay (also increases migration frequency)
        diaspora_delay_years = self.config.get('simulation.diaspora.population_center_delay_years', 0) if self.config else 0
        is_diaspora_delay_active = (
            diaspora_delay_years > 0 and 
            self.current_year < diaspora_delay_years
        )
        
        # Increase baseline migration chance during diaspora delay
        if is_diaspora_delay_active:
            diaspora_migration_multiplier = self.config.get('simulation.diaspora.migration_frequency_multiplier', 3.0) if self.config else 3.0
            baseline_chance *= diaspora_migration_multiplier

        population_center_lookup: Set[int] = {
            center.tile_index
            for center in self.population_centers
            if 0 <= center.tile_index < len(self.tiles)
        }
        dp_array = self._calculate_destination_potential_array(population_center_lookup)

        # Precompute arrays used by baseline migration to avoid per-tile overhead
        land_mask = np.fromiter((not t.is_water for t in self.tiles), dtype=bool, count=len(self.tiles))
        polity_arr = np.fromiter((t.polity_id for t in self.tiles), dtype=int, count=len(self.tiles))
        elev_arr = np.fromiter((t.elevation for t in self.tiles), dtype=float, count=len(self.tiles))

        def get_destination_potential(idx: int) -> float:
            if idx < 0 or idx >= len(dp_array):
                return 0.0
            return float(dp_array[idx])

        profile_enabled = self.tick_profiling_enabled
        baseline_elapsed = 0.0
        exodus_elapsed = 0.0
        dp_elapsed = 0.0
        
        # Only check tiles with population
        populated_tiles = [(i, tile) for i, tile in enumerate(self.tiles) 
                          if tile.population > 0 and not tile.is_water]
        
        for tile_idx, tile in populated_tiles:
            is_population_center = tile_idx in population_center_lookup
            # First, apply baseline migration for population diffusion
            baseline_start = time.perf_counter() if profile_enabled else None
            self._apply_baseline_migration(
                tile,
                tile_idx,
                baseline_chance,
                baseline_min,
                baseline_max,
                cross_polity_baseline_multiplier,
                land_mask=land_mask,
                polity_arr=polity_arr,
                elev_arr=elev_arr,
            )
            if baseline_start is not None:
                baseline_elapsed += time.perf_counter() - baseline_start
            
            # Apply Exodus migration (region-wide search for better destinations)
            exodus_start = time.perf_counter() if profile_enabled else None
            self._apply_exodus_migration(
                tile,
                tile_idx,
                dp_lookup=get_destination_potential,
                population_center_lookup=population_center_lookup,
            )
            if exodus_start is not None:
                exodus_elapsed += time.perf_counter() - exodus_start
            
            # Then apply destination-potential based migration (major migration)
            # Only allow major migrations in tiles with development >= 1 (lowered from 3)
            if tile.development < 1:
                continue
            is_severely_overcrowded = (
                tile.population > 0 and tile.development > 0 and
                tile.population / max(1, tile.development) > 2.0
            )
            if not self._should_run_major_migration(
                tile,
                min_population=major_min_population,
                pressure_threshold=major_pressure_threshold,
                cooldown_ticks=major_cooldown_ticks,
                is_severely_overcrowded=is_severely_overcrowded,
                loss_ratio_trigger=major_loss_ratio_trigger,
                is_population_center=is_population_center,
            ):
                continue
            dp_start = time.perf_counter() if profile_enabled else None
            # Pass origin population to DP calculation for population differential logic
            current_dp = self._calculate_destination_potential(
                tile,
                tile_idx=tile_idx,
                population_center_lookup=population_center_lookup,
                origin_population=tile.population
            )
            
            # Skip if destination potential is high enough (not desperate to leave)
            if current_dp >= dp_threshold:
                # For Mixed/Migratory, allow major migration check even if DP is high
                dominant_culture_name = self._get_tile_majority_culture(tile)
                itinerance_label = "Sedentary"
                if dominant_culture_name:
                    culture_obj = self._find_culture_by_name(dominant_culture_name)
                    if culture_obj:
                        self._ensure_culture_ideas(culture_obj)
                        itin = culture_obj.ideas.get("itinerance", 0.25)
                        label = self._idea_label_for_value("itinerance", itin)
                        if label:
                            itinerance_label = label
                if itinerance_label not in ("Mixed", "Migratory"):
                    if dp_start is not None:
                        dp_elapsed += time.perf_counter() - dp_start
                    # Debug: uncomment to see why tiles aren't migrating
                    # print(f"Tile {tile_idx}: DP {current_dp:.1f} >= threshold {dp_threshold}, not desperate enough to migrate")
                    continue
            
            # Evaluate all neighbors and find the best migration destinations
            neighbor_indices = tile.neighbors.copy()
            random.shuffle(neighbor_indices)
            
            # Store potential destinations with their scores
            destination_candidates = []  # Store (neighbor_idx, dp_diff, migrants) tuples
            
            for neighbor_idx in neighbor_indices:
                if neighbor_idx >= len(self.tiles):
                    continue
                    
                neighbor = self.tiles[neighbor_idx]
                
                # Skip water tiles as migration destinations
                if neighbor.is_water:
                    continue
                    
                neighbor_dp = self._calculate_destination_potential(
                    self.tiles[neighbor_idx],
                    tile_idx=neighbor_idx,
                    population_center_lookup=population_center_lookup,
                    origin_population=tile.population
                )
                
                # Calculate DP differential and apply polity reluctance penalties
                dp_diff = neighbor_dp - current_dp
                if cross_polity_dp_penalty > 0 and self._is_cross_polity_migration(tile, neighbor):
                    dp_diff -= cross_polity_dp_penalty
                
                # Apply elevation penalty for uphill migration from higher elevations
                elevation_penalty_threshold = self.config.get('simulation.migration.elevation_penalty_threshold', 0.4) if self.config else 0.4
                elevation_penalty_factor = self.config.get('simulation.migration.elevation_penalty_factor', 50.0) if self.config else 50.0
                elevation_penalty_exponent = self.config.get('simulation.migration.elevation_penalty_curve_exponent', 2.0) if self.config else 2.0
                
                if tile.elevation > elevation_penalty_threshold and neighbor.elevation > tile.elevation:
                    elevation_diff = neighbor.elevation - tile.elevation
                    # Only penalize upward migration, using exponential curve for higher elevations
                    elevation_penalty = elevation_penalty_factor * (elevation_diff ** elevation_penalty_exponent)
                    dp_diff -= elevation_penalty
                
                # Lower threshold for severely overcrowded areas
                effective_threshold = migration_threshold if not is_severely_overcrowded else migration_threshold * 0.5
                
                if dp_diff > effective_threshold:  # Significant difference
                    # Calculate migration amount - higher rates for desperate situations
                    base_migration_factor = min(migration_rate_max, dp_diff / 100)

                    # --- Itinerance integration ---
                    # Get dominant culture for the tile
                    dominant_culture_name = self._get_tile_majority_culture(tile)
                    itinerance = 0.25  # Default baseline
                    if dominant_culture_name:
                        culture_obj = self._find_culture_by_name(dominant_culture_name)
                        if culture_obj:
                            self._ensure_culture_ideas(culture_obj)
                            itinerance = culture_obj.ideas.get("itinerance", 0.25)
                    # New scaling: nonlinear, up to 50% at high itinerance, low end nearly unchanged
                    # At itinerance=0.25 (baseline), scale=1.0; at itinerance=1.0, scale=2.0 (for up to 50% eligible)
                    # Use a smoothstep curve for a gentle ramp-up
                    def itinerance_migration_scale(itin):
                        # Clamp to [0,1]
                        x = max(0.0, min(1.0, (itin - 0.25) / 0.75))
                        # Smoothstep: 3x^2 - 2x^3
                        return 1.0 + (3 * x**2 - 2 * x**3)  # 1.0 to 2.0
                    scale = itinerance_migration_scale(itinerance)
                    itinerance_label = self._idea_label_for_value("itinerance", itinerance)
                    migratory_flat_boost = 0.15 if itinerance_label == "Migratory" else 0.0
                    if is_severely_overcrowded:
                        overcrowding_multiplier = min(2.0, tile.population / max(1, tile.development))
                        migration_factor = base_migration_factor * overcrowding_multiplier * (scale + migratory_flat_boost)
                    else:
                        migration_factor = base_migration_factor * (scale + migratory_flat_boost)
                    # Cap: at high itinerance, allow up to 50% of population to move
                    max_total_migration = 0.5 if scale > 1.99 else (0.25 if is_severely_overcrowded else 0.15)
                    migrants = int(tile.population * migration_factor)

                    if migrants > 0:
                        destination_candidates.append((neighbor_idx, dp_diff, migrants))
            
            # Sort candidates by DP difference (best destinations first)
            destination_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Apply migrations to the best destinations, respecting migration caps
            total_migrants = 0
            migrations = []  # Store (neighbor_idx, migrants) tuples
            max_total_migration = 0.25 if is_severely_overcrowded else 0.15
            migration_cap = tile.population * max_total_migration
            
            for neighbor_idx, dp_diff, migrants in destination_candidates:
                if total_migrants >= migration_cap:
                    break
                    
                # Adjust migration amount if it would exceed the cap
                available_migration_capacity = migration_cap - total_migrants
                actual_migrants = min(migrants, int(available_migration_capacity))
                
                if actual_migrants > 0:
                    migrations.append((neighbor_idx, actual_migrants))
                    total_migrants += actual_migrants
            
            # Apply all migrations and handle polity expansion
            for neighbor_idx, migrants in migrations:
                if tile.population > migrants:  # Ensure we don't migrate more than available
                    # Get config values for polity expansion
                    control_threshold = self.config.get('simulation.migration.polity_expansion_control_threshold', 50) if self.config else 50
                    pop_threshold = self.config.get('simulation.migration.polity_expansion_population_threshold', 10) if self.config else 10

                    # Perform migration
                    tile.population -= migrants
                    self.tiles[neighbor_idx].population += migrants

                    # --- Development transfer for regular migration (itinerance-based) ---
                    # Determine itinerance label for the migrating population
                    dominant_culture_name = self._get_tile_majority_culture(tile)
                    itinerance_label = "Sedentary"
                    if dominant_culture_name:
                        culture_obj = self._find_culture_by_name(dominant_culture_name)
                        if culture_obj:
                            self._ensure_culture_ideas(culture_obj)
                            itin = culture_obj.ideas.get("itinerance", 0.25)
                            label = self._idea_label_for_value("itinerance", itin)
                            if label:
                                itinerance_label = label
                    if itinerance_label == "Migratory":
                        transfer_ratio = self.config.get('simulation.migration.itinerant_migration_development_transfer_ratio', 0.7) if self.config else 0.7
                    elif itinerance_label == "Mixed":
                        transfer_ratio = self.config.get('simulation.migration.mixed_migration_development_transfer_ratio', 0.5) if self.config else 0.5
                    else:
                        transfer_ratio = self.config.get('simulation.migration.sedentary_migration_development_transfer_ratio', 0.3) if self.config else 0.3
                    loss_percent = self.config.get('simulation.migration.migration_development_loss_percent', 0.25) if self.config else 0.25
                    dev_transfer = migrants * transfer_ratio
                    dev_net_loss = dev_transfer * loss_percent
                    dev_loss = dev_transfer
                    dev_gain = dev_transfer - dev_net_loss
                    if dev_loss > 0:
                        tile.development = max(0.0, tile.development - dev_loss)
                    if dev_gain > 0:
                        self.tiles[neighbor_idx].development += dev_gain

                    # Apply control effects from migration
                    self._apply_migration_control_effects(
                        tile,
                        self.tiles[neighbor_idx],
                        migrants,
                        source_idx=tile_idx,
                        dest_idx=neighbor_idx
                    )

                    # Check for polity expansion
                    source_tile = tile
                    dest_tile = self.tiles[neighbor_idx]

                    # Conditions for polity expansion:
                    # 1. Source tile is controlled by a polity (polity_id >= 0)
                    # 2. Source tile has sufficient control (>=50%)
                    # 3. Destination tile is uncontrolled (polity_id == -1)
                    # 4. Destination tile has sufficient population after migration (>=10)
                    if (source_tile.polity_id >= 0 and 
                        source_tile.control_level >= control_threshold and
                        dest_tile.polity_id == -1 and
                        dest_tile.population >= pop_threshold):

                        # Assimilate destination tile into source polity
                        previous_polity = dest_tile.polity_id
                        dest_tile.polity_id = source_tile.polity_id
                        # Apply assimilation penalty
                        assimilation_penalty = self.config.get('simulation.control.assimilation_penalty', 10) if self.config else 10
                        initial_control = max(1, source_tile.control_level - assimilation_penalty)
                        before_control = dest_tile.control_level
                        dest_tile.control_level = initial_control
                        self._record_control_change(
                            dest_tile,
                            before_control,
                            dest_tile.control_level,
                            label="Assimilation Penalty",
                            category="flat",
                            details=f"penalty {assimilation_penalty}",
                        )
                        self._on_tile_polity_changed(neighbor_idx, previous_polity, dest_tile.polity_id)
                        self._apply_control_alignment_cap(neighbor_idx)

                        # Update polity's tile list for border rendering
                        if source_tile.polity_id < len(self.polities):
                            polity = self.polities[source_tile.polity_id]
                            if neighbor_idx not in polity.tile_indices:
                                polity.tile_indices.append(neighbor_idx)

                        # Expansion events are intentionally silent in debug logs to reduce noise
            if dp_start is not None:
                dp_elapsed += time.perf_counter() - dp_start

        if profile_enabled:
            self._accumulate_tick_profile_section('migration:baseline', baseline_elapsed)
            self._accumulate_tick_profile_section('migration:exodus', exodus_elapsed)
            self._accumulate_tick_profile_section('migration:dp', dp_elapsed)
    
    def _apply_baseline_migration(
        self,
        tile: Tile,
        tile_idx: int,
        baseline_chance: float,
        baseline_min: int,
        baseline_max: int,
        cross_polity_multiplier: float,
        *,
        land_mask: Optional[np.ndarray] = None,
        polity_arr: Optional[np.ndarray] = None,
        elev_arr: Optional[np.ndarray] = None,
    ) -> None:
        """Apply baseline migration for population diffusion.
        
        Args:
            tile: Source tile for migration
            tile_idx: Index of the source tile
            baseline_chance: Probability of baseline migration occurring
            baseline_min: Minimum number of people to migrate
            baseline_max: Maximum number of people to migrate
        """
        import random

        if tile.population < 3:
            return

        # Use precomputed arrays when available; fall back to attributes otherwise
        if land_mask is not None:
            dest_indices = [i for i in tile.neighbors if 0 <= i < len(self.tiles) and land_mask[i]]
        else:
            dest_indices = [i for i in tile.neighbors if 0 <= i < len(self.tiles) and not self.tiles[i].is_water]
        if not dest_indices:
            return

        tile_elev = tile.elevation
        neighbor_elev = (elev_arr[dest_indices] if elev_arr is not None
                         else np.fromiter((self.tiles[i].elevation for i in dest_indices), dtype=float, count=len(dest_indices)))

        adjusted_chances = np.full(len(dest_indices), baseline_chance, dtype=float)

        if cross_polity_multiplier < 1.0:
            dest_polity = (polity_arr[dest_indices] if polity_arr is not None
                           else np.fromiter((self.tiles[i].polity_id for i in dest_indices), dtype=int, count=len(dest_indices)))
            cross_mask = dest_polity != tile.polity_id
            adjusted_chances[cross_mask] *= max(0.0, cross_polity_multiplier)

        elevation_penalty_threshold = self.config.get('simulation.migration.elevation_penalty_threshold', 0.4) if self.config else 0.4
        elevation_penalty_factor = self.config.get('simulation.migration.elevation_penalty_factor', 50.0) if self.config else 50.0
        elevation_penalty_exponent = self.config.get('simulation.migration.elevation_penalty_curve_exponent', 2.0) if self.config else 2.0

        uphill_mask = (tile_elev > elevation_penalty_threshold) & (neighbor_elev > tile_elev)
        if np.any(uphill_mask):
            elev_diff = neighbor_elev[uphill_mask] - tile_elev
            penalties = elevation_penalty_factor * np.power(elev_diff, elevation_penalty_exponent)
            chance_mult = np.maximum(0.0, 1.0 - (penalties / 100.0))
            adjusted_chances[uphill_mask] *= chance_mult

        rolls = np.random.random(len(dest_indices))
        for i, neighbor_idx in enumerate(dest_indices):
            if rolls[i] >= adjusted_chances[i]:
                continue
            migration_amount = random.randint(baseline_min, baseline_max)
            migrants = min(migration_amount, tile.population - 1)
            if migrants <= 0:
                continue
            tile.population -= migrants
            self.tiles[neighbor_idx].population += migrants

            # --- Development transfer for baseline migration (if enabled) ---
            # Always initialize itinerance and itinerance_label to safe defaults
            itinerance = 0.25  # Default baseline
            itinerance_label = "Sedentary"
            dominant_culture_name = self._get_tile_majority_culture(tile)
            if dominant_culture_name:
                culture_obj = self._find_culture_by_name(dominant_culture_name)
                if culture_obj:
                    self._ensure_culture_ideas(culture_obj)
                    itin = culture_obj.ideas.get("itinerance", 0.25)
                    label = self._idea_label_for_value("itinerance", itin)
                    if label:
                        itinerance_label = label
                    itinerance = itin
            # Use a single config value for baseline migration dev transfer, or fallback to sedentary ratio
            transfer_ratio = self.config.get('simulation.migration.sedentary_migration_development_transfer_ratio', 0.25) if self.config else 0.25
            loss_percent = self.config.get('simulation.migration.migration_development_loss_percent', 0.2) if self.config else 0.2
            dev_transfer = migrants * transfer_ratio
            dev_net_loss = dev_transfer * loss_percent
            dev_loss = dev_transfer
            dev_gain = dev_transfer - dev_net_loss
            if dev_loss > 0:
                tile.development = max(0.0, tile.development - dev_loss)
            if dev_gain > 0:
                self.tiles[neighbor_idx].development += dev_gain

            self._apply_migration_control_effects(
                tile,
                self.tiles[neighbor_idx],
                migrants,
                source_idx=tile_idx,
                dest_idx=neighbor_idx
            )

    def _should_run_major_migration(
        self,
        tile: Tile,
        min_population: int,
        pressure_threshold: float,
        cooldown_ticks: int,
        is_severely_overcrowded: bool,
        loss_ratio_trigger: float,
        is_population_center: bool,
    ) -> bool:
        """Return True when the expensive DP-based migration step should run for the tile."""
        if tile.population <= 0:
            tile.major_migration_cooldown = 0
            return False
        try:
            min_population = max(0, int(min_population))
        except (TypeError, ValueError):
            min_population = 0
        if tile.population < min_population:
            return False
        if is_severely_overcrowded:
            tile.major_migration_cooldown = 0
            return True
        if loss_ratio_trigger > 0 and tile.last_tick_deaths > 0:
            recent_population = tile.population + tile.last_tick_deaths
            loss_ratio = tile.last_tick_deaths / max(1, recent_population)
            if loss_ratio >= loss_ratio_trigger:
                tile.major_migration_cooldown = 0
                return True
        try:
            pressure_threshold = float(pressure_threshold)
        except (TypeError, ValueError):
            pressure_threshold = 0.0
        pressure_ratio = tile.population / max(1.0, tile.development)
        if pressure_threshold <= 0.0 or pressure_ratio >= pressure_threshold:
            tile.major_migration_cooldown = 0
            return True
        cooldown_ticks = max(0, int(cooldown_ticks))
        cooldown_remaining = getattr(tile, 'major_migration_cooldown', 0)
        if cooldown_remaining > 0:
            tile.major_migration_cooldown = cooldown_remaining - 1
            return False
        tile.major_migration_cooldown = cooldown_ticks
        return True

    def _apply_exodus_migration(
        self,
        tile: Tile,
        tile_idx: int,
        dp_lookup: Optional[Callable[[int], float]] = None,
        population_center_lookup: Optional[Set[int]] = None,
    ) -> None:
        """Apply Exodus migration - sophisticated region-wide migration for developed settlements.
        
        Args:
            tile: Source tile for exodus migration
            tile_idx: Index of the source tile
            dp_lookup: Optional callable that returns cached destination potential for a tile index
            population_center_lookup: Optional lookup set for population center indices
        """
        import random
        
        mass_death_ratio = self.config.get('simulation.migration.mass_death_exodus_ratio', 0.1) if self.config else 0.1
        mass_death_min_pop = self.config.get('simulation.migration.mass_death_exodus_population', 50) if self.config else 50
        dev_transfer_ratio = self.config.get('simulation.migration.exodus_development_transfer_ratio', 0.5) if self.config else 0.5
        forced_exodus = (
            tile.population >= mass_death_min_pop and
            tile.last_tick_deaths >= tile.population * mass_death_ratio
        )
        trigger_reason: Optional[str] = None

        if not self._is_coastal_tile(tile_idx):
            return

        if not forced_exodus:
            # Exodus migration requirements
            if (tile.population < 50 or 
                tile.development < tile._calculate_development_cap(tile) * 0.5 if hasattr(tile, '_calculate_development_cap') else 
                tile.development < self._calculate_development_cap(tile) * 0.5):
                return  # Must have at least 50 population and be at 50% of development cap

            # --- Itinerance integration for exodus frequency ---
            dominant_culture_name = self._get_tile_majority_culture(tile)
            itinerance = 0.25  # Default baseline
            if dominant_culture_name:
                culture_obj = self._find_culture_by_name(dominant_culture_name)
                if culture_obj:
                    self._ensure_culture_ideas(culture_obj)
                    itinerance = culture_obj.ideas.get("itinerance", 0.25)
            # Scale exodus chance: 0.25 = baseline, <0.25 = less, >0.25 = more
            exodus_chance = self.config.get('simulation.migration.exodus_migration_chance', 0.02) if self.config else 0.02
            exodus_chance_scaled = exodus_chance * (itinerance / 0.25 if 0.25 > 0 else 1.0)
            # Threshold-based boost for 'Migratory' societies
            itinerance_label = self._idea_label_for_value("itinerance", itinerance)
            if itinerance_label == "Migratory":
                exodus_chance_scaled += 0.03  # Flat +3% chance
            if random.random() > exodus_chance_scaled:
                return
            trigger_reason = f"opportunity (chance={exodus_chance_scaled:.2f})"
        else:
            # Ensure forced exodus still respects minimum population requirement
            if tile.population < mass_death_min_pop:
                return
            ratio = tile.last_tick_deaths / max(1, tile.population)
            trigger_reason = f"mass-death (deaths={tile.last_tick_deaths}, ratio={ratio:.2f})"
            
        raw_distance_cap = self.config.get('simulation.migration.exodus_max_distance', 30) if self.config else 30
        try:
            distance_cap = max(1, int(raw_distance_cap))
        except (TypeError, ValueError):
            distance_cap = 30
        candidate_tiles = self._get_coastal_tiles_within_distance(tile_idx, distance_cap)
        if not candidate_tiles:
            return
        if dp_lookup is not None:
            current_dp = dp_lookup(tile_idx)
        else:
            current_dp = self._calculate_destination_potential(
                tile,
                tile_idx=tile_idx,
                population_center_lookup=population_center_lookup,
            )
        cross_polity_penalty = self.config.get('simulation.migration.cross_polity_reluctance', 15) if self.config else 15
        dp_threshold = self.config.get('simulation.migration.migration_threshold', 10) if self.config else 10
        if forced_exodus:
            dp_threshold *= 0.5
        diff_region_bonus = self.config.get('simulation.migration.exodus_cross_region_bonus', 5) if self.config else 5
        pop_center_threshold = self.config.get('simulation.development.population_center_threshold', 50) if self.config else 50
        source_region_id = tile.region_id if 0 <= tile.region_id < len(self.regions) else -1
        random.shuffle(candidate_tiles)
        candidate_scores: List[Tuple[int, float, int]] = []
        for dest_idx in candidate_tiles:
            if dest_idx == tile_idx:
                continue
            dest_tile = self.tiles[dest_idx]
            if dest_tile.population > pop_center_threshold:
                continue
            if dp_lookup is not None:
                dest_dp = dp_lookup(dest_idx)
            else:
                dest_dp = self._calculate_destination_potential(
                    dest_tile,
                    tile_idx=dest_idx,
                    population_center_lookup=population_center_lookup,
                    origin_population=tile.population
                )
            dp_diff = dest_dp - current_dp
            # Apply linear distance penalty
            distance = self._calculate_tile_distance(tile_idx, dest_idx) if hasattr(self, '_calculate_tile_distance') else abs(tile_idx - dest_idx)
            distance_penalty_scale = self.config.get('simulation.migration.exodus_distance_penalty_scale', 1.0) if self.config else 1.0
            dp_diff -= distance * distance_penalty_scale
            if cross_polity_penalty > 0 and self._is_cross_polity_migration(tile, dest_tile):
                dp_diff -= cross_polity_penalty
            if diff_region_bonus and dest_tile.region_id != source_region_id:
                dp_diff += diff_region_bonus
            # Apply elevation penalty for uphill migration from higher elevations
            elevation_penalty_threshold = self.config.get('simulation.migration.elevation_penalty_threshold', 0.4) if self.config else 0.4
            elevation_penalty_factor = self.config.get('simulation.migration.elevation_penalty_factor', 50.0) if self.config else 50.0
            elevation_penalty_exponent = self.config.get('simulation.migration.elevation_penalty_curve_exponent', 2.0) if self.config else 2.0
            if tile.elevation > elevation_penalty_threshold and dest_tile.elevation > tile.elevation:
                elevation_diff = dest_tile.elevation - tile.elevation
                # Only penalize upward migration, using exponential curve for higher elevations
                elevation_penalty = elevation_penalty_factor * (elevation_diff ** elevation_penalty_exponent)
                dp_diff -= elevation_penalty
            if dp_diff <= dp_threshold:
                continue
            candidate_scores.append((dest_idx, dp_diff, dest_tile.population))
        if not candidate_scores:
            return
        best_destination, _, _ = max(candidate_scores, key=lambda entry: (entry[1], -entry[2]))

        min_ratio = self.config.get('simulation.migration.exodus_min_population_ratio', 0.05) if self.config else 0.05
        max_ratio = self.config.get('simulation.migration.exodus_max_population_ratio', 0.2) if self.config else 0.2
        try:
            min_ratio = max(0.0, float(min_ratio))
        except (TypeError, ValueError):
            min_ratio = 0.05
        try:
            max_ratio = max(min_ratio, float(max_ratio))
        except (TypeError, ValueError):
            max_ratio = max(min_ratio, 0.2)
        if max_ratio < min_ratio:
            max_ratio = min_ratio
        # New scaling: nonlinear, up to 50% at high itinerance, low end nearly unchanged
        def itinerance_exodus_scale(itin):
            x = max(0.0, min(1.0, (itin - 0.25) / 0.75))
            return 1.0 + (3 * x**2 - 2 * x**3)  # 1.0 to 2.0
        scale = itinerance_exodus_scale(itinerance)
        min_ratio *= scale
        max_ratio *= scale
        # Threshold-based boost for 'Migratory' societies
        if itinerance_label == "Migratory":
            min_ratio += 0.05
            max_ratio += 0.05
        # Cap: at high itinerance, allow up to 50% of population to move
        max_cap = 0.5
        migration_ratio = min(max_cap, random.uniform(min_ratio, max_ratio))
        migrants = max(10, int(tile.population * migration_ratio))
        migrants = min(migrants, tile.population - 10)
        
        if migrants > 0:
            pre_migration_population = tile.population
            pre_migration_development = tile.development
            tile.population -= migrants
            self.tiles[best_destination].population += migrants

            # Transfer a share of development with the migrating population (itinerance-based)
            ratio = migrants / max(1, pre_migration_population)
            # Determine itinerance label for the migrating population
            dominant_culture_name = self._get_tile_majority_culture(tile)
            itinerance_label = "Sedentary"
            if dominant_culture_name:
                culture_obj = self._find_culture_by_name(dominant_culture_name)
                if culture_obj:
                    self._ensure_culture_ideas(culture_obj)
                    itin = culture_obj.ideas.get("itinerance", 0.25)
                    label = self._idea_label_for_value("itinerance", itin)
                    if label:
                        itinerance_label = label
            if itinerance_label == "Migratory":
                transfer_ratio = self.config.get('simulation.migration.itinerant_migration_development_transfer_ratio', 0.7) if self.config else 0.7
            elif itinerance_label == "Mixed":
                transfer_ratio = self.config.get('simulation.migration.mixed_migration_development_transfer_ratio', 0.5) if self.config else 0.5
            else:
                transfer_ratio = self.config.get('simulation.migration.sedentary_migration_development_transfer_ratio', 0.3) if self.config else 0.3
            loss_percent = self.config.get('simulation.migration.migration_development_loss_percent', 0.25) if self.config else 0.25
            dev_transfer = min(pre_migration_development, pre_migration_development * ratio * transfer_ratio)
            dev_loss = dev_transfer * loss_percent
            dev_gain = dev_transfer - dev_loss
            if dev_loss > 0:
                tile.development = max(0.0, tile.development - dev_loss)
            if dev_gain > 0:
                self.tiles[best_destination].development += dev_gain
                # Respect destination development cap softly by trimming overflow
                dest_cap = self._calculate_development_cap(self.tiles[best_destination])
                if self.tiles[best_destination].development > dest_cap:
                    self.tiles[best_destination].development = dest_cap

            # Apply control effects
            cultural_multiplier = self.config.get('simulation.culture.exodus_cultural_multiplier', 1.35) if self.config else 1.35
            self._apply_migration_control_effects(
                tile,
                self.tiles[best_destination],
                migrants,
                source_idx=tile_idx,
                dest_idx=best_destination,
                cultural_multiplier=cultural_multiplier
            )
            
            trigger_text = trigger_reason or "unspecified"
            source_name = self._get_population_center_name(tile_idx) or f"Tile_{tile_idx}"
            dest_name = self._get_population_center_name(best_destination) or f"Tile_{best_destination}"
            self._log_event(
                "migration",
                f"Exodus migration ({trigger_text}): {migrants} people left {source_name} "
                f"(prev pop {pre_migration_population}) for {dest_name}; dev moved {dev_transfer:.1f}"
            )

    def _apply_migration_control_effects(
        self,
        source_tile: Tile,
        dest_tile: Tile,
        migrants: int,
        source_idx: Optional[int] = None,
        dest_idx: Optional[int] = None,
        cultural_multiplier: float = 1.0,
    ) -> None:
        """Apply control and cultural effects when population migrates."""
        # Always propagate cultural influence regardless of ownership
        self._apply_migration_cultural_effects(source_tile, dest_tile, migrants, impact_multiplier=cultural_multiplier)
        if dest_idx is None:
            dest_idx = self._find_tile_index_by_identity(dest_tile)
        if dest_idx is not None:
            self._maybe_assign_capital_primary_culture(dest_idx)

        # Control effects only matter when both tiles are governed
        if source_tile.polity_id == -1 or dest_tile.polity_id == -1:
            return
        
        # Get config values
        influence_factor = self.config.get('simulation.control.migration_influence_factor', 0.5) if self.config else 0.5
        # --- Itinerance integration for migration influence ---
        dominant_culture_name = self._get_tile_majority_culture(source_tile)
        itinerance = 0.25  # Default baseline
        if dominant_culture_name:
            culture_obj = self._find_culture_by_name(dominant_culture_name)
            if culture_obj:
                self._ensure_culture_ideas(culture_obj)
                itinerance = culture_obj.ideas.get("itinerance", 0.25)
        itinerance_scale = itinerance / 0.25 if 0.25 > 0 else 1.0
        influence_factor *= itinerance_scale

        # Calculate migration influence based on source control level
        before = dest_tile.control_level
        if source_tile.control_level >= 50:
            # High control source increases destination control
            control_influence = (source_tile.control_level - 50) / 50  # 0.0 to 1.0
            control_change = int(migrants * influence_factor * control_influence)
            dest_tile.control_level = min(100, dest_tile.control_level + control_change)
        else:
            # Low control source decreases destination control
            control_influence = (50 - source_tile.control_level) / 50  # 0.0 to 1.0
            control_change = int(migrants * influence_factor * control_influence)
            dest_tile.control_level = max(1, dest_tile.control_level - control_change)
        details = f"{migrants} migrants ({before}->{dest_tile.control_level})"
        if source_idx is not None and dest_idx is not None:
            details += f" via {source_idx}->{dest_idx}"
        self._record_control_change(
            dest_tile,
            before,
            dest_tile.control_level,
            label="Migration Influence",
            details=details,
        )
        if dest_idx is not None:
            self._apply_control_alignment_cap(dest_idx)

    def _is_cross_polity_migration(self, source_tile: Tile, dest_tile: Tile) -> bool:
        """Return True when both tiles are owned by different polities."""
        return (
            source_tile.polity_id >= 0 and
            dest_tile.polity_id >= 0 and
            source_tile.polity_id != dest_tile.polity_id
        )

    def _apply_migration_cultural_effects(
        self,
        source_tile: Tile,
        dest_tile: Tile,
        migrants: int,
        impact_multiplier: float = 1.0,
    ) -> None:
        """Apply cultural effects when population migrates between tiles.
        
        Args:
            source_tile: Tile population is migrating from
            dest_tile: Tile population is migrating to
            migrants: Number of people migrating
            impact_multiplier: Additional scaling factor for cultural transfer
        """
        # Only apply cultural effects if enough population units are moved
        culture_threshold = self.config.get('simulation.culture.migration_culture_threshold', 10) if self.config else 10
        if migrants < culture_threshold:
            return
        
        if not self._tile_supports_culture_tracking(source_tile):
            return
        if dest_tile.population == 0:
            dest_tile.cultural_makeup = {}
            return
        # Skip if source has no cultural makeup
        if not source_tile.cultural_makeup:
            return
        
        # Get config values
        cultural_influence_factor = self.config.get('simulation.culture.migration_cultural_influence', 0.1) if self.config else 0.1
        
        # Calculate cultural influence based on migration size relative to destination population
        if dest_tile.population > 0:
            influence_ratio = migrants / max(dest_tile.population, migrants)  # Ensure we don't divide by zero after migration
        else:
            influence_ratio = 1.0  # Migrants become dominant in empty tile
        
        # Ensure destination has cultural makeup
        if not dest_tile.cultural_makeup:
            dest_tile.cultural_makeup = {}
        
        original_cultures = set(dest_tile.cultural_makeup.keys())
        
        cultural_transfer = cultural_influence_factor * influence_ratio * max(0.0, impact_multiplier)
        if cultural_transfer <= 0:
            return
        
        culture_shares = self._calculate_biased_culture_shares(source_tile, dest_tile)
        if not culture_shares:
            culture_shares = source_tile.cultural_makeup
        for culture_name, source_percentage in culture_shares.items():
            current_dest_percentage = dest_tile.cultural_makeup.get(culture_name, 0.0)
            transfer_amount = source_percentage * cultural_transfer
            dest_tile.cultural_makeup[culture_name] = current_dest_percentage + transfer_amount
        
        # Reset cooldown for newly introduced non-syncretic cultures (only if tile already had culture)
        if original_cultures:  # Only reset cooldown if tile previously had cultural composition
            for culture_name in culture_shares:
                if (culture_name not in original_cultures and 
                    dest_tile.cultural_makeup.get(culture_name, 0.0) > 0.0 and
                    not self._get_syncretic_parents_for(culture_name)):
                    dest_tile.last_culture_spawn_year = self.current_year
        
        # Ensure cultural composition sums to 100%
        self._normalize_tile_culture(dest_tile)

    def _calculate_biased_culture_shares(self, source_tile: Tile, dest_tile: Tile) -> Dict[str, float]:
        """Return normalized culture shares with destination bias for minorities."""
        if not source_tile.cultural_makeup:
            return {}
        bias_scale = self.config.get('simulation.migration.cultural_bias_scale', 1.0) if self.config else 1.0
        weights: Dict[str, float] = {}
        for culture_name, source_share in source_tile.cultural_makeup.items():
            if source_share <= 0:
                continue
            weight = source_share
            dest_share = 0.0
            if getattr(dest_tile, 'cultural_makeup', None):
                dest_share = dest_tile.cultural_makeup.get(culture_name, 0.0)
            diff = dest_share - source_share
            if bias_scale > 0 and diff > 0:
                weight *= (1.0 + min(diff, 1.0) * bias_scale)
            weights[culture_name] = weight
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return {}
        return {name: weight / total_weight for name, weight in weights.items()}

    def _normalize_tile_culture(self, tile: Tile) -> None:
        """Normalize a tile's cultural makeup to sum to 1.0 and remove noise."""
        if not tile.cultural_makeup:
            return
        
        positive_values = {name: max(0.0, value) for name, value in tile.cultural_makeup.items()}
        total = sum(positive_values.values())
        if total <= 0:
            tile.cultural_makeup = {}
            return
        
        normalized = {name: value / total for name, value in positive_values.items()}
        min_fraction = 0.01
        dominant = max(normalized.items(), key=lambda x: x[1])
        dominant_name = dominant[0]
        removed_total = 0.0
        filtered: Dict[str, float] = {}
        for name, value in normalized.items():
            if value >= min_fraction:
                filtered[name] = value
            else:
                removed_total += value
        
        if not filtered:
            # Preserve the dominant culture if everything was below the threshold
            tile.cultural_makeup = {dominant_name: 1.0}
            return
        
        if removed_total > 0:
            filtered[dominant_name] = filtered.get(dominant_name, 0.0) + removed_total
        
        final_total = sum(filtered.values())
        if final_total <= 0:
            tile.cultural_makeup = {}
            return
        for name in filtered:
            filtered[name] /= final_total
        
        tile.cultural_makeup = filtered

    def _tile_supports_culture_tracking(self, tile: Optional[Tile]) -> bool:
        """Return True if the tile meets the population threshold for culture data."""
        if tile is None or tile.is_water:
            return False
        threshold = getattr(self, 'culture_population_threshold', 0)
        if threshold <= 0:
            return True
        return tile.population >= threshold

    def _enforce_culture_population_threshold_for_tile(self, tile: Optional[Tile]) -> None:
        """Clear cultural makeup on tiles that fall below the population cutoff."""
        if tile is None or tile.is_water:
            return
        threshold = getattr(self, 'culture_population_threshold', 0)
        if threshold <= 0:
            return
        if tile.population == 0 and tile.cultural_makeup:
            tile.cultural_makeup = {}

    def _enforce_culture_population_thresholds(self) -> None:
        """Apply the population threshold rule to every tile in the world."""
        threshold = getattr(self, 'culture_population_threshold', 0)
        if threshold <= 0:
            return
        for tile in self.tiles:
            self._enforce_culture_population_threshold_for_tile(tile)

    def _apply_control_alignment_cap(self, tile_idx: int) -> None:
        """Clamp control when a tile's alignment with its polity is too low."""
        if tile_idx < 0 or tile_idx >= len(self.tiles):
            return
        tile = self.tiles[tile_idx]
        if tile.polity_id == -1 or not tile.cultural_makeup:
            return
        polity = self._get_polity(tile.polity_id)
        if not polity or not polity.primary_culture:
            return
        primary_share = tile.cultural_makeup.get(polity.primary_culture, 0.0)
        base_threshold = self.config.get('simulation.culture.low_alignment_threshold', 0.5) if self.config else 0.5
        threshold = self._get_alignment_threshold(polity, base_threshold)
        if threshold <= 0 or primary_share >= threshold:
            return
        bias = self._get_tolerance_bias(polity)
        cap_span = self.config.get('simulation.culture.tolerance_alignment_cap_span', 12) if self.config else 12
        min_cap = 75 + int(round(cap_span * bias)) + self._get_tolerance_cap_bonus(polity)
        min_cap = max(10, min(100, min_cap))
        max_cap = 100
        ratio = max(0.0, min(1.0, primary_share / max(threshold, 1e-6)))
        max_allowed = int(round(min_cap + (max_cap - min_cap) * ratio))
        floor_base = self.config.get('simulation.culture.tolerance_alignment_floor_base', 1) if self.config else 1
        floor_span = self.config.get('simulation.culture.tolerance_alignment_floor_span', 30) if self.config else 30
        positive_bias = 0.5 * (bias + 1.0)
        min_allowed = floor_base + int(round(floor_span * positive_bias))
        min_allowed = max(1, min(99, min_allowed))
        if min_allowed > max_allowed:
            min_allowed = max_allowed
        new_level = tile.control_level
        if tile.control_level > max_allowed:
            new_level = max_allowed
        elif tile.control_level < min_allowed:
            new_level = min_allowed
        if new_level != tile.control_level:
            before_cap = tile.control_level
            tile.control_level = new_level
            details = (
                f"primary {primary_share:.0%} vs threshold {threshold:.0%} "
                f"(range {min_allowed}-{max_allowed}%)"
            )
            self._record_control_change(
                tile,
                before_cap,
                tile.control_level,
                label="Alignment Cap",
                details=details,
            )
            self._log_control_cap(tile, "Alignment Cap", tile.control_level, details)

    def _maybe_assign_capital_primary_culture(self, tile_idx: int) -> None:
        """Assign a polity's primary culture if its capital gains a majority culture."""
        if tile_idx < 0 or tile_idx >= len(self.tiles):
            return
        tile = self.tiles[tile_idx]
        if tile.polity_id == -1:
            return
        polity = self._get_polity(tile.polity_id)
        if not polity or polity.primary_culture:
            return
        if getattr(polity, 'capital_tile_index', -1) != tile_idx:
            return
        culture_name = self._get_tile_majority_culture(tile)
        if culture_name:
            previous_name = polity.name
            self._log_event(
                "culture_debug",
                f"[culture] Polity '{previous_name}' adopted capital culture '{culture_name}'"
            )
            self._apply_polity_primary_culture(polity, culture_name)

    def _is_valid_polity_capital(self, polity: Polity, capital_idx: int) -> bool:
        if capital_idx is None or capital_idx < 0 or capital_idx >= len(self.tiles):
            return False
        tile = self.tiles[capital_idx]
        return (not tile.is_water) and tile.polity_id == polity.id

    def _select_new_capital_tile(self, polity: Polity) -> Optional[int]:
        candidates = [idx for idx in getattr(polity, 'tile_indices', [])
                      if 0 <= idx < len(self.tiles) and not self.tiles[idx].is_water and self.tiles[idx].polity_id == polity.id]
        if not candidates:
            return None
        return max(candidates, key=lambda idx: (self.tiles[idx].population, self.tiles[idx].development, -idx))

    def _ensure_polity_capital(self, polity: Polity) -> Optional[int]:
        if polity is None or not getattr(polity, 'is_active', True):
            return None
        capital_idx = getattr(polity, 'capital_tile_index', -1)
        if self._is_valid_polity_capital(polity, capital_idx):
            return capital_idx
        new_capital = self._select_new_capital_tile(polity)
        if new_capital is not None:
            polity.capital_tile_index = new_capital
            self._log_event(
                "culture_debug",
                f"[capital] Reassigned capital of '{polity.name}' to tile {new_capital}"
            )
            return new_capital
        return None

    def _find_tile_index_by_identity(self, tile: Tile) -> Optional[int]:
        try:
            return self.tiles.index(tile)
        except ValueError:
            return None

    def _apply_polity_primary_culture(
        self,
        polity: Optional[Polity],
        culture_name: Optional[str],
    ) -> None:
        """Assign a polity's primary culture and refresh its language-derived name."""
        if polity is None or not culture_name:
            return
        previous_culture = getattr(polity, 'primary_culture', None)
        changed = previous_culture != culture_name
        if changed and getattr(polity, 'dynasty_title_token', None):
            self._release_dynasty_title(polity)
        polity.primary_culture = culture_name
        if polity.leader:
            polity.leader.culture = culture_name
            if not previous_culture:
                generation_index = max(1, getattr(polity, 'leader_generation', 1))
                leader_name = self._generate_leader_name(polity, culture_name, generation_index)
                if isinstance(leader_name, tuple):
                    display_name, dynasty_name = leader_name
                else:
                    display_name, dynasty_name = leader_name, None
                polity.leader.name = display_name
                polity.leader.dynasty = dynasty_name
                self._log_event(
                    'polity',
                    f"{polity.name} renamed leader to {display_name} after adopting primary culture {culture_name}"
                )
                self._refresh_polity_naming(polity, reason="primary_culture_assignment")
        # Do not rename polities when culture shifts; keep existing naming stable

    def _assign_polity_culture_if_needed(self, tile: Tile, culture_name: str) -> None:
        """Let a cultureless polity adopt the first culture appearing in its capital."""
        if not culture_name:
            return
        polity_id = getattr(tile, 'polity_id', -1)
        if polity_id == -1:
            return
        polity = self._get_polity(polity_id)
        if not polity or polity.primary_culture:
            return
        capital_idx = polity.capital_tile_index
        if not (0 <= capital_idx < len(self.tiles)):
            return
        if self.tiles[capital_idx] is tile:
            self._apply_polity_primary_culture(polity, culture_name)

    def _set_tile_culture_full(self, tile: Tile, culture_name: str) -> None:
        """Convert a tile fully to a single culture and handle polity adoption."""
        if tile.cultural_makeup is None:
            tile.cultural_makeup = {}
        if tile.population == 0:
            tile.cultural_makeup = {}
            return
        if not culture_name:
            tile.cultural_makeup = {}
            return
        was_new_culture = culture_name not in tile.cultural_makeup
        had_prior_culture = bool(tile.cultural_makeup)
        tile.cultural_makeup = {culture_name: 1.0}
        # Reset cooldown if this is a new culture introduction to a tile that already had culture, and not syncretic
        if was_new_culture and had_prior_culture and not self._get_syncretic_parents_for(culture_name):
            tile.last_culture_spawn_year = self.current_year
        self._assign_polity_culture_if_needed(tile, culture_name)

    def _seed_first_culture_if_needed(self, tile: Tile, culture_name: Optional[str]) -> bool:
        """Assign the first culture a tile ever receives, returning True if applied."""
        if tile.cultural_makeup:
            return False
        if tile.population == 0:
            tile.cultural_makeup = {}
            return False
        if culture_name:
            self._set_tile_culture_full(tile, culture_name)
            return True
        return False

    def _seed_first_culture_from_dict(self, tile: Tile, culture_map: Dict[str, float]) -> bool:
        """Seed a tile with the dominant culture from a provided share map."""
        dominant = self._get_dominant_culture_from_dict(culture_map)
        if dominant:
            return self._seed_first_culture_if_needed(tile, dominant)
        return False

    def _get_dominant_culture_from_dict(self, culture_map: Dict[str, float]) -> Optional[str]:
        """Return the highest-share culture name from an arbitrary mapping."""
        if not culture_map:
            return None
        return max(culture_map.items(), key=lambda item: item[1])[0]

    def _map_capital_connected_tiles(self, owned_capitals: Dict[int, "Polity"]) -> Dict[int, Set[int]]:
        """Return land tiles connected to each polity's capital via same-polity territory."""
        connections: Dict[int, Set[int]] = {}
        for capital_idx, polity in owned_capitals.items():
            if polity is None:
                continue
            polity_id = getattr(polity, 'id', -1)
            if polity_id < 0:
                continue
            if capital_idx < 0 or capital_idx >= len(self.tiles):
                continue
            capital_tile = self.tiles[capital_idx]
            if capital_tile.is_water:
                continue
            visited: Set[int] = {capital_idx}
            queue: deque[int] = deque([capital_idx])
            while queue:
                current_idx = queue.popleft()
                current_tile = self.tiles[current_idx]
                for neighbor_idx in current_tile.neighbors:
                    if neighbor_idx < 0 or neighbor_idx >= len(self.tiles):
                        continue
                    if neighbor_idx in visited:
                        continue
                    neighbor_tile = self.tiles[neighbor_idx]
                    if neighbor_tile.is_water or neighbor_tile.polity_id != polity_id:
                        continue
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)
            connections[polity_id] = visited
        return connections
    
    def _process_control(self) -> None:
        """Process control level changes for all tiles."""
        # Get config values
        decay_rate = self.config.get('simulation.control.overdevelopment_decay_rate', 0.02) if self.config else 0.02
        threshold_ratio = self.config.get('simulation.control.overdevelopment_threshold_ratio', 1.0) if self.config else 1.0
        border_decay_rate = self.config.get('simulation.control.border_decay_rate', 1) if self.config else 1
        border_decay_threshold = self.config.get('simulation.control.border_decay_threshold', 30) if self.config else 30
        non_center_bleed_chance = self.config.get('simulation.control.non_center_control_bleed_chance', 0.02) if self.config else 0.02
        exclave_penalty_factor = self.config.get('simulation.control.exclave_penalty_factor', 0.8) if self.config else 0.8
        try:
            exclave_penalty_factor = float(exclave_penalty_factor)
        except (TypeError, ValueError):
            exclave_penalty_factor = 0.8
        exclave_penalty_factor = max(0.0, min(1.0, exclave_penalty_factor))
        population_center_indices = {
            center.tile_index for center in self.population_centers
            if 0 <= center.tile_index < len(self.tiles)
        }
        owned_capitals = self._get_owned_capital_tiles()
        capital_connections = self._map_capital_connected_tiles(owned_capitals) if exclave_penalty_factor < 1.0 else {}
        
        for idx, tile in enumerate(self.tiles):
            # Skip water tiles and uncontrolled tiles
            if tile.is_water or tile.polity_id == -1:
                continue

            if idx in owned_capitals:
                before_cap = tile.control_level
                tile.control_level = 100
                self._record_control_change(
                    tile,
                    before_cap,
                    tile.control_level,
                    label="Capital Control Lock",
                )
                self._log_control_cap(tile, "Capital Control Lock", 100, "Capital tile")
                self._apply_control_alignment_cap(idx)
                continue
            
            # Check for border with other polities and apply decay
            if tile.control_level > border_decay_threshold:
                # Check if this tile borders a different polity
                borders_other_polity = False
                for neighbor_idx in tile.neighbors:
                    if neighbor_idx < len(self.tiles):
                        neighbor = self.tiles[neighbor_idx]
                        # Border exists if neighbor belongs to different polity (including uncontrolled)
                        if (not neighbor.is_water and 
                            neighbor.polity_id != tile.polity_id):
                            borders_other_polity = True
                            break
                
                if borders_other_polity:
                    # Apply border control decay
                    before_border = tile.control_level
                    new_level = max(border_decay_threshold, tile.control_level - border_decay_rate)
                    if new_level != tile.control_level:
                        tile.control_level = new_level
                        self._record_control_change(
                            tile,
                            before_border,
                            tile.control_level,
                            label="Border Pressure",
                            details=f"toward {border_decay_threshold}%",
                        )
            
            # Apply control decay in overdeveloped areas
            if tile.population > 0 and tile.development > tile.population * threshold_ratio:
                # Calculate overdevelopment ratio
                overdevelopment_ratio = tile.development / (tile.population * threshold_ratio)
                
                # Apply exponential decay based on how overdeveloped the area is
                decay_amount = decay_rate * (overdevelopment_ratio - 1.0)
                decay_points = int(tile.control_level * decay_amount)
                if decay_points > 0:
                    before_over = tile.control_level
                    tile.control_level = max(1, tile.control_level - decay_points)
                    self._record_control_change(
                        tile,
                        before_over,
                        tile.control_level,
                        label="Overdevelopment Stress",
                        details=f"ratio {overdevelopment_ratio:.2f}",
                    )

            if (idx not in population_center_indices and
                tile.control_level > 1 and
                non_center_bleed_chance > 0 and
                random.random() < non_center_bleed_chance):
                before_bleed = tile.control_level
                tile.control_level -= 1
                self._record_control_change(
                    tile,
                    before_bleed,
                    tile.control_level,
                    label="Peripheral Bleed",
                )
            polity_connected = capital_connections.get(tile.polity_id)
            if (polity_connected is not None and
                idx not in polity_connected and
                exclave_penalty_factor < 1.0 and
                tile.control_level > 1):
                before_exclave = tile.control_level
                penalized_level = max(1, int(tile.control_level * exclave_penalty_factor))
                if penalized_level < tile.control_level:
                    tile.control_level = penalized_level
                    penalty_percent = int(round((1.0 - exclave_penalty_factor) * 100))
                    self._record_control_change(
                        tile,
                        before_exclave,
                        tile.control_level,
                        label="Exclave Isolation",
                        details=f"-{penalty_percent}%",
                    )
            self._apply_control_alignment_cap(idx)
            # Apply temporary control bonus from war victories
            if tile.temporary_control_bonus > 0:
                before_bonus = tile.control_level
                tile.control_level = min(100, tile.control_level + tile.temporary_control_bonus)
                if tile.control_level != before_bonus:
                    self._record_control_change(
                        tile,
                        before_bonus,
                        tile.control_level,
                        label="War Victory Boost",
                        details=f"+{tile.temporary_control_bonus:.1f}%",
                    )

            polity_ref = self._get_polity(tile.polity_id)
            if polity_ref and self._polity_has_idea_label(polity_ref, 'kinship_state', 'aristocratic'):
                majority_culture = self._get_tile_majority_culture(tile)
                if majority_culture and majority_culture == getattr(polity_ref, 'primary_culture', None):
                    before_aristocratic = tile.control_level
                    tile.control_level = min(100, tile.control_level + 5)
                    if tile.control_level != before_aristocratic:
                        self._record_control_change(
                            tile,
                            before_aristocratic,
                            tile.control_level,
                            label="Aristocratic Primacy",
                            category='flat',
                            details="+5 flat control",
                        )
        
        # Decay temporary control bonuses
        for tile in self.tiles:
            if tile.temporary_control_bonus > 0:
                tile.temporary_control_bonus *= 0.95  # Decay by 5% per tick
        
        # Process low-control tile defection
        self._process_tile_defection()
        self._process_population_center_control_projection()
        
        # Process culture spawning and cultural effects
        self._process_culture_spawning()
        self._process_cultural_assimilation()
        self._apply_leader_tile_bonuses()
    
    def _process_population_center_control_projection(self) -> None:
        """Process control projection from population centers to their neighbors."""
        # Get config values
        projection_rate = self.config.get('simulation.control.control_projection_rate', 2) if self.config else 2
        
        # Only process tiles that are population centers with assigned control
        for center in self.population_centers:
            tile_idx = center.tile_index
            if tile_idx >= len(self.tiles):
                continue
                
            center_tile = self.tiles[tile_idx]
            center_polity = self._get_polity(center_tile.polity_id) if center_tile.polity_id != -1 else None
            projection_multiplier = 1.0
            if self._polity_has_idea_label(center_polity, 'kinship_state', 'oligarchic'):
                projection_multiplier = 1.75
            effective_projection_rate = max(0.0, projection_rate * projection_multiplier)
            
            # Project control to all neighbors
            for neighbor_idx in center_tile.neighbors:
                if neighbor_idx >= len(self.tiles):
                    continue
                    
                neighbor_tile = self.tiles[neighbor_idx]
                
                # Skip water tiles and tiles from different polities
                if (neighbor_tile.is_water or 
                    neighbor_tile.polity_id != center_tile.polity_id or
                    neighbor_tile.polity_id == -1):
                    continue
                
                control_difference = center_tile.control_level - neighbor_tile.control_level
                if control_difference == 0:
                    continue

                # Move neighbor control toward the center in either direction
                adjustment = min(effective_projection_rate, abs(control_difference))
                if control_difference > 0:
                    before_projection = neighbor_tile.control_level
                    neighbor_tile.control_level = min(100, neighbor_tile.control_level + adjustment)
                else:
                    before_projection = neighbor_tile.control_level
                    neighbor_tile.control_level = max(1, neighbor_tile.control_level - adjustment)
                self._record_control_change(
                    neighbor_tile,
                    before_projection,
                    neighbor_tile.control_level,
                    label="Population Center Projection",
                    details=f"{center.name or 'Center'}->{neighbor_idx} (x{projection_multiplier:.2f})",
                )
                self._apply_control_alignment_cap(neighbor_idx)

    def _process_tile_defection(self) -> None:
        """Process defection of low-control non-pop-center tiles to neighboring high-control polities."""
        population_center_indices = {
            center.tile_index for center in self.population_centers
            if 0 <= center.tile_index < len(self.tiles)
        }
        
        for idx, tile in enumerate(self.tiles):
            if (tile.is_water or tile.polity_id == -1 or 
                tile.control_level > 1 or idx in population_center_indices):
                continue
            
            # Find eligible neighboring polities with 100% control
            eligible_neighbors = []
            for neighbor_idx in tile.neighbors:
                if neighbor_idx < len(self.tiles):
                    neighbor = self.tiles[neighbor_idx]
                    if (not neighbor.is_water and neighbor.polity_id != -1 and 
                        neighbor.polity_id != tile.polity_id and neighbor.control_level >= 100):
                        # Check if not at war
                        relationship = self._get_relationship(tile.polity_id, neighbor.polity_id)
                        if relationship and relationship.status != "war":
                            eligible_neighbors.append((neighbor.polity_id, neighbor.development, neighbor_idx))
            
            if not eligible_neighbors:
                continue
            
            # Sort by development descending, then by tile index for tie-breaking
            eligible_neighbors.sort(key=lambda x: (-x[1], x[2]))
            best_polity, best_dev, best_neighbor_idx = eligible_neighbors[0]
            
            # Defect to the best polity
            old_polity = tile.polity_id
            tile.polity_id = best_polity
            tile.control_level = 100
            tile.temporary_control_bonus = 0  # Reset bonus on defection
            
            # Update polity tile indices
            if old_polity != -1 and old_polity < len(self.polities):
                self.polities[old_polity].tile_indices.remove(idx)
            if best_polity < len(self.polities):
                self.polities[best_polity].tile_indices.append(idx)
            
            self._record_control_change(
                tile,
                1,
                100,
                label="Tile Defection",
                details=f"from polity {old_polity} to {best_polity} (neighbor {best_neighbor_idx} dev {best_dev:.1f})",
            )
            self._log_event(
                "control",
                f"Tile {idx} defected from polity {old_polity} to {best_polity} due to low control and high-control neighbor"
            )

    def _culture_matches_region(self, culture: Optional[Culture], region_id: int) -> bool:
        """Return True when the provided culture is native to the supplied region."""
        if region_id < 0:
            return True
        if culture is None:
            return False
        if culture.home_region_id < 0:
            return True
        return culture.home_region_id == region_id

    def _are_direct_parent_cultures(self, culture_a: str, culture_b: str) -> bool:
        """Return True if one culture lists the other as an immediate parent."""
        if not culture_a or not culture_b or culture_a == culture_b:
            return False
        culture_obj_a = self._find_culture_by_name(culture_a)
        culture_obj_b = self._find_culture_by_name(culture_b)
        if not culture_obj_a or not culture_obj_b:
            return False
        return (
            culture_b in culture_obj_a.heritage or
            culture_a in culture_obj_b.heritage
        )

    def _spawn_new_culture_at_center(self, center: PopulationCenter, immunity_years: int, reason: str) -> bool:
        """Create a new culture at the provided population center and convert the tile."""
        self._log_event(
            "culture_debug",
            f"[culture] New culture spawn attempt at tile {center.tile_index} (center {center.name}) in Year {self.current_year} for reason: {reason}"
        )
        tile_index = center.tile_index
        if tile_index < 0 or tile_index >= len(self.tiles):
            return False
        culture_name: Optional[str] = None
        reserved_token: Optional[str] = None
        try:
            tile = self.tiles[tile_index]
            if not self._tile_supports_culture_tracking(tile):
                self._log_event(
                    "culture_debug",
                    f"[culture] Skipped new culture spawn at tile {tile_index} due to low population"
                )
                return False
            descriptor = 'R' if (reason or '').lower().startswith('regional') else 'D'

            heritage: Dict[str, float] = {}
            parent_color: Optional[Tuple[int, int, int]] = None
            parent_culture_name: Optional[str] = None
            if tile.cultural_makeup:
                total_influence = sum(tile.cultural_makeup.values())
                if total_influence > 0:
                    for culture, percentage in tile.cultural_makeup.items():
                        heritage[culture] = percentage / total_influence
                    parent_culture_name = max(heritage.items(), key=lambda x: x[1])[0]
                    parent_culture = self._find_culture_by_name(parent_culture_name)
                    if parent_culture:
                        parent_color = parent_culture.color

            home_region_id = self._determine_home_region_id(tile_index)
            # Create culture with temporary name for dynamic naming
            temp_name = self._next_culture_name(descriptor)
            culture_name, reserved_token = self._reserve_unique_culture_name(temp_name, tile_index)

            culture_color = self._generate_child_culture_color(parent_color)
            new_culture = Culture(
                name=culture_name,
                color=culture_color,
                heritage=heritage,
                origin_tile_index=tile_index,
                birth_year=self.current_year,
                home_region_id=home_region_id,
                immunity_end_year=self.current_year + immunity_years,
                is_initial=False
            )

            jitter = 0.08 if (reason or '').lower().startswith('regional') else 0.12
            self._ensure_culture_ideas(new_culture, heritage_override=heritage, jitter=jitter)

            self.cultures.append(new_culture)
            if reserved_token:
                self._commit_reserved_culture_name(culture_name)
                reserved_token = None
            else:
                self._register_culture_name(culture_name)
            
            # Assign language before naming region
            if parent_culture_name:
                time_depth = 0 if (reason or '').lower().startswith('regional') else 1
                self.assign_derivative_language(new_culture, parent_culture_name, time_depth=time_depth)
            else:
                self.assign_base_language(new_culture)
            
            # Generate a culture name that anchors to the home region before falling back to settlements
            best_name: Optional[str] = None
            region_name: Optional[str] = None
            if home_region_id is not None and home_region_id >= 0:
                region_name = self.ensure_region_language_name(home_region_id, new_culture.name)
                if region_name:
                    best_name = self._generate_culture_demonym_from_region(region_name, tile_index)
                if not best_name:
                    best_name = self._generate_regional_culture_name(
                        home_region_id,
                        parent_culture_name,
                        tile_index
                    )

            if not best_name:
                settlement_label = self._derive_city_label(tile_index)
                if settlement_label:
                    best_name = self._make_unique_culture_name(settlement_label, tile_index)

            final_name = best_name or culture_name
            if final_name != new_culture.name:
                old_name = new_culture.name
                new_culture.name = final_name
                culture_name = final_name
                # Update name registry
                self._culture_name_registry.discard(self._normalize_culture_name_token(old_name))
                self._register_culture_name(culture_name)
                # Update culture languages dict
                if old_name in self.culture_languages:
                    self.culture_languages[culture_name] = self.culture_languages.pop(old_name)

            self.ensure_region_name_for_culture(new_culture)
            self._set_tile_culture_full(tile, culture_name)
            self._rename_population_center_for_tile(tile_index)

            chain_conversions = 0
            if (reason or '').lower().startswith('regional'):
                chain_conversions = self._propagate_regional_culture_chain(tile_index, culture_name)

            center_display_name = center.name
            if not center_display_name:
                center_display_name, _ = self._generate_settlement_name(tile_index)
            parent_label = parent_culture_name or "none"
            chain_suffix = f", chain_tiles={chain_conversions}" if chain_conversions else ""
            self._log_event(
                "culture",
                f"New culture '{culture_name}' spawned at {center_display_name} in Year {self.current_year} "
                f"(reason: {reason}, parent: {parent_label}, color: {culture_color}{chain_suffix})"
            )
            return True
        except Exception as exc:
            center_name = getattr(center, 'name', f"Tile_{tile_index}")
            if reserved_token:
                self._release_reserved_culture_name(culture_name)
            self._log_event(
                "culture_debug",
                f"[culture] Failed to spawn new culture at {center_name} (tile {tile_index}) in Year {self.current_year}: {exc}"
            )
            traceback.print_exc()
            return False

    def _adopt_neighboring_culture_for_initial_center(self, center: PopulationCenter) -> bool:
        """Let an unseeded center adopt any adjacent culture before creating a brand new one."""
        tile_index = center.tile_index
        if tile_index < 0 or tile_index >= len(self.tiles):
            return False
        tile = self.tiles[tile_index]
        if tile.is_water or tile.cultural_makeup:
            return False
        for neighbor_idx in getattr(tile, 'neighbors', []) or []:
            if neighbor_idx < 0 or neighbor_idx >= len(self.tiles):
                continue
            neighbor = self.tiles[neighbor_idx]
            if neighbor.is_water or not neighbor.cultural_makeup:
                continue
            adopted_culture = self._get_tile_majority_culture(neighbor)
            if not adopted_culture:
                continue
            self._set_tile_culture_full(tile, adopted_culture)
            tile.last_culture_spawn_year = self.current_year
            self._rename_population_center_for_tile(tile_index)
            center_label = center.name or f"Center_{tile_index}"
            self._log_event(
                "culture",
                f"Population center '{center_label}' adopted neighboring culture {adopted_culture} instead of spawning a new root culture",
            )
            return True
        return False

    def _process_culture_spawning(self) -> None:
        """Process spawning of new cultures at population centers."""
        new_culture_chance = self.config.get('simulation.culture.new_culture_chance', 0.02) if self.config else 0.02
        min_center_ticks = self.config.get('simulation.culture.new_culture_min_center_ticks', 40) if self.config else 40
        min_center_years_config = self.config.get('simulation.culture.new_culture_min_center_years') if self.config else None
        immunity_years = self.config.get('simulation.culture.new_culture_immunity_years', 10) if self.config else 10
        regional_home_threshold = self.config.get('simulation.culture.region_home_mismatch_years', 100) if self.config else 100
        tile_spawn_cooldown_years = self.config.get('simulation.culture.tile_spawn_cooldown_years', 20) if self.config else 20
        ticks_per_year = max(1, self.ticks_per_year)
        if min_center_years_config is None:
            if min_center_ticks <= 0:
                min_center_years = 1
            else:
                min_center_years = max(1, (min_center_ticks + ticks_per_year - 1) // ticks_per_year)
        else:
            try:
                min_center_years = max(1, int(min_center_years_config))
            except (TypeError, ValueError):
                min_center_years = 1
        summary = {
            'centers_checked': 0,
            'invalid_tile': 0,
            'water_or_empty': 0,
            'cooldown_active': 0,
            'regional_spawns': 0,
            'regional_failures': 0,
            'already_cultured': 0,
            'unowned_tile': 0,
            'too_young': 0,
            'spawn_attempts': 0,
            'spawn_success': 0,
            'spawn_failures': 0,
            'chance_skips': 0
        }
        
        for center in self.population_centers:
            summary['centers_checked'] += 1
            if center.tile_index >= len(self.tiles):
                summary['invalid_tile'] += 1
                continue
            
            tile = self.tiles[center.tile_index]
            established_year = center.established_year if center.established_year is not None else self.current_year
            center_age_years = max(0, self.current_year - established_year)
            polity = self._get_polity(tile.polity_id) if tile.polity_id != -1 else None
            effective_regional_threshold = regional_home_threshold
            if polity and regional_home_threshold > 0:
                effective_regional_threshold = self._get_regional_culture_spawn_threshold(
                    regional_home_threshold,
                    polity
                )

            if tile.is_water or tile.population == 0:
                summary['water_or_empty'] += 1
                continue

            if (tile_spawn_cooldown_years > 0 and tile.last_culture_spawn_year is not None and
                    self.current_year - tile.last_culture_spawn_year < tile_spawn_cooldown_years):
                summary['cooldown_active'] += 1
                continue

            if (effective_regional_threshold > 0 and center_age_years >= effective_regional_threshold and
                    tile.region_id >= 0):
                primary_culture_name = self._get_tile_majority_culture(tile)
                if primary_culture_name:
                    primary_culture = self._find_culture_by_name(primary_culture_name)
                    if not self._culture_matches_region(primary_culture, tile.region_id):
                        summary['spawn_attempts'] += 1
                        if self._spawn_new_culture_at_center(center, immunity_years, reason="regional divergence"):
                            summary['regional_spawns'] += 1
                            summary['spawn_success'] += 1
                            continue
                        summary['regional_failures'] += 1
                        self._log_event(
                            "culture_debug",
                            f"[culture] Regional divergence spawn failed at tile {center.tile_index} (center {center.name})"
                        )

            if tile.cultural_makeup:
                summary['already_cultured'] += 1
                continue
            if tile.polity_id == -1:
                summary['unowned_tile'] += 1
                continue
            if center_age_years < min_center_years:
                summary['too_young'] += 1
                continue

            roll = random.random()
            if roll < new_culture_chance:
                summary['spawn_attempts'] += 1
                if self._adopt_neighboring_culture_for_initial_center(center):
                    summary['spawn_success'] += 1
                    continue
                if not self._spawn_new_culture_at_center(center, immunity_years, reason="first culture"):
                    summary['spawn_failures'] += 1
                    self._log_event(
                        "culture_debug",
                        f"[culture] Spawn attempt failed at tile {center.tile_index} (center {center.name})"
                    )
                else:
                    summary['spawn_success'] += 1
            else:
                summary['chance_skips'] += 1
                self._log_event(
                    "culture_debug",
                    f"[culture] First-culture skipped at tile {center.tile_index} (center {center.name}); "
                    f"roll={roll:.3f}, threshold={new_culture_chance:.3f}"
                )
        should_log_summary = summary['centers_checked'] == 0 or summary['spawn_success'] == 0
        if should_log_summary:
            if summary['centers_checked'] == 0:
                self._log_event(
                    "culture_debug",
                    f"[culture] Spawn summary Year {self.current_year}: no population centers to evaluate"
                )
            else:
                reason_labels = [
                    ('water_or_empty', 'water/empty'),
                    ('cooldown_active', 'cooldown'),
                    ('already_cultured', 'already-had-culture'),
                    ('unowned_tile', 'unowned'),
                    ('too_young', 'too-young'),
                    ('chance_skips', 'roll-failed'),
                    ('spawn_failures', 'spawn-failed'),
                    ('regional_failures', 'regional-failed')
                ]
                reason_bits = []
                for key, label in reason_labels:
                    count = summary.get(key, 0)
                    if count:
                        reason_bits.append(f"{label}={count}")
                if not reason_bits:
                    reason_bits.append("no-eligible-centers")
                self._log_event(
                    "culture_debug",
                    f"[culture] Spawn summary Year {self.current_year}: centers={summary['centers_checked']}, "
                    f"success={summary['spawn_success']}, attempts={summary['spawn_attempts']} | "
                    + ", ".join(reason_bits)
                )

    def _process_cultural_assimilation(self) -> None:
        """Process cultural assimilation, immunity effects, and syncretism."""
        assimilation_rate = self.config.get('simulation.culture.cultural_assimilation_rate', 0.005) if self.config else 0.005
        control_decay_rate = self.config.get('simulation.culture.cultural_control_decay_rate', 1) if self.config else 1
        misalignment_multiplier = self.config.get('simulation.culture.misalignment_control_multiplier', 4.0) if self.config else 4.0
        soft_threshold_base = self.config.get('simulation.culture.assimilation_soft_threshold_base', 0.2) if self.config else 0.2
        soft_threshold_control_factor = self.config.get('simulation.culture.assimilation_soft_threshold_control_factor', 0.3) if self.config else 0.3
        assimilation_min_factor = self.config.get('simulation.culture.assimilation_min_factor', 0.1) if self.config else 0.1
        low_alignment_threshold = self.config.get('simulation.culture.low_alignment_threshold', 0.5) if self.config else 0.5
        low_alignment_penalty_multiplier = (
            self.config.get(
                'simulation.culture.low_alignment_penalty_multiplier',
                self.config.get('simulation.culture.low_alignment_penalty_modifier', 3.5)
            ) if self.config else 3.5
        )
        owned_capitals = self._get_owned_capital_tiles()
        
        for tile_idx, tile in enumerate(self.tiles):
            if tile.is_water or not tile.cultural_makeup:
                continue
            if tile.population == 0:
                tile.cultural_makeup = {}
                continue

            if tile.polity_id == -1:
                immune_cultures = self._get_immune_cultures(tile)
                self._apply_new_culture_growth(tile, immune_cultures)
                self._normalize_tile_culture(tile)
                self._update_syncretism_tracker(tile_idx, tile)
                continue
            
            polity = self._get_polity(tile.polity_id)
            if not polity:
                self._normalize_tile_culture(tile)
                continue
            is_capital = tile_idx in owned_capitals
            
            immune_cultures = self._get_immune_cultures(tile)

            if not polity.primary_culture:
                self._apply_new_culture_growth(tile, immune_cultures)
                self._normalize_tile_culture(tile)
                self._update_syncretism_tracker(tile_idx, tile)
                continue

            primary_culture_percentage = tile.cultural_makeup.get(polity.primary_culture, 0.0)
            alignment_share = self._compute_primary_alignment_share(tile, polity)
            alignment_threshold = self._get_alignment_threshold(polity, low_alignment_threshold)
            penalty_multiplier = self._get_alignment_penalty_multiplier(polity, low_alignment_penalty_multiplier)
            tolerance_factor = self._get_misalignment_tolerance_factor(polity)
            if alignment_share < 0.5 and not is_capital:
                misalignment_factor = 1.0 - alignment_share
                severity_scale = 1.0
                if alignment_share < alignment_threshold:
                    threshold_gap = alignment_threshold - alignment_share
                    severity_scale += (threshold_gap / max(0.01, alignment_threshold)) * penalty_multiplier
                decay_amount = (
                    control_decay_rate *
                    misalignment_factor *
                    misalignment_multiplier *
                    severity_scale *
                    tolerance_factor
                )
                decay_points = max(1, int(decay_amount))
                before_culture_decay = tile.control_level
                tile.control_level = max(1, tile.control_level - decay_points)
                self._record_control_change(
                    tile,
                    before_culture_decay,
                    tile.control_level,
                    label="Cultural Misalignment",
                    details=f"aligned {alignment_share:.0%}",
                )
            
            if polity.primary_culture not in tile.cultural_makeup:
                tile.cultural_makeup[polity.primary_culture] = 0.0
            
            old_percentage = tile.cultural_makeup[polity.primary_culture]
            minority_share = max(0.0, 1.0 - old_percentage)
            raw_control_ratio = 1.0 if is_capital else tile.control_level / 100.0
            raw_control_ratio = max(0.0, min(1.0, raw_control_ratio))
            effective_control_ratio = self._apply_tolerance_control_bias(
                polity,
                raw_control_ratio,
                is_capital=is_capital
            )
            soft_threshold = max(0.01, soft_threshold_base + soft_threshold_control_factor * raw_control_ratio)
            if minority_share < soft_threshold:
                threshold_ratio = minority_share / soft_threshold if soft_threshold > 0 else 0.0
                assimilation_multiplier = max(assimilation_min_factor, threshold_ratio)
            else:
                assimilation_multiplier = 1.0
            trait_rate = self._get_trait_value(polity, 'TRADITIONAL', 'assimilation_rate_multiplier', 1.0)
            tolerance_rate = self._get_assimilation_rate_multiplier(polity)
            desired_increase = (
                assimilation_rate *
                trait_rate *
                effective_control_ratio *
                assimilation_multiplier *
                tolerance_rate
            )
            donor_exclusions = set(immune_cultures)
            donor_exclusions.add(polity.primary_culture)
            actual_increase = self._redistribute_culture_share(tile, donor_exclusions, desired_increase)
            if actual_increase > 0:
                tile.cultural_makeup[polity.primary_culture] += actual_increase
            
            self._apply_new_culture_growth(tile, immune_cultures)
            self._normalize_tile_culture(tile)
            self._update_syncretism_tracker(tile_idx, tile)
            if is_capital and tile.control_level != 100:
                before_capital = tile.control_level
                tile.control_level = 100
                self._record_control_change(
                    tile,
                    before_capital,
                    tile.control_level,
                    label="Capital Control Lock",
                )
                self._log_control_cap(tile, "Capital Control Lock", 100, "Capital tile")
            self._apply_control_alignment_cap(tile_idx)

    def _ensure_population_center_entry(self, tile_index: int, name: Optional[str] = None, threshold: Optional[int] = None) -> None:
        """Force a population center entry for a tile if one does not exist."""
        if tile_index < 0 or tile_index >= len(self.tiles):
            return
        tile = self.tiles[tile_index]
        if tile.is_water:
            return
        if any(center.tile_index == tile_index for center in self.population_centers):
            return
        center_name, center_phonetic = name or self._generate_settlement_name(tile_index)
        original_threshold = threshold if threshold is not None else max(1, int(tile.population) or 1)
        new_center = PopulationCenter(
            tile_index=tile_index,
            name=center_name,
            phonetic_name=center_phonetic,
            original_threshold=original_threshold,
            established_year=self.current_year,
            established_tick=self.total_ticks
        )
        self.population_centers.append(new_center)
        culture = self._get_tile_majority_culture(tile)
        self._initialize_population_center_history(
            new_center,
            culture=culture,
            reason="ensured_center",
            note="forced",
        )

    def _update_population_centers(self) -> None:
        """Update population centers based on dynamic thresholds."""
        # Calculate total land development (used for scaling thresholds)
        total_development = sum(tile.development for tile in self.tiles if not tile.is_water)
        development_reference = max(1.0, float(total_development))
        
        # Get config values
        base_threshold = self.config.get('simulation.development.population_center_threshold', 50) if self.config else 50
        threshold_scaling = self.config.get('simulation.development.population_center_threshold_scaling', 0.002) if self.config else 0.002
        percentage = self.config.get('simulation.development.population_center_percentage', 0.05) if self.config else 0.05
        absolute_population_threshold = self.config.get('simulation.development.population_center_absolute_population', 10000) if self.config else 10000
        absolute_development_threshold = self.config.get('simulation.development.population_center_absolute_development', 10000) if self.config else 10000
        creation_hysteresis_ratio = self.config.get('simulation.development.population_center_creation_hysteresis_ratio', 0.15) if self.config else 0.15
        demotion_grace_years = self.config.get('simulation.development.population_center_demotion_grace_years', 3) if self.config else 3
        demotion_population_ratio = self.config.get('simulation.development.population_center_demotion_population_ratio', 0.9) if self.config else 0.9
        demotion_development_ratio = self.config.get('simulation.development.population_center_demotion_development_ratio', 0.8) if self.config else 0.8
        demotion_buffer_years = self.config.get('simulation.development.population_center_demotion_buffer_years', 2) if self.config else 2
        try:
            creation_hysteresis_ratio = max(0.0, float(creation_hysteresis_ratio))
        except (TypeError, ValueError):
            creation_hysteresis_ratio = 0.0
        try:
            demotion_grace_years = max(0.0, float(demotion_grace_years))
        except (TypeError, ValueError):
            demotion_grace_years = 0.0
        try:
            demotion_population_ratio = max(0.0, min(1.0, float(demotion_population_ratio)))
        except (TypeError, ValueError):
            demotion_population_ratio = 0.9
        try:
            demotion_development_ratio = max(0.0, min(1.0, float(demotion_development_ratio)))
        except (TypeError, ValueError):
            demotion_development_ratio = 0.8
        try:
            demotion_buffer_years = max(0.0, float(demotion_buffer_years))
        except (TypeError, ValueError):
            demotion_buffer_years = 0.0
        demotion_grace_ticks = int(round(self.ticks_per_year * demotion_grace_years)) if demotion_grace_years > 0 else 0
        if demotion_grace_ticks <= 0 and demotion_grace_years > 0:
            demotion_grace_ticks = 1
        demotion_buffer_ticks = int(round(self.ticks_per_year * demotion_buffer_years)) if demotion_buffer_years > 0 else 0
        if demotion_buffer_ticks <= 0 and demotion_buffer_years > 0:
            demotion_buffer_ticks = 1
        
        # Calculate scaled base threshold that grows with total development
        scaled_base_threshold = max(base_threshold, int(round(base_threshold + (development_reference * threshold_scaling))))
        
        # Calculate current dynamic threshold (percentage of world development, minimum of scaled base threshold)
        current_threshold = max(scaled_base_threshold, int(round(development_reference * percentage)))
        creation_threshold = current_threshold
        if creation_hysteresis_ratio > 0.0:
            creation_threshold = max(
                current_threshold,
                int(round(current_threshold * (1.0 + creation_hysteresis_ratio)))
            )
        capital_tiles = self._get_owned_capital_tiles()

        def meets_absolute_threshold(tile: Tile) -> bool:
            return (
                tile.population >= absolute_population_threshold and
                tile.development >= absolute_development_threshold
            )
        
        # Remove centers that no longer qualify
        qualified_centers: List[PopulationCenter] = []
        for center in self.population_centers:
            idx = center.tile_index
            name = center.name
            original_threshold = center.original_threshold
            if idx < len(self.tiles):
                tile = self.tiles[idx]

                center.demotion_buffer_ticks = getattr(center, 'demotion_buffer_ticks', 0)

                if getattr(center, 'demotion_grace_ticks', 0) > 0:
                    center.demotion_grace_ticks -= 1
                    center.demotion_buffer_ticks = 0
                    qualified_centers.append(center)
                    continue

                if idx in capital_tiles:
                    center.low_control_ticks = 0
                    center.demotion_buffer_ticks = 0
                    qualified_centers.append(center)
                    continue

                if meets_absolute_threshold(tile):
                    center.demotion_buffer_ticks = 0
                    qualified_centers.append(center)
                    continue

                demotion_triggered = False
                demotion_reason = ""

                # Automatic demotion if population falls below scaled base threshold
                if tile.population < scaled_base_threshold:
                    demotion_triggered = True
                    demotion_reason = f"population {tile.population} below scaled threshold {scaled_base_threshold}"
                else:
                    pop_ratio_against_original = tile.population / max(1.0, float(original_threshold))
                    dev_ratio_against_population = tile.development / max(1.0, tile.population)
                    if (
                        pop_ratio_against_original < demotion_population_ratio or
                        dev_ratio_against_population < demotion_development_ratio
                    ):
                        demotion_triggered = True
                        demotion_reason = (
                            f"pop ratio {pop_ratio_against_original:.2f} < {demotion_population_ratio:.2f} "
                            f"or dev ratio {dev_ratio_against_population:.2f} < {demotion_development_ratio:.2f}"
                        )

                if demotion_triggered:
                    center.demotion_buffer_ticks += 1
                    if demotion_buffer_ticks > 0 and center.demotion_buffer_ticks < demotion_buffer_ticks:
                        qualified_centers.append(center)
                        continue
                    buffer_note = ""
                    if demotion_buffer_ticks > 0:
                        buffer_note = (
                            f" after {center.demotion_buffer_ticks} low ticks (buffer {demotion_buffer_ticks})"
                        )
                    self._log_event(
                        "population_center",
                        f"Population center demoted: {name} ({demotion_reason}){buffer_note}"
                    )
                    continue

                center.demotion_buffer_ticks = 0
                qualified_centers.append(center)
            else:
                self._log_event(
                    "population_center",
                    f"Population center demoted: {name} (invalid tile index {idx})"
                )
        
        self.population_centers = qualified_centers
        
        # Add new population centers using current dynamic threshold
        existing_center_indices = {center.tile_index for center in self.population_centers}
        
        # Check for diaspora population center delay
        diaspora_delay_years = self.config.get('simulation.diaspora.population_center_delay_years', 0) if self.config else 0
        is_diaspora_delay_active = (
            diaspora_delay_years > 0 and 
            self.current_year < diaspora_delay_years
        )
        
        for i, tile in enumerate(self.tiles):
            is_capital_tile = i in capital_tiles
            if tile.is_water:
                continue
            if i in existing_center_indices:
                continue

            added = False

            if is_capital_tile:
                # Skip capital center creation during diaspora delay (except for manually ensured centers)
                if is_diaspora_delay_active:
                    continue
                polity = capital_tiles[i]
                display_name, phonetic_name = self._generate_settlement_name(i, prefer_capital=True)
                new_center = PopulationCenter(
                    tile_index=i,
                    name=display_name,
                    phonetic_name=phonetic_name,
                    original_threshold=1,
                    established_year=self.current_year,
                    established_tick=self.total_ticks,
                    demotion_grace_ticks=demotion_grace_ticks,
                )
                self.population_centers.append(new_center)
                culture = self._get_tile_majority_culture(tile)
                self._initialize_population_center_history(
                    new_center,
                    culture=culture,
                    reason="capital_foundation",
                    note="capital",
                )
                self._log_event(
                    "population_center",
                    f"New population center established: {display_name} (reason: capital)"
                )
                added = True

            elif tile.population >= creation_threshold:
                # Skip population center creation during diaspora delay
                if is_diaspora_delay_active:
                    continue
                display_name, phonetic_name = self._generate_settlement_name(i)
                new_center = PopulationCenter(
                    tile_index=i,
                    name=display_name,
                    phonetic_name=phonetic_name,
                    original_threshold=creation_threshold,
                    established_year=self.current_year,
                    established_tick=self.total_ticks,
                    demotion_grace_ticks=demotion_grace_ticks,
                )
                self.population_centers.append(new_center)
                culture = self._get_tile_majority_culture(tile)
                self._initialize_population_center_history(
                    new_center,
                    culture=culture,
                    reason="population_threshold",
                    note=f"dynamic≥{creation_threshold}",
                )
                self._log_event(
                    "population_center",
                    f"New population center established: {display_name} (threshold: {creation_threshold}, reason: dynamic)"
                )
                added = True

            elif meets_absolute_threshold(tile):
                display_name, phonetic_name = self._generate_settlement_name(i)
                new_center = PopulationCenter(
                    tile_index=i,
                    name=display_name,
                    phonetic_name=phonetic_name,
                    original_threshold=absolute_population_threshold,
                    established_year=self.current_year,
                    established_tick=self.total_ticks,
                    demotion_grace_ticks=demotion_grace_ticks,
                )
                self.population_centers.append(new_center)
                culture = self._get_tile_majority_culture(tile)
                self._initialize_population_center_history(
                    new_center,
                    culture=culture,
                    reason="absolute_threshold",
                    note="absolute",
                )
                self._log_event(
                    "population_center",
                    f"New population center established: {display_name} (threshold: {absolute_population_threshold}, reason: absolute)"
                )
                added = True

            if added:
                existing_center_indices.add(i)

        # Evaluate whether any population centers should spawn new polities
        self._evaluate_population_center_control()

        # Refresh placeholder/default names so new lexemes or placeholders propagate quickly
        for center in self.population_centers:
            self._rename_population_center_for_tile(center.tile_index)

    def _evaluate_population_center_control(self) -> None:
        """Track low-control streaks and trigger breakaway polities."""
        enable_breakaways = self.config.get('simulation.polity.enable_breakaway_polities', True) if self.config else True
        if not enable_breakaways:
            return

        low_control_years = self.config.get('simulation.polity.breakaway_low_control_years', 15) if self.config else 15
        unowned_spawn_years = (
            self.config.get('simulation.polity.unowned_spawn_years', low_control_years)
            if self.config else low_control_years
        )
        low_control_threshold = self.config.get('simulation.polity.breakaway_low_control_threshold', 5) if self.config else 5
        admin_max_bonus = self.config.get('simulation.polity.breakaway_administrative_burden_max_bonus', 20) if self.config else 20
        admin_reference_share = self.config.get('simulation.polity.breakaway_administrative_burden_reference_share', 0.35) if self.config else 0.35
        admin_scale_exponent = self.config.get('simulation.polity.breakaway_administrative_burden_exponent', 1.25) if self.config else 1.25
        admin_max_multiplier = self.config.get('simulation.polity.breakaway_administrative_burden_max_multiplier', 1.5) if self.config else 1.5
        admin_reference_share = max(0.01, min(1.0, admin_reference_share))
        try:
            admin_scale_exponent = max(0.1, float(admin_scale_exponent))
        except (TypeError, ValueError):
            admin_scale_exponent = 1.25
        try:
            admin_max_multiplier = max(1.0, float(admin_max_multiplier))
        except (TypeError, ValueError):
            admin_max_multiplier = 1.5
        administrative_burden: Dict[int, int] = {}
        if admin_max_bonus > 0:
            land_tile_count = max(1, sum(1 for tile in self.tiles if not tile.is_water))
            for polity in self.polities:
                if polity is None or not getattr(polity, 'tile_indices', None):
                    continue
                land_owned = 0
                for idx in getattr(polity, 'tile_indices', []):
                    if 0 <= idx < len(self.tiles):
                        tile_ref = self.tiles[idx]
                        if not tile_ref.is_water:
                            land_owned += 1
                if land_owned <= 0:
                    continue
                share = land_owned / land_tile_count
                share_ratio = max(0.0, share / admin_reference_share)
                normalized = pow(share_ratio, admin_scale_exponent)
                capped_ratio = min(admin_max_multiplier, normalized)
                bonus = int(round(admin_max_bonus * capped_ratio))
                if self._polity_has_idea_label(polity, 'kinship_state', 'meritocratic'):
                    bonus = int(round(bonus * 1.75))
                if bonus > 0:
                    administrative_burden[polity.id] = bonus
            self.polity_administrative_burden = administrative_burden
        low_control_required_ticks = int(self.ticks_per_year * max(0, low_control_years))
        if low_control_required_ticks <= 0:
            low_control_required_ticks = 1
        unowned_required_ticks = int(self.ticks_per_year * max(0, unowned_spawn_years))
        if unowned_required_ticks <= 0:
            unowned_required_ticks = 1
        for center in self.population_centers:
            idx = center.tile_index
            if idx >= len(self.tiles):
                continue
            tile = self.tiles[idx]
            if tile.is_water:
                center.low_control_ticks = 0
                continue
            polity_id = tile.polity_id
            is_unowned = polity_id == -1
            effective_threshold = low_control_threshold
            if not is_unowned:
                effective_threshold += administrative_burden.get(polity_id, 0)
                effective_threshold = max(0, min(99, effective_threshold))
            has_low_control = (not is_unowned) and tile.control_level <= effective_threshold
            if is_unowned:
                center.low_control_ticks += 1
                if center.low_control_ticks > unowned_required_ticks:
                    self._spawn_breakaway_polity(center)
                continue
            if has_low_control:
                center.low_control_ticks += 1
                if center.low_control_ticks > low_control_required_ticks:
                    self._spawn_breakaway_polity(center)
            else:
                center.low_control_ticks = 0

    def _spawn_breakaway_polity(self, center: PopulationCenter) -> None:
        """Create a new polity when a population center resists control."""
        if center.tile_index >= len(self.tiles):
            return
        center_tile = self.tiles[center.tile_index]
        parent_polity_id = center_tile.polity_id
        majority_culture = self._get_tile_majority_culture(center_tile)
        if not majority_culture:
            # Leave culture empty so first-culture spawning can occur naturally
            center_tile.cultural_makeup = {}
        captured_tiles = self._collect_breakaway_tiles(center.tile_index, parent_polity_id, majority_culture)
        if not captured_tiles:
            captured_tiles = [center.tile_index]
        new_polity_id = len(self.polities)
        polity_name = self._generate_breakaway_polity_name(center.name, majority_culture)
        color = self._generate_polity_color(new_polity_id)
        new_polity = Polity(
            id=new_polity_id,
            name=polity_name,
            color=color,
            leader=None,
            primary_culture=majority_culture,
            tile_indices=list(captured_tiles),
            capital_tile_index=center.tile_index
        )

        # Remove captured tiles from parent polity, if any
        if parent_polity_id != -1 and parent_polity_id < len(self.polities):
            parent_polity = self.polities[parent_polity_id]
            parent_polity.tile_indices = [idx for idx in parent_polity.tile_indices if idx not in captured_tiles]

        initial_control = 75
        capital_control = max(initial_control, 100)
        for tile_idx in captured_tiles:
            if tile_idx >= len(self.tiles):
                continue
            tile = self.tiles[tile_idx]
            old_polity = tile.polity_id
            tile.polity_id = new_polity_id
            tile.control_level = capital_control if tile_idx == center.tile_index else initial_control
            self._apply_control_alignment_cap(tile_idx)
            if majority_culture:
                if tile.cultural_makeup is None:
                    tile.cultural_makeup = {}
                tile.cultural_makeup.setdefault(majority_culture, 0.5)
                self._normalize_tile_culture(tile)
            self._on_tile_polity_changed(tile_idx, old_polity, new_polity_id)

        self.polities.append(new_polity)
        self._appoint_new_leader(new_polity, reason='breakaway')
        self._assign_polity_language_name(new_polity)
        polity_name = new_polity.name
        self._ensure_relationships_for_polity(new_polity_id)
        self._ensure_population_center_entry(new_polity.capital_tile_index, name=f"{polity_name} Capital", threshold=1)
        center.low_control_ticks = 0
        parent_polity = self._get_polity(parent_polity_id) if parent_polity_id != -1 else None
        self._log_polity_event(
            "polity",
            f"Breakaway polity formed: {polity_name} with {len(captured_tiles)} tile(s)",
            new_polity,
            parent_polity,
        )
        if parent_polity_id != -1:
            relationship = self._get_relationship(new_polity_id, parent_polity_id, create=True)
            if relationship:
                relationship.met = True
                self._begin_war(
                    relationship,
                    initiator_id=new_polity_id,
                    target_id=parent_polity_id,
                )
            hostility = -50.0
            self._modify_ticking_modifier(new_polity_id, parent_polity_id, hostility)
            self._modify_ticking_modifier(parent_polity_id, new_polity_id, hostility)
            # Breakaway shakes confidence in the parent overlord
            self._apply_vassal_integration_burst(parent_polity_id, delta=-50.0)

    def _collect_breakaway_tiles(
        self,
        center_index: int,
        parent_polity_id: int,
        anchor_culture: Optional[str] = None,
    ) -> List[int]:
        """Collect culturally aligned tiles that join a breakaway polity via low-control chaining."""
        collected: Set[int] = {center_index}
        if parent_polity_id == -1:
            return list(collected)

        chain_threshold = self.config.get('simulation.polity.breakaway_chain_control_threshold', 10) if self.config else 10

        def _tile_culture_matches(tile: Tile) -> bool:
            if not anchor_culture:
                return True
            majority = self._get_tile_majority_culture(tile)
            if not majority:
                return False
            if majority == anchor_culture:
                return True
            culture_obj = self._find_culture_by_name(majority)
            if culture_obj and anchor_culture in culture_obj.heritage:
                return True
            return False

        def qualifies(index: int) -> bool:
            if index >= len(self.tiles):
                return False
            tile = self.tiles[index]
            return (not tile.is_water and 
                    tile.polity_id == parent_polity_id and 
                    tile.control_level <= chain_threshold and
                    _tile_culture_matches(tile))

        visited: Set[int] = set(collected)
        queue = deque()

        for neighbor_idx in self.tiles[center_index].neighbors:
            if neighbor_idx not in visited and qualifies(neighbor_idx):
                visited.add(neighbor_idx)
                queue.append(neighbor_idx)

        while queue:
            current_idx = queue.popleft()
            collected.add(current_idx)
            for neighbor_idx in self.tiles[current_idx].neighbors:
                if neighbor_idx not in visited and qualifies(neighbor_idx):
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)

        return sorted(collected)

    def _propagate_regional_culture_chain(self, origin_tile_index: int, culture_name: str) -> int:
        """Spread a regional culture to nearby low-control tiles in the same region."""
        if origin_tile_index < 0 or origin_tile_index >= len(self.tiles):
            return 0
        origin_tile = self.tiles[origin_tile_index]
        region_id = origin_tile.region_id
        if region_id is None or region_id < 0:
            return 0

        def qualifies(tile_index: int) -> bool:
            if tile_index < 0 or tile_index >= len(self.tiles):
                return False
            tile = self.tiles[tile_index]
            if tile.is_water or tile.region_id != region_id:
                return False
            if tile.control_level >= 100:
                return False
            return True

        visited: Set[int] = {origin_tile_index}
        queue = deque()
        for neighbor_idx in origin_tile.neighbors:
            if neighbor_idx not in visited and qualifies(neighbor_idx):
                visited.add(neighbor_idx)
                queue.append(neighbor_idx)

        conversions = 0
        while queue:
            current_idx = queue.popleft()
            tile = self.tiles[current_idx]
            self._set_tile_culture_full(tile, culture_name)
            self._rename_population_center_for_tile(current_idx)
            conversions += 1
            for neighbor_idx in tile.neighbors:
                if neighbor_idx not in visited and qualifies(neighbor_idx):
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)
        return conversions

    def _generate_breakaway_polity_name(self, center_name: str, culture_name: str) -> str:
        """Generate a compact, non-generic name for a newly formed polity."""
        max_length = self.config.get('linguistics.polity_naming.max_name_length', 25) if self.config else 25

        def _sanitize(label: Optional[str]) -> Optional[str]:
            formatted = self._format_polity_name_component(label or "")
            return formatted or (label.replace('_', ' ').strip() if label else None)

        # Prefer a language-derived token to avoid generic "Free State"/"League" names
        base: Optional[str] = None
        if culture_name:
            token = self._select_culture_language_token(culture_name, POLITY_COMPONENT_CATEGORIES, allow_fallback=True)
            base = _sanitize(token)

        if not base:
            base = _sanitize(center_name)

        if not base:
            base = _sanitize(culture_name) or "Breakaway Realm"

        candidate = base[:max_length].strip() if len(base) > max_length else base
        return self._ensure_unique_polity_name(candidate, culture_name=culture_name)

    def _generate_polity_color(self, polity_id: int) -> Tuple[int, int, int]:
        """Generate a color for a newly spawned polity."""
        import colorsys
        hue = (polity_id * 0.37) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.6, 0.95)
        return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

    def _generate_vassal_color(self, overlord_color: Optional[Tuple[int, int, int]], subject_id: int) -> Tuple[int, int, int]:
        """Bias a vassal's color toward its overlord for easy visual grouping."""
        if not overlord_color:
            return self._generate_polity_color(subject_id)
        r, g, b = [c / 255.0 for c in overlord_color]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        rng = random.Random(f"vassal_color:{self.world_seed}:{subject_id}")
        h = (h + rng.uniform(-0.035, 0.035)) % 1.0
        s = max(0.0, min(1.0, s * rng.uniform(0.78, 0.92)))
        v = max(0.0, min(1.0, v * rng.uniform(0.88, 1.02)))
        nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
        return (int(nr * 255), int(ng * 255), int(nb * 255))

    def _get_tile_majority_culture(self, tile: Tile) -> Optional[str]:
        """Return the dominant culture for a tile, if any."""
        if not tile.cultural_makeup:
            return None
        return max(tile.cultural_makeup.items(), key=lambda item: item[1])[0]

    def _get_immune_cultures(self, tile: Tile) -> Set[str]:
        immune: Set[str] = set()
        for culture_name in tile.cultural_makeup.keys():
            culture = self._find_culture_by_name(culture_name)
            if (culture and not culture.is_initial and culture.immunity_end_year is not None and
                    self.current_year < culture.immunity_end_year):
                immune.add(culture_name)
        return immune

    def _redistribute_culture_share(self, tile: Tile, excluded_names: Set[str], desired_amount: float) -> float:
        """Redistribute cultural share from donors not in excluded set."""
        if desired_amount <= 0:
            return 0.0
        donors = [name for name, share in tile.cultural_makeup.items()
                  if name not in excluded_names and share > 0.0]
        donor_total = sum(tile.cultural_makeup[name] for name in donors)
        if donor_total <= 0:
            return 0.0
        actual_amount = min(desired_amount, donor_total)
        if actual_amount <= 0:
            return 0.0
        for name in donors:
            reduction = (tile.cultural_makeup[name] / donor_total) * actual_amount
            tile.cultural_makeup[name] = max(0.0, tile.cultural_makeup[name] - reduction)
        return actual_amount

    def _apply_new_culture_growth(self, tile: Tile, immune_cultures: Optional[Set[str]] = None) -> None:
        """Apply positive growth pressure to recently formed cultures."""
        growth_rate = self.config.get('simulation.culture.new_culture_growth_rate', 0.05) if self.config else 0.05
        if growth_rate <= 0:
            return
        if tile.population == 0:
            tile.cultural_makeup = {}
            return
        if immune_cultures is None:
            immune_cultures = self._get_immune_cultures(tile)
        if not immune_cultures:
            return
        for culture_name in list(immune_cultures):
            share = tile.cultural_makeup.get(culture_name, 0.0)
            culture_obj = self._find_culture_by_name(culture_name)
            if (not culture_obj or culture_obj.is_initial or
                    culture_obj.immunity_end_year is None or self.current_year >= culture_obj.immunity_end_year):
                continue
            remaining = max(0.0, 1.0 - share)
            if remaining <= 0:
                continue
            desired_growth = growth_rate * remaining
            exclusions = set(immune_cultures)
            exclusions.add(culture_name)
            actual_growth = self._redistribute_culture_share(tile, exclusions, desired_growth)
            if actual_growth > 0:
                tile.cultural_makeup[culture_name] = share + actual_growth

    def _update_syncretism_tracker(self, tile_idx: int, tile: Tile) -> None:
        """Track long-lived culture pairs to spawn syncretic cultures."""
        base_threshold = self.config.get('simulation.culture.syncretism_threshold', 0.3) if self.config else 0.3
        duration_years = self.config.get('simulation.culture.syncretism_duration_years', 20) if self.config else 20
        required_ticks = max(1, int(duration_years * self.ticks_per_year))
        polity = self._get_polity(tile.polity_id)
        threshold = self._get_syncretism_threshold(base_threshold, polity)
        eligible_cultures = [name for name, share in tile.cultural_makeup.items() if share >= threshold]
        existing_keys = [key for key in list(self.syncretism_tracker.keys()) if key[0] == tile_idx]
        eligible_pairs: Set[Tuple[str, str]] = set()
        if len(eligible_cultures) >= 2:
            for pair in itertools.combinations(sorted(eligible_cultures), 2):
                if self._are_direct_parent_cultures(pair[0], pair[1]):
                    continue
                eligible_pairs.add(pair)
                key = (tile_idx, pair)
                self.syncretism_tracker[key] = self.syncretism_tracker.get(key, 0) + 1
                if self.syncretism_tracker[key] >= required_ticks:
                    self._create_syncretic_culture(tile_idx, tile, pair)
                    self.syncretism_tracker.pop(key, None)
        for key in existing_keys:
            if key[1] not in eligible_pairs:
                self.syncretism_tracker.pop(key, None)

    def _generate_syncretic_culture_name(
        self,
        parent1: Optional[str],
        parent2: Optional[str],
        origin_tile_index: int
    ) -> Optional[str]:
        """Blend parent phonology to name syncretic cultures with some uniqueness bias."""
        if not parent1 or not parent2:
            return None
        catalog_a = self.get_culture_language(parent1)
        catalog_b = self.get_culture_language(parent2)

        token_a = self._select_language_token(catalog_a, CULTURE_NAME_CATEGORIES, allow_fallback=True) if catalog_a else None
        token_b = self._select_language_token(catalog_b, CULTURE_NAME_CATEGORIES, allow_fallback=True) if catalog_b else None

        candidates: List[str] = []

        def blend(a: str, b: str) -> str:
            a_mid = max(1, len(a) // 2)
            b_mid = max(1, len(b) // 2)
            return f"{a[:a_mid]}{b[b_mid:]}"

        for a, b in ((token_a, token_b), (token_b, token_a)):
            if a and b:
                blended = self._format_polity_name_component(blend(a, b))
                if blended:
                    candidates.append(blended)
        if token_a and token_b:
            hyphen = self._format_polity_name_component(f"{token_a}-{token_b}")
            if hyphen:
                candidates.append(hyphen)
        for token in (token_a, token_b):
            if token:
                formatted = self._format_polity_name_component(token)
                if formatted:
                    candidates.append(formatted)

        for candidate in candidates:
            unique = self._make_unique_culture_name(candidate, origin_tile_index)
            if unique:
                return unique
        return None

    def _create_syncretic_culture(self, tile_idx: int, tile: Tile, parent_pair: Tuple[str, str]) -> None:
        """Form or reuse a syncretic culture from two parent cultures."""
        if tile.population == 0:
            tile.cultural_makeup = {}
            return
        parent1, parent2 = parent_pair
        if self._are_direct_parent_cultures(parent1, parent2):
            return
        share1 = tile.cultural_makeup.get(parent1, 0.0)
        share2 = tile.cultural_makeup.get(parent2, 0.0)
        total_share = share1 + share2
        if total_share <= 0:
            return
        parent_set = frozenset(parent_pair)
        existing_name = self.syncretic_cultures.get(parent_set)
        culture_obj = self._find_culture_by_name(existing_name) if existing_name else None
        culture_name = existing_name
        reserved_token: Optional[str] = None
        if not culture_obj:
            immunity_years = self.config.get('simulation.culture.new_culture_immunity_years', 10) if self.config else 10
            dominant_parent = parent1 if share1 >= share2 else parent2
            dominant_culture = self._find_culture_by_name(dominant_parent)
            parent_color = dominant_culture.color if dominant_culture else None
            heritage_total = share1 + share2
            heritage = {}
            if heritage_total > 0:
                heritage[parent1] = share1 / heritage_total
                heritage[parent2] = share2 / heritage_total
            base_name = existing_name or self._generate_syncretic_culture_name(parent1, parent2, tile_idx) or self._next_culture_name('S')
            culture_name, reserved_token = self._reserve_unique_culture_name(base_name, tile_idx)
            try:
                new_culture = Culture(
                    name=culture_name,
                    color=self._generate_child_culture_color(parent_color),
                    heritage=heritage,
                    origin_tile_index=tile_idx,
                    birth_year=self.current_year,
                    home_region_id=self._determine_home_region_id(tile_idx),
                    immunity_end_year=self.current_year + immunity_years,
                    is_initial=False
                )
                self._ensure_culture_ideas(new_culture, heritage_override=heritage, jitter=0.05)
                self.cultures.append(new_culture)
                if reserved_token:
                    self._commit_reserved_culture_name(culture_name)
                    reserved_token = None
                else:
                    self._register_culture_name(culture_name)
                self.assign_syncretic_language(new_culture, parent1, parent2, share1, share2)
                self.syncretic_cultures[parent_set] = culture_name
                self._syncretic_parent_lookup[culture_name] = parent_set
                self._log_event(
                    "culture",
                    f"Syncretic culture '{culture_name}' formed at tile {tile_idx} in Year {self.current_year} "
                    f"from parents {parent1}+{parent2}, color: {new_culture.color}"
                )
                culture_obj = new_culture
            except Exception:
                if reserved_token:
                    self._release_reserved_culture_name(culture_name)
                raise
        else:
            culture_name = culture_obj.name
            self._syncretic_parent_lookup.setdefault(culture_name, parent_set)
            self._ensure_culture_ideas(culture_obj)
        for parent, share in ((parent1, share1), (parent2, share2)):
            if share <= 0:
                continue
            current_share = tile.cultural_makeup.get(parent, 0.0)
            new_share = max(0.0, current_share - share)
            if new_share <= 0.0:
                tile.cultural_makeup.pop(parent, None)
            else:
                tile.cultural_makeup[parent] = new_share
        tile.cultural_makeup[culture_name] = tile.cultural_makeup.get(culture_name, 0.0) + total_share
        self._normalize_tile_culture(tile)

    # ------------------------------------------------------------------
    # Relationship and conflict helpers
    # ------------------------------------------------------------------

    def _relationship_key(self, polity_a: int, polity_b: int) -> Tuple[int, int]:
        return tuple(sorted((polity_a, polity_b)))

    def _cleanup_stale_relationships(self) -> None:
        """Remove relationships that reference non-existent or inactive polities."""
        if not self.relationships:
            return
        retained: List[Relationship] = []
        removed_count = 0
        for relationship in self.relationships:
            alive_a = self._get_polity(relationship.polity_a) is not None
            alive_b = self._get_polity(relationship.polity_b) is not None
            if alive_a and alive_b:
                retained.append(relationship)
                continue
            if relationship.status == "war":
                self._clear_occupations_for_relationship(relationship)
            key = self._relationship_key(relationship.polity_a, relationship.polity_b)
            self.relationship_lookup.pop(key, None)
            removed_count += 1
            self._log_event(
                "diplomacy_debug",
                f"Removed stale relationship between Polity {relationship.polity_a} and Polity {relationship.polity_b} (polity missing)"
            )
        if removed_count > 0:
            self.relationships = retained
            self._increment_workload_counter('relationships_stale_removed', removed_count)

    def _on_tile_polity_changed(self, tile_index: int, old_polity_id: int, new_polity_id: int) -> None:
        """Update relationship metadata when ownership of a tile changes."""
        if old_polity_id == new_polity_id or tile_index < 0 or tile_index >= len(self.tiles):
            return
        self._invalidate_frontline_cache_for_tile(tile_index, {old_polity_id, new_polity_id})
        tile = self.tiles[tile_index]
        for neighbor_idx in tile.neighbors:
            if neighbor_idx >= len(self.tiles):
                continue
            neighbor = self.tiles[neighbor_idx]
            neighbor_polity = neighbor.polity_id
            if old_polity_id >= 0 and neighbor_polity >= 0 and neighbor_polity != old_polity_id:
                self._adjust_shared_border(old_polity_id, neighbor_polity, -1)
            if new_polity_id >= 0 and neighbor_polity >= 0 and neighbor_polity != new_polity_id:
                self._adjust_shared_border(new_polity_id, neighbor_polity, 1)

    def _adjust_shared_border(self, polity_a: int, polity_b: int, delta: int) -> None:
        if delta == 0 or polity_a == polity_b or polity_a < 0 or polity_b < 0:
            return
        relationship = self._get_relationship(polity_a, polity_b, create=True)
        if relationship is None:
            return
        relationship.shared_border_tiles = max(0, relationship.shared_border_tiles + delta)
        relationship.met = relationship.shared_border_tiles > 0
        self._relationship_borders_initialized = True

    def _get_frontline_polity_version(self, polity_id: int) -> int:
        if polity_id < 0:
            return -1
        return self._war_frontline_polity_versions.get(polity_id, 0)

    def _bump_frontline_polity_version(self, polity_id: int) -> None:
        if polity_id < 0:
            return
        current = self._war_frontline_polity_versions.get(polity_id, 0)
        self._war_frontline_polity_versions[polity_id] = current + 1

    def _invalidate_frontline_cache_for_tile(self, tile_idx: int, extra_polities: Optional[Set[int]] = None) -> None:
        if tile_idx < 0 or tile_idx >= len(self.tiles):
            return
        tile = self.tiles[tile_idx]
        touched: Set[int] = set(extra_polities or set())
        if tile.polity_id >= 0:
            touched.add(tile.polity_id)
        if tile.occupied_by_polity_id >= 0:
            touched.add(tile.occupied_by_polity_id)
        for neighbor_idx in tile.neighbors:
            if neighbor_idx < 0 or neighbor_idx >= len(self.tiles):
                continue
            neighbor_tile = self.tiles[neighbor_idx]
            if neighbor_tile.polity_id >= 0:
                touched.add(neighbor_tile.polity_id)
            if neighbor_tile.occupied_by_polity_id >= 0:
                touched.add(neighbor_tile.occupied_by_polity_id)
        for polity_id in touched:
            self._bump_frontline_polity_version(polity_id)

    def _ensure_border_cache(self) -> None:
        if self._relationship_borders_initialized:
            return
        for relationship in self.relationships:
            relationship.shared_border_tiles = 0
            relationship.met = False
        for idx, tile in enumerate(self.tiles):
            if tile.polity_id < 0:
                continue
            for neighbor_idx in tile.neighbors:
                if neighbor_idx <= idx or neighbor_idx >= len(self.tiles):
                    continue
                neighbor = self.tiles[neighbor_idx]
                if neighbor.polity_id < 0 or neighbor.polity_id == tile.polity_id:
                    continue
                self._adjust_shared_border(tile.polity_id, neighbor.polity_id, 1)
        self._relationship_borders_initialized = True

    def _ensure_relationship_fields(self, relationship: Relationship) -> None:
        for polity_id in (relationship.polity_a, relationship.polity_b):
            if polity_id not in relationship.ticking_modifiers:
                relationship.ticking_modifiers[polity_id] = 0.0
            if polity_id not in relationship.war_exhaustion:
                relationship.war_exhaustion[polity_id] = 0.0
            if polity_id not in relationship.war_exhaustion_anchor_values:
                relationship.war_exhaustion_anchor_values[polity_id] = relationship.war_exhaustion.get(polity_id, 0.0)
            if polity_id not in relationship.war_exhaustion_anchor_years:
                relationship.war_exhaustion_anchor_years[polity_id] = self.current_year
        if not getattr(relationship, 'occupied_tiles', None):
            relationship.occupied_tiles = {}

    def _clamp_relation_score(self, value: float) -> float:
        return max(-100.0, min(100.0, value))

    def _clamp_exhaustion(self, value: float) -> float:
        return max(0.0, min(100.0, value))

    def _get_war_config(self) -> Dict[str, float]:
        return self.config.get('simulation.war', {}) if self.config else {}

    def _cleanup_capture_cooldowns(self) -> None:
        if not self.capture_cooldowns:
            return
        expired = [tile_idx for tile_idx, expiry in self.capture_cooldowns.items() if expiry <= self.total_ticks]
        for tile_idx in expired:
            self.capture_cooldowns.pop(tile_idx, None)

    def _is_tile_on_capture_cooldown(self, tile_idx: int) -> bool:
        if tile_idx < 0:
            return False
        expiry = self.capture_cooldowns.get(tile_idx)
        return bool(expiry and expiry > self.total_ticks)

    def _start_capture_cooldown(self, tile_idx: int, war_config: Dict[str, float]) -> None:
        cooldown_ticks = int(war_config.get('capture_cooldown_ticks', 4))
        if cooldown_ticks <= 0:
            self.capture_cooldowns.pop(tile_idx, None)
            return
        self.capture_cooldowns[tile_idx] = self.total_ticks + cooldown_ticks

    def _ensure_relationships_for_polity(self, polity_id: int) -> None:
        polity = self._get_polity(polity_id)
        if polity is None:
            return
        for other in self.polities:
            if other is None or not getattr(other, 'is_active', True):
                continue
            if other.id == polity_id:
                continue
            self._get_relationship(polity_id, other.id, create=True)

    def _sync_overlord_relationships_for_vassal(self, vassal_id: int) -> None:
        vassal = self._get_polity(vassal_id)
        if not vassal:
            return
        overlord_id = int(getattr(vassal, 'suzerain_id', -1))
        if overlord_id < 0:
            return
        for relationship in list(self.relationships):
            if relationship is None:
                continue
            other_id: Optional[int] = None
            if relationship.polity_a == vassal_id:
                other_id = relationship.polity_b
            elif relationship.polity_b == vassal_id:
                other_id = relationship.polity_a
            if other_id is None or other_id == overlord_id:
                continue
            target_rel = self._get_relationship(overlord_id, other_id, create=True)
            if not target_rel:
                continue
            target_rel.met = True
            proxy_border = max(0, getattr(relationship, 'shared_border_tiles', 0))
            if proxy_border > getattr(target_rel, 'shared_border_tiles', 0):
                target_rel.shared_border_tiles = proxy_border

    def _propagate_overlord_relationships(self) -> None:
        """Ensure overlords know and count borders through their vassals."""
        if not self.relationships:
            return
        for relationship in list(self.relationships):
            if relationship is None:
                continue
            a, b = relationship.polity_a, relationship.polity_b
            shared = max(0, getattr(relationship, 'shared_border_tiles', 0))
            for child_id, other_id in ((a, b), (b, a)):
                child = self._get_polity(child_id)
                if not child:
                    continue
                overlord_id = int(getattr(child, 'suzerain_id', -1))
                if overlord_id < 0 or overlord_id == other_id:
                    continue
                target_rel = self._get_relationship(overlord_id, other_id, create=True)
                if not target_rel:
                    continue
                target_rel.met = True
                if shared > getattr(target_rel, 'shared_border_tiles', 0):
                    target_rel.shared_border_tiles = shared

    def _get_polity(self, polity_id: int) -> Optional[Polity]:
        if polity_id < 0 or polity_id >= len(self.polities):
            return None
        polity = self.polities[polity_id]
        if polity is None or not getattr(polity, 'is_active', True):
            return None
        return polity

    def get_polity_cultural_tolerance(self, polity_or_id: Optional[Union[int, Polity]]) -> Optional[float]:
        polity: Optional[Polity]
        if isinstance(polity_or_id, Polity):
            polity = polity_or_id
        elif isinstance(polity_or_id, int):
            polity = self._get_polity(polity_or_id)
        else:
            polity = None
        if polity is None:
            return None
        return self._get_polity_tolerance(polity)

    def get_polity_administrative_burden(self, polity_or_id: Optional[Union[int, Polity]]) -> Optional[int]:
        polity: Optional[Polity]
        if isinstance(polity_or_id, Polity):
            polity = polity_or_id
        elif isinstance(polity_or_id, int):
            polity = self._get_polity(polity_or_id)
        else:
            polity = None
        if polity is None:
            return None
        burden_map = getattr(self, 'polity_administrative_burden', {}) or {}
        return burden_map.get(polity.id, 0)

    def _get_relationship(self, polity_a: int, polity_b: int, create: bool = False) -> Optional[Relationship]:
        if polity_a == polity_b or polity_a < 0 or polity_b < 0:
            return None
        key = self._relationship_key(polity_a, polity_b)
        relationship = self.relationship_lookup.get(key)
        if relationship is None and create:
            relationship = Relationship(
                polity_a=key[0],
                polity_b=key[1],
                last_status_change_tick=self.total_ticks
            )
            self.relationships.append(relationship)
            self.relationship_lookup[key] = relationship
            self._log_event(
                "diplomacy_debug",
                f"Relationship initialized between Polity {key[0]} and Polity {key[1]}"
            )
        if relationship:
            self._ensure_relationship_fields(relationship)
        return relationship

    def _relationship_setting(self, key: str, default: float) -> float:
        if not self.config:
            return default
        return self.config.get(f'simulation.relationships.{key}', default)

    def _get_yearly_relationship_bucket_state(self) -> Tuple[bool, int, int, int]:
        ticks_per_year = max(1, int(self.ticks_per_year))
        cadence_years = max(1, int(self.relationship_yearly_check_interval_years))
        if cadence_years > 1:
            cycle_ticks = ticks_per_year * cadence_years
            if cycle_ticks <= 0 or (self.total_ticks % cycle_ticks) >= ticks_per_year:
                return False, 0, 1, 1
        bucket_count = max(1, int(self.relationship_yearly_check_buckets))
        if bucket_count == 1:
            should_run = (self.total_ticks % ticks_per_year == 0)
            return should_run, 0, 1, 1
        buckets_per_tick = max(1, int(math.ceil(bucket_count / ticks_per_year)))
        current_bucket = (self.total_ticks * buckets_per_tick) % bucket_count
        return True, current_bucket, bucket_count, buckets_per_tick

    def _relationship_matches_yearly_bucket(self, relationship: Relationship, current_bucket: int, bucket_count: int, buckets_per_tick: int) -> bool:
        if bucket_count <= 1:
            return True
        key = self._relationship_key(relationship.polity_a, relationship.polity_b)
        bucket_hash = ((key[0] * 73856093) ^ (key[1] * 19349663)) & 0x7FFFFFFF
        relation_bucket = bucket_hash % bucket_count
        width = max(1, buckets_per_tick)
        span = min(bucket_count, width)
        end_bucket = current_bucket + span
        if end_bucket <= bucket_count:
            return current_bucket <= relation_bucket < end_bucket
        wrapped_end = end_bucket % bucket_count
        return relation_bucket >= current_bucket or relation_bucket < wrapped_end

    def _relationship_matches_tick_stride(self, relationship: Relationship, stride_ticks: int) -> bool:
        stride = max(1, int(stride_ticks))
        if stride <= 1:
            return True
        key = self._relationship_key(relationship.polity_a, relationship.polity_b)
        bucket_hash = ((key[0] * 83492791) ^ (key[1] * 2654435761)) & 0x7FFFFFFF
        return (self.total_ticks % stride) == (bucket_hash % stride)

    def _is_hot_war_relationship(self, relationship: Relationship) -> bool:
        if relationship.status != "war":
            return False
        if relationship.shared_border_tiles > 0:
            return True
        if getattr(relationship, 'occupied_tiles', None):
            for tiles in relationship.occupied_tiles.values():
                if tiles:
                    return True
        start_year = relationship.war_start_year
        if start_year is None:
            return True
        return (self.current_year - start_year) <= self.hot_war_recent_years

    def _modify_ticking_modifier(self, observer_id: int, subject_id: int, delta: float) -> None:
        relationship = self._get_relationship(observer_id, subject_id, create=True)
        if relationship is None:
            return
        current = relationship.ticking_modifiers.get(observer_id, 0.0)
        relationship.ticking_modifiers[observer_id] = self._clamp_relation_score(current + delta)

    def _decay_ticking_modifiers(self, relationship: Relationship) -> None:
        for polity_id, value in list(relationship.ticking_modifiers.items()):
            if value > 0:
                relationship.ticking_modifiers[polity_id] = max(0.0, value - 1.0)
            elif value < 0:
                relationship.ticking_modifiers[polity_id] = min(0.0, value + 1.0)

    def _get_primary_culture_parent(self, culture_name: Optional[str]) -> Optional[str]:
        if not culture_name:
            return None
        culture = self._find_culture_by_name(culture_name)
        if culture is None:
            return culture_name
        if not culture.heritage:
            return culture_name
        return max(culture.heritage.items(), key=lambda item: item[1])[0]

    def _get_polity_population_center_count(self, polity_id: int) -> int:
        if polity_id < 0 or polity_id >= len(self.polities):
            return 0
        count = 0
        for center in self.population_centers:
            if 0 <= center.tile_index < len(self.tiles):
                tile = self.tiles[center.tile_index]
                if tile.polity_id == polity_id:
                    count += 1
        return count

    def _get_capital_region_id(self, polity: Optional[Polity]) -> Optional[int]:
        if polity is None:
            return None
        capital_idx = getattr(polity, 'capital_tile_index', -1)
        if 0 <= capital_idx < len(self.tiles):
            return self.tiles[capital_idx].region_id
        return None

    def _estimate_tile_distance(self, tile_a_idx: Optional[int], tile_b_idx: Optional[int]) -> Optional[float]:
        """Approximate the hop distance between two tiles using geometric centers."""
        if tile_a_idx is None or tile_b_idx is None:
            return None
        if not (0 <= tile_a_idx < len(self.tiles)):
            return None
        if not (0 <= tile_b_idx < len(self.tiles)):
            return None
        tile_a = self.tiles[tile_a_idx]
        tile_b = self.tiles[tile_b_idx]
        dx = float(tile_a.center[0]) - float(tile_b.center[0])
        dy = float(tile_a.center[1]) - float(tile_b.center[1])
        pixel_distance = math.hypot(dx, dy)
        if pixel_distance <= 0:
            return 0.0
        total_area = max(1.0, float(self.width) * float(self.height))
        avg_tile_area = total_area / max(1, len(self.tiles))
        scale = max(1e-6, math.sqrt(avg_tile_area))
        return pixel_distance / scale

    def _capital_distance_war_bias(self, observer: Optional[Polity], subject: Optional[Polity]) -> float:
        """Return a relation modifier based on capital proximity scaled by polity size."""
        if observer is None or subject is None:
            return 0.0
        observer_capital = self._ensure_polity_capital(observer)
        subject_capital = self._ensure_polity_capital(subject)
        raw_distance = self._estimate_tile_distance(observer_capital, subject_capital)
        if raw_distance is None:
            return 0.0
        midpoint = self._relationship_setting('capital_distance_midpoint', 8.0)
        distance_scale = self._relationship_setting('capital_distance_scale', 2.0)
        size_scale = self._relationship_setting('capital_distance_size_scale', 0.35)
        observer_tiles = max(1, len(getattr(observer, 'tile_indices', []) or []))
        subject_tiles = max(1, len(getattr(subject, 'tile_indices', []) or []))
        average_radius = 0.5 * (math.sqrt(observer_tiles) + math.sqrt(subject_tiles))
        effective_distance = max(0.0, raw_distance - average_radius * size_scale)
        delta = midpoint - effective_distance
        bias = -delta * distance_scale
        return max(-20.0, min(20.0, bias))

    def _calculate_base_relation(self, observer_id: int, subject_id: int, relationship: Relationship) -> float:
        self._increment_workload_counter('relation_base_evals')
        observer = self._get_polity(observer_id)
        subject = self._get_polity(subject_id)
        if observer is None or subject is None:
            return 0.0
        base = 0.0
        border_penalty_cap = self._relationship_setting('border_penalty_cap', 50.0)
        border_penalty_scale = self._relationship_setting('border_penalty_scale', 12.0)
        shared_culture_bonus = self._relationship_setting('shared_culture_bonus', 20.0)
        different_culture_penalty = self._relationship_setting('different_culture_penalty', 10.0)
        shared_parent_penalty = self._relationship_setting('shared_parent_penalty', 10.0)
        dev_superiority_bonus = self._relationship_setting('development_superiority_bonus', 50.0)
        dev_peer_penalty = self._relationship_setting('development_peer_penalty', 10.0)
        center_advantage_penalty = self._relationship_setting('population_center_advantage_penalty', 10.0)
        center_gap_penalty = self._relationship_setting('population_center_gap_penalty', 0.0)
        shared_region_penalty = self._relationship_setting('shared_region_penalty', 10.0)
        dominance_dev_ratio = self._relationship_setting('dominance_development_ratio', 0.0)
        dominance_dev_penalty = self._relationship_setting('dominance_development_penalty', 0.0)
        dominance_tile_ratio = self._relationship_setting('dominance_tile_ratio', 0.0)
        dominance_tile_penalty = self._relationship_setting('dominance_tile_penalty', 0.0)
        if relationship.shared_border_tiles > 0:
            # Penalty is linear in number of shared border tiles, capped at border_penalty_cap
            border_penalty = -min(border_penalty_cap, border_penalty_scale * relationship.shared_border_tiles)
            base += border_penalty
        observer_culture = getattr(observer, 'primary_culture', None)
        subject_culture = getattr(subject, 'primary_culture', None)
        if observer_culture and subject_culture:
            if observer_culture == subject_culture:
                base += shared_culture_bonus
            else:
                base -= different_culture_penalty
                parent_cache = (self._relation_eval_cache or {}).get('culture_parents', {})
                observer_parent = parent_cache.get(observer_culture)
                subject_parent = parent_cache.get(subject_culture)
                if observer_parent is None:
                    observer_parent = self._get_primary_culture_parent(observer_culture)
                if subject_parent is None:
                    subject_parent = self._get_primary_culture_parent(subject_culture)
                if observer_parent and subject_parent and observer_parent == subject_parent:
                    base -= shared_parent_penalty
        relation_cache = self._relation_eval_cache or {}
        dev_cache = relation_cache.get('development', {})
        center_cache = relation_cache.get('centers', {})
        tile_count_cache = relation_cache.get('tile_counts', {})
        region_cache = relation_cache.get('regions', {})

        observer_dev = float(dev_cache.get(observer.id, self._calculate_polity_development_value(observer.id)))
        subject_dev = float(dev_cache.get(subject.id, self._calculate_polity_development_value(subject.id)))
        if observer_dev > 0:
            if subject_dev >= observer_dev * 2.0:
                base += dev_superiority_bonus
            elif subject_dev > observer_dev * 1.25 and subject_dev < observer_dev * 1.75:
                base -= dev_peer_penalty
        if dominance_dev_penalty > 0 and dominance_dev_ratio > 0:
            threshold = max(1.0, dominance_dev_ratio)
            ratio = observer_dev / max(1.0, subject_dev)
            if ratio >= threshold:
                severity = min(2.5, ratio / threshold)
                base -= dominance_dev_penalty * severity
        observer_centers = int(center_cache.get(observer.id, self._get_polity_population_center_count(observer.id)))
        subject_centers = int(center_cache.get(subject.id, self._get_polity_population_center_count(subject.id)))
        if subject_centers < observer_centers:
            gap = observer_centers - subject_centers
            penalty = center_advantage_penalty
            if center_gap_penalty > 0 and gap > 0:
                penalty += min(center_gap_penalty * gap, center_gap_penalty * 10)
            base -= penalty
        observer_tile_count = int(tile_count_cache.get(observer.id, len(getattr(observer, 'tile_indices', []) or [])))
        subject_tile_count = int(tile_count_cache.get(subject.id, len(getattr(subject, 'tile_indices', []) or [])))
        if dominance_tile_penalty > 0 and dominance_tile_ratio > 0 and subject_tile_count > 0:
            tile_ratio = observer_tile_count / max(1, subject_tile_count)
            if tile_ratio >= dominance_tile_ratio:
                severity = min(2.5, tile_ratio / dominance_tile_ratio)
                base -= dominance_tile_penalty * severity
        observer_region = region_cache.get(observer.id, self._get_capital_region_id(observer))
        subject_region = region_cache.get(subject.id, self._get_capital_region_id(subject))
        if observer_region is not None and subject_region is not None and observer_region == subject_region:
            base -= shared_region_penalty
        base += self._capital_distance_war_bias(observer, subject)
        diplomat_bonus = self._get_trait_value(subject, 'DIPLOMAT', 'opinion_bonus', 0.0)
        if diplomat_bonus:
            base += diplomat_bonus

        # Flat vassal-overlord bonus (configurable)
        vassal_overlord_bonus = self._relationship_setting('vassal_overlord_bonus', 30.0)
        # Vassal's opinion of overlord
        if getattr(observer, 'suzerain_id', -1) == subject.id:
            base += vassal_overlord_bonus
            if self._polity_has_idea_label(subject, 'legitimacy', 'divine right'):
                base += 20.0
            if self._polity_has_idea_label(subject, 'kinship_state', 'aristocratic'):
                base += 20.0
        # Overlord's opinion of vassal
        if getattr(subject, 'suzerain_id', -1) == observer.id:
            base += vassal_overlord_bonus
        return self._clamp_relation_score(base)

    def _compute_long_peace_bonus(self, relationship: Relationship) -> float:
        if relationship.status != "peace":
            return 0.0
        per_year = self._relationship_setting('long_peace_bonus_per_year', 0.0)
        cap = self._relationship_setting('long_peace_bonus_cap', 0.0)
        grace = self._relationship_setting('long_peace_grace_years', 0.0)
        if per_year <= 0 or cap <= 0:
            return 0.0
        last_end = relationship.last_war_end_year
        if last_end is None:
            return 0.0
        years_at_peace = max(0.0, float(self.current_year - last_end))
        effective_years = max(0.0, years_at_peace - max(0.0, grace))
        return min(cap, effective_years * per_year)

    def _get_current_relation(self, observer_id: int, subject_id: int, relationship: Relationship) -> float:
        base = self._calculate_base_relation(observer_id, subject_id, relationship)
        # Tolerance opinion bonus re-enabled, modulated by config
        subject = self._get_polity(subject_id)
        tolerance_bonus_scale = self._relationship_setting('tolerance_opinion_bonus_scale', 50.0)
        tolerance_bonus = 0.0
        if subject is not None:
            tolerance = self._get_polity_tolerance(subject)
            tolerance_bonus = tolerance_bonus_scale * (tolerance)
        modifier = relationship.ticking_modifiers.get(observer_id, 0.0)
        peace_bonus = self._compute_long_peace_bonus(relationship)
        return self._clamp_relation_score(base + tolerance_bonus + modifier + peace_bonus)

    def _get_active_war_count(self, polity_id: int) -> int:
        if polity_id < 0:
            return 0
        count = 0
        for relationship in self.relationships:
            if relationship is None or relationship.status != "war":
                continue
            if polity_id in (relationship.polity_a, relationship.polity_b):
                count += 1
        return count

    def _calculate_war_chance(
        self,
        relation_value: float,
        initiator_id: Optional[int] = None,
        target_id: Optional[int] = None
    ) -> float:
        if relation_value >= 0:
            return 0.0
        war_config = self._get_war_config()
        severity = min(100.0, abs(relation_value))
        base_factor = war_config.get('declaration_base_chance', 0.00065)
        growth_rate = war_config.get('declaration_growth_rate', 0.045)
        max_chance = war_config.get('declaration_max_chance', 0.5)
        chance = base_factor * math.exp(growth_rate * severity)
        initiator_penalty = war_config.get('declaration_initiator_war_penalty', 0.35)
        target_bonus = war_config.get('declaration_target_war_bonus', 0.15)
        if initiator_id is not None and initiator_penalty > 0:
            wars = self._get_active_war_count(initiator_id)
            if wars > 0:
                chance *= max(0.0, 1.0 - initiator_penalty * wars)
        if target_id is not None and target_bonus > 0:
            wars = self._get_active_war_count(target_id)
            if wars > 0:
                chance *= 1.0 + target_bonus * wars
        chance *= self._get_war_chance_trait_multiplier(initiator_id)
        return min(max_chance, max(0.0, chance))

    def _war_allows_vassal_participation(self, initiator_id: Optional[int], target_id: Optional[int]) -> bool:
        """Block wars where a vassal would fight without its overlord, except independence wars."""
        if initiator_id is None or target_id is None:
            return True
        initiator = self._get_polity(initiator_id)
        target = self._get_polity(target_id)
        if not initiator or not target:
            return True
        initiator_overlord = int(getattr(initiator, 'suzerain_id', -1))
        target_overlord = int(getattr(target, 'suzerain_id', -1))

        # Independence wars are allowed: vassal attacking its overlord.
        if initiator_overlord >= 0 and initiator_overlord == target_id:
            return True

        # Overlords can attack or defend their own subjects.
        if target_overlord >= 0 and target_overlord == initiator_id:
            return True

        # Otherwise, block any war where either side is a vassal without its overlord involved.
        if initiator_overlord >= 0:
            return False
        if target_overlord >= 0:
            return False
        return True

    def _is_independence_war(self, relationship: Relationship) -> bool:
        polity_a = self._get_polity(relationship.polity_a)
        polity_b = self._get_polity(relationship.polity_b)
        if not polity_a or not polity_b:
            return False
        return (
            getattr(polity_a, 'suzerain_id', -1) == polity_b.id or
            getattr(polity_b, 'suzerain_id', -1) == polity_a.id
        )

    def _get_overlord_war(
        self,
        subject_id: int,
        enemy_id: int,
        current_relationship: Optional[Relationship] = None,
    ) -> Optional[Relationship]:
        subject = self._get_polity(subject_id)
        if not subject:
            return None
        overlord_id = int(getattr(subject, 'suzerain_id', -1))
        if overlord_id < 0:
            return None
        rel = self._get_relationship(overlord_id, enemy_id)
        if rel is None or rel is current_relationship:
            return None
        if getattr(rel, 'status', None) != "war":
            return None
        return rel

    def _should_defer_to_overlord_resolution(
        self,
        subject_id: int,
        enemy_id: int,
        relationship: Relationship,
    ) -> bool:
        if subject_id < 0 or enemy_id < 0:
            return False
        if self._is_independence_war(relationship):
            return False
        overlord_war = self._get_overlord_war(subject_id, enemy_id, relationship)
        return overlord_war is not None

    def _attempt_relation_based_war_declaration(self, relationship: Relationship) -> None:
        self._increment_workload_counter('war_declaration_checks')
        if relationship.status == "war" or not relationship.met:
            return
        if relationship.truce_until_year is not None and self.current_year < relationship.truce_until_year:
            return
        for initiator, subject in ((relationship.polity_a, relationship.polity_b),
                                   (relationship.polity_b, relationship.polity_a)):
            self._increment_workload_counter('war_declaration_directions')
            relation_value = self._get_current_relation(initiator, subject, relationship)
            if relation_value >= 0:
                continue
            chance = self._calculate_war_chance(relation_value, initiator, subject)
            if chance <= 0 or random.random() >= chance:
                continue
            self._begin_war(relationship, initiator_id=initiator, target_id=subject)
            break

    def _check_high_relation_annexation(self, relationship: Relationship) -> None:
        # Peaceful annexation is disabled.
        return

    def _apply_grudge_penalty(self, relationship: Relationship) -> None:
        penalty = self._relationship_setting('grudge_penalty', 40.0)
        self._modify_ticking_modifier(relationship.polity_a, relationship.polity_b, -penalty)
        self._modify_ticking_modifier(relationship.polity_b, relationship.polity_a, -penalty)

    def _apply_warmongering_penalty(self, aggressor_id: Optional[int], exclude_id: Optional[int] = None) -> None:
        if aggressor_id is None:
            return
        penalty = self._relationship_setting('warmongering_penalty', 10.0)
        for polity in self.polities:
            if polity is None or not getattr(polity, 'is_active', True):
                continue
            if polity.id == aggressor_id or (exclude_id is not None and polity.id == exclude_id):
                continue
            relationship = self._get_relationship(polity.id, aggressor_id)
            if relationship is None or not relationship.met:
                continue
            self._modify_ticking_modifier(polity.id, aggressor_id, -penalty)

    def _begin_war(
        self,
        relationship: Relationship,
        initiator_id: Optional[int] = None,
        target_id: Optional[int] = None
    ) -> None:
        if relationship.status == "war":
            return
        relationship.status = "war"
        relationship.war_start_year = self.current_year
        relationship.last_status_change_tick = self.total_ticks
        relationship.truce_until_year = None
        relationship.war_exhaustion[relationship.polity_a] = 0.0
        relationship.war_exhaustion[relationship.polity_b] = 0.0
        relationship.war_exhaustion_anchor_values[relationship.polity_a] = 0.0
        relationship.war_exhaustion_anchor_values[relationship.polity_b] = 0.0
        relationship.war_exhaustion_anchor_years[relationship.polity_a] = self.current_year
        relationship.war_exhaustion_anchor_years[relationship.polity_b] = self.current_year
        self._log_event(
            "war",
            f"War declared between Polity {relationship.polity_a} and Polity {relationship.polity_b} "
            f"in Year {self.current_year}"
        )
        self._apply_warmongering_penalty(initiator_id, exclude_id=target_id)

        # Pull in vassals of each belligerent on the same side
        self._summon_vassals_to_war(relationship.polity_a, relationship.polity_b)
        self._summon_vassals_to_war(relationship.polity_b, relationship.polity_a)

    def _summon_vassals_to_war(self, overlord_id: int, enemy_id: int) -> None:
        overlord = self._get_polity(overlord_id)
        if not overlord:
            return
        for vassal_id in list(getattr(overlord, 'vassal_ids', []) or []):
            vassal = self._get_polity(vassal_id)
            if not vassal or not getattr(vassal, 'is_active', True):
                continue
            rel = self._get_relationship(vassal_id, enemy_id, create=True)
            if not rel or rel.status == "war":
                continue
            rel.met = True
            self._begin_war(rel, initiator_id=vassal_id, target_id=enemy_id)


    def _white_peace_polity_wars(self, polity_id: int, reason: str = "vassalization") -> int:
        """Immediately end all active wars involving the provided polity."""
        if polity_id < 0 or not self.relationships:
            return 0
        ended = 0
        for relationship in list(self.relationships):
            if getattr(relationship, 'status', None) != "war":
                continue
            if relationship.polity_a == polity_id or relationship.polity_b == polity_id:
                self._end_war(relationship, reason)
                ended += 1
        return ended

    def _end_war(self, relationship: Relationship, reason: str, *, propagate_dependents: bool = True) -> None:
        if relationship.status != "war":
            return
        independence_war = self._is_independence_war(relationship)
        self._clear_occupations_for_relationship(relationship)
        relationship.status = "peace"
        relationship.last_war_end_year = self.current_year
        relationship.war_start_year = None
        relationship.last_status_change_tick = self.total_ticks
        self._log_event(
            "war_end",
            f"War ended between Polity {relationship.polity_a} and Polity {relationship.polity_b} due to {reason} in Year {self.current_year}"
        )
        self._apply_grudge_penalty(relationship)
        war_config = self._get_war_config()
        truce_years = int(war_config.get('truce_years', 10))
        relationship.truce_until_year = self.current_year + truce_years
        for polity_id in (relationship.polity_a, relationship.polity_b):
            relationship.war_exhaustion[polity_id] = 0.0
            relationship.war_exhaustion_anchor_values[polity_id] = 0.0
            relationship.war_exhaustion_anchor_years[polity_id] = self.current_year

        if propagate_dependents and not independence_war:
            self._cascade_dependent_war_resolution(relationship, reason)

    def _cascade_dependent_war_resolution(self, base_relationship: Relationship, reason: str) -> None:
        """End wars between each participant's vassals and the opposing belligerent to keep peace synchronized."""
        suffix_reason = f"{reason} (overlord resolution)"
        seen: Set[int] = set()
        for overlord_id, enemy_id in (
            (base_relationship.polity_a, base_relationship.polity_b),
            (base_relationship.polity_b, base_relationship.polity_a),
        ):
            overlord = self._get_polity(overlord_id)
            if not overlord:
                continue
            stack = list(getattr(overlord, 'vassal_ids', []) or [])
            while stack:
                vassal_id = stack.pop()
                if vassal_id in seen:
                    continue
                seen.add(vassal_id)
                dep_rel = self._get_relationship(vassal_id, enemy_id)
                if dep_rel and dep_rel is not base_relationship and getattr(dep_rel, 'status', None) == "war" and not self._is_independence_war(dep_rel):
                    self._end_war(dep_rel, suffix_reason, propagate_dependents=False)
                vassal = self._get_polity(vassal_id)
                if vassal and getattr(vassal, 'vassal_ids', None):
                    stack.extend(vassal.vassal_ids)

    def _process_relationships(self) -> None:
        if not self.relationships:
            return
        self._increment_workload_counter('relationships_total', len(self.relationships))
        self._profiled_call('diplomacy:cleanup', self._cleanup_stale_relationships)
        self._profiled_call('diplomacy:border_cache', self._ensure_border_cache)
        self._profiled_call('diplomacy:overlord_sync', self._propagate_overlord_relationships)
        yearly_war_roll, yearly_bucket_index, yearly_bucket_count, yearly_buckets_per_tick = self._get_yearly_relationship_bucket_state()

        self._relation_eval_cache = self._build_relation_eval_cache() if yearly_war_roll else None

        profile_enabled = self.tick_profiling_enabled
        decay_elapsed = 0.0
        yearly_checks_elapsed = 0.0
        try:
            for relationship in list(self.relationships):
                self._increment_workload_counter('relationships_iterated')
                self._ensure_relationship_fields(relationship)
                decay_start = time.perf_counter() if profile_enabled else 0.0
                self._decay_ticking_modifiers(relationship)
                if profile_enabled:
                    decay_elapsed += time.perf_counter() - decay_start
                if yearly_war_roll:
                    self._increment_workload_counter('relationships_yearly_considered')
                    if not relationship.met and relationship.status != "war":
                        self._increment_workload_counter('relationships_yearly_not_met')
                        continue
                    if not self._relationship_matches_yearly_bucket(
                        relationship,
                        yearly_bucket_index,
                        yearly_bucket_count,
                        yearly_buckets_per_tick,
                    ):
                        self._increment_workload_counter('relationships_yearly_deferred')
                        continue
                    self._increment_workload_counter('relationships_yearly_checked')
                    yearly_start = time.perf_counter() if profile_enabled else 0.0
                    self._attempt_relation_based_war_declaration(relationship)
                    self._check_high_relation_annexation(relationship)
                    # --- BEGIN: Cultural Tolerance Opinion Bonus ---
                    # Polity A's opinion of B is increased by up to +50 based on B's tolerance
                    # Tolerance bonus is now only applied as a flat bonus in _get_current_relation
                    pass
                    # --- END: Cultural Tolerance Opinion Bonus ---
                    if profile_enabled:
                        yearly_checks_elapsed += time.perf_counter() - yearly_start
        finally:
            self._relation_eval_cache = None

        if profile_enabled:
            self._accumulate_tick_profile_section('diplomacy:modifier_decay', decay_elapsed)
            if yearly_war_roll:
                self._accumulate_tick_profile_section('diplomacy:yearly_checks', yearly_checks_elapsed)

        self._profiled_call(
            'diplomacy:vassal_integration',
            self._process_vassal_integration,
            yearly_bucket_index,
            yearly_bucket_count,
            yearly_buckets_per_tick,
        )

        self._profiled_call('war:active_processing', self._process_active_wars)

    def _process_active_wars(self) -> None:
        self._profiled_call('war:occupation_cleanup', self._cleanup_stale_occupations)
        war_relationships = [rel for rel in self.relationships if rel.status == "war"]
        if not war_relationships:
            return
        war_config = self._get_war_config()
        self._profiled_call('war:capture_cooldowns', self._cleanup_capture_cooldowns)
        self._profiled_call('war:frontline_decay', self._decay_frontline_supply, war_config)
        self._profiled_call('war:capture_progress_decay', self._decay_capture_progress, war_config)
        full_resolution_relationships: List[Relationship] = []
        cold_war_stride = max(1, int(self.cold_war_stride_ticks))
        self._increment_workload_counter('war_relationships_total', len(war_relationships))
        for relationship in war_relationships:
            if self._is_hot_war_relationship(relationship):
                full_resolution_relationships.append(relationship)
                self._increment_workload_counter('war_relationships_hot')
                continue
            if self._relationship_matches_tick_stride(relationship, cold_war_stride):
                full_resolution_relationships.append(relationship)
                self._increment_workload_counter('war_relationships_cold_processed')
            else:
                self._increment_workload_counter('war_relationships_cold_deferred')

        if not full_resolution_relationships:
            return

        war_counts: Counter = Counter()
        for relationship in full_resolution_relationships:
            war_counts[relationship.polity_a] += 1
            war_counts[relationship.polity_b] += 1

        profile_enabled = self.tick_profiling_enabled
        supply_prep_elapsed = 0.0
        supply_dist_elapsed = 0.0
        battle_resolution_elapsed = 0.0
        post_battle_elapsed = 0.0

        for polity_id in war_counts.keys():
            supply_prep_start = time.perf_counter() if profile_enabled else 0.0
            self._prepare_polity_war_supply(polity_id, war_config)
            if profile_enabled:
                supply_prep_elapsed += time.perf_counter() - supply_prep_start

        self._increment_workload_counter('war_relationships_resolved', len(full_resolution_relationships))
        for relationship in list(full_resolution_relationships):
            polity_a = self._get_polity(relationship.polity_a)
            polity_b = self._get_polity(relationship.polity_b)
            if polity_a is None or polity_b is None:
                self._end_war(relationship, "polity collapse")
                continue
            spent_supply: Dict[int, float] = {}
            supply_share_a = self._get_polity_supply_share(polity_a.id, war_counts, war_config)
            supply_share_b = self._get_polity_supply_share(polity_b.id, war_counts, war_config)
            if supply_share_a > 0:
                supply_dist_start = time.perf_counter() if profile_enabled else 0.0
                spent_a = self._distribute_supply_to_frontlines(polity_a.id, polity_b.id, supply_share_a, war_config)
                if profile_enabled:
                    supply_dist_elapsed += time.perf_counter() - supply_dist_start
                self.polity_war_supply[polity_a.id] = max(0.0, self.polity_war_supply.get(polity_a.id, 0.0) - spent_a)
                spent_supply[polity_a.id] = spent_a
            if supply_share_b > 0:
                supply_dist_start = time.perf_counter() if profile_enabled else 0.0
                spent_b = self._distribute_supply_to_frontlines(polity_b.id, polity_a.id, supply_share_b, war_config)
                if profile_enabled:
                    supply_dist_elapsed += time.perf_counter() - supply_dist_start
                self.polity_war_supply[polity_b.id] = max(0.0, self.polity_war_supply.get(polity_b.id, 0.0) - spent_b)
                spent_supply[polity_b.id] = spent_b

            battle_start = time.perf_counter() if profile_enabled else 0.0
            occupation_events, battle_report = self._resolve_border_battles(
                relationship, polity_a, polity_b, war_config
            )
            if profile_enabled:
                battle_resolution_elapsed += time.perf_counter() - battle_start

            battle_report['supply_spent'] = spent_supply
            engagements = battle_report.get('engagements', 0)
            name_a = getattr(polity_a, 'name', f"Polity {polity_a.id if polity_a else relationship.polity_a}")
            name_b = getattr(polity_b, 'name', f"Polity {polity_b.id if polity_b else relationship.polity_b}")
            if engagements > 0:
                losses = battle_report.get('losses', {})
                loss_a = losses.get(polity_a.id, 0)
                loss_b = losses.get(polity_b.id, 0)
                self._log_event(
                    'battle',
                    f"Battle: {name_a} vs {name_b} | engagements={engagements}, losses={loss_a}/{loss_b}"
                )
            if occupation_events:
                self._log_event(
                    'war',
                    f"{len(occupation_events)} tile(s) newly occupied in {name_a} vs {name_b}"
                )

            post_battle_start = time.perf_counter() if profile_enabled else 0.0
            self._tick_war_exhaustion(relationship, war_config, battle_report)
            self._enforce_war_limits(relationship, polity_a, polity_b, war_config, battle_report)
            if profile_enabled:
                post_battle_elapsed += time.perf_counter() - post_battle_start

        if profile_enabled:
            self._accumulate_tick_profile_section('war:supply_prep', supply_prep_elapsed)
            self._accumulate_tick_profile_section('war:supply_distribution', supply_dist_elapsed)
            self._accumulate_tick_profile_section('war:battle_resolution', battle_resolution_elapsed)
            self._accumulate_tick_profile_section('war:post_battle', post_battle_elapsed)

    def _decay_frontline_supply(self, war_config: Dict[str, float]) -> None:
        if not self.frontline_supply:
            return
        decay = war_config.get('supply_decay_per_tick', 0.85)
        noise_floor = war_config.get('supply_noise_floor', 0.5)
        expired: List[Tuple[int, int]] = []
        for key, value in self.frontline_supply.items():
            new_value = value * decay
            if new_value <= noise_floor:
                expired.append(key)
            else:
                self.frontline_supply[key] = new_value
        for key in expired:
            self.frontline_supply.pop(key, None)

    def _decay_capture_progress(self, war_config: Dict[str, float]) -> None:
        if not self.capture_progress:
            return
        decay = float(war_config.get('capture_progress_decay', 0.94) or 0.94)
        decay = max(0.0, min(0.999, decay))
        noise_floor = float(war_config.get('capture_progress_noise_floor', 0.5) or 0.5)
        to_remove: List[Tuple[int, int]] = []
        for key, value in list(self.capture_progress.items()):
            attacker_id, tile_idx = key
            if tile_idx < 0 or tile_idx >= len(self.tiles):
                to_remove.append(key)
                continue
            new_value = value * decay
            owner = self._get_effective_tile_owner(tile_idx)
            if owner == attacker_id or new_value <= noise_floor:
                to_remove.append(key)
            else:
                self.capture_progress[key] = new_value
        for key in to_remove:
            self.capture_progress.pop(key, None)

    def _prepare_polity_war_supply(self, polity_id: int, war_config: Dict[str, float]) -> None:
        polity = self._get_polity(polity_id)
        if polity is None or not polity.tile_indices:
            self.polity_war_supply.pop(polity_id, None)
            return
        base = war_config.get('supply_base_per_tick', 25.0)
        per_center = war_config.get('supply_per_population_center', 40.0)
        per_development = war_config.get('supply_per_development', 0.01)
        storage_cap = war_config.get('supply_storage_cap', 2500.0)
        center_count = 0
        for center in self.population_centers:
            if center.tile_index >= len(self.tiles):
                continue
            tile = self.tiles[center.tile_index]
            if tile.polity_id == polity_id and tile.occupied_by_polity_id == -1:
                center_count += 1
        development_sum = 0.0
        for tile_idx in polity.tile_indices:
            if 0 <= tile_idx < len(self.tiles):
                tile = self.tiles[tile_idx]
                if tile.occupied_by_polity_id != -1:
                    continue
                development_sum += tile.development
        gain = base + per_center * center_count + development_sum * per_development
        gain *= self._get_trait_value(polity_id, 'TACTICAL_GENIUS', 'supply_generation_multiplier', 1.0)
        exhaustion_penalty = war_config.get('exhaustion_supply_penalty_per_point', 0.0)
        if exhaustion_penalty > 0:
            current_exhaustion = self._get_polity_war_exhaustion(polity_id)
            if current_exhaustion > 0:
                min_multiplier = war_config.get('exhaustion_supply_min_multiplier', 0.25)
                multiplier = max(min_multiplier, 1.0 - current_exhaustion * exhaustion_penalty)
                multiplier = min(1.0, multiplier)
                gain *= multiplier
        reserve = self.polity_war_supply.get(polity_id, 0.0)
        net = reserve + max(0.0, gain)
        penalty_per_tile = war_config.get('occupation_supply_penalty_per_tile', 0.0)
        if penalty_per_tile > 0:
            occupied_tiles = self._count_tiles_occupied_by_polity(polity_id)
            if occupied_tiles > 0:
                net -= penalty_per_tile * occupied_tiles
        self.polity_war_supply[polity_id] = min(storage_cap, max(0.0, net))

    def _get_polity_supply_share(
        self,
        polity_id: int,
        war_counts: Counter,
        war_config: Dict[str, float]
    ) -> float:
        total_supply = self.polity_war_supply.get(polity_id, 0.0)
        if total_supply <= 0:
            return 0.0
        wars = max(1, war_counts.get(polity_id, 1))
        per_war_cap = war_config.get('supply_commitment_cap', 250.0)
        share = min(per_war_cap, total_supply / wars)
        return max(0.0, share)

    def _distribute_supply_to_frontlines(
        self,
        polity_id: int,
        enemy_id: int,
        supply_share: float,
        war_config: Dict[str, float]
    ) -> float:
        frontline_tiles = self._get_frontline_tiles(polity_id, enemy_id)
        if not frontline_tiles:
            return 0.0
        available = min(supply_share, self.polity_war_supply.get(polity_id, 0.0))
        if available <= 0:
            return 0.0
        per_tile = available / len(frontline_tiles)
        reinforcement_decay = war_config.get('supply_stack_decay', 0.65)
        for tile_idx in frontline_tiles:
            key = (polity_id, tile_idx)
            current = self.frontline_supply.get(key, 0.0)
            self.frontline_supply[key] = current * reinforcement_decay + per_tile
            if 0 <= tile_idx < len(self.tiles):
                self.tiles[tile_idx].last_war_supply_tick = self.total_ticks
        return available

    def _apply_capture_pressure(
        self,
        relationship: Relationship,
        attacker_id: int,
        tile_idx: int,
        delta: float,
        war_config: Dict[str, float],
    ) -> bool:
        if delta <= 0 or attacker_id < 0 or tile_idx < 0 or tile_idx >= len(self.tiles):
            return False
        threshold = float(war_config.get('capture_progress_threshold', 100.0) or 100.0)
        control_bleed = float(war_config.get('capture_pressure_control_bleed', 0.35) or 0.0)
        clamp_delta = max(0.0, delta)
        tile = self.tiles[tile_idx]
        if control_bleed > 0:
            drop = control_bleed * clamp_delta
            tile.control_level = max(0, int(tile.control_level - drop))
        key = (attacker_id, tile_idx)
        progress = self.capture_progress.get(key, 0.0) + clamp_delta
        on_cooldown = self._is_tile_on_capture_cooldown(tile_idx)
        if progress >= threshold and not on_cooldown:
            success = self._set_tile_occupation(relationship, tile_idx, attacker_id, war_config)
            if success:
                self.capture_progress.pop(key, None)
                self._reset_capture_progress_for_tile(tile_idx)
                return True
            progress = min(progress, threshold * 0.95)
        elif on_cooldown:
            progress = min(progress, threshold * 0.95)
        self.capture_progress[key] = progress
        return False

    def _reduce_capture_progress(self, attacker_id: int, tile_idx: int, amount: float) -> None:
        if amount <= 0 or not self.capture_progress:
            return
        key = (attacker_id, tile_idx)
        current = self.capture_progress.get(key)
        if current is None:
            return
        new_value = current - amount
        if new_value <= 0.5:
            self.capture_progress.pop(key, None)
        else:
            self.capture_progress[key] = new_value

    def _collect_border_pairs(self, polity_a_id: int, polity_b_id: int) -> List[Tuple[int, int]]:
        return list(self._get_frontline_pair_cache(polity_a_id, polity_b_id).get('pairs', []))

    def _collect_border_pairs_uncached(self, polity_a_id: int, polity_b_id: int) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        controlled_tiles = self._get_polity_controlled_tiles(polity_a_id)
        if not controlled_tiles:
            return pairs
        for tile_idx in controlled_tiles:
            if tile_idx < 0 or tile_idx >= len(self.tiles):
                continue
            tile = self.tiles[tile_idx]
            if tile.is_water:
                continue
            for neighbor_idx in tile.neighbors:
                if neighbor_idx < 0 or neighbor_idx >= len(self.tiles):
                    continue
                neighbor_owner = self._get_effective_tile_owner(neighbor_idx)
                if neighbor_owner != polity_b_id:
                    continue
                key = (min(tile_idx, neighbor_idx), max(tile_idx, neighbor_idx))
                if key not in seen:
                    seen.add(key)
                    pairs.append((tile_idx, neighbor_idx))
        return pairs

    def _get_frontline_pair_cache(self, polity_a_id: int, polity_b_id: int) -> Dict[str, Any]:
        key = self._relationship_key(polity_a_id, polity_b_id)
        version_a = self._get_frontline_polity_version(key[0])
        version_b = self._get_frontline_polity_version(key[1])
        cached = self._war_frontline_pair_cache.get(key)
        if cached and cached.get('version_a') == version_a and cached.get('version_b') == version_b:
            self._increment_workload_counter('frontline_cache_hits')
            return cached

        self._increment_workload_counter('frontline_cache_misses')
        if polity_a_id == key[0]:
            pairs = self._collect_border_pairs_uncached(key[0], key[1])
        else:
            raw_pairs = self._collect_border_pairs_uncached(key[0], key[1])
            pairs = [(tile_b, tile_a) for tile_a, tile_b in raw_pairs]

        frontline_a: Set[int] = set()
        frontline_b: Set[int] = set()
        for tile_a, tile_b in pairs:
            frontline_a.add(tile_a)
            frontline_b.add(tile_b)

        entry = {
            'version_a': version_a,
            'version_b': version_b,
            'pairs': list(pairs),
            'frontline_a': list(frontline_a),
            'frontline_b': list(frontline_b),
            'a_id': polity_a_id,
            'b_id': polity_b_id,
        }
        self._war_frontline_pair_cache[key] = entry
        return entry

    def _calculate_tile_battle_strength(
        self,
        polity_id: int,
        tile_index: int,
        war_config: Dict[str, float]
    ) -> float:
        if tile_index < 0 or tile_index >= len(self.tiles):
            return 0.0
        tile = self.tiles[tile_index]
        pop_factor = war_config.get('strength_population_factor', 0.5)
        dev_factor = war_config.get('strength_development_factor', 0.15)
        control_factor = war_config.get('strength_control_factor', 0.35)
        supply_factor = war_config.get('strength_supply_factor', 1.0)
        supply = self.frontline_supply.get((polity_id, tile_index), 0.0)
        strength = (
            tile.population * pop_factor +
            tile.development * dev_factor +
            tile.control_level * control_factor +
            supply * supply_factor
        )
        return strength * self._get_trait_value(polity_id, 'TACTICAL_GENIUS', 'battle_strength_multiplier', 1.0)

    def _resolve_border_battles(
        self,
        relationship: Relationship,
        polity_a: Polity,
        polity_b: Polity,
        war_config: Dict[str, float]
    ) -> Tuple[List[Tuple[int, int]], Dict[str, Dict[int, float]]]:
        occupation_events: List[Tuple[int, int]] = []
        battle_report = {
            'losses': {
                polity_a.id: 0,
                polity_b.id: 0
            },
            'engagements': 0
        }
        contested_pairs = self._collect_border_pairs(polity_a.id, polity_b.id)
        if not contested_pairs:
            return occupation_events, battle_report
        # Compute capital connections to discourage exclave expansion
        owned_capitals = {polity.capital_tile_index: polity for polity in [polity_a, polity_b] if polity.capital_tile_index >= 0}
        capital_connections = self._map_capital_connected_tiles(owned_capitals)
        capture_margin = war_config.get('capture_margin', 40.0)
        skirmish_loss_ratio = war_config.get('skirmish_population_loss_ratio', 0.01)
        battle_loss_ratio = war_config.get('battle_population_loss_ratio', 0.05)
        dev_loss_per_pop = war_config.get('development_loss_per_population', 0.02)
        capture_control = war_config.get('capture_control_level', 35)
        river_penalty = war_config.get('river_crossing_penalty', 0.0)
        default_flux_threshold = self.config.get('world.rivers.min_flux', 0.12) if self.config else 0.12
        river_flux_threshold = war_config.get('river_flux_threshold', default_flux_threshold)
        for tile_a_idx, tile_b_idx in contested_pairs:
            strength_a = self._calculate_tile_battle_strength(polity_a.id, tile_a_idx, war_config)
            strength_b = self._calculate_tile_battle_strength(polity_b.id, tile_b_idx, war_config)
            if river_penalty > 0.0 and self._tiles_divided_by_river(tile_a_idx, tile_b_idx, river_flux_threshold):
                if strength_a > strength_b:
                    strength_a *= max(0.0, 1.0 - river_penalty)
                elif strength_b > strength_a:
                    strength_b *= max(0.0, 1.0 - river_penalty)
                else:
                    strength_a *= max(0.0, 1.0 - river_penalty * 0.5)
                    strength_b *= max(0.0, 1.0 - river_penalty * 0.5)
            if strength_a <= 0 and strength_b <= 0:
                continue
            battle_report['engagements'] += 1
            total_strength = max(strength_a + strength_b, 1.0)
            margin = strength_a - strength_b
            intense = abs(margin) >= capture_margin
            loss_ratio = battle_loss_ratio if intense else skirmish_loss_ratio
            tile_a = self.tiles[tile_a_idx]
            tile_b = self.tiles[tile_b_idx]
            loss_a = int(tile_a.population * loss_ratio * (strength_b / total_strength))
            loss_b = int(tile_b.population * loss_ratio * (strength_a / total_strength))
            if loss_a > 0:
                tile_a.population = max(0, tile_a.population - loss_a)
                tile_a.development = max(0.0, tile_a.development - loss_a * dev_loss_per_pop)
                battle_report['losses'][polity_a.id] += loss_a
            if loss_b > 0:
                tile_b.population = max(0, tile_b.population - loss_b)
                tile_b.development = max(0.0, tile_b.development - loss_b * dev_loss_per_pop)
                battle_report['losses'][polity_b.id] += loss_b
            # Add development gain to the stronger side (spoils of war)
            gain_ratio = self.config.get('simulation.war.victory_development_gain_ratio', 0.05)
            if strength_a > strength_b:
                tile_a.development += loss_b * dev_loss_per_pop * gain_ratio
            elif strength_b > strength_a:
                tile_b.development += loss_a * dev_loss_per_pop * gain_ratio
            if margin == 0:
                continue
            if margin > 0:
                winner_id = polity_a.id
                loser_id = polity_b.id
                winner_tile_idx = tile_a_idx
                target_tile_idx = tile_b_idx
            else:
                winner_id = polity_b.id
                loser_id = polity_a.id
                winner_tile_idx = tile_b_idx
                target_tile_idx = tile_a_idx

            winner_tile = self.tiles[winner_tile_idx]
            if winner_tile.occupied_by_polity_id == loser_id:
                self._clear_tile_occupation(winner_tile_idx)
                self._reduce_capture_progress(loser_id, winner_tile_idx, war_config.get('capture_pressure_relief', 24.0))
                continue

            if winner_tile_idx not in capital_connections.get(winner_id, set()):
                continue

            margin_ratio = abs(margin) / max(1.0, total_strength)
            base_gain = float(war_config.get('capture_pressure_gain_skirmish', 12.0) or 0.0)
            margin_gain = float(war_config.get('capture_pressure_margin_weight', 55.0) or 0.0) * margin_ratio
            pressure_gain = max(0.0, base_gain + margin_gain)
            if intense:
                pressure_gain += float(war_config.get('capture_pressure_gain_breakthrough', 65.0) or 0.0)

            captured = False
            if pressure_gain > 0:
                captured = self._apply_capture_pressure(
                    relationship,
                    winner_id,
                    target_tile_idx,
                    pressure_gain,
                    war_config,
                )
            if captured:
                occupation_events.append((target_tile_idx, winner_id))
                continue

            relief_scale = float(war_config.get('capture_pressure_relief', 24.0) or 0.0)
            if relief_scale > 0:
                relief_multiplier = float(war_config.get('capture_pressure_relief_breakthrough_multiplier', 1.4) or 1.0)
                if not intense:
                    relief_multiplier = 1.0
                relief_amount = relief_scale * relief_multiplier * max(0.05, margin_ratio)
                self._reduce_capture_progress(loser_id, winner_tile_idx, relief_amount)

        return occupation_events, battle_report

    def _tick_war_exhaustion(
        self,
        relationship: Relationship,
        war_config: Dict[str, float],
        battle_report: Dict[str, Dict[int, float]]
    ) -> None:
        base = war_config.get('exhaustion_base_per_tick', 0.5)
        casualty_weight = war_config.get('exhaustion_loss_weight', 0.05)
        supply_relief = war_config.get('exhaustion_supply_relief', 0.002)
        occupation_weight = war_config.get('occupation_exhaustion_per_tile', 0.0)
        anchor_delta = max(0.0, float(war_config.get('status_quo_exhaustion_delta', 5.0)))
        for polity_id in (relationship.polity_a, relationship.polity_b):
            exhaustion = relationship.war_exhaustion.get(polity_id, 0.0)
            losses = battle_report.get('losses', {}).get(polity_id, 0)
            spent = battle_report.get('supply_spent', {}).get(polity_id, 0.0)
            delta = base + losses * casualty_weight - spent * supply_relief
            if occupation_weight > 0:
                enemy_id = relationship.polity_b if polity_id == relationship.polity_a else relationship.polity_a
                if enemy_id is not None and relationship.occupied_tiles:
                    occupied_tiles = relationship.occupied_tiles.get(enemy_id)
                    if occupied_tiles:
                        total_dev = max(1.0, self._calculate_polity_development_value(enemy_id))
                        occupied_dev = 0.0
                        for tile_idx in occupied_tiles:
                            if 0 <= tile_idx < len(self.tiles):
                                occupied_dev += max(0.0, self.tiles[tile_idx].development)
                        if occupied_dev > 0 and total_dev > 0:
                            delta += occupation_weight * (occupied_dev / total_dev)
            if math.isclose(delta, 0.0, abs_tol=1e-6):
                continue
            new_value = self._clamp_exhaustion(exhaustion + delta)
            if math.isclose(new_value, exhaustion, abs_tol=1e-6):
                continue
            relationship.war_exhaustion[polity_id] = new_value
            if anchor_delta > 0:
                anchor_value = relationship.war_exhaustion_anchor_values.get(polity_id, new_value)
                if abs(new_value - anchor_value) >= anchor_delta:
                    relationship.war_exhaustion_anchor_values[polity_id] = new_value
                    relationship.war_exhaustion_anchor_years[polity_id] = self.current_year

    def _clear_tile_occupation(self, tile_idx: int) -> None:
        if tile_idx < 0 or tile_idx >= len(self.tiles):
            return
        tile = self.tiles[tile_idx]
        occupier = tile.occupied_by_polity_id
        if occupier == -1:
            tile.occupation_relation = None
            return
        previous_owner = tile.polity_id
        relation_key = tile.occupation_relation
        if relation_key and relation_key in self.relationship_lookup:
            relationship = self.relationship_lookup[relation_key]
            if relationship and relationship.occupied_tiles:
                tiles = relationship.occupied_tiles.get(occupier)
                if tiles and tile_idx in tiles:
                    tiles.remove(tile_idx)
                    if not tiles:
                        relationship.occupied_tiles.pop(occupier, None)
        tile.occupied_by_polity_id = -1
        tile.occupation_since_tick = -1
        tile.occupation_relation = None
        self._reset_capture_progress_for_tile(tile_idx)
        self._invalidate_frontline_cache_for_tile(tile_idx, {occupier, previous_owner})

    def _reset_capture_progress_for_tile(self, tile_idx: int) -> None:
        if tile_idx < 0 or not self.capture_progress:
            return
        to_remove = [key for key in self.capture_progress if key[1] == tile_idx]
        for key in to_remove:
            self.capture_progress.pop(key, None)

    def _set_tile_occupation(
        self,
        relationship: Relationship,
        tile_idx: int,
        occupier_id: int,
        war_config: Dict[str, float]
    ) -> bool:
        if tile_idx < 0 or tile_idx >= len(self.tiles):
            return False
        tile = self.tiles[tile_idx]
        if tile.is_water or tile.polity_id == occupier_id:
            return False
        relation_key = self._relationship_key(relationship.polity_a, relationship.polity_b)
        if tile.occupation_relation and tile.occupation_relation != relation_key:
            # Clear occupation from other war before applying
            self._clear_tile_occupation(tile_idx)
        if tile.occupied_by_polity_id == occupier_id:
            return False
        if tile.occupied_by_polity_id != -1:
            self._clear_tile_occupation(tile_idx)
        occupation_control = war_config.get('occupation_control_level', 20.0)
        tile.occupied_by_polity_id = occupier_id
        tile.occupation_since_tick = self.total_ticks
        tile.occupation_relation = relation_key
        tile.control_level = min(tile.control_level, int(occupation_control))
        tile.last_war_supply_tick = self.total_ticks
        relationship.occupied_tiles.setdefault(occupier_id, set()).add(tile_idx)
        self._reset_capture_progress_for_tile(tile_idx)
        self._invalidate_frontline_cache_for_tile(tile_idx, {occupier_id, tile.polity_id})
        return True

    def _clear_occupations_for_relationship(self, relationship: Relationship) -> int:
        if not relationship or not getattr(relationship, 'occupied_tiles', None):
            return 0
        cleared = 0
        for occupier_tiles in list(relationship.occupied_tiles.values()):
            for tile_idx in list(occupier_tiles):
                self._clear_tile_occupation(tile_idx)
                cleared += 1
        relationship.occupied_tiles.clear()
        return cleared

    def _clear_polity_occupations(self, polity_id: int) -> None:
        if polity_id < 0:
            return
        for relationship in self.relationships:
            if relationship is None or not getattr(relationship, 'occupied_tiles', None):
                continue
            tiles = relationship.occupied_tiles.get(polity_id)
            if not tiles:
                continue
            for tile_idx in list(tiles):
                self._clear_tile_occupation(tile_idx)

    def _cleanup_stale_occupations(self) -> None:
        if not self.tiles:
            return
        for tile_idx, tile in enumerate(self.tiles):
            occupier = tile.occupied_by_polity_id
            if occupier == -1:
                continue
            relation_key = tile.occupation_relation
            if not relation_key:
                self._clear_tile_occupation(tile_idx)
                continue
            relationship = self.relationship_lookup.get(tuple(sorted(relation_key)))
            if relationship is None or relationship.status != "war":
                self._clear_tile_occupation(tile_idx)
                continue
            if occupier not in (relationship.polity_a, relationship.polity_b):
                self._clear_tile_occupation(tile_idx)
                continue
            if self._get_polity(occupier) is None:
                self._clear_tile_occupation(tile_idx)

    def _count_tiles_occupied_by_polity(self, polity_id: int) -> int:
        if polity_id < 0:
            return 0
        return len(self._get_tiles_occupied_by_polity(polity_id))

    def _get_tiles_occupied_by_polity(self, polity_id: int) -> List[int]:
        occupied: List[int] = []
        if polity_id < 0:
            return occupied
        for relationship in self.relationships:
            if relationship is None or not getattr(relationship, 'occupied_tiles', None):
                continue
            tiles = relationship.occupied_tiles.get(polity_id)
            if not tiles:
                continue
            for tile_idx in tiles:
                if 0 <= tile_idx < len(self.tiles):
                    occupied.append(tile_idx)
        return occupied

    def _get_polity_controlled_tiles(self, polity_id: int) -> List[int]:
        controlled_set: Set[int] = set()
        polity = self._get_polity(polity_id)
        if polity is not None:
            for tile_idx in getattr(polity, 'tile_indices', []):
                if 0 <= tile_idx < len(self.tiles):
                    tile = self.tiles[tile_idx]
                    occupier = tile.occupied_by_polity_id
                    if occupier not in (-1, polity_id):
                        continue
                    controlled_set.add(tile_idx)
        for tile_idx in self._get_tiles_occupied_by_polity(polity_id):
            controlled_set.add(tile_idx)
        return list(controlled_set)

    def _get_effective_tile_owner(self, tile_idx: int) -> int:
        if tile_idx < 0 or tile_idx >= len(self.tiles):
            return -1
        tile = self.tiles[tile_idx]
        if tile.occupied_by_polity_id != -1:
            return tile.occupied_by_polity_id
        return tile.polity_id

    def _get_polity_war_exhaustion(self, polity_id: int) -> float:
        if polity_id < 0:
            return 0.0
        max_exhaustion = 0.0
        for relationship in self.relationships:
            if relationship is None or relationship.status != "war":
                continue
            if polity_id not in (relationship.polity_a, relationship.polity_b):
                continue
            exhaustion_value = relationship.war_exhaustion.get(polity_id, 0.0)
            if exhaustion_value > max_exhaustion:
                max_exhaustion = exhaustion_value
        return max_exhaustion

    def _transfer_tile_to_polity(
        self,
        tile_idx: int,
        new_owner: int,
        control_level: float,
        war_config: Optional[Dict[str, float]] = None
    ) -> bool:
        if tile_idx < 0 or tile_idx >= len(self.tiles):
            return False
        tile = self.tiles[tile_idx]
        old_owner = tile.polity_id
        if old_owner == new_owner:
            self._clear_tile_occupation(tile_idx)
            return False
        tile.polity_id = new_owner
        tile.control_level = int(max(1.0, control_level))
        tile.last_war_supply_tick = self.total_ticks
        self._clear_tile_occupation(tile_idx)
        self._reset_capture_progress_for_tile(tile_idx)
        self._on_tile_polity_changed(tile_idx, old_owner, new_owner)
        old_polity = self._get_polity(old_owner)
        if old_polity and tile_idx in old_polity.tile_indices:
            old_polity.tile_indices.remove(tile_idx)
        new_polity = self._get_polity(new_owner)
        if new_polity and tile_idx not in new_polity.tile_indices:
            new_polity.tile_indices.append(tile_idx)
        if new_polity and new_polity.primary_culture:
            tile.cultural_makeup = tile.cultural_makeup or {}
            tile.cultural_makeup[new_polity.primary_culture] = tile.cultural_makeup.get(new_polity.primary_culture, 0.0) + 0.25
            self._normalize_tile_culture(tile)
        if war_config:
            self._start_capture_cooldown(tile_idx, war_config)
        else:
            self.capture_cooldowns.pop(tile_idx, None)
        return True

    def _annex_occupied_tiles(
        self,
        relationship: Relationship,
        winner_id: int,
        war_config: Dict[str, float]
    ) -> int:
        if not relationship or not relationship.occupied_tiles:
            return 0
        annex_control = war_config.get('occupation_annex_control_level', 45.0)
        occupied = relationship.occupied_tiles.get(winner_id, set())
        if not occupied:
            return 0
        annexed = 0
        for tile_idx in list(occupied):
            if self._transfer_tile_to_polity(tile_idx, winner_id, annex_control, war_config):
                annexed += 1
        relationship.occupied_tiles[winner_id] = set()
        if annexed > 0:
            self._apply_war_victory_boost(winner_id)
        return annexed

    def _attempt_pressure_annexation(
        self,
        relationship: Relationship,
        winner_id: int,
        loser_id: int,
        war_config: Dict[str, float],
        pressure_bonus: float,
        occupied_seed: Set[int],
    ) -> int:
        if pressure_bonus <= 0:
            return 0
        bonus_factor = min(2.0, 1.0 + pressure_bonus)
        bonus_cap = int(max(0.0, war_config.get('pressure_annexation_cap', 0)))
        if bonus_cap <= 0:
            return 0
        occupation_control = war_config.get('occupation_annex_control_level', 45.0)
        loser = self._get_polity(loser_id)
        if not loser:
            return 0
        loser_tile_set: Set[int] = {
            tile for tile in loser.tile_indices if 0 <= tile < len(self.tiles)
        }
        if not loser_tile_set:
            return 0
        limit = int(min(len(loser_tile_set), round(bonus_cap * bonus_factor)))
        transferred = 0
        if limit <= 0:
            return 0
        queue = deque()
        for tile_idx in occupied_seed:
            if tile_idx in loser_tile_set:
                queue.append(tile_idx)
        if not queue:
            border_tiles = [
                tile_idx
                for tile_idx in loser_tile_set
                if any(
                    0 <= neighbor < len(self.tiles)
                    and self.tiles[neighbor].polity_id == winner_id
                    for neighbor in self.tiles[tile_idx].neighbors
                )
            ]
            random.shuffle(border_tiles)
            for tile_idx in border_tiles:
                queue.append(tile_idx)
        visited: Set[int] = set()
        while queue and transferred < limit:
            tile_idx = queue.popleft()
            if tile_idx in visited or tile_idx not in loser_tile_set:
                continue
            visited.add(tile_idx)
            if self._transfer_tile_to_polity(tile_idx, winner_id, occupation_control, None):
                transferred += 1
                loser_tile_set.discard(tile_idx)
                for neighbor in self.tiles[tile_idx].neighbors:
                    if neighbor in loser_tile_set and neighbor not in visited:
                        queue.append(neighbor)
        if transferred > 0:
            self._apply_war_victory_boost(winner_id)
        return transferred

    def _apply_war_victory_boost(self, polity_id: int) -> None:
        """Apply temporary control boost to all tiles of a polity after winning a war."""
        polity = self._get_polity(polity_id)
        if not polity:
            return
        boost_amount = self.config.get('simulation.war.war_victory_control_boost', 6.0) if self.config else 6.0
        boosted_tiles = 0
        for tile_idx in polity.tile_indices:
            if 0 <= tile_idx < len(self.tiles):
                tile = self.tiles[tile_idx]
                tile.temporary_control_bonus += boost_amount
                boosted_tiles += 1
        if boosted_tiles > 0:
            self._log_event(
                "war",
                f"Polity {polity_id} ({getattr(polity, 'name', 'Unknown')}) gained war victory boost: +{boost_amount}% control to {boosted_tiles} tiles"
            )

    def _apply_war_loss_tolerance_relief(self, polity_id: int, war_intensity: float = 0.0) -> None:
        """Soften cultural rigidity after a loss to model post-war openness."""
        polity = self._get_polity(polity_id)
        if not polity:
            return
        base_bump = self.config.get('simulation.culture.tolerance_war_loss_bump', 0.06) if self.config else 0.06
        try:
            war_intensity = max(0.0, float(war_intensity))
        except (TypeError, ValueError):
            war_intensity = 0.0
        intensity_factor = min(1.0, war_intensity if war_intensity > 0 else 0.5)
        tolerance = self._get_polity_tolerance(polity)
        bump = base_bump * (1.0 - tolerance) * max(0.35, intensity_factor)
        if bump <= 0:
            return
        polity.cultural_tolerance = min(1.0, tolerance + bump)
        self._log_event(
            "culture",
            f"{getattr(polity, 'name', 'A polity')} became more tolerant after losing a war (+{bump:.3f})"
        )

    def _calculate_integration_cost(self, overlord: Optional[Polity], subject: Optional[Polity]) -> float:
        """Compute integration cost based on overlord-to-vassal development ratio (clamped 50-200)."""
        if not overlord or not subject:
            return 400.0  # doubled from 200.0
        overlord_dev = max(1.0, self._calculate_polity_development_value(overlord.id))
        subject_dev = max(1.0, self._calculate_polity_development_value(subject.id))
        # Stronger overlord -> lower cost; equal size -> 400; weaker overlord -> clamp at 400.
        ratio = subject_dev / overlord_dev
        cost = 400.0 * ratio  # doubled from 200.0
        return max(100.0, min(400.0, cost))  # doubled min/max

    def _apply_vassal_integration_burst(self, overlord_id: int, delta: float) -> None:
        """Apply a one-time integration level change to all direct vassals of an overlord."""
        if overlord_id < 0:
            return
        for polity in self.polities:
            if polity and getattr(polity, 'suzerain_id', -1) == overlord_id:
                polity.integration_level = max(0.0, min(100.0, float(getattr(polity, 'integration_level', 0)) + delta))

    def _white_peace_polity_and_dependents(self, polity_id: int, reason: str) -> None:
        """White-peace all wars for a polity and its vassal tree."""
        if polity_id < 0:
            return
        self._white_peace_polity_wars(polity_id, reason=reason)
        polity = self._get_polity(polity_id)
        for child_id in list(getattr(polity, 'vassal_ids', []) or []) if polity else []:
            self._white_peace_polity_and_dependents(child_id, reason)

    def _integrate_vassal_chain(self, overlord_id: int, subject_id: int) -> None:
        """Annex a vassal into its overlord and integrate any nested vassals."""
        overlord = self._get_polity(overlord_id)
        subject = self._get_polity(subject_id)
        if not overlord or not subject or getattr(subject, 'suzerain_id', -1) != overlord_id:
            return

        # Integrate nested vassals first
        for child_id in list(getattr(subject, 'vassal_ids', []) or []):
            self._integrate_vassal_chain(overlord_id, child_id)

        # Clear wars before annexation to avoid dangling relationships
        self._white_peace_polity_wars(subject_id, reason="integration")

        transferred = self._annex_polity_tiles(overlord_id, subject_id, control_level=65.0)
        subject.integration_level = 100.0
        subject.suzerain_id = -1
        subject.vassal_ids = []
        if transferred > 0:
            self._log_event(
                'polity',
                f"{getattr(overlord, 'name', f'Polity {overlord_id}')} integrated {getattr(subject, 'name', f'Polity {subject_id}')} ({transferred} tile(s))"
            )
        if not subject.tile_indices:
            subject.is_active = False
            self._log_event(
                'polity_status',
                f"{getattr(subject, 'name', f'Polity {subject_id}')} ceased to exist after integration into {getattr(overlord, 'name', f'Polity {overlord_id}')}"
            )

    def _process_vassal_integration(
        self,
        yearly_bucket_index: int,
        yearly_bucket_count: int,
        yearly_buckets_per_tick: int,
    ) -> None:
        for polity in self.polities:
            if not polity or not getattr(polity, 'is_active', True):
                continue
            suzerain_id = getattr(polity, 'suzerain_id', -1)
            if suzerain_id < 0:
                continue
            overlord = self._get_polity(suzerain_id)
            if not overlord or not getattr(overlord, 'is_active', True):
                continue
            relationship = self._get_relationship(polity.id, suzerain_id, create=True)
            if not relationship:
                continue
            if not self._relationship_matches_yearly_bucket(
                relationship,
                yearly_bucket_index,
                yearly_bucket_count,
                yearly_buckets_per_tick,
            ):
                continue

            relation_value = self._get_current_relation(polity.id, suzerain_id, relationship)
            delta = 0.0
            delta += 1.0 if relation_value > 0 else -1.0
            if self._polity_has_trait(overlord, 'DIPLOMAT'):
                delta += 1.0
            if self._polity_has_trait(overlord, 'MACHIAVELLIAN'):
                delta += 1.0

            cost = self._calculate_integration_cost(overlord, polity)
            scale = max(0.0, min(1.0, 50.0 / cost))
            gain = delta * scale
            if abs(gain) < 1e-6:
                continue
            polity.integration_level = max(0.0, min(100.0, float(getattr(polity, 'integration_level', 0)) + gain))
            if polity.integration_level >= 100.0:
                self._integrate_vassal_chain(overlord.id, polity.id)

    def _calculate_vassalization_chance(self, winner: Optional[Polity], loser: Optional[Polity]) -> float:
        """Return percent chance (0-100) that a victor opts to vassalize the loser."""
        if not winner or not loser:
            return 0.0

        chance = 0.0

        # Ideological tilt: pluralism vs universalism (centered at 0.5 -> ±50%).
        pluralism_value = 0.5
        culture = self._find_culture_by_name(getattr(winner, 'primary_culture', None))
        if culture:
            try:
                self._ensure_culture_ideas(culture)
                pluralism_value = float(culture.ideas.get('universalism', pluralism_value))
            except (TypeError, ValueError):
                pluralism_value = 0.5
        pluralism_value = max(0.0, min(1.0, pluralism_value))
        chance += (pluralism_value - 0.5) * 100.0  # +50 at max pluralism, -50 at max universalism

        # Cultural difference encourages subject status over annexation.
        winner_culture = getattr(winner, 'primary_culture', None)
        loser_culture = getattr(loser, 'primary_culture', None)
        if winner_culture and loser_culture and winner_culture != loser_culture:
            chance += 20.0

        # Diplomatic and justice-leaning leaders make vassalization more likely.
        if self._polity_has_trait(loser, 'DIPLOMAT'):
            chance += 20.0
        if self._polity_has_trait(winner, 'JUST'):
            chance += 20.0

        # Underdog conquerors are less able to impose vassalage.
        winner_dev = self._calculate_polity_development_value(winner.id)
        loser_dev = self._calculate_polity_development_value(loser.id)
        if winner_dev < loser_dev:
            chance -= 75.0

        return max(0.0, min(100.0, chance))

    def _maybe_adopt_conquered_culture(
        self,
        winner: Optional[Polity],
        loser: Optional[Polity],
        annexed_tiles: int,
        pre_conquest_size: int,
    ) -> None:
        """Small chance for the victor to adopt the conquered culture based on conquest scale and pluralism."""
        if not winner or not loser or annexed_tiles <= 0:
            return
        if not getattr(loser, 'primary_culture', None):
            return
        config_block = self.config.get('simulation.culture.conquest_culture_adoption') if self.config else None
        enabled = True if config_block is None else bool(config_block.get('enabled', True))
        if not enabled:
            return
        stance = (self._get_polity_idea_label(winner, 'universalism') or '').lower()
        if 'universalist' in stance:
            return
        pluralist_base = 0.40
        hybrid_base = 0.20
        max_cap = 0.8
        if isinstance(config_block, dict):
            try:
                pluralist_base = float(config_block.get('pluralist_base_chance', pluralist_base))
            except (TypeError, ValueError):
                pluralist_base = 0.40
            try:
                hybrid_base = float(config_block.get('hybrid_base_chance', hybrid_base))
            except (TypeError, ValueError):
                hybrid_base = 0.20
            try:
                max_cap = float(config_block.get('max_chance_cap', max_cap))
            except (TypeError, ValueError):
                max_cap = 0.8
        if 'pluralist' in stance:
            base = pluralist_base
        elif 'hybrid' in stance:
            base = hybrid_base
        else:
            base = 0.0
        if base <= 0:
            return
        winner_size = max(1, pre_conquest_size)
        ratio = max(0.0, annexed_tiles / winner_size)
        probability = min(max_cap, base * ratio)
        if probability <= 0:
            return
        generation_index = max(1, getattr(winner, 'leader_generation', 1))
        rng = self._get_leader_rng(winner.id, generation_index, f"culture_adopt_{self.current_year}")
        if rng.random() < probability and winner.primary_culture != loser.primary_culture:
            self._apply_polity_primary_culture(winner, loser.primary_culture)
            self._log_event(
                "culture",
                (
                    f"{getattr(winner, 'name', 'A polity')} adopted conquered culture "
                    f"{loser.primary_culture} after annexing {annexed_tiles} tile(s) from {getattr(loser, 'name', 'a rival')} "
                    f"(ratio={ratio:.2f}, p={probability:.2f})"
                ),
            )

    def _resolve_war_victory(
        self,
        relationship: Relationship,
        loser_id: int,
        war_config: Dict[str, float]
    ) -> None:
        winner_id = relationship.polity_b if relationship.polity_a == loser_id else relationship.polity_a
        winner_polity = self._get_polity(winner_id)
        loser_polity = self._get_polity(loser_id)
        winner_name = getattr(winner_polity, 'name', f"Polity {winner_id}")
        loser_name = getattr(loser_polity, 'name', f"Polity {loser_id}")
        loss_intensity = 0.0
        if getattr(relationship, 'war_exhaustion', None):
            loss_intensity = relationship.war_exhaustion.get(loser_id, 0.0)

        vassal_chance = self._calculate_vassalization_chance(winner_polity, loser_polity)
        if vassal_chance > 0.0 and winner_polity and loser_polity:
            generation_index = max(1, getattr(winner_polity, 'leader_generation', 1))
            rng = self._get_leader_rng(winner_polity.id, generation_index, f"vassalize_{self.current_year}")
            if rng.random() < (vassal_chance / 100.0):
                self._update_polity_suzerain(loser_polity, winner_id)
                self._clear_occupations_for_relationship(relationship)
                self._apply_war_loss_tolerance_relief(loser_id, war_intensity=loss_intensity)
                self._log_event(
                    'war',
                    f"{winner_name} imposed vassalage on {loser_name} (p={vassal_chance:.1f}%)",
                )
                # Subject vassals become vassals of the overlord and lose integration progress
                for child_id in list(getattr(loser_polity, 'vassal_ids', []) or []):
                    self._update_polity_suzerain(self._get_polity(child_id), winner_id, refresh_name=False)
                    child = self._get_polity(child_id)
                    if child:
                        child.integration_level = 0.0
                self._end_war(relationship, "vassalized")
                return

        winner_size_before = len(getattr(winner_polity, 'tile_indices', []) or []) if winner_polity else 0
        occupied_snapshot: Dict[int, Set[int]] = {}
        if getattr(relationship, 'occupied_tiles', None):
            occupied_snapshot = {
                occupier: set(tiles)
                for occupier, tiles in relationship.occupied_tiles.items()
            }
        pressure_bonus = self._calculate_annexation_pressure(relationship, winner_id, loser_id, war_config)
        frontline_annexed = self._annex_occupied_tiles(relationship, winner_id, war_config)
        frontline_annexed += self._attempt_pressure_annexation(
            relationship,
            winner_id,
            loser_id,
            war_config,
            pressure_bonus,
            occupied_snapshot.get(winner_id, set()),
        )
        full_annexed = self._attempt_full_annexation(relationship, winner_id, loser_id, war_config)
        self._clear_occupations_for_relationship(relationship)
        total_annexed = frontline_annexed + full_annexed
        if total_annexed > 0:
            self._maybe_adopt_conquered_culture(winner_polity, loser_polity, total_annexed, winner_size_before)
        if full_annexed:
            self._log_event(
                'war',
                f"{winner_name} completely absorbed {loser_name} ({total_annexed} tile(s)) after victory"
            )
            self._log_event(
                'polity_status',
                f"{loser_name} ceased to exist after conquest by {winner_name}"
            )
        elif frontline_annexed:
            self._log_event(
                'war',
                f"{winner_name} annexed {frontline_annexed} tile(s) after defeating {loser_name}"
            )
        self._apply_war_loss_tolerance_relief(loser_id, war_intensity=loss_intensity)
        # Losing a war makes it harder to hold onto subjects (only if the loser survives as a polity)
        if loser_polity and getattr(loser_polity, 'is_active', False):
            self._apply_vassal_integration_burst(loser_id, delta=-10.0)
        self._end_war(relationship, f"Polity {loser_id} exhausted")

    def _perform_peaceful_annexation(self, winner_id: int, loser_id: int, control_level: float = 50.0) -> bool:
        winner = self._get_polity(winner_id)
        loser = self._get_polity(loser_id)
        if not winner or not loser or winner_id == loser_id:
            return False
        transferred = self._annex_polity_tiles(winner_id, loser_id, control_level)
        if transferred:
            self._log_event(
                'polity',
                f"{getattr(winner, 'name', f'Polity {winner_id}')} peacefully annexed "
                f"{getattr(loser, 'name', f'Polity {loser_id}')} ({transferred} tile(s))"
            )
        return transferred > 0

    def _annex_polity_tiles(self, winner_id: int, loser_id: int, control_level: float) -> int:
        winner = self._get_polity(winner_id)
        loser = self._get_polity(loser_id)
        if not winner or not loser or winner_id == loser_id:
            return 0
        tiles_to_transfer = list(loser.tile_indices)
        if not tiles_to_transfer:
            return 0
        transferred = 0
        for tile_idx in tiles_to_transfer:
            if self._transfer_tile_to_polity(tile_idx, winner_id, control_level, None):
                transferred += 1
        if transferred:
            loser.tile_indices = []
            loser.is_active = False
            loser.capital_tile_index = -1
        return transferred

    def _calculate_annexation_pressure(
        self,
        relationship: Relationship,
        winner_id: int,
        loser_id: int,
        war_config: Dict[str, float]
    ) -> float:
        diff = 0.0
        exhaustion_weight = float(war_config.get('annexation_exhaustion_weight', 0.0))
        supply_weight = float(war_config.get('annexation_supply_weight', 0.0))
        if exhaustion_weight > 0:
            winner_exhaustion = relationship.war_exhaustion.get(winner_id, 0.0)
            loser_exhaustion = relationship.war_exhaustion.get(loser_id, 0.0)
            diff += exhaustion_weight * max(0.0, loser_exhaustion - winner_exhaustion)
        if supply_weight > 0:
            winner_supply = self.polity_war_supply.get(winner_id, 0.0)
            loser_supply = self.polity_war_supply.get(loser_id, 0.0)
            diff += supply_weight * max(0.0, winner_supply - loser_supply)
        return diff

    def _attempt_full_annexation(
        self,
        relationship: Relationship,
        winner_id: int,
        loser_id: int,
        war_config: Dict[str, float]
    ) -> int:
        loser = self._get_polity(loser_id)
        winner = self._get_polity(winner_id)
        if not loser or not winner:
            return 0
        loser_tiles = len(getattr(loser, 'tile_indices', []) or [])
        if loser_tiles == 0:
            return 0
        tile_cap = int(war_config.get('full_annexation_tile_cap', 0))
        ratio_threshold = float(war_config.get('full_annexation_ratio', 0.0))
        pressure_bonus = self._calculate_annexation_pressure(relationship, winner_id, loser_id, war_config)
        if self._polity_has_trait(winner, 'DIPLOMAT'):
            trait_settings = self._get_trait_settings('DIPLOMAT')
            tile_cap += int(trait_settings.get('annexation_tile_cap_bonus', 0))
            reduction = float(trait_settings.get('annexation_ratio_reduction', 0.2))
            ratio_threshold *= max(0.25, 1.0 - reduction)
        tile_cap = max(0, tile_cap) + int(round(pressure_bonus))
        ratio_threshold = max(0.0, ratio_threshold * (1.0 - min(0.75, pressure_bonus * 0.05)))
        qualifies = tile_cap > 0 and loser_tiles <= tile_cap
        if not qualifies and ratio_threshold > 0:
            winner_tiles = len(getattr(winner, 'tile_indices', []) or [])
            qualifies = winner_tiles >= loser_tiles * ratio_threshold
        if not qualifies:
            return 0
        control_level = war_config.get('occupation_annex_control_level', 45.0)
        transferred = self._annex_polity_tiles(winner_id, loser_id, control_level)
        if transferred:
            relationship.occupied_tiles.clear()
        return transferred

    def _status_quo_stalemate(self, relationship: Relationship, war_config: Dict[str, float]) -> bool:
        if relationship.status != 'war':
            return False
        years_required = int(war_config.get('status_quo_years_without_exhaustion_change', 5))
        delta_threshold = float(war_config.get('status_quo_exhaustion_delta', 5.0))
        if years_required <= 0 or delta_threshold <= 0:
            return False
        if relationship.war_start_year is None:
            return False
        if (self.current_year - relationship.war_start_year) < years_required:
            return False
        anchor_years = getattr(relationship, 'war_exhaustion_anchor_years', {})
        anchor_values = getattr(relationship, 'war_exhaustion_anchor_values', {})
        for polity_id in (relationship.polity_a, relationship.polity_b):
            anchor_year = anchor_years.get(polity_id)
            if anchor_year is None:
                return False
            if (self.current_year - anchor_year) < years_required:
                return False
            anchor_value = anchor_values.get(polity_id, relationship.war_exhaustion.get(polity_id, 0.0))
            current_value = relationship.war_exhaustion.get(polity_id, 0.0)
            if abs(current_value - anchor_value) >= delta_threshold:
                return False
        return True

    def _enforce_war_limits(
        self,
        relationship: Relationship,
        polity_a: Polity,
        polity_b: Polity,
        war_config: Dict[str, float],
        battle_report: Dict[str, Dict[int, float]]
    ) -> None:
        # Subjects should not resolve non-independence wars separately from their overlord.
        if self._should_defer_to_overlord_resolution(polity_a.id, polity_b.id, relationship):
            return
        if self._should_defer_to_overlord_resolution(polity_b.id, polity_a.id, relationship):
            return

        exhaustion_threshold = war_config.get('exhaustion_end_threshold', 75.0)
        supply_floor = war_config.get('supply_starvation_threshold', 5.0)
        exhaustion_a = relationship.war_exhaustion.get(polity_a.id, 0.0)
        exhaustion_b = relationship.war_exhaustion.get(polity_b.id, 0.0)
        if exhaustion_a >= exhaustion_threshold and exhaustion_b >= exhaustion_threshold:
            self._end_war(relationship, "mutual exhaustion")
            return
        if exhaustion_a >= exhaustion_threshold:
            self._resolve_war_victory(relationship, polity_a.id, war_config)
            return
        if exhaustion_b >= exhaustion_threshold:
            self._resolve_war_victory(relationship, polity_b.id, war_config)
            return
        if self._status_quo_stalemate(relationship, war_config):
            self._end_war(relationship, "status_quo")
            return
        if not polity_a.tile_indices:
            self._eradicate_polity(polity_a.id)
            self._end_war(relationship, f"Polity {polity_a.id} eradicated")
            return
        if not polity_b.tile_indices:
            self._eradicate_polity(polity_b.id)
            self._end_war(relationship, f"Polity {polity_b.id} eradicated")
            return
        if not self._get_frontline_tiles(polity_a.id, polity_b.id):
            self._end_war(relationship, "no remaining fronts")
            return
        if self.polity_war_supply.get(polity_a.id, 0.0) <= supply_floor and \
           self.polity_war_supply.get(polity_b.id, 0.0) <= supply_floor and \
           battle_report.get('engagements', 0) == 0:
            self._end_war(relationship, "stalemate")

    def _get_frontline_tiles(self, polity_id: int, enemy_id: int) -> List[int]:
        entry = self._get_frontline_pair_cache(polity_id, enemy_id)
        if polity_id == entry.get('a_id'):
            return list(entry.get('frontline_a', []))
        if polity_id == entry.get('b_id'):
            return list(entry.get('frontline_b', []))
        return []

    def _eradicate_polity(self, polity_id: int) -> None:
        polity = self._get_polity(polity_id)
        if polity is None:
            return
        # Release subjects before deactivating the overlord
        if getattr(polity, 'vassal_ids', None):
            for vassal_id in list(polity.vassal_ids):
                vassal = self._get_polity(vassal_id)
                if vassal:
                    self._update_polity_suzerain(vassal, None, refresh_name=True)
            polity.vassal_ids = []
        self._release_dynasty_title(polity)
        conquered_culture = getattr(polity, 'primary_culture', None)
        self._clear_polity_occupations(polity_id)
        for tile_idx in list(polity.tile_indices):
            if 0 <= tile_idx < len(self.tiles):
                self._clear_tile_occupation(tile_idx)
                tile = self.tiles[tile_idx]
                old_polity = tile.polity_id
                tile.polity_id = -1
                tile.control_level = 0
                self._on_tile_polity_changed(tile_idx, old_polity, -1)
        polity.tile_indices = []
        polity.is_active = False
        self.population_centers = [center for center in self.population_centers if self.tiles[center.tile_index].polity_id != polity_id]
        self._log_event(
            "polity_status",
            f"Polity {polity_id} has been eradicated from the simulation"
        )

    def _calculate_polity_development_value(self, polity_id: int) -> float:
        polity = self._get_polity(polity_id)
        if polity is None:
            return 0.0
        total = 0.0
        for tile_idx in polity.tile_indices:
            if 0 <= tile_idx < len(self.tiles):
                total += self.tiles[tile_idx].development
        return total

    def get_polity_development_history(self) -> Dict[int, List[Dict[str, Any]]]:
        """Return recorded development history per polity."""
        return self.polity_development_history

    def record_polity_development_snapshot(self) -> None:
        """Manually capture a development snapshot for graph consumers."""
        self._record_polity_development_history()

    def _record_polity_development_history(self) -> None:
        """Capture current development totals so UI overlays can graph them."""
        if not self.polities:
            return
        if not isinstance(self.polity_development_history, dict):
            self.polity_development_history = {}
        season_label = None
        if 0 <= self.current_season < len(self.season_names):
            season_label = self.season_names[self.current_season]
        max_points_conf = self.config.get('simulation.polity.max_development_history_points', 2000) if self.config else 2000
        try:
            max_points = int(max_points_conf)
        except (TypeError, ValueError):
            max_points = 2000
        max_points = max(0, max_points)
        for polity in self.polities:
            if polity is None:
                continue
            series = self.polity_development_history.setdefault(polity.id, [])
            series.append({
                'tick': self.total_ticks,
                'year': self.current_year,
                'season': season_label,
                'development': self._calculate_polity_development_value(polity.id),
                'is_active': getattr(polity, 'is_active', True),
            })
            if max_points > 0 and len(series) > max_points:
                series[:] = series[-max_points:]

    def purge_tileless_polities(self) -> int:
        """Eradicate active tile-less polities once and return newly purged count."""
        purged = 0
        for polity in self.polities:
            if polity is None:
                continue
            is_active = bool(getattr(polity, 'is_active', True))
            has_tiles = bool(getattr(polity, 'tile_indices', None))

            if not is_active:
                # Already eradicated; only sanitize inconsistent remnants if tiles still linger.
                if has_tiles:
                    self._eradicate_polity(polity.id)
                    self.polity_development_history.pop(polity.id, None)
                continue

            if has_tiles:
                continue

            self._eradicate_polity(polity.id)
            self.polity_development_history.pop(polity.id, None)
            purged += 1
        if purged:
            self._log_event('polity_status', f"Purged {purged} tile-less polities")
        return purged

    def _find_culture_by_name(self, name: Optional[str]) -> Optional[Culture]:
        """Return the culture object matching the provided name."""
        if not name:
            return None
        for culture in self.cultures:
            if culture.name == name:
                return culture
        return None

    def _generate_child_culture_color(self, parent_color: Optional[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """Return a color for a new culture, biased toward its parent color."""
        hue_shift = self.config.get('simulation.culture.color_variation_hue', 0.08) if self.config else 0.08
        sat_shift = self.config.get('simulation.culture.color_variation_saturation', 0.12) if self.config else 0.12
        value_shift = self.config.get('simulation.culture.color_variation_value', 0.12) if self.config else 0.12

        if parent_color:
            r, g, b = [c / 255.0 for c in parent_color]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            # Treat near-grayscale parents as having no useful hue reference
            if s >= 0.15:
                h = (h + random.uniform(-hue_shift, hue_shift)) % 1.0
                s = min(1.0, max(0.1, s + random.uniform(-sat_shift, sat_shift)))
                v = min(1.0, max(0.2, v + random.uniform(-value_shift, value_shift)))
                nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
                return (int(nr * 255), int(ng * 255), int(nb * 255))

        # Either no parent color or the parent provided no real saturation
        parent_color = None

        # Fallback color when no parent exists: pick a vivid random hue
        base_hue = random.random()
        saturation = random.uniform(0.55, 0.9)
        value = random.uniform(0.65, 0.95)
        fr, fg, fb = colorsys.hsv_to_rgb(base_hue, saturation, value)
        return (int(fr * 255), int(fg * 255), int(fb * 255))

    def _next_culture_name(self, descriptor: str) -> str:
        """Return the next culture name with a descriptor tag for debugging."""
        suffix = (descriptor or '').strip().upper()
        base_name = f"Culture_{len(self.cultures) + 1}"
        return f"{base_name} ({suffix})" if suffix else base_name

    def to_save_payload(self) -> Dict[str, Any]:
        """Serialize the complete world state into JSON-friendly data."""
        metadata = {
            'width': self.width,
            'height': self.height,
            'sea_level': self.sea_level,
            'current_year': self.current_year,
            'current_season': self.current_season,
            'season_names': list(self.season_names),
            'ticks_per_year': self.ticks_per_year,
            'total_ticks': self.total_ticks,
            'current_map_mode': self.current_map_mode,
            'auto_tick_enabled': self.auto_tick_enabled,
            'tick_interval': self.tick_interval,
            'default_tick_interval': self.default_tick_interval,
            'tick_speed_multiplier': self.tick_speed_multiplier,
            'last_tick_time': self.last_tick_time,
            'water_tiles': self.water_tiles,
            'land_tiles': self.land_tiles,
            'relationship_borders_initialized': self._relationship_borders_initialized,
            'relationship_tick_interval': self.relationship_tick_interval,
            'relationship_yearly_check_buckets': self.relationship_yearly_check_buckets,
            'relationship_yearly_check_interval_years': self.relationship_yearly_check_interval_years,
            'world_seed': self.world_seed,
            'mandate_region_event_years': {
                str(region_id): year
                for region_id, year in (self._mandate_region_event_years or {}).items()
            },
            'language_system': self._serialize_language_system_state()
        }

        def serialize_color(color: Tuple[int, int, int]) -> List[int]:
            return [int(color[0]), int(color[1]), int(color[2])]

        tiles_data: List[Dict[str, Any]] = []
        for tile in self.tiles:
            tiles_data.append({
                'vertices': [[float(x), float(y)] for x, y in tile.vertices],
                'center': [float(tile.center[0]), float(tile.center[1])],
                'elevation': tile.elevation,
                'is_water': tile.is_water,
                'color': serialize_color(tile.color),
                'neighbors': list(tile.neighbors),
                'temperature': tile.temperature,
                'rainfall': tile.rainfall,
                'biome': tile.biome,
                'region_id': tile.region_id,
                'polity_id': tile.polity_id,
                'control_level': tile.control_level,
                'population': tile.population,
                'development': tile.development,
                'cultural_makeup': tile.cultural_makeup,
                'last_tick_deaths': tile.last_tick_deaths,
                'last_culture_spawn_year': tile.last_culture_spawn_year,
                'last_war_supply_tick': tile.last_war_supply_tick,
                'occupied_by_polity_id': tile.occupied_by_polity_id,
                'occupation_since_tick': tile.occupation_since_tick,
                'occupation_relation': list(tile.occupation_relation) if tile.occupation_relation else None,
                'river_ids': list(tile.river_ids),
                'river_flux': tile.river_flux,
                'river_neighbors': [
                    {'neighbor': int(neighbor), 'flux': flux}
                    for neighbor, flux in tile.river_neighbors.items()
                ],
                'is_river_lake': tile.is_river_lake
            })

        regions_data = [
            {
                'id': region.id,
                'name': region.name,
                'color': serialize_color(region.color),
                'tile_indices': list(region.tile_indices),
                'center_tile_index': region.center_tile_index
            }
            for region in self.regions
        ]

        def serialize_leader(leader: Optional[Leader]) -> Optional[Dict[str, Any]]:
            if leader is None:
                return None
            return {
                'name': leader.name,
                'dynasty': getattr(leader, 'dynasty', None),
                'age': leader.age,
                'culture': leader.culture,
                'traits': list(leader.traits or []),
                'accession_year': leader.accession_year,
                'term_years': leader.term_years
            }

        polities_data = [
            {
                'id': polity.id,
                'name': polity.name,
                'name_gloss': getattr(polity, 'name_gloss', None),
                'color': serialize_color(polity.color),
                'leader': serialize_leader(polity.leader),
                'primary_culture': polity.primary_culture,
                'tile_indices': list(polity.tile_indices),
                'suzerain_id': polity.suzerain_id,
                'vassal_ids': list(polity.vassal_ids or []),
                'integration_level': polity.integration_level,
                'is_active': polity.is_active,
                'capital_tile_index': polity.capital_tile_index,
                'leader_generation': polity.leader_generation,
                'title_rank': polity.title_rank,
                'language_name_component': polity.language_name_component,
                'name_from_language': polity.name_from_language,
                'cultural_tolerance': getattr(polity, 'cultural_tolerance', 0.5),
                'dynasty_title_token': getattr(polity, 'dynasty_title_token', None),
                'dynasty_title_source': getattr(polity, 'dynasty_title_source', None),
                'dynasty_title_culture': getattr(polity, 'dynasty_title_culture', None),
                'spawned_from_explosion': getattr(polity, 'spawned_from_explosion', False),
                'spawned_from_succession_crisis': getattr(polity, 'spawned_from_succession_crisis', False),
            }
            for polity in self.polities
        ]

        cultures_data = [
            {
                'name': culture.name,
                'color': serialize_color(culture.color),
                'heritage': culture.heritage,
                'origin_tile_index': culture.origin_tile_index,
                'birth_year': culture.birth_year,
                'home_region_id': culture.home_region_id,
                'immunity_end_year': culture.immunity_end_year,
                'is_initial': culture.is_initial,
                'language_name': culture.language_name,
                'language_parent': culture.language_parent,
                'language_last_update_year': culture.language_last_update_year,
                'language_time_depth': culture.language_time_depth,
                'language_transformation': culture.language_transformation,
                'language_catalog': self._serialize_culture_language(culture.name),
                'ideas': dict(culture.ideas or {}),
            }
            for culture in self.cultures
        ]

        population_centers_data = [
            {
                'tile_index': center.tile_index,
                'name': center.name,
                'phonetic_name': center.phonetic_name,
                'original_threshold': center.original_threshold,
                'established_year': center.established_year,
                'established_tick': center.established_tick,
                'low_control_ticks': center.low_control_ticks
            }
            for center in self.population_centers
        ]

        relationships_data: List[Dict[str, Any]] = []
        for relationship in self.relationships:
            relationships_data.append({
                'polity_a': relationship.polity_a,
                'polity_b': relationship.polity_b,
                'status': relationship.status,
                'war_start_year': relationship.war_start_year,
                'last_war_end_year': relationship.last_war_end_year,
                'last_status_change_tick': relationship.last_status_change_tick,
                'met': relationship.met,
                'shared_border_tiles': relationship.shared_border_tiles,
                'ticking_modifiers': relationship.ticking_modifiers,
                'war_exhaustion': relationship.war_exhaustion,
                'war_exhaustion_anchor_values': relationship.war_exhaustion_anchor_values,
                'war_exhaustion_anchor_years': relationship.war_exhaustion_anchor_years,
                'truce_until_year': relationship.truce_until_year,
                'occupied_tiles': {
                    str(occupier): sorted(list(tiles))
                    for occupier, tiles in relationship.occupied_tiles.items()
                }
            })

        syncretism_data = [
            {
                'tile_index': key[0],
                'culture_pair': list(key[1]),
                'ticks': value
            }
            for key, value in self.syncretism_tracker.items()
        ]

        syncretic_cultures_data = [
            {
                'parents': list(key),
                'name': value
            }
            for key, value in self.syncretic_cultures.items()
        ]

        history_payload = {
            str(polity_id): [
                {
                    'tick': entry.get('tick', 0),
                    'year': entry.get('year', 0),
                    'season': entry.get('season'),
                    'development': entry.get('development', 0.0),
                    'is_active': entry.get('is_active', True),
                }
                for entry in entries
            ]
            for polity_id, entries in self.polity_development_history.items()
            if entries
        }

        frontline_supply_data = [
            {
                'polity_id': key[0],
                'tile_index': key[1],
                'value': value
            }
            for key, value in self.frontline_supply.items()
        ]

        capture_progress_data = [
            {
                'attacker_id': key[0],
                'tile_index': key[1],
                'value': value,
            }
            for key, value in self.capture_progress.items()
        ]

        payload = {
            'metadata': metadata,
            'tiles': tiles_data,
            'regions': regions_data,
            'polities': polities_data,
            'cultures': cultures_data,
            'population_centers': population_centers_data,
            'relationships': relationships_data,
            'polity_war_supply': {str(pid): supply for pid, supply in self.polity_war_supply.items()},
            'frontline_supply': frontline_supply_data,
            'capture_cooldowns': {str(idx): ticks for idx, ticks in self.capture_cooldowns.items()},
            'capture_progress': capture_progress_data,
            'syncretism_tracker': syncretism_data,
            'syncretic_cultures': syncretic_cultures_data,
            'region_name_tokens': {str(rid): token for rid, token in self.region_name_tokens.items()},
            'polity_development_history': history_payload,
            'rivers': [
                {
                    'id': river.id,
                    'tile_indices': list(river.tile_indices),
                    'points': [[float(x), float(y)] for x, y in river.points],
                    'flux': river.flux,
                    'terminates_in_sea': river.terminates_in_sea,
                    'terminates_in_lake': river.terminates_in_lake,
                }
                for river in self.rivers
            ],
        }
        return payload

    @classmethod
    def from_save_payload(cls, payload: Dict[str, Any], config=None) -> 'World':
        """Create a world instance from serialized save data."""
        metadata = payload.get('metadata', {})
        width = metadata.get('width')
        height = metadata.get('height')
        if width is None or height is None:
            default_width = config.get('world.width', 800) if config else 800
            default_height = config.get('world.height', 600) if config else 600
            width = width or default_width
            height = height or default_height
        world = cls(int(width), int(height), config)
        world._apply_save_payload(payload)
        return world

    def _apply_save_payload(self, payload: Dict[str, Any]) -> None:
        """Apply serialized data to this world instance."""
        metadata = payload.get('metadata', {})
        self.width = metadata.get('width', self.width)
        self.height = metadata.get('height', self.height)
        self.sea_level = metadata.get('sea_level', self.sea_level)
        self.current_year = metadata.get('current_year', self.current_year)
        self.current_season = metadata.get('current_season', self.current_season)
        self.world_seed = metadata.get('world_seed', self.world_seed)
        season_names = metadata.get('season_names')
        if season_names:
            self.season_names = list(season_names)
            self.ticks_per_year = len(self.season_names)
        self.ticks_per_year = metadata.get('ticks_per_year', self.ticks_per_year)
        self.total_ticks = metadata.get('total_ticks', self.total_ticks)
        self.current_map_mode = metadata.get('current_map_mode', self.current_map_mode)
        self.auto_tick_enabled = metadata.get('auto_tick_enabled', self.auto_tick_enabled)
        self.tick_interval = metadata.get('tick_interval', self.tick_interval)
        self.default_tick_interval = metadata.get('default_tick_interval', self.default_tick_interval)
        self.tick_speed_multiplier = metadata.get('tick_speed_multiplier', self.tick_speed_multiplier)
        self.last_tick_time = metadata.get('last_tick_time', self.last_tick_time)
        self.water_tiles = metadata.get('water_tiles', self.water_tiles)
        self.land_tiles = metadata.get('land_tiles', self.land_tiles)
        self._relationship_borders_initialized = metadata.get('relationship_borders_initialized', self._relationship_borders_initialized)
        self.relationship_tick_interval = metadata.get('relationship_tick_interval', self.relationship_tick_interval)
        saved_relationship_buckets = metadata.get('relationship_yearly_check_buckets', self.relationship_yearly_check_buckets)
        try:
            parsed_relationship_buckets = int(saved_relationship_buckets)
        except (TypeError, ValueError):
            parsed_relationship_buckets = self.relationship_yearly_check_buckets
        self.relationship_yearly_check_buckets = max(1, parsed_relationship_buckets)
        saved_relationship_year_cadence = metadata.get('relationship_yearly_check_interval_years', self.relationship_yearly_check_interval_years)
        try:
            parsed_relationship_year_cadence = int(saved_relationship_year_cadence)
        except (TypeError, ValueError):
            parsed_relationship_year_cadence = self.relationship_yearly_check_interval_years
        self.relationship_yearly_check_interval_years = max(1, parsed_relationship_year_cadence)
        mandate_schedule = metadata.get('mandate_region_event_years', {})
        self._mandate_region_event_years = {}
        if isinstance(mandate_schedule, dict):
            for region_id, year in mandate_schedule.items():
                try:
                    region_key = int(region_id)
                    next_year = int(year)
                except (TypeError, ValueError):
                    continue
                if region_key < 0:
                    continue
                self._mandate_region_event_years[region_key] = max(1, next_year)
        language_system_payload = metadata.get('language_system') or {}
        saved_interval = language_system_payload.get('time_shift_years')
        if saved_interval is not None:
            try:
                self.language_time_shift_years = max(1, int(saved_interval))
            except (TypeError, ValueError):
                pass

        def to_tuple_color(color_list: Optional[List[int]]) -> Tuple[int, int, int]:
            if not color_list:
                return (0, 0, 0)
            return (int(color_list[0]), int(color_list[1]), int(color_list[2]))

        self.tiles = []
        self.water_tiles = 0
        self.land_tiles = 0
        for tile_data in payload.get('tiles', []):
            river_neighbor_entries = tile_data.get('river_neighbors', []) or []
            river_neighbor_map: Dict[int, float] = {}
            if isinstance(river_neighbor_entries, dict):
                for key, value in river_neighbor_entries.items():
                    try:
                        neighbor_idx = int(key)
                        river_neighbor_map[neighbor_idx] = float(value)
                    except (TypeError, ValueError):
                        continue
            else:
                for entry in river_neighbor_entries:
                    if not isinstance(entry, dict):
                        continue
                    neighbor_idx = entry.get('neighbor')
                    flux_value = entry.get('flux', 0.0)
                    if neighbor_idx is None:
                        continue
                    try:
                        neighbor_idx_int = int(neighbor_idx)
                    except (TypeError, ValueError):
                        continue
                    river_neighbor_map[neighbor_idx_int] = float(flux_value)
            tile = Tile(
                vertices=[tuple(vertex) for vertex in tile_data.get('vertices', [])],
                center=tuple(tile_data.get('center', (0.0, 0.0))),
                elevation=tile_data.get('elevation', 0.0),
                is_water=tile_data.get('is_water', False),
                color=to_tuple_color(tile_data.get('color')),
                neighbors=tile_data.get('neighbors', []),
                temperature=tile_data.get('temperature', 0.0),
                rainfall=tile_data.get('rainfall', 0.0),
                biome=tile_data.get('biome', 'unknown'),
                region_id=tile_data.get('region_id', -1),
                polity_id=tile_data.get('polity_id', -1),
                control_level=tile_data.get('control_level', 50),
                population=tile_data.get('population', 0),
                development=tile_data.get('development', 0.0),
                cultural_makeup=tile_data.get('cultural_makeup', {}) or {},
                last_tick_deaths=tile_data.get('last_tick_deaths', 0),
                last_culture_spawn_year=tile_data.get('last_culture_spawn_year'),
                last_war_supply_tick=tile_data.get('last_war_supply_tick', -1),
                occupied_by_polity_id=tile_data.get('occupied_by_polity_id', -1),
                occupation_since_tick=tile_data.get('occupation_since_tick', -1),
                occupation_relation=tuple(tile_data.get('occupation_relation')) if tile_data.get('occupation_relation') else None,
                river_ids=tile_data.get('river_ids', []) or [],
                river_flux=float(tile_data.get('river_flux', 0.0) or 0.0),
                river_neighbors=river_neighbor_map,
                is_river_lake=bool(tile_data.get('is_river_lake', False))
            )
            self.tiles.append(tile)
            if tile.is_water:
                self.water_tiles += 1
            else:
                self.land_tiles += 1

        self.rivers = []
        raw_rivers = payload.get('rivers', []) or []
        for river_data in raw_rivers:
            points_payload = river_data.get('points', []) or []
            river = River(
                id=river_data.get('id', len(self.rivers)),
                tile_indices=river_data.get('tile_indices', []) or [],
                points=[tuple(point) for point in points_payload],
                flux=float(river_data.get('flux', 0.0) or 0.0),
                terminates_in_sea=bool(river_data.get('terminates_in_sea', False)),
                terminates_in_lake=bool(river_data.get('terminates_in_lake', False)),
            )
            self.rivers.append(river)
        if self.rivers:
            self._rebuild_river_neighbors_from_paths()
        else:
            for tile in self.tiles:
                tile.river_ids = tile.river_ids or []
                tile.river_neighbors = tile.river_neighbors or {}
        self.river_lakes = {
            idx for idx, tile in enumerate(self.tiles) if getattr(tile, 'is_river_lake', False)
        }

        self.regions = [
            Region(
                id=region_data.get('id', -1),
                name=region_data.get('name', f"Region_{region_data.get('id', -1)}"),
                color=to_tuple_color(region_data.get('color')),
                tile_indices=region_data.get('tile_indices', []),
                center_tile_index=region_data.get('center_tile_index', -1)
            )
            for region_data in payload.get('regions', [])
        ]

        raw_region_tokens = payload.get('region_name_tokens', {}) or {}
        region_token_map: Dict[int, str] = {}
        for key, value in raw_region_tokens.items():
            try:
                region_id = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, str) and value:
                region_token_map[region_id] = value
        self.region_name_tokens = region_token_map
        # Check for English words in region tokens and rename if needed
        for region_id, token in list(self.region_name_tokens.items()):
            if self._token_contains_english_words(token):
                old_token = token
                new_token = self._generate_non_english_region_token(region_id)
                self.region_name_tokens[region_id] = new_token
                self._log_event(
                    "culture_debug",
                    f"[region] Renamed region token from '{old_token}' to '{new_token}' due to English word filtering"
                )
        for region in self.regions:
            token = self.region_name_tokens.get(region.id)
            if token:
                self._apply_region_display_name(region.id, token)
        self._initialize_region_seed_words_from_existing()

        self.cultures = [
            Culture(
                name=culture_data.get('name', f"Culture_{idx}"),
                color=to_tuple_color(culture_data.get('color')),
                heritage=culture_data.get('heritage', {}),
                origin_tile_index=culture_data.get('origin_tile_index', -1),
                birth_year=culture_data.get('birth_year', 0),
                home_region_id=culture_data.get('home_region_id', -1),
                immunity_end_year=culture_data.get('immunity_end_year'),
                is_initial=culture_data.get('is_initial', False),
                language_name=culture_data.get('language_name'),
                language_parent=culture_data.get('language_parent'),
                language_last_update_year=culture_data.get('language_last_update_year'),
                language_time_depth=culture_data.get('language_time_depth', 0),
                language_transformation=culture_data.get('language_transformation'),
                ideas=culture_data.get('ideas', {}),
            )
            for idx, culture_data in enumerate(payload.get('cultures', []), start=1)
        ]

        self.culture_languages = {}
        for culture, culture_data in zip(self.cultures, payload.get('cultures', [])):
            language_catalog = self._deserialize_language_catalog(culture_data.get('language_catalog'))
            if language_catalog:
                self.culture_languages[culture.name] = language_catalog
        self._rebuild_culture_name_registry()
        self._backfill_culture_ideas()

        def deserialize_leader(leader_data: Optional[Dict[str, Any]]) -> Optional[Leader]:
            if not leader_data:
                return None
            return Leader(
                name=leader_data.get('name', 'Unknown'),
                dynasty=leader_data.get('dynasty'),
                age=leader_data.get('age', 0),
                culture=leader_data.get('culture', 'Unknown'),
                traits=leader_data.get('traits', []) or [],
                accession_year=leader_data.get('accession_year', 0),
                term_years=leader_data.get('term_years', 0)
            )

        self.polities = []
        for polity_data in payload.get('polities', []):
            leader = deserialize_leader(polity_data.get('leader'))
            polity = Polity(
                id=polity_data.get('id', len(self.polities)),
                name=polity_data.get('name', f"Polity_{len(self.polities)}"),
                name_gloss=polity_data.get('name_gloss'),
                color=to_tuple_color(polity_data.get('color')),
                leader=leader if leader else Leader(name='Unknown', age=0, culture='Unknown'),
                primary_culture=polity_data.get('primary_culture'),
                tile_indices=polity_data.get('tile_indices', []),
                suzerain_id=polity_data.get('suzerain_id', -1),
                vassal_ids=polity_data.get('vassal_ids', []),
                integration_level=polity_data.get('integration_level', 100),
                is_active=polity_data.get('is_active', True),
                capital_tile_index=polity_data.get('capital_tile_index', -1),
                leader_generation=polity_data.get('leader_generation', 0),
                title_rank=polity_data.get('title_rank'),
                language_name_component=polity_data.get('language_name_component'),
                name_from_language=polity_data.get('name_from_language', False),
                cultural_tolerance=polity_data.get('cultural_tolerance', 0.5),
                dynasty_title_token=polity_data.get('dynasty_title_token'),
                dynasty_title_source=polity_data.get('dynasty_title_source'),
                dynasty_title_culture=polity_data.get('dynasty_title_culture'),
                spawned_from_explosion=polity_data.get('spawned_from_explosion', False),
                spawned_from_succession_crisis=polity_data.get('spawned_from_succession_crisis', False),
            )
            self.polities.append(polity)

        # Rebuild dynasty title reservations for catalog-backed titles
        self._culture_dynasty_title_reservations = {}
        for polity in self.polities:
            token = getattr(polity, 'dynasty_title_token', None)
            if not token or getattr(polity, 'dynasty_title_source', None) != 'catalog':
                continue
            culture_key = getattr(polity, 'dynasty_title_culture', None) or getattr(polity, 'primary_culture', None)
            if not culture_key:
                continue
            reservations = self._culture_dynasty_title_reservations.setdefault(culture_key, {})
            reservations[token.lower()] = {'owner': polity.id, 'token': token}

        self.population_centers = [
            PopulationCenter(
                tile_index=center_data.get('tile_index', -1),
                name=center_data.get('name', 'Settlement'),
                phonetic_name=center_data.get('phonetic_name'),
                original_threshold=center_data.get('original_threshold', 0),
                established_year=center_data.get('established_year', 0),
                established_tick=center_data.get('established_tick', 0),
                low_control_ticks=center_data.get('low_control_ticks', 0)
            )
            for center_data in payload.get('population_centers', [])
        ]

        self.relationships = []
        for rel_data in payload.get('relationships', []):
            relationship = Relationship(
                polity_a=rel_data.get('polity_a', -1),
                polity_b=rel_data.get('polity_b', -1),
                status=rel_data.get('status', 'peace'),
                war_start_year=rel_data.get('war_start_year'),
                last_war_end_year=rel_data.get('last_war_end_year'),
                last_status_change_tick=rel_data.get('last_status_change_tick', 0),
                met=rel_data.get('met', False),
                shared_border_tiles=rel_data.get('shared_border_tiles', 0),
                ticking_modifiers={},
                war_exhaustion={},
                war_exhaustion_anchor_values={},
                war_exhaustion_anchor_years={},
                truce_until_year=rel_data.get('truce_until_year')
            )
            ticking_modifiers = {}
            for key, value in rel_data.get('ticking_modifiers', {}).items():
                try:
                    ticking_modifiers[int(key)] = value
                except (TypeError, ValueError):
                    continue
            relationship.ticking_modifiers = ticking_modifiers
            war_exhaustion = {}
            for key, value in rel_data.get('war_exhaustion', {}).items():
                try:
                    war_exhaustion[int(key)] = value
                except (TypeError, ValueError):
                    continue
            relationship.war_exhaustion = war_exhaustion
            anchor_values: Dict[int, float] = {}
            for key, value in rel_data.get('war_exhaustion_anchor_values', {}).items():
                try:
                    anchor_values[int(key)] = max(0.0, float(value))
                except (TypeError, ValueError):
                    continue
            if not anchor_values:
                anchor_values = dict(war_exhaustion)
            relationship.war_exhaustion_anchor_values = anchor_values
            anchor_years: Dict[int, int] = {}
            for key, value in rel_data.get('war_exhaustion_anchor_years', {}).items():
                try:
                    anchor_years[int(key)] = int(value)
                except (TypeError, ValueError):
                    continue
            if not anchor_years:
                default_year = rel_data.get('war_start_year', self.current_year)
                for polity_id in (relationship.polity_a, relationship.polity_b):
                    anchor_years[polity_id] = default_year if default_year is not None else self.current_year
            relationship.war_exhaustion_anchor_years = anchor_years
            occupied_tiles: Dict[int, Set[int]] = {}
            for occupier, tiles in rel_data.get('occupied_tiles', {}).items():
                try:
                    occupier_id = int(occupier)
                except (TypeError, ValueError):
                    continue
                occupied_tiles[occupier_id] = set(int(t) for t in tiles)
            relationship.occupied_tiles = occupied_tiles
            self._ensure_relationship_fields(relationship)
            self.relationships.append(relationship)

        self.relationship_lookup = {}
        for relationship in self.relationships:
            key = self._relationship_key(relationship.polity_a, relationship.polity_b)
            self.relationship_lookup[key] = relationship
        self._war_frontline_pair_cache = {}
        self._war_frontline_polity_versions = {}

        self.polity_war_supply = {
            int(pid): float(value)
            for pid, value in payload.get('polity_war_supply', {}).items()
        }

        self.frontline_supply = {}
        for entry in payload.get('frontline_supply', []):
            polity_id = entry.get('polity_id')
            tile_idx = entry.get('tile_index')
            value = entry.get('value', 0.0)
            if polity_id is None or tile_idx is None:
                continue
            self.frontline_supply[(int(polity_id), int(tile_idx))] = float(value)

        self.capture_cooldowns = {
            int(idx): int(ticks)
            for idx, ticks in payload.get('capture_cooldowns', {}).items()
        }

        self.capture_progress = {}
        for entry in payload.get('capture_progress', []):
            attacker_id = entry.get('attacker_id')
            tile_idx = entry.get('tile_index')
            value = entry.get('value', 0.0)
            if attacker_id is None or tile_idx is None:
                continue
            self.capture_progress[(int(attacker_id), int(tile_idx))] = float(value)

        self.syncretism_tracker = {}
        for entry in payload.get('syncretism_tracker', []):
            tile_idx = entry.get('tile_index')
            culture_pair = entry.get('culture_pair', [])
            ticks = entry.get('ticks', 0)
            if tile_idx is None or len(culture_pair) != 2:
                continue
            key = (int(tile_idx), (culture_pair[0], culture_pair[1]))
            self.syncretism_tracker[key] = int(ticks)

        self.syncretic_cultures = {}
        for entry in payload.get('syncretic_cultures', []):
            parents = entry.get('parents', [])
            name = entry.get('name')
            if not parents or name is None:
                continue
            self.syncretic_cultures[frozenset(parents)] = name
        self._syncretic_parent_lookup = {
            name: parents for parents, name in self.syncretic_cultures.items()
        }

        raw_history = payload.get('polity_development_history', {}) or {}
        history: Dict[int, List[Dict[str, Any]]] = {}
        for key, series in raw_history.items():
            try:
                polity_id = int(key)
            except (TypeError, ValueError):
                continue
            cleaned: List[Dict[str, Any]] = []
            for entry in series or []:
                if not isinstance(entry, dict):
                    continue
                cleaned.append({
                    'tick': int(entry.get('tick', 0) or 0),
                    'year': int(entry.get('year', 0) or 0),
                    'season': entry.get('season'),
                    'development': float(entry.get('development', 0.0) or 0.0),
                    'is_active': bool(entry.get('is_active', True)),
                })
            if cleaned:
                history[polity_id] = cleaned
        self.polity_development_history = history

        self._restore_language_system_state(language_system_payload)
        self.initialize_leader_system()
        if self.polities and not any(self.polity_development_history.values()):
            self.record_polity_development_snapshot()