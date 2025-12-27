High-Level Implementation Plan: Diachronic Language Evolution System
Project Overview

Build a Python system that applies systematic phonological and typological transformations to language catalogs, supporting both time-based evolution and contact-induced change through substrate/superstratum relationships.

Core Components
1. Data Structures
LanguageCatalog Class
pythonclass LanguageCatalog:
    """
    Represents a complete language snapshot with vocabulary and typology.
    """
    def __init__(self, name, words, typology, metadata=None):
        self.name = str              # Language identifier
        self.words = List[Word]      # Vocabulary samples (minimum 25)
        self.typology = Typology     # Phonological/grammatical parameters
        self.metadata = dict         # Optional: time period, geographic location, parent language

Word Class
pythonclass Word:
    """
    Individual lexical item with phonetic representation and meaning.
    """
    def __init__(self, phonetic_form, gloss, category=None):
        self.phonetic_form = str     # IPA representation
        self.gloss = str             # English translation/meaning
        self.category = str          # Optional: 'geographic', 'military', 'personal_name', etc.
        self.is_irregular = bool     # Flag for high-frequency forms resistant to change

Typology Class
pythonclass Typology:
    """
    Encodes systematic phonological and structural patterns of a language.
    """
    def __init__(self):
        # PHONEME INVENTORY
        self.consonants = set()           # Set of consonant phonemes (IPA)
        self.vowels = set()               # Set of vowel phonemes (IPA)
        
        # PHONOTACTIC CONSTRAINTS
        self.max_onset_complexity = int   # Max consonants in syllable onset
        self.max_coda_complexity = int    # Max consonants in syllable coda
        self.allowed_onsets = List[str]   # Permitted onset clusters
        self.allowed_codas = List[str]    # Permitted coda clusters
        self.syllable_patterns = List[str] # e.g., ['CV', 'CVC', 'CCVC']
        
        # PROSODY
        self.stress_pattern = str         # 'initial', 'penultimate', 'weight-sensitive', etc.
        self.has_tone = bool
        
        # ACTIVE SOUND LAWS
        self.sound_changes = List[SoundChange]  # Ordered list of transformation rules
        
        # MORPHOPHONOLOGY
        self.lenition_contexts = List[str]      # Environments triggering weakening
        self.fortition_contexts = List[str]     # Environments triggering strengthening
        self.vowel_harmony = bool
        self.consonant_mutation = bool

SoundChange Class
pythonclass SoundChange:
    """
    Represents a single phonological transformation rule.
    """
    def __init__(self, source, target, environment_before="", environment_after="", condition=None):
        self.source = str           # Pattern to match (can be regex or IPA)
        self.target = str           # Replacement pattern
        self.env_before = str       # Required preceding context
        self.env_after = str        # Required following context
        self.condition = callable   # Optional: function for complex conditions
        self.probability = float    # Optional: for stochastic changes (default 1.0)
    
    def apply(self, word):
        """Apply this sound change to a word."""
        pass

2. Core Operations

TypologyBlender Module
pythondef blend_typologies(primary_typology, influence_typology, influence_weight=0.3):
    """
    Create syncretic typology from two languages.
    
    Args:
        primary_typology: The dominant/superstrate language typology
        influence_typology: The influencing/substrate language typology
        influence_weight: Float 0-1, how much substrate influences result
    
    Returns:
        New Typology object representing the blended system
    
    Logic:
        - Phoneme inventories: UNION (expand to include both)
        - Phonotactic constraints: WEIGHTED BLEND (loosen restrictions)
        - Stress pattern: CATEGORICAL SELECTION (weighted random choice)
        - Sound changes: COMBINE with substrate rules applied first
    """
    pass

DiachronicEvolution Module

pythondef evolve_catalog(catalog, target_typology):
    """
    Transform an entire language catalog through a new typology.
    
    Args:
        catalog: LanguageCatalog to transform
        target_typology: Typology to apply (from time evolution or blending)
    
    Returns:
        New LanguageCatalog with transformed words and updated typology
    
    Process:
        1. Apply sound changes in order to each word
        2. Enforce phonotactic constraints (repair strategies)
        3. Apply stress pattern
        4. Handle irregular forms specially (resistance or analogical leveling)
    """
    pass

def apply_time_shift(typology, time_depth):
    """
    Evolve a typology forward in time.
    
    Args:
        typology: Starting typology
        time_depth: Relative time units (could represent centuries)
    
    Returns:
        Evolved Typology with drift applied
    
    Transformations (probabilistic):
        - Lenition tendencies (voicing, spirantization, deletion)
        - Vowel mergers or splits
        - Consonant chain shifts
        - Stress pattern drift
        - Phonotactic simplification or complexification
    """
    pass

ContactScenario Module

pythondef apply_substrate_influence(superstrate_catalog, substrate_typology, influence_strength=0.2):
    """
    Model language contact with substrate influence.
    
    Args:
        superstrate_catalog: Dominant language catalog
        substrate_typology: Influencing substrate typology
        influence_strength: How strongly substrate affects superstrate
    
    Returns:
        New catalog with contact effects applied
    
    Process:
        1. Blend typologies (substrate as influence)
        2. Apply blended typology to superstrate vocabulary
        3. Optional: add substrate loanwords to catalog
    """
    pass

def apply_adstratum_influence(catalog_a, catalog_b, bidirectional_weight=0.5):
    """
    Model mutual influence between peer languages (trade contact, bilingualism).
    
    Returns:
        Tuple of (modified_catalog_a, modified_catalog_b)
    """
    pass

3. Data I/O

JSON Schema for Language Catalogs
json{
    "name": "Proto-Germanic",
    "metadata": {
        "time_period": "0",
        "associated_culture": "Proto-Indo-European",
    },
    "typology": {
        "consonants": ["p", "t", "k", "b", "d", "g", "f", "θ", "s", "h", "m", "n", "r", "l", "w", "j"],
        "vowels": ["i", "e", "a", "o", "u", "ī", "ē", "ā", "ō", "ū"],
        "max_onset_complexity": 3,
        "max_coda_complexity": 2,
        "syllable_patterns": ["V", "CV", "CVC", "CCVC"],
        "stress_pattern": "initial",
        "sound_changes": [
            {
                "name": "Grimm's Law p>f",
                "source": "p",
                "target": "f",
                "env_after": "[aeiou]"
            }
        ]
    },
    "words": [
        {
            "phonetic_form": "faðēr",
            "gloss": "father",
            "category": "kinship"
        },
        {
            "phonetic_form": "mōdēr",
            "gloss": "mother",
            "category": "kinship"
        }
    ]
}

Loader/Saver Functions

pythondef load_catalog_from_json(filepath):
    """Load LanguageCatalog from JSON file."""
    pass

def save_catalog_to_json(catalog, filepath):
    """Export LanguageCatalog to JSON file."""
    pass

def load_all_catalogs(directory_path):
    """Load all JSON catalogs from a directory."""
    pass

4. Transformation Pipeline

Main Workflow

pythondef transform_language(
    input_catalog,
    transformation_type='time_evolution',  # or 'substrate', 'adstratum'
    **kwargs
):
    """
    Main entry point for language transformation.
    
    Examples:
        # Time evolution
        transform_language(pie_catalog, 'time_evolution', time_depth=5)
        
        # Substrate influence
        transform_language(
            germanic_catalog, 
            'substrate', 
            substrate_typology=celtic_typology,
            influence_strength=0.25
        )
    """
    pass
```

## Implementation Phases

### Phase 1: Core Data Structures
- Implement `Word`, `LanguageCatalog`, `Typology`, `SoundChange` classes
- Create JSON loader/saver functions
- Validate against existing PIE-derived language JSON data

### Phase 2: Basic Sound Change Engine
- Implement `SoundChange.apply()` with regex-based pattern matching
- Handle IPA tokenization (use `segments` library if helpful)
- Test with simple rules (Grimm's Law, Great Vowel Shift examples)

### Quick Console Harness for Smoke Tests
- Module `linguistics.console` now loads every JSON catalog in `init_catalogs`, applies the configured sound changes, and prints paired summaries for the base and evolved variants.
- Run it from the repo root after activating the virtualenv:
    - `python -m linguistics.console --catalog-dir init_catalogs --seed 1337 --max-examples 6`
- Flags:
    - `--catalog-dir` points at any folder of catalogs (defaults to `init_catalogs`).
    - `--seed` keeps probabilistic rules deterministic between runs.
    - `--max-examples` controls how many words show up per summary line.
    - `--limit` restricts how many catalogs print (useful when the directory grows).
    - `--show-diffs` appends `Δ gloss: original -> evolved` lines for each catalog so it is obvious which lexemes changed.
    - `--diff-limit` constrains how many changed words display per catalog when `--show-diffs` is enabled (default 5).
- Use this harness before writing automated tests to ensure the JSON catalogs deserialize correctly and that the basic sound-change pipeline behaves as expected.

### Phase 3: Phonotactic Enforcement
- Implement syllable structure validation
- Add repair strategies (epenthesis, deletion, metathesis) when changes violate constraints
- Test with consonant cluster simplification scenarios
- Current state: `phonotactics.enforce_phonotactics_form()` now runs after each catalog's sound-change pass, trimming codas that exceed `max_coda_complexity`, inserting default vowels to break up overly complex onsets, and attempting simple metathesis when an allowed onset/coda ordering exists. This gives us a first-pass safeguard before richer syllabification lands.

### Phase 4: Typology Blending
- Implement `blend_typologies()` with different strategies per parameter type
- Test with historical substrate examples (Celtic substrate in Germanic)
- Current state: `typology_blender.blend_typologies()` now unions phoneme inventories, weight-blends onset/coda limits, randomly selects stress according to the influence weight, and prepends substrate sound changes before the dominant rules. Flags such as `has_tone`, `vowel_harmony`, and `consonant_mutation` flip probabilistically based on the same weight so we can experiment with contact scenarios immediately.

### Phase 5: Time Evolution
- Implement `apply_time_shift()` with stochastic drift
- Model common diachronic tendencies (lenition, vowel raising/lowering)
- Current state: `time_evolution.apply_time_shift()` now deep-copies a typology and runs configurable "time steps" of drift. Each step may append lenition/vowel-shift sound changes, loosen or tighten syllable complexity, flip vowel-harmony/mutation flags, and occasionally select a new stress pattern. The helper returns the evolved typology plus the freshly generated sound laws so cultures can request time-based divergence directly.

### Phase 6: Full Pipeline Integration
- Connect all components into `transform_language()` workflow
- Add validation and error handling
- Integrate into Cultures system
- Current state: `linguistics.pipeline` exposes `transform_language()`, `evolve_catalog()`, and the `apply_substrate_influence()` / `apply_adstratum_influence()` helpers. These wrap time shifts, typology blending, and catalog evolution (with metadata breadcrumbs) so the culture system can request time, substrate, or peer-contact scenarios directly.

## Vocabulary Categories for 25+ Word Minimum

Essential semantic domains to include in each catalog:
1. **Geographic**: mountain, river, sea, forest, valley
2. **Settlement**: village, house, fort, boundary, path
3. **Kinship**: father, mother, brother, sister, son, daughter
4. **Military**: warrior, spear, shield, horse, battle
5. **Natural world**: sun, moon, water, fire, earth, sky
6. **Personal names**: 3-5 typical given names, 2-3 surnames/epithets
7. **Numbers**: one, two, three, five, ten

## Optional Enhancements for Later

- **Analogical leveling**: High-frequency irregular forms can regularize
- **Loanword detection**: Flag words borrowed vs. inherited
- **Feature-based rules**: Use `panphon` for natural class targeting ([+voice], [-continuant], etc.)

## Integration with Simulation

Each new culture will generate an associated language.
- Default cultures draw an unused culture from init_catalogs, but if all init_catalogs have already been taken, we'll generate a new one by performing a time transformation on a random existing language. 
- Regional cultures will create a new language catalog by applying a 100-year time-change to their parent culture's language (call `transform_language(..., transformation_type='time_evolution', time_depth=1)`).
- Syncretic cultures will use the blending system, and the weights for the input languages will be determined by the origin tile's cultural composition. `transform_language(..., transformation_type='substrate')` or `apply_adstratum_influence()` covers these cases once the culture engine calculates weights.

Existing cultures will also undergo a lingustic time-shift every 200 years, so we'll need to keep track of each language's age and perform updates when appropriate. 

Cultures will influence Polity names, and eventually names for other things as well. A Polity without a culture will take a default name ([Region_Name] Tribe), but upon adopting a primary culture for the first time, their name will be randomly selected from the catalog (geographic, natural_world, or settlement name) preceded by a rank descriptor (Empire for 70th percentile of dev or higher, Kingdom for 30th percentile, and Duchy for the rest). This rank descriptor has potential to update every year if the Polity's relative size changes-- though it will never downgrade, it can only upgrade. The Polity name, however, will always remain constant. 