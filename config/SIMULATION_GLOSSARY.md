# Simulation Config Glossary

This glossary provides detailed definitions for all fields under the main simulation categories in `config/world_generation.json`. It is essential to keep this document up to date whenever fields are added, removed, or changed.

## Categories Covered
- population
- development
- migration
- control
- polity
- culture
- war
- relationships

---

## population
- **base_birth_rate**: The base annual birth rate per population unit (e.g., 0.033 = 3.3% per year).
- **base_death_rate**: The base annual death rate per population unit (e.g., 0.015 = 1.5% per year).
- **climate_death_penalty_max**: Maximum additional death rate due to harsh climate conditions.
- **rainfall_death_penalty_max**: Maximum additional death rate due to suboptimal rainfall.
- **rainfall_optimal_min**: Minimum rainfall value considered optimal for population survival.
- **rainfall_optimal_max**: Maximum rainfall value considered optimal for population survival.
- **development_death_reduction**: Fractional reduction in death rate in well-developed areas.
- **development_threshold_ratio**: Ratio of development to population above which death rate is reduced.
- **overcap_birth_penalty**: Multiplier reducing birth rate when population exceeds carrying capacity.
- **small_population_threshold**: Population size below which special rules may apply (currently unused).

## development
- **rapid_increase_rate**: Growth rate of development when development is less than population.
- **decay_rate**: Rate at which development decays when population is insufficient.
- **decay_threshold_ratio**: Population-to-development ratio below which decay begins.
- **population_center_threshold**: Minimum development required for a tile to become a population center.
- **population_center_threshold_scaling**: Scaling factor for population center threshold based on world size. Higher values make it harder for tiles to become population centers (fewer centers overall); lower values make it easier (more centers).
- **population_center_percentage**: Percentage of total world development required for a tile to qualify as a population center (not a cap on number of centers). Higher values require more development for new centers to form; lower values make it easier.
- **population_center_creation_hysteresis_ratio**: Ratio to prevent rapid toggling of population center status. Higher values make it harder to regain center status after losing it; lower values make status changes more responsive.
- **population_center_demotion_grace_years**: Years a center can remain below threshold before demotion. Higher values give more time before demotion; lower values cause quicker demotion.
- **population_center_demotion_population_ratio**: Population ratio below which demotion occurs. Lower values make demotion less likely; higher values make demotion more likely.
- **population_center_demotion_development_ratio**: Development ratio below which demotion occurs. Lower values make demotion less likely; higher values make demotion more likely.
- **population_center_absolute_population**: Absolute population required for center status (overrides ratio).
- **population_center_absolute_development**: Absolute development required for center status (overrides ratio).
- **new_culture_control_bonus**: Bonus to control for new cultures in population centers.
- **population_center_culture_impact**: Impact of population centers on culture spread.
- **coastal_bonus**: Bonus to development for coastal tiles. Higher values make coastal locations more advantageous for development.
- **temperature_penalty_severity**: Severity of development penalty for suboptimal temperature. Higher values increase the penalty for non-ideal temperatures.
- **rainfall_penalty_severity**: Severity of development penalty for suboptimal rainfall. Higher values increase the penalty for non-ideal rainfall.
- **rainfall_optimal_min**: Minimum rainfall for optimal development.
- **rainfall_optimal_max**: Maximum rainfall for optimal development.
- **river_bonus**: Bonus to development for tiles adjacent to rivers.
- **river_dryness_multiplier**: Multiplier for river bonus in dry regions.
- **river_cap_bonus**: Maximum development bonus from rivers.
- **development_cap_population_ratio_max**: Maximum multiplier for development cap based on population advantage over neighbors. Higher values allow greater development caps for population-dense areas; lower values limit the advantage.

## migration
- **destination_potential_threshold**: Minimum potential required for a tile to be considered a migration destination.
- **development_ratio_factor**: Weight of development/population ratio in migration potential.
- **death_penalty_factor**: Weight of recent deaths in reducing migration potential.
- **overcrowding_penalty_factor**: Penalty for overcrowding in migration potential.
- **migration_threshold**: Minimum migration pressure required to trigger migration.
- **migration_rate_max**: Maximum fraction of population that can migrate per tick.
- **development_migration_bonus**: Bonus to migration potential for well-developed destinations.
- **population_center_dp_bonus**: Bonus to migration potential for population centers.
- **population_center_dp_bonus_scaling**: Scaling factor for population center migration bonus.
- **population_center_dp_bonus_max_multiplier**: Maximum multiplier for population center migration bonus.
- **population_center_overcrowding_penalty_multiplier**: Multiplier for overcrowding penalty in centers.
- **overcap_dp_penalty**: Penalty to migration potential when over carrying capacity.
- **baseline_migration_chance**: Base chance for migration to occur regardless of other factors.
- **baseline_migration_min**: Minimum number of migrants in baseline migration.
- **baseline_migration_max**: Maximum number of migrants in baseline migration.
- **river_dp_bonus**: Bonus to migration potential for river-adjacent tiles.
- **major_migration_min_population**: Minimum population required to trigger major migration.
- **major_migration_pressure_threshold**: Pressure threshold for major migration events.
- **major_migration_cooldown_ticks**: Cooldown (in ticks) between major migrations.
- **major_migration_loss_ratio_trigger**: Population loss ratio that triggers major migration.
- **exodus_migration_chance**: Chance for exodus migration under extreme conditions.
- **exodus_max_distance**: Maximum distance for exodus migration.
- **exodus_min_population_ratio**: Minimum population ratio for exodus.
- **exodus_max_population_ratio**: Maximum population ratio for exodus.
- **exodus_cross_region_bonus**: Bonus for exodus migration across regions.
- **mass_death_exodus_ratio**: Population loss ratio that triggers mass exodus.
- **mass_death_exodus_population**: Minimum population loss to trigger mass exodus.
- **polity_expansion_control_threshold**: Control threshold for polity expansion via migration.
- **polity_expansion_population_threshold**: Population threshold for polity expansion via migration.
- **cultural_bias_scale**: Scale of cultural preference in migration decisions.

## control
- **assimilation_penalty**: Penalty to control for newly assimilated territories.
- **migration_influence_factor**: Influence of migration on control changes.
- **control_projection_rate**: Rate at which control spreads from centers.
- **border_decay_rate**: Rate at which control decays at borders.
- **border_decay_threshold**: Control value below which decay accelerates at borders.
- **overdevelopment_decay_rate**: Rate of control decay in overdeveloped areas.
- **overdevelopment_threshold_ratio**: Development-to-population ratio above which overdevelopment decay applies.
- **non_center_control_bleed_chance**: Chance for control to "bleed" from non-center tiles.

## polity
- **enable_breakaway_polities**: Whether polities can break away and form new entities.
- **breakaway_low_control_threshold**: Control threshold below which breakaway is possible.
- **breakaway_low_control_years**: Years of low control required for breakaway.
- **unowned_spawn_years**: Years before unowned regions can spawn new polities.
- **breakaway_chain_control_threshold**: Control threshold for chain breakaways.
- **breakaway_administrative_burden_reference_share**: Reference share of administrative burden for breakaway calculations.
- **breakaway_administrative_burden_max_bonus**: Maximum bonus to breakaway chance from administrative burden.
- **rank_percentiles**: Percentile thresholds for kingdom and empire ranks.

## culture
- **new_culture_chance**: Chance for a new culture to spawn in a population center.
- **new_culture_min_center_ticks**: Minimum ticks before a new culture can spawn in a center.
- **new_culture_min_center_years**: Minimum years before a new culture can spawn in a center.
- **new_culture_immunity_years**: Years of immunity after a new culture spawns.
- **region_home_mismatch_years**: Years before a culture is considered mismatched to its region.
- **tile_spawn_cooldown_years**: Cooldown in years before a tile can spawn a new culture again. This cooldown is reset whenever a tile is introduced to any culture (via spawning, migration, or other means), except for syncretic cultures which are exempt from this restriction.
- **new_culture_growth_rate**: Growth rate of new cultures.
- **migration_cultural_influence**: Influence of migration on cultural change.
- **linguistics.catalog_dir**: Directory containing initial language catalogs.
- **linguistics.seed**: Random seed for linguistics operations.
- **linguistics.time_shift_years**: Years between automatic language time shifts.
- **assimilation_soft_threshold_base**: Base threshold for soft assimilation.
- **assimilation_soft_threshold_control_factor**: Control factor for soft assimilation threshold.
- **assimilation_min_factor**: Minimum assimilation factor.
- **syncretism_threshold**: Threshold for syncretism (cultural blending).
- **syncretism_duration_years**: Duration of syncretism state.
- **low_alignment_threshold**: Threshold for low cultural alignment.
- **low_alignment_penalty_multiplier**: Penalty multiplier for low alignment.
- **tolerance_base_rate**: Base rate of cultural tolerance increase.
- **tolerance_max_rate**: Maximum rate of cultural tolerance increase.
- **tolerance_diversity_reference**: Reference value for diversity in tolerance calculations.
- **tolerance_diversity_neutral**: Neutral value for diversity in tolerance.
- **tolerance_war_penalty_step**: Penalty step to tolerance during war.
- **tolerance_diversity_positive_multiplier**: Multiplier for positive diversity effects.
- **tolerance_diversity_negative_multiplier**: Multiplier for negative diversity effects.
- **tolerance_extreme_damping**: Damping factor for extreme tolerance values.
- **tolerance_dp_span**: Span of development/population for tolerance calculations.
- **tolerance_alignment_threshold_span**: Span for alignment threshold in tolerance.
- **tolerance_alignment_cap_span**: Cap span for alignment in tolerance.
- **tolerance_alignment_floor_base**: Base floor for alignment in tolerance.
- **tolerance_alignment_floor_span**: Floor span for alignment in tolerance.

## war
- **truce_years**: Minimum years of truce after a war.
- **declaration_base_chance**: Base chance per tick for war declaration.
- **declaration_growth_rate**: Growth rate of war declaration chance over time.
- **declaration_max_chance**: Maximum chance for war declaration.
- **declaration_initiator_war_penalty**: Penalty to initiator's war chance if already at war.
- **declaration_target_war_bonus**: Bonus to target's war chance if already at war.
- **supply_base_per_tick**: Base supply income per tick for polities at war.
- **supply_per_population_center**: Additional supply per population center.
- **supply_per_development**: Additional supply per development point.
- **supply_storage_cap**: Maximum supply that can be stored.
- **river_crossing_penalty**: Penalty to strength when crossing rivers in battle.
- **river_flux_threshold**: Minimum river flux for penalty to apply.
- **supply_commitment_cap**: Maximum supply that can be committed to a single battle.
- **supply_decay_per_tick**: Fraction of supply lost per tick.
- **supply_stack_decay**: Additional decay for stacked supply.
- **supply_noise_floor**: Minimum supply value after decay.
- **supply_starvation_threshold**: Supply level below which starvation occurs.
- **strength_population_factor**: Contribution of population to military strength.
- **strength_development_factor**: Contribution of development to military strength.
- **strength_control_factor**: Contribution of control to military strength.
- **strength_supply_factor**: Contribution of supply to military strength.
- **capture_margin**: Margin required to capture a tile.
- **capture_control_level**: Control level required to capture a tile.
- **capture_cooldown_ticks**: Cooldown between capture attempts.
- **occupation_control_level**: Control level for occupation.
- **occupation_annex_control_level**: Control level for annexation.
- **occupation_overlay_alpha**: Transparency of occupation overlay on map.
- **occupation_supply_penalty_per_tile**: Supply penalty per occupied tile.
- **occupation_exhaustion_per_tile**: Exhaustion per occupied tile.
- **annexation_exhaustion_weight**: Weight of exhaustion in annexation calculations.
- **annexation_supply_weight**: Weight of supply in annexation calculations.
- **pressure_annexation_cap**: Cap on pressure for annexation.
- **skirmish_population_loss_ratio**: Population loss ratio in skirmishes.
- **battle_population_loss_ratio**: Population loss ratio in battles.
- **development_loss_per_population**: Development loss per population lost in war.
- **exhaustion_base_per_tick**: Base exhaustion gained per tick during war.
- **exhaustion_loss_weight**: Weight of losses in exhaustion calculations.
- **exhaustion_supply_relief**: Relief to exhaustion from supply.
- **exhaustion_supply_penalty_per_point**: Penalty to supply per exhaustion point.
- **exhaustion_supply_min_multiplier**: Minimum multiplier for supply under exhaustion.
- **exhaustion_end_threshold**: Exhaustion level at which war ends.
- **full_annexation_tile_cap**: Maximum number of tiles for full annexation.
- **full_annexation_ratio**: Ratio for full annexation eligibility.
- **victory_development_gain_ratio**: Fraction of development lost by the defeated side that is gained by the victor. Lower values reduce the spoils of war; higher values increase them.
- **war_victory_control_boost**: Temporary control percentage boost applied to all tiles of a polity after winning a war (by annexing tiles). This bonus decays over time.

## relationships
- **border_penalty_scale**: Scale of penalty for shared borders.
- **border_penalty_cap**: Maximum penalty for shared borders.
- **shared_culture_bonus**: Bonus to relations for shared culture.
- **different_culture_penalty**: Penalty to relations for different cultures.
- **shared_parent_penalty**: Penalty for shared parent polity.
- **development_superiority_bonus**: Bonus for higher development compared to neighbor.
- **development_peer_penalty**: Penalty for similar development to neighbor.
- **population_center_advantage_penalty**: Penalty for having more population centers than neighbor.
- **population_center_gap_penalty**: Penalty for gap in population centers.
- **shared_region_penalty**: Penalty for sharing a region.
- **dominance_development_ratio**: Ratio for dominance based on development.
- **dominance_development_penalty**: Penalty for dominance in development.
- **dominance_tile_ratio**: Ratio for dominance based on tile count.
- **dominance_tile_penalty**: Penalty for dominance in tile count.
- **grudge_penalty**: Penalty for past grievances.
- **warmongering_penalty**: Penalty for history of warlike behavior.

---

*This glossary must be updated whenever simulation config fields are added, removed, or changed.*
