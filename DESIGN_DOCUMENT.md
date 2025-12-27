# Etymon Design Document

## Implementation Phase I

### Specifications
Tech stack must be able to handle drawing operations as well as flexible UI systems. Prioritize modernity, maintainability, and robustness.

### World Generation

#### Tiles
- Divide the world into polygonal tiles of assorted shapes (perhaps utilizing Voronoi math or something to accomplish this)
- Use Perlin Noise to determine the elevation of each tile, and then apply sea level to form geography

### Presentation

#### Map Appearance
- Below sea level: shades of blue depending on depth (deeper = darker blue like #024B86, shallower = closer to something like #8CF6FF)
- Above sea level: spectrum from verdant green to brownish gray depending on altitude
- This will change in the future.

### Maintainability
- "Dependency injection"-esque approach for designing component interfaces
- Config files for all world generation settings & assets
  - Sea level
  - Number of provinces
  - Map colors
  - Perlin noise fine-tuning

## Implementation Phase II

### Presentation

#### Map views/modes
A plethora of map overlay modes that can be toggled. For now, all we'll want to display is the "regions" mode, since that's all we'll have implemented anyway. We should be able to toggle these via a simple GUI.

### World Generation

#### Heightmap support
- If heightmap folder is empty, generate based on perlin noise
- If not, try to use grayscale values of PNG inside of the folder to generate height values instead (black = 0, white = 1)

#### Intelligent region mapping
These regions could be islands, continents, or subcontinents. Generally all contiguous landmasses will be their own regions, except for especially large landmasses, which might need to be subdivided into two or more regions. Also, if a particularly large contiguous landmass has an obvious "isthmus" or "choke point," the simulation ought to recognize that as a potential boundary point for regional separation.

#### Climate Details
- Temperature based on distance from poles & altitude
- Rainfall based on another layer of perlin noise (for now)
- Map coloration on land now depends on temperature & rainfall

### Maintainability
- Equator position configuration
- Region sizing details

## Implementation Phase III

### Polities
A polity could represent a lot of things– a country, a city-state, a fiefdom, a colony, a state… The list goes on. 
- All polities have several things in common:
  - A leader (for now just a dummy object with a dummy name & age, will have more later)
  - A primary culture
  - Borders
  - Relationships with other polities
- Polities can also exist in a hierarchy. One polity can “own” another– so Polities must keep track of their “suzerain” (parent) and their “vassals” (children) through an encapsulated Relationship. This relationship object keeps track of who the suzerain is and who the vassal is, as well as the “integration” level, which exists on a scale of 1-100. If integration reaches 100, the vassal is dissolved and "annexed" into the suzerain polity. If it reaches 0, it will become independent, gaining its own unique color.
- Border coloration and design depends on these relationships. A top-level polity has its own color. Vassals inherit their colors from their suzerains, using a slightly different shade so as to differentiate themselves, and their borders will be much thinner. 
Tiles will now have a “control” level, which represents local “layman” loyalty to their immediate subordinate, on a scale of 1-100. If a tile with >100 population reaches 0 control, the province and any provinces nearby with <=10% integration will coalesce to form a new, independent state.
  - This “nearby” calculation can chain– it’ll keep spreading to check for adjacent low-control (<=10%) tiles that haven’t revolted yet.
- In order to represent polities, we’ll need to first work on a few systems:
  - Border drawing
    - A mass of contiguous tiles owned by the same polity would have a simple border (in that polity’s color) drawn around the peripherie of the outermost tiles. 
  - Map text
    - Intelligently positioned toward the center of the polity’s borders
  - Cultures
    - For now just dummy placeholders, see below
  - Populations & Population centers
    - Tiles keep track of their own Population, Cultural Makeup, Resource (taken from a simple enum and determined by climate & chance), Development (value from 0-100), and the Polity that controls them.
    - Population movement will need to be an encapsulated process that always factors in cultural makeup. If 200 people migrate from the tile, we’ll need to first determine which culture(s) those 200 are from, and then modify the cultural makeup percentage of both the origin and destination accordingly. 
  - Cultural Makeup corresponds to our Cultures system, detailed later. For now, cultures are indistinct from one another except for their color.
- To make this more debuggable, we’ll need to have robust console output displaying the status of each polity and relationship when created or altered.

### Gameticks
The simulation should run on a tick system, where each tick we do calculations, update populations, etc. For now we can also probably get away with only redrawing the map each tick as well. 

Each tick will represent 3 months (which we can represent as seasons-- Fall, Winter, Spring, Summer of year X). We'll keep track of the simulation date in the UI. We won't worry about seasons affecting climate.

### Development & Population
Higher development = Higher population growth, more migration destination priority
Higher population = Higher population growth
If population > development:
- Development ticks upward
- Population growth boost from high population is negated
If population < development:
- Development ticks downward very, very slowly
If development reaches 0.2, we establish a new “population center.” Right now, mechanically, population centers don’t do anything, but they’ll display as a city dot on the map and will have a small name label associated with them. We’ll use placeholder text for now. Population centers can also disappear, if they fall beneath the threshold. 
- All of this should be configurable, of course. We’ll also need to find a way to factor in exponential population growth: At a certain point every tile will be a population center, which will be messy, and we don’t want that.

Each tick, populations will have a chance to migrate. We should only check tiles that have at least one population. We'll want to be as efficient as possible for this calculation, so apply any performance optimizations you can think of.

Destination Potential:
- (Base of 0)
- Factors in Development (positive), Altitude (negative), Temperature (extremes are negative), Access to coast (slight positive), 

If our Destination Potential is >= 90%, we don't bother checking migration at all.

For each eligible tile, we'll compare our Desination Potential against our neighbors' DP. We'll check our neighbors in a random order. If one of our neighbors has a significantly higher DP, an amount of population (scaled with tile population) will migrate, and we'll break the loop.

DP also influences how many pops travel: If there's a huge DP differential, a lot of pops move. If it's relatively small, only a couple might move.


### Linguistics
Each culture group has a word for themself, their neighbors, and distant lands. These words of course vary from the names those places use.

When a province is conquered, it may be renamed to the conquering culture's preference. But that culture will retain their own naming conventions, and if liberated, would revert back to it. On the other hand, if that culture were to be eradicated or assimilated, the original placename could vanish.