# Ecosystem Simulation Project

A Python-based ecological system simulator that models complex predator-prey dynamics with environmental factors and social behaviors. This project implements an advanced agent-based model to simulate realistic interactions between predators and prey in a grid-based environment with resources, seasons, and group dynamics.

## Features

### Core Components
- **Grid-based Environment**: 
  - Customizable grid size with wrapped boundaries
  - Resource distribution system
  - Obstacle generation
- **Multiple Agent Types**:
  - Predators that hunt in groups and have cooperative behaviors
  - Prey with flocking behavior and camouflage abilities
- **Environmental Elements**:
  - Dynamic resource generation and regeneration
  - Seasonal changes affecting resource availability
  - Random obstacles affecting movement patterns
  - Camouflage tiles for prey protection

### Advanced Agent Behaviors

#### Predator Social Structure
- Group-based hunting strategies
- Pack formation with group IDs
- Cooperative resource sharing
- Energy-based reproduction system
- Territory-based movement patterns

#### Prey Social Behaviors
- Flocking mechanics with flock IDs
- Camouflage abilities in specific tiles
- Resource-seeking behavior
- Group-based survival strategies
- Predator avoidance mechanisms

### Simulation Mechanics
- **Energy System**:
  - Agents consume energy when moving
  - Predators gain energy from successful hunts (10 units)
  - Predators can opportunistically consume resources (5 units)
  - Prey gain energy from resources (5 units)
- **Population Dynamics**:
  - Agent reproduction at energy thresholds (Predators: 15, Prey: 10)
  - Energy splitting between parent and offspring
  - Natural population cycles
  - Environmental carrying capacity

### Visualization and Analysis
- **Dual-View Animation**:
  - Resource distribution with color gradient
  - Agent positions with population tracking
  - Real-time population counts
- **Population Metrics**:
  - Dynamic population trends
  - Predator-prey relationship visualization
  - Season impact analysis
- **State Management**:
  - Comprehensive state saving/loading
  - JSON-based persistence
  - Detailed logging system

## Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv eco
source eco/bin/activate  # On Unix
# or
.\eco\Scripts\activate  # On Windows
```
3. Install required packages:
```bash
pip install numpy matplotlib pydantic
```

## Usage

Basic simulation:
```python
python main.py
```

Customized simulation:
```python
simulation = EcosystemSimulation(
    grid_size=20,         # Environment size
    num_prey=15,          # Initial prey count
    num_predators=8,      # Initial predator count
    resource_regen_rate=0.3  # Resource regeneration rate
)
```

## Output Files
- `ecosystem_simulation.mp4`: Side-by-side animation of resources and agents
- `population_metrics.png`: High-resolution population dynamics graph
- `final_state.json`: Complete simulation state
- `simulation.log`: Detailed behavior logs

## Project Structure
```
Ecosystem/
├── agents/
│   ├── predator.py       # Predator agent with group behaviors
│   └── prey.py          # Prey agent with flocking mechanics
├── simulation/
│   └── eco_simulation.py # Core simulation engine
├── main.py              # Entry point
└── README.md
```

## Environmental Factors

### Seasons
Dynamic seasonal system affecting:
- Resource regeneration rates (Spring/Summer: 0.3, Fall/Winter: 0.1)
- Agent behavior patterns
- Population dynamics

### Resources and Obstacles
- Resource regeneration based on seasonal cycles
- Random obstacle generation (10% of grid)
- Camouflage tiles for prey (10% of grid)
- Resource consumption hierarchy

## Logging and Monitoring
- Detailed event logging system
- Agent interaction tracking
- Population dynamics monitoring
- Resource consumption tracking

## Contributing
Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License
[APL 2.0]

## Author
Girish Kumar Adari