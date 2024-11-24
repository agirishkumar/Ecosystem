# Ecosystem Simulation Project

A Python-based ecological system simulator that models predator-prey dynamics with environmental factors. This project implements an agent-based model to simulate the interactions between predators and prey in a grid-based environment with resources and seasonal changes.

## Features

### Core Components
- **Grid-based Environment**: Customizable grid size with wrapped boundaries
- **Multiple Agent Types**:
  - Predators that hunt prey and gain energy from successful hunts
  - Prey that forage for resources and try to avoid predators
- **Environmental Elements**:
  - Dynamic resource generation and regeneration
  - Seasonal changes affecting resource availability
  - Random obstacles in the environment

### Simulation Mechanics
- **Energy System**:
  - Agents consume energy when moving
  - Predators gain energy by catching prey
  - Prey gain energy by consuming resources
- **Population Dynamics**:
  - Agent reproduction when energy levels are sufficient
  - Agent death when energy is depleted
  - Natural population cycles emerge from interactions

### Visualization and Analysis
- **Real-time Animation**:
  - Visual representation of predators (red) and prey (green)
  - Resource distribution visualization
  - Population counts displayed in title
- **Population Metrics**:
  - Population trends plotting
  - Separate graphs for predator and prey populations
- **State Management**:
  - Save/Load simulation states
  - JSON-based state persistence

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
   pip install -r requirements.txt
   ```

## Usage

Run the simulation with default parameters:
```python
python main.py
```

Customize the simulation:
```python
simulation = EcosystemSimulation(
    grid_size=20,         # Size of the environment grid
    num_prey=15,          # Initial number of prey
    num_predators=8,      # Initial number of predators
    resource_regen_rate=0.3  # Rate of resource regeneration
)
```

## Output Files
- `ecosystem_simulation.mp4`: Animation of the simulation
- `population_metrics.png`: Graph of population dynamics
- `final_state.json`: Final state of the simulation

## Project Structure
```
Ecosystem/
├── agents/
│   ├── predator.py       # Predator agent implementation
│   └── prey.py          # Prey agent implementation
├── simulation/
│   └── eco_simulation.py # Main simulation logic
├── main.py              # Entry point
└── README.md
```

## Agent Behaviors

### Predator Agents
- Hunt nearest prey using Manhattan distance
- Gain energy from successful hunts
- Reproduce when energy exceeds threshold
- Die when energy depleted
- Random movement when no prey nearby

### Prey Agents
- Random movement strategy
- Consume resources for energy
- Reproduce when energy threshold met
- Die when energy depleted

## Environmental Factors

### Seasons
The simulation includes seasonal cycles that affect resource regeneration:
- Spring/Summer: Higher resource regeneration rate (0.3)
- Fall/Winter: Lower resource regeneration rate (0.1)

### Resources
- Randomly generated across the grid
- Regenerate based on seasonal rates
- Consumed by prey for energy

## Contributing
Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License
[APL 2.0]

## Author
Girish Kumar Adari