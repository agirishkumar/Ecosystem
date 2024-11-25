# Ecosystem Simulation Project

## Overview
A sophisticated Python-based ecosystem simulation that models complex predator-prey dynamics, environmental interactions, and emergent behaviors. This project implements an advanced agent-based model that simulates realistic ecological interactions with features like seasonal changes, territory management, and social behaviors.

## Key Features

### Environmental System
- **Dynamic Grid-based Environment**
  - Configurable grid size with wrapped boundaries
  - Seasonal resource regeneration cycles
  - Obstacle generation and management
  - Pheromone trail system for agent communication

### Agent Types

#### Predator Agents
- **Hunting Strategies**
  - Pack hunting with coordination
  - Ambush tactics
  - Chase behavior with energy management
  - Adaptive strategy learning

- **Territory System**
  - Territory marking and maintenance
  - Group-based territory defense
  - Influence radius mechanics
  - Territory decay over time

- **Pack Behavior**
  - Role-based actions (Scout, Hunter, Defender)
  - Pack coordination for hunting
  - Memory mapping of prey locations
  - Group-based decision making

#### Prey Agents
- **Survival Mechanisms**
  - Flocking behavior with configurable parameters
  - Camouflage in specific tiles
  - Predator evasion tactics
  - Resource foraging strategies

- **Social Structure**
  - Dynamic role assignment
  - Group-based survival strategies
  - Resource information sharing
  - Collective decision making

### Environment Features
- **Resource Management**
  - Dynamic resource regeneration
  - Seasonal variation in resource availability
  - Energy transfer system
  - Resource competition

- **Seasonal System**
  - Four-season cycle affecting resources
  - Variable regeneration rates
  - Impact on agent behavior
  - Configurable season lengths

### Advanced Mechanics
- **Energy System**
  - Detailed energy accounting
  - Movement costs
  - Hunting rewards
  - Reproduction energy requirements

- **Population Dynamics**
  - Natural reproduction cycles
  - Energy-based reproduction
  - Population control mechanisms
  - Carrying capacity considerations

## Swarm Agent Integration

### OpenAI Swarm Framework
The simulation leverages OpenAI's Swarm framework for agent-based modeling, providing several key advantages:

- **Agent Communication Protocol**: Built-in communication mechanisms between agents
- **State Management**: Efficient handling of agent states and transitions
- **Behavior Organization**: Structured approach to implementing agent behaviors
- **Scalability**: Better performance for large numbers of agents

### Swarm-based Features
1. **Agent Base Class**
   - Inheritance from Swarm's `Agent` class
   - Built-in coordination mechanisms
   - Standardized agent interfaces
   - Event handling capabilities

2. **Communication Infrastructure**
   - Local agent detection
   - State sharing between agents
   - Group coordination primitives
   - Behavior synchronization

3. **Optimization Benefits**
   - Efficient agent updates
   - Optimized state propagation
   - Better memory management
   - Improved performance for large simulations

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ecosystem-simulation.git
cd ecosystem-simulation
```

2. Create and activate virtual environment:
```bash
python -m venv eco
source eco/bin/activate  # On Unix/macOS
# or
.\eco\Scripts\activate  # On Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Simulation
Run the default simulation:
```python
python3 main.py
```

### Custom Configuration
Create a custom simulation configuration in main.py:
```python
from config import SimulationConfig
from simulation.eco_simulation import run_simulation_with_config

config = SimulationConfig(
    grid_size=30,
    num_prey=25,
    num_predators=5,
    resource_regen_rate=0.15,
    # Add other parameters as needed
)

simulation = run_simulation_with_config(config, steps=200)
```

## Configuration Parameters

### Environment Parameters
- `grid_size`: Size of the simulation grid
- `resource_regen_rate`: Base resource regeneration rate
- `obstacle_density`: Density of obstacles in the environment
- `pheromone_decay`: Decay rate of pheromone trails

### Prey Parameters
- `num_prey`: Initial number of prey
- `prey_initial_energy`: Starting energy for prey
- `prey_vision_range`: Prey vision range
- `prey_reproduction_threshold`: Energy needed for reproduction
- `prey_movement_cost`: Energy cost for movement
- `prey_camouflage_chance`: Probability of successful camouflage

### Predator Parameters
- `num_predators`: Initial number of predators
- `predator_initial_energy`: Starting energy for predators
- `predator_vision_range`: Predator vision range
- `predator_reproduction_threshold`: Energy needed for reproduction
- `predator_hunt_reward`: Energy gained from successful hunt
- `predator_movement_cost`: Energy cost for movement

### Seasonal Parameters
- `season_length`: Number of steps per season
- `summer_resource_rate`: Resource regeneration rate in summer
- `winter_resource_rate`: Resource regeneration rate in winter

## Output and Visualization

### Simulation Outputs
- **Animation**: MP4 file showing simulation progress
- **Population Metrics**: PNG plots of population dynamics
- **Advanced Metrics**: Detailed analysis of simulation behavior
- **State Files**: JSON files containing simulation state data

### Available Metrics
1. Population Dynamics
   - Prey population over time
   - Predator population over time
   - Population ratios

2. Hunting Success
   - Success rate over time
   - Strategy effectiveness
   - Pack hunting efficiency

3. Territory Analysis
   - Territory changes
   - Group distributions
   - Influence maps

4. Resource Metrics
   - Resource availability
   - Consumption patterns
   - Seasonal effects

## Project Structure
```
ecosystem-simulation/
├── agents/
│   ├── __init__.py
│   ├── predator.py
│   └── prey.py
├── simulation/
│   ├── __init__.py
│   └── eco_simulation.py
├── config.py
├── main.py
└── README.md
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Known Limitations
- High computation cost for large grids
- Memory intensive for long simulations
- Limited to 2D environment
- Single-threaded execution

## Future Improvements
- Multi-threading support
- 3D environment options
- GPU acceleration
- Advanced visualization options
- Machine learning integration
- More sophisticated agent behaviors
- ### Enhanced Swarm Capabilities

1. **Advanced Communication Systems**
   - **Broadcast Messaging**
     - Group-wide announcements
     - Emergency signals (predator warnings)
     - Resource location sharing
     - Territory marking broadcasts

   - **Direct Messaging**
     - One-to-one agent communication
     - Pack coordination messages
     - Leader-follower communication
     - Resource negotiation

2. **Swarm Intelligence Enhancements**
   - **Collective Decision Making**
     - Group consensus mechanisms
     - Distributed problem solving
     - Emergent leadership selection
     - Resource allocation optimization

   - **Dynamic Role Assignment**
     - Adaptive role switching
     - Need-based role distribution
     - Experience-based specialization
     - Group composition optimization

3. **Advanced Coordination Features**
   - **Stigmergy**
     - Enhanced pheromone systems
     - Environmental marking
     - Indirect communication
     - Trail optimization

   - **Hierarchical Organization**
     - Multi-level group structures
     - Specialized task forces
     - Dynamic hierarchy formation
     - Leadership succession

4. **Communication Patterns**
   - **Event-Driven Communication**
     - Threat detection broadcasts
     - Resource discovery alerts
     - Territory invasion warnings
     - Mating availability signals

   - **Information Sharing**
     - Knowledge pooling
     - Experience sharing
     - Strategy adaptation
     - Resource mapping



## License
[APL 2.0]

## Author
Girish Kumar Adari