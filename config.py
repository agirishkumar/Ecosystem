# config.py
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    # Environment parameters
    grid_size: int = 30
    resource_regen_rate: float = 0.2
    resource_energy_value: int = 3
    pheromone_decay: float = 0.95
    obstacle_density: float = 0.1
    
    # Prey parameters
    num_prey: int = 20
    prey_initial_energy: float = 8.0
    prey_vision_range: int = 4
    prey_reproduction_threshold: float = 12.0
    prey_reproduction_cost: float = 4.0
    prey_movement_cost: float = 0.5
    prey_camouflage_chance: float = 0.15
    prey_starve_threshold: float = 2.0
    prey_max_energy: float = 20.0
    
    # Predator parameters
    num_predators: int = 4
    predator_initial_energy: float = 12.0
    predator_vision_range: int = 5
    predator_reproduction_threshold: float = 20.0
    predator_reproduction_cost: float = 6.0
    predator_movement_cost: float = 0.8
    predator_hunt_reward: float = 8.0
    predator_starve_threshold: float = 3.0
    predator_max_energy: float = 30.0
    
    # Group behavior parameters
    flock_cohesion_range: int = 3
    pack_hunt_range: int = 4
    territory_influence_range: int = 2
    min_pack_size: int = 2
    
    # Seasonal parameters
    season_length: int = 15
    summer_resource_rate: float = 0.3
    winter_resource_rate: float = 0.05
    
    # Simulation control
    max_steps: int = 1000
    min_population: int = 2
    max_population: int = 100
    
    # Advanced mechanics
    energy_decay_rate: float = 0.98
    territory_decay_rate: float = 0.9
    learning_rate: float = 0.1
