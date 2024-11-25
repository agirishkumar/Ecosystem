from swarm import Agent
from pydantic import BaseModel, Field
from typing import Tuple, List, Dict, ClassVar, Optional
from config import SimulationConfig
import random
import numpy as np

class PredatorAgent(Agent, BaseModel):
    # Required attributes
    name: str
    position: Tuple[int, int]
    grid_size: int
    config: SimulationConfig
    
    # Basic attributes
    energy: float = Field(default=10.0)
    vision_range: int = 4
    group_id: Optional[int] = None
    territory_influence: float = 1.0
    
    # Status tracking
    age: int = 0
    successful_hunts: int = 0
    hunt_attempts: int = 0
    last_prey_sighting: Optional[Tuple[int, int]] = None
    memory_map: np.ndarray = Field(default_factory=lambda: np.array([]))
    
    # Strategy attributes
    success_rate: Dict[str, float] = {
        'ambush': 0.5,
        'chase': 0.5,
        'group_hunt': 0.5
    }
    strategy_weights: Dict[str, float] = {
        'ambush': 0.5,
        'chase': 0.5,
        'group_hunt': 0.5
    }
    
    # Pack behavior
    pack_role: str = Field(default='hunter')
    memory_decay: float = 0.95

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.memory_map = np.zeros((self.grid_size, self.grid_size))
        self.pack_role = random.choice(['scout', 'hunter', 'defender'])
        self.hunt_attempts = 0
        self.successful_hunts = 0

    def move_towards(self, target_position):
        """
        Move towards target with energy consideration and strategy
        """
        if self.energy < self.config.predator_movement_cost:
            return False
            
        # Don't move if using ambush strategy successfully
        if (self.strategy_weights['ambush'] > max(self.strategy_weights.values()) * 0.8 
            and random.random() < 0.3):
            return False
                
        dx = target_position[0] - self.position[0]
        dy = target_position[1] - self.position[1]
        
        # Normalized movement
        new_x = self.position[0] + (1 if dx > 0 else -1 if dx < 0 else 0)
        new_y = self.position[1] + (1 if dy > 0 else -1 if dy < 0 else 0)
        
        self.position = (new_x % self.grid_size, new_y % self.grid_size)
        self.energy -= self.config.predator_movement_cost
        return True

    def coordinate_pack_hunt(self, pack_members: List['PredatorAgent'], 
                           prey_positions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Coordinate hunting strategies with pack members
        """
        if not prey_positions or not pack_members:
            return None

        # Find most vulnerable prey (closest to most pack members)
        prey_scores = {}
        for prey_pos in prey_positions:
            pack_distances = [member._distance(prey_pos) for member in pack_members]
            prey_scores[prey_pos] = sum(1/d if d > 0 else float('inf') for d in pack_distances)
        
        target_prey = max(prey_scores.items(), key=lambda x: x[1])[0]
        
        # Calculate optimal surrounding positions
        surrounding_positions = [
            ((target_prey[0] + dx) % self.grid_size, 
             (target_prey[1] + dy) % self.grid_size)
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
        ]
        
        # Assign positions to pack members based on their current positions
        for member in pack_members:
            if surrounding_positions:
                closest_pos = min(surrounding_positions, 
                                key=lambda p: member._distance(p))
                member.move_towards(closest_pos)
                surrounding_positions.remove(closest_pos)

        return target_prey

    def update_territory(self, territory_map: np.ndarray) -> np.ndarray:
        """
        Update and maintain territory with influence radius
        """
        if self.group_id is None:
            return territory_map
            
        radius = int(self.territory_influence)
        x, y = self.position
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    tx = (x + dx) % self.grid_size
                    ty = (y + dy) % self.grid_size
                    current_value = territory_map[tx, ty]
                    if current_value == 0 or current_value == self.group_id:
                        territory_map[tx, ty] = self.group_id
                    
        return territory_map

    def update_memory(self, prey_positions: List[Tuple[int, int]]):
        """
        Update memory map with prey sightings and decay old memories
        """
        self.memory_map *= self.memory_decay
        
        for pos in prey_positions:
            if self._distance(pos) <= self.vision_range:
                self.memory_map[pos[0], pos[1]] = 1.0
                self.last_prey_sighting = pos

    def act(self, prey_positions: List[Tuple[int, int]], 
            other_predators: List['PredatorAgent'], 
            resources: List[Tuple[int, int]], 
            territory_map: Optional[np.ndarray] = None) -> str:
        """
        Enhanced decision making with energy management and pack behavior
        """
        self.age += 1
        self.energy -= self.config.predator_movement_cost * 0.5  # Base energy cost
        
        # Die if no energy
        if self.energy <= 0:
            return "die"
            
        # Update territory and memory
        if territory_map is not None:
            territory_map = self.update_territory(territory_map)
        self.update_memory(prey_positions)
        
        # Find pack members
        pack_members = [p for p in other_predators 
                       if p.group_id == self.group_id and p != self]
        
        # Emergency energy replenishment
        if self.energy < self.config.predator_movement_cost * 3:
            if resources:
                nearest_resource = min(resources, 
                    key=lambda r: self._distance(r))
                if self._distance(nearest_resource) <= self.vision_range:
                    if self.move_towards(nearest_resource):
                        return "forage"
            return "rest"

        # Role-based behavior
        if self.pack_role == 'scout':
            if prey_positions:
                distant_prey = [p for p in prey_positions 
                              if all(pred._distance(p) > pred.vision_range 
                                   for pred in other_predators if pred != self)]
                if distant_prey:
                    target = min(distant_prey, key=lambda p: self._distance(p))
                    if self.move_towards(target):
                        return "scout"
                        
        elif self.pack_role == 'hunter' and self.energy > self.config.predator_movement_cost * 5:
            if prey_positions and pack_members:
                target = self.coordinate_pack_hunt(pack_members, prey_positions)
                if target and self.move_towards(target):
                    return "hunt"

        # Standard hunting based on current strategy
        if prey_positions and self.energy > self.config.predator_movement_cost * 4:
            strategy = max(self.strategy_weights.items(), 
                         key=lambda x: x[1])[0]
            
            if strategy == 'ambush':
                if random.random() < 0.7:  # 70% chance to stay still
                    if any(self._distance(prey) <= self.vision_range/2 
                          for prey in prey_positions):
                        return "ambush"
                    
            elif strategy == 'group_hunt' and len(pack_members) >= 2:
                target = self.coordinate_pack_hunt(pack_members, prey_positions)
                if target and self.move_towards(target):
                    self.energy -= self.config.predator_movement_cost * 0.5
                    return "hunt"
                    
            else:  # chase strategy
                nearest_prey = min(prey_positions, 
                    key=lambda p: self._distance(p))
                if self._distance(nearest_prey) <= self.vision_range:
                    if self.move_towards(nearest_prey):
                        return "hunt"

        # Random movement with territory consideration
        if random.random() < 0.5 and self.energy > self.config.predator_movement_cost * 2:
            if territory_map is not None:
                # Prefer movement within own territory
                possible_moves = []
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    new_x = (self.position[0] + dx) % self.grid_size
                    new_y = (self.position[1] + dy) % self.grid_size
                    territory_value = territory_map[new_x, new_y]
                    if territory_value == 0 or territory_value == self.group_id:
                        possible_moves.append((new_x, new_y))
                
                if possible_moves:
                    self.position = random.choice(possible_moves)
                    self.energy -= self.config.predator_movement_cost
                    return "move"
            
            # Default random movement
            self.position = (
                (self.position[0] + random.choice([-1, 0, 1])) % self.grid_size,
                (self.position[1] + random.choice([-1, 0, 1])) % self.grid_size
            )
            self.energy -= self.config.predator_movement_cost
            return "move"
            
        return "rest"

    def adapt_strategy(self, success: bool, strategy: str):
        """
        Update strategy success rates with improved learning
        """
        if strategy not in self.success_rate:
            self.success_rate[strategy] = 0.5
            
        # Update success tracking
        self.hunt_attempts += 1
        if success:
            self.successful_hunts += 1
            
        # Update success rate with decay
        decay = 0.95
        self.success_rate[strategy] = (
            self.success_rate[strategy] * decay + 
            (1.0 if success else 0.0) * (1 - decay)
        )
        
        # Update strategy weights
        total_success = sum(self.success_rate.values())
        if total_success > 0:
            for s in self.strategy_weights:
                self.strategy_weights[s] = self.success_rate[s] / total_success
        else:
            # Reset weights if no success
            for s in self.strategy_weights:
                self.strategy_weights[s] = 1.0 / len(self.strategy_weights)

    def _distance(self, target_position: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance to target
        """
        return abs(self.position[0] - target_position[0]) + \
               abs(self.position[1] - target_position[1])