# agents/prey.py
from swarm import Agent
from pydantic import BaseModel, Field
from typing import Tuple, List, ClassVar, Optional
from config import SimulationConfig
import random
import numpy as np

class PreyAgent(Agent, BaseModel):
    # Required attributes
    name: str
    position: Tuple[int, int]
    grid_size: int
    config: SimulationConfig
    
    # Basic attributes with defaults
    energy: float = Field(default=5.0)
    vision_range: int = 3
    camouflage_tiles: List[Tuple[int, int]] = None
    flock_id: Optional[int] = None
    
    # Flocking parameters
    separation_radius: float = 2.0
    alignment_weight: float = 1.0
    cohesion_weight: float = 1.0
    separation_weight: float = 1.5
    
    # Status tracking
    age: int = 0
    successful_escapes: int = 0
    resource_findings: int = 0
    current_role: str = 'forager'
    has_spotted_predator: bool = False
    has_found_resource: bool = False
    last_predator_position: Optional[Tuple[int, int]] = None
    last_resource_position: Optional[Tuple[int, int]] = None

    # Class variables
    roles: ClassVar[List[str]] = ['scout', 'forager', 'defender']

    class Config:
        arbitrary_types_allowed = True

    def move_towards(self, target_position):
        """
        Move towards target position with energy cost
        """
        if self.energy < self.config.prey_movement_cost:
            return False
            
        new_x = self.position[0] + (1 if target_position[0] > self.position[0] 
                                   else -1 if target_position[0] < self.position[0] else 0)
        new_y = self.position[1] + (1 if target_position[1] > self.position[1] 
                                   else -1 if target_position[1] < self.position[1] else 0)
        
        self.position = (new_x % self.grid_size, new_y % self.grid_size)
        self.energy -= self.config.prey_movement_cost
        return True

    def act(self, resources: List[Tuple[int, int]], 
            predator_positions: List[Tuple[int, int]], 
            other_prey: List['PreyAgent']):
        """
        Enhanced decision making with energy management and role-based behavior
        """
        # Age and basic energy consumption
        self.age += 1
        self.energy -= self.config.prey_movement_cost * 0.5  # Base energy cost per turn
        
        # Die if no energy
        if self.energy <= 0:
            return "die"
        
        # Assess threats
        nearby_predators = [
            predator for predator in predator_positions
            if self._distance(predator) <= self.vision_range
        ]
        
        # Emergency energy conservation
        if self.energy < self.config.prey_movement_cost * 2:
            if self.position in self.camouflage_tiles:
                return "camouflage"
            return "rest"

        # Handle predator evasion with role-specific behavior
        if nearby_predators:
            self.has_spotted_predator = True
            self.last_predator_position = nearby_predators[0]
            
            if self.current_role == 'defender' and self.energy > self.config.prey_movement_cost * 4:
                # Defenders might stay to warn others
                nearby_prey = [prey for prey in other_prey 
                             if self._distance(prey.position) <= self.vision_range]
                if len(nearby_prey) > 2:
                    return "warn"
            
            # All roles try to escape if too close
            safe_tile = self._find_safe_tile(predator_positions)
            if safe_tile:
                if self.move_towards(safe_tile):
                    self.successful_escapes += 1
                    return "evade"

        # Role-specific behaviors
        if self.current_role == 'scout' and self.energy > self.config.prey_movement_cost * 3:
            # Scouts explore more and share information
            if resources:
                unexplored_resources = [r for r in resources 
                                      if not any(self._distance(prey.position) <= self.vision_range 
                                               for prey in other_prey)]
                if unexplored_resources:
                    nearest = min(unexplored_resources, key=lambda r: self._distance(r))
                    if self.move_towards(nearest):
                        return "explore"

        # Resource foraging when energy is low or role is forager
        if self.energy < self.config.prey_reproduction_threshold * 0.7 or self.current_role == 'forager':
            if resources:
                nearest_resource = min(resources, key=lambda r: self._distance(r))
                if self._distance(nearest_resource) <= self.vision_range:
                    if self.move_towards(nearest_resource):
                        self.has_found_resource = True
                        self.last_resource_position = nearest_resource
                        self.resource_findings += 1
                        return "forage"

        # Flocking behavior when safe
        if (not nearby_predators and other_prey and 
            self.energy > self.config.prey_movement_cost * 2):
            flock_mates = [prey for prey in other_prey 
                          if prey.flock_id == self.flock_id and prey != self]
            if flock_mates:
                # Apply flocking rules
                center = self.calculate_flock_center(flock_mates)
                separation = self.calculate_separation(flock_mates)
                alignment = self.calculate_alignment(flock_mates)
                
                # Decide movement based on flocking rules
                if self._distance(center) > self.separation_radius:
                    if self.move_towards(center):
                        return "flock"

        # Random movement with energy consideration
        if (random.random() < 0.7 and 
            self.energy > self.config.prey_movement_cost * 2):
            self.position = (
                (self.position[0] + random.choice([-1, 0, 1])) % self.grid_size,
                (self.position[1] + random.choice([-1, 0, 1])) % self.grid_size
            )
            self.energy -= self.config.prey_movement_cost
            return "move"
        
        return "rest"

    def calculate_flock_center(self, flock_mates):
        """Calculate the center of the flock"""
        if not flock_mates:
            return self.position
        avg_x = sum(prey.position[0] for prey in flock_mates) / len(flock_mates)
        avg_y = sum(prey.position[1] for prey in flock_mates) / len(flock_mates)
        return (int(avg_x), int(avg_y))

    def calculate_separation(self, flock_mates):
        """Calculate separation vector from nearby flock mates"""
        separation = [0, 0]
        for mate in flock_mates:
            dist = self._distance(mate.position)
            if dist < self.separation_radius:
                separation[0] += self.position[0] - mate.position[0]
                separation[1] += self.position[1] - mate.position[1]
        return separation

    def calculate_alignment(self, flock_mates):
        """Calculate average movement direction of flock mates"""
        if not flock_mates:
            return (0, 0)
        avg_dx = sum(mate.position[0] - mate.last_position[0] 
                    if hasattr(mate, 'last_position') else 0 
                    for mate in flock_mates) / len(flock_mates)
        avg_dy = sum(mate.position[1] - mate.last_position[1] 
                    if hasattr(mate, 'last_position') else 0 
                    for mate in flock_mates) / len(flock_mates)
        return (avg_dx, avg_dy)

    def _distance(self, target_position):
        """Calculate Manhattan distance to target"""
        return abs(self.position[0] - target_position[0]) + \
               abs(self.position[1] - target_position[1])

    def _find_safe_tile(self, predator_positions):
        """Find a tile away from predators"""
        if not predator_positions:
            return None
            
        # Get all possible tiles within vision range
        possible_tiles = [
            (x, y) 
            for x in range(max(0, self.position[0] - self.vision_range),
                         min(self.grid_size, self.position[0] + self.vision_range + 1))
            for y in range(max(0, self.position[1] - self.vision_range),
                         min(self.grid_size, self.position[1] + self.vision_range + 1))
        ]
        
        # Filter for safe tiles
        safe_tiles = [
            tile for tile in possible_tiles
            if all(self._distance(predator) > self.vision_range 
                  for predator in predator_positions)
        ]
        
        # Return the closest safe tile
        if safe_tiles:
            return min(safe_tiles, key=lambda t: self._distance(t))
        return None

    def update_role(self, flock):
        """Update role based on flock needs and individual status"""
        if not flock:
            return
            
        current_roles = [mate.current_role for mate in flock]
        
        # Scouts should be high-energy agents
        if ('scout' not in current_roles and 
            self.energy > self.config.prey_reproduction_threshold * 0.8):
            self.current_role = 'scout'
            return
            
        # Defenders should be experienced agents
        if ('defender' not in current_roles and 
            self.successful_escapes > 5):
            self.current_role = 'defender'
            return
            
        # Default to forager
        if current_roles.count('forager') < len(flock) * 0.6:
            self.current_role = 'forager'
