from swarm import Agent
from pydantic import BaseModel
from typing import Tuple, List
import random

class PreyAgent(Agent, BaseModel):
    name: str
    position: Tuple[int, int]
    energy: int = 5
    grid_size: int
    vision_range: int = 3
    camouflage_tiles: List[Tuple[int, int]] = None  # Tiles where prey can blend
    flock_id: int = None  # Identifier for flocking behaviors

    def move_towards(self, target_position):
        """
        Move towards the target position using a basic step-by-step approach.
        """
        new_x = self.position[0] + (1 if target_position[0] > self.position[0] else -1 if target_position[0] < self.position[0] else 0)
        new_y = self.position[1] + (1 if target_position[1] > self.position[1] else -1 if target_position[1] < self.position[1] else 0)
        self.position = (new_x % self.grid_size, new_y % self.grid_size)

    def act(self, resources: List[Tuple[int, int]], predator_positions: List[Tuple[int, int]], other_prey: List['PreyAgent']):
        """
        Decide action based on energy level, nearby resources, predators, and flocking behavior.
        """
        # Risk management: Avoid predators if nearby
        nearby_predators = [
            predator for predator in predator_positions
            if self._distance(predator) <= self.vision_range
        ]
        if self.energy > 10 and nearby_predators:
            # High energy: Move away from predator zones
            safe_tile = self._find_safe_tile(predator_positions)
            if safe_tile:
                self.move_towards(safe_tile)
                return "evade"

        # Resource prioritization
        if self.energy <= 5:
            # Low energy: Seek nearest resource
            if resources:
                nearest_resource = min(resources, key=lambda r: self._distance(r))
                self.move_towards(nearest_resource)
                return "forage"

        # Camouflage
        if self.position in self.camouflage_tiles:
            return "camouflage"  # Blend in; no movement

        # Social behaviors: Flocking
        if self.flock_id is not None and other_prey:
            flock_mates = [
                prey for prey in other_prey if prey.flock_id == self.flock_id and prey != self
            ]
            if flock_mates:
                avg_x = sum(prey.position[0] for prey in flock_mates) // len(flock_mates)
                avg_y = sum(prey.position[1] for prey in flock_mates) // len(flock_mates)
                self.move_towards((avg_x, avg_y))  # Cohesion
                return "flock"

        # Random movement (fallback)
        self.position = (
            (self.position[0] + random.choice([-1, 0, 1])) % self.grid_size,
            (self.position[1] + random.choice([-1, 0, 1])) % self.grid_size,
        )
        return "move"

    def _distance(self, target_position):
        """
        Calculate Manhattan distance to a target position.
        """
        return abs(self.position[0] - target_position[0]) + abs(self.position[1] - target_position[1])

    def _find_safe_tile(self, predator_positions):
        """
        Find a tile away from predators.
        """
        all_tiles = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
        ]
        safe_tiles = [
            tile for tile in all_tiles
            if all(self._distance(tile) > self.vision_range for predator in predator_positions)
        ]
        return random.choice(safe_tiles) if safe_tiles else None
