from swarm import Agent
from pydantic import BaseModel
from typing import Tuple, List
import random

class PredatorAgent(Agent, BaseModel):
    name: str
    position: Tuple[int, int]
    energy: int = 10
    grid_size: int
    vision_range: int = 3
    group_id: int = None  # Group identifier for coordination

    def move_towards(self, target_position):
        # Move towards the target position
        new_x = self.position[0] + (1 if target_position[0] > self.position[0] else -1)
        new_y = self.position[1] + (1 if target_position[1] > self.position[1] else -1)
        self.position = (new_x % self.grid_size, new_y % self.grid_size)

    def communicate(self, other_predators: List['PredatorAgent'], prey_positions: List[Tuple[int, int]]):
        """
        Share prey location with nearby predators in the same group.
        """
        if not prey_positions:
            return None  # No prey to communicate

        for predator in other_predators:
            if predator.group_id == self.group_id and self._distance(predator.position) <= self.vision_range:
                # Share the nearest prey position
                nearest_prey = min(prey_positions, key=lambda p: abs(p[0] - self.position[0]) + abs(p[1] - self.position[1]))
                predator.move_towards(nearest_prey)

    def act(self, prey_positions: List[Tuple[int, int]], other_predators: List['PredatorAgent'], resources: List[Tuple[int, int]]):
        # Energy-based decision making
        if self.energy > 15:
            self.communicate(other_predators, prey_positions)  # Active group hunting
        elif 5 < self.energy <= 15:
            # Move strategically towards prey or stay within resource-dense areas
            if prey_positions:
                nearest_prey = min(prey_positions, key=lambda p: abs(p[0] - self.position[0]) + abs(p[1] - self.position[1]))
                self.move_towards(nearest_prey)
        else:
            # Low energy: Rest or move to resource areas
            if resources:
                nearest_resource = min(resources, key=lambda r: abs(r[0] - self.position[0]) + abs(r[1] - self.position[1]))
                self.move_towards(nearest_resource)
            else:
                self.energy -= 1  # Rest and conserve energy
        
        # Check for energy depletion
        if self.energy <= 0:
            return "die"
        
        self.energy -= 1  # Moving costs energy
        return "move"

    def _distance(self, target_position):
        """
        Calculate Manhattan distance to a target position.
        """
        return abs(self.position[0] - target_position[0]) + abs(self.position[1] - target_position[1])
