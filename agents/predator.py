import random
from pydantic import BaseModel
from swarm import Agent
from typing import Tuple

class PredatorAgent(Agent, BaseModel):
    name: str
    position: Tuple[int, int]
    energy: int = 10
    grid_size: int

    def move_towards(self, target_position):
        # Move towards the target position
        new_x = self.position[0] + (1 if target_position[0] > self.position[0] else -1)
        new_y = self.position[1] + (1 if target_position[1] > self.position[1] else -1)
        self.position = (new_x % self.grid_size, new_y % self.grid_size)

    def act(self, prey_positions):
        if prey_positions:
            # Move towards the nearest prey
            nearest_prey = min(prey_positions, key=lambda p: abs(p[0] - self.position[0]) + abs(p[1] - self.position[1]))
            self.move_towards(nearest_prey)
            if self.position == nearest_prey:
                self.energy += 10  # Gain energy on eating prey
                return "eat"
        else:
            # Random movement if no prey is nearby
            self.position = (
                (self.position[0] + random.choice([-1, 0, 1])) % self.grid_size,
                (self.position[1] + random.choice([-1, 0, 1])) % self.grid_size,
            )
        self.energy -= 1  # Energy decreases with movement
        if self.energy <= 0:
            return "die"
        return "move"
