import random
from pydantic import BaseModel
from swarm import Agent
from typing import Tuple

class PreyAgent(Agent, BaseModel):
    name: str
    position: Tuple[int, int]
    energy: int = 5
    grid_size: int

    def move(self):
        # Random movement within the grid
        self.position = (
            (self.position[0] + random.choice([-1, 0, 1])) % self.grid_size,
            (self.position[1] + random.choice([-1, 0, 1])) % self.grid_size,
        )

    def act(self):
        self.move()
        self.energy -= 1  # Energy decreases on every move
        if self.energy <= 0:
            return "die"
        return "move"
