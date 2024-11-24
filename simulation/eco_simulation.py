import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from agents.prey import PreyAgent
from agents.predator import PredatorAgent
import logging
import json
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EcosystemSimulation:
    def __init__(self, grid_size=10, num_prey=5, num_predators=2, resource_regen_rate=0.2):
        self.grid_size = grid_size
        self.resource_regen_rate = resource_regen_rate

        # Initialize resources: 1 indicates a resource present, 0 indicates absent
        self.resources = np.random.randint(0, 2, size=(grid_size, grid_size))

        # Initialize obstacles: 2 indicates an obstacle
        self.obstacles = np.random.choice([0, 2], size=(grid_size, grid_size), p=[0.9, 0.1])

        self.prey_agents = [
            PreyAgent(
                name=f"Prey{i}",
                position=(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)),
                grid_size=grid_size
            ) for i in range(num_prey)
        ]
        self.predator_agents = [
            PredatorAgent(
                name=f"Predator{i}",
                position=(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)),
                grid_size=grid_size,
                group_id=random.randint(1, 3)  # Assign to one of 3 groups
            ) for i in range(num_predators)
        ]

        self.frames = []
        self.prey_population = []
        self.predator_population = []

    def step(self):
        # Determine the current season
        seasons = ["Spring", "Summer", "Fall", "Winter"]
        current_season = (len(self.frames) // 10) % len(seasons)
        if current_season in ["Spring", "Summer"]:
            self.resource_regen_rate = 0.3
        else:
            self.resource_regen_rate = 0.1

        for prey in self.prey_agents[:]:
            x, y = prey.position
            if self.resources[x, y] == 1:  # Resource is present
                prey.energy += 5  # Gain energy
                self.resources[x, y] = 0  # Consume the resource
                logger.info(f"{prey.name} consumed a resource at {prey.position}.")

            action = prey.act()
            if action == "die":
                self.prey_agents.remove(prey)
                logger.info(f"{prey.name} has died.")
            elif prey.energy >= 10:  # Reproduction
                new_prey = PreyAgent(
                    name=f"Prey{len(self.prey_agents)}",
                    position=prey.position,
                    grid_size=self.grid_size
                )
                self.prey_agents.append(new_prey)
                prey.energy //= 2

        prey_positions = [prey.position for prey in self.prey_agents]
        resources = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if self.resources[x, y] == 1]

        for predator in self.predator_agents[:]:
            action = predator.act(prey_positions, self.predator_agents, resources)
            if action == "die":
                self.predator_agents.remove(predator)
            elif action == "move":
                # Check if the predator is on the same position as prey
                prey_at_position = [prey for prey in self.prey_agents if prey.position == predator.position]
                if prey_at_position:
                    # Predator eats the prey
                    for prey in prey_at_position:
                        self.prey_agents.remove(prey)
                        predator.energy += 10  # Gain energy for successful hunt
                        logger.info(f"{predator.name} ate prey at {predator.position}.")
                elif predator.position in resources:
                    # Consume a resource if predator happens to land on it
                    resources.remove(predator.position)  # Remove the resource
                    predator.energy += 5  # Gain energy from the resource
                    logger.info(f"{predator.name} consumed a resource at {predator.position}.")
                else:
                    # No interaction; just a movement
                    logger.info(f"{predator.name} moved to {predator.position}.")

                # Handle reproduction
                if predator.energy >= 15:  # Energy threshold for reproduction
                    new_predator = PredatorAgent(
                        name=f"Predator{len(self.predator_agents)}",
                        position=predator.position,  # Newborn appears at the parent's position
                        energy=predator.energy // 2,  # Split energy with offspring
                        grid_size=self.grid_size,
                        group_id=predator.group_id  # Inherit parent's group
                    )
                    self.predator_agents.append(new_predator)
                    predator.energy //= 2  # Reduce parent's energy after reproduction
                    logger.info(f"{predator.name} reproduced at {predator.position}. New predator: {new_predator.name}")

        # Regenerate resources based on the current season
        self.resources += np.random.choice([0, 1], size=self.resources.shape, p=[1 - self.resource_regen_rate, self.resource_regen_rate])
        self.resources = np.clip(self.resources, 0, 1)

        frame_data = {
            "prey": [prey.position for prey in self.prey_agents],
            "predators": [predator.position for predator in self.predator_agents],
            "resources": self.resources.copy()
        }
        self.frames.append(frame_data)

        self.prey_population.append(len(self.prey_agents))
        self.predator_population.append(len(self.predator_agents))

        if not self.prey_agents or not self.predator_agents:
            return False
        return True

    def run_simulation(self, steps=20):
        for step in range(steps):
            if not self.step():
                break

    def save_state(self, filename="simulation_state.json"):
        state = {
            "prey": [{"name": p.name, "position": p.position, "energy": p.energy} for p in self.prey_agents],
            "predators": [{"name": p.name, "position": p.position, "energy": p.energy} for p in self.predator_agents],
            "resources": self.resources.tolist(),
            "obstacles": self.obstacles.tolist()
        }
        with open(filename, "w") as f:
            json.dump(state, f)

    def load_state(self, filename="simulation_state.json"):
        with open(filename, "r") as f:
            state = json.load(f)
        self.resources = np.array(state["resources"])
        self.obstacles = np.array(state["obstacles"])
        self.prey_agents = [PreyAgent(**p) for p in state["prey"]]
        self.predator_agents = [PredatorAgent(**p) for p in state["predators"]]

    def create_animation(self, filename="ecosystem_simulation.mp4"):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1, self.grid_size)
        ax.set_ylim(-1, self.grid_size)
        ax.grid(True)

        prey_scatter = ax.scatter([], [], color="green", label="Prey", s=100)
        predator_scatter = ax.scatter([], [], color="red", label="Predators", s=100)
        resource_plot = ax.imshow(self.resources, cmap="Greens", extent=(-0.5, self.grid_size - 0.5, -0.5, self.grid_size - 0.5), alpha=0.3)
        ax.legend()

        def update(frame):
            prey_positions = frame["prey"]
            predator_positions = frame["predators"]
            resources = frame["resources"]

            if prey_positions:
                prey_x, prey_y = zip(*prey_positions)
                prey_coords = np.column_stack((prey_x, prey_y))
            else:
                prey_coords = np.empty((0, 2))

            if predator_positions:
                pred_x, pred_y = zip(*predator_positions)
                pred_coords = np.column_stack((pred_x, pred_y))
            else:
                pred_coords = np.empty((0, 2))

            prey_scatter.set_offsets(prey_coords)
            predator_scatter.set_offsets(pred_coords)
            resource_plot.set_data(resources)

            ax.set_title(f"Step {self.frames.index(frame) + 1} | Prey: {len(prey_positions)} | Predators: {len(predator_positions)}")

            return prey_scatter, predator_scatter, resource_plot

        ani = FuncAnimation(fig, update, frames=self.frames, interval=500)
        ani.save(filename, writer="ffmpeg")
        plt.close()

    def plot_population_metrics(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.prey_population, label="Prey Population", color="green")
        plt.plot(self.predator_population, label="Predator Population", color="red")
        plt.xlabel("Steps")
        plt.ylabel("Population")
        plt.title("Population Dynamics")
        plt.legend()
        plt.savefig("population_metrics.png")
        plt.show()
