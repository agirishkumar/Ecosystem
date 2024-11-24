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
                position=(random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)),
                grid_size=self.grid_size,
                camouflage_tiles=[(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if random.random() < 0.1],  # 10% grid tiles for camouflage
                flock_id=random.randint(1, 3)  # Assign to one of 3 flocks
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

            action = prey.act(
                resources=[(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if self.resources[x, y] == 1],
                predator_positions=[predator.position for predator in self.predator_agents],
                other_prey=self.prey_agents
            )
            if action == "die":
                self.prey_agents.remove(prey)
                logger.info(f"{prey.name} has died.")
            elif prey.energy >= 10:  # Reproduction
                new_prey = PreyAgent(
                    name=f"Prey{len(self.prey_agents)}",
                    position=prey.position,
                    grid_size=self.grid_size,
                    camouflage_tiles=prey.camouflage_tiles,
                    flock_id=prey.flock_id
                )
                self.prey_agents.append(new_prey)
                prey.energy //= 2
                logger.info(f"{prey.name} reproduced at {prey.position}. New prey: {new_prey.name}")

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
        """
        Creates an animation of the ecosystem simulation showing predators, prey, and resources.
        
        Args:
            filename (str): Name of the output animation file
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Setup resource plot
        ax1.set_xlim(-1, self.grid_size)
        ax1.set_ylim(-1, self.grid_size)
        ax1.set_title("Resource Distribution")
        ax1.grid(True)
        
        # Setup population plot
        ax2.set_xlim(-1, self.grid_size)
        ax2.set_ylim(-1, self.grid_size)
        ax2.set_title("Agent Positions")
        ax2.grid(True)

        # Initialize plots
        resource_plot = ax1.imshow(
            self.resources, 
            cmap="YlGn",  # Yellow-Green colormap for better resource visibility
            extent=(-0.5, self.grid_size - 0.5, -0.5, self.grid_size - 0.5),
            alpha=0.7,
            vmin=0,
            vmax=1
        )
        plt.colorbar(resource_plot, ax=ax1, label="Resource Level")
        
        prey_scatter = ax2.scatter([], [], color="green", label="Prey", s=100)
        predator_scatter = ax2.scatter([], [], color="red", label="Predators", s=100)
        ax2.legend()

        def update(frame):
            prey_positions = frame["prey"]
            predator_positions = frame["predators"]
            resources = frame["resources"]

            # Update resource plot
            resource_plot.set_array(resources)
            
            # Update prey positions
            if prey_positions:
                prey_x, prey_y = zip(*prey_positions)
                prey_coords = np.column_stack((prey_x, prey_y))
            else:
                prey_coords = np.empty((0, 2))
                
            # Update predator positions
            if predator_positions:
                pred_x, pred_y = zip(*predator_positions)
                pred_coords = np.column_stack((pred_x, pred_y))
            else:
                pred_coords = np.empty((0, 2))

            prey_scatter.set_offsets(prey_coords)
            predator_scatter.set_offsets(pred_coords)

            # Update titles with current population counts
            frame_index = self.frames.index(frame)
            ax1.set_title(f"Resource Distribution (Step {frame_index + 1})")
            ax2.set_title(f"Agents (Prey: {len(prey_positions)}, Predators: {len(predator_positions)})")

            return resource_plot, prey_scatter, predator_scatter

        ani = FuncAnimation(fig, update, frames=self.frames, interval=500)
        ani.save(filename, writer="ffmpeg")
        plt.close()

    def plot_population_metrics(self, filename="population_metrics.png"):
        """
        Plots and saves the population metrics of prey and predator over the simulation steps.
        
        Args:
            filename (str): Name of the output file for the population plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot populations
        plt.plot(self.prey_population, label="Prey Population", color="green", linewidth=2)
        plt.plot(self.predator_population, label="Predator Population", color="red", linewidth=2)
        
        # Add labels and title
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Population Count", fontsize=12)
        plt.title("Population Dynamics Over Time", fontsize=14, pad=15)
        
        # Customize the plot
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # Add some padding to the plot
        plt.margins(x=0.02)
        
        # Customize the tick labels
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        # Ensure the plot fits within the figure bounds
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        # Display the plot
        plt.show()
        
        # Close the figure to free up memory
        plt.close()

