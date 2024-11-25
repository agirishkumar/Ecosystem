import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from agents.prey import PreyAgent
from agents.predator import PredatorAgent
import logging
import json
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from typing import List, Dict, Any, Optional
from config import SimulationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EcosystemSimulation:
    def __init__(self, config: SimulationConfig):
        """
        Initialize the ecosystem simulation with configuration.
        
        Args:
            config (SimulationConfig): Configuration object with simulation parameters
        """
        self.config = config
        self.grid_size = config.grid_size
        
        # Initialize environmental maps
        self.resources = np.random.randint(0, 2, size=(self.grid_size, self.grid_size))
        self.obstacles = np.random.choice(
            [0, 2], 
            size=(self.grid_size, self.grid_size), 
            p=[1 - config.obstacle_density, config.obstacle_density]
        )
        self.territory_map = np.zeros((self.grid_size, self.grid_size))
        self.pheromone_map = np.zeros((self.grid_size, self.grid_size))
        
        # Initialize metrics tracking
        self.frames: List[Dict[str, Any]] = []
        self.prey_population: List[int] = []
        self.predator_population: List[int] = []
        self.hunting_success_rate: List[float] = []
        self.territory_changes: List[float] = []
        self.energy_distribution: List[Dict[str, List[float]]] = []
        self.resource_availability: List[float] = []
        
        # Initialize agents
        self.initialize_agents()

    def initialize_agents(self):
        """Initialize prey and predator agents with configuration parameters"""
        # Initialize prey agents
        self.prey_agents = [
            PreyAgent(
                name=f"Prey{i}",
                position=self.get_random_empty_position(),
                grid_size=self.grid_size,
                config=self.config,
                energy=self.config.prey_initial_energy,
                vision_range=self.config.prey_vision_range,
                camouflage_tiles=self.generate_camouflage_tiles(),
                flock_id=random.randint(1, 3)
            ) for i in range(self.config.num_prey)
        ]

        # Initialize predator agents
        self.predator_agents = [
            PredatorAgent(
                name=f"Predator{i}",
                position=self.get_random_empty_position(),
                grid_size=self.grid_size,
                config=self.config,
                energy=self.config.predator_initial_energy,
                vision_range=self.config.predator_vision_range,
                group_id=random.randint(1, 3)
            ) for i in range(self.config.num_predators)
        ]

        # Initialize territories
        self.update_territory_map(initial=True)

    def get_random_empty_position(self) -> tuple:
        """Get a random position that's not occupied by an obstacle"""
        while True:
            pos = (random.randint(0, self.grid_size - 1), 
                  random.randint(0, self.grid_size - 1))
            if self.obstacles[pos[0], pos[1]] == 0:
                return pos

    def generate_camouflage_tiles(self) -> List[tuple]:
        """Generate camouflage tiles based on configuration"""
        return [(x, y) 
                for x in range(self.grid_size) 
                for y in range(self.grid_size) 
                if random.random() < self.config.prey_camouflage_chance 
                and self.obstacles[x, y] == 0]

    def update_territory_map(self, initial: bool = False):
        """Update territory map with predator influences"""
        if not initial:
            self.territory_map *= self.config.territory_decay_rate
        
        for predator in self.predator_agents:
            self.territory_map = predator.update_territory(self.territory_map)
            
        if not initial:
            self.territory_changes.append(
                np.sum(np.abs(np.diff(self.territory_map)))
            )

    def process_resources(self, season: str):
        """Process resource regeneration based on season"""
        regen_rate = (self.config.summer_resource_rate 
                     if season in ["Spring", "Summer"] 
                     else self.config.winter_resource_rate)
        
        # Regenerate resources
        new_resources = np.random.choice(
            [0, 1], 
            size=self.resources.shape, 
            p=[1 - regen_rate, regen_rate]
        )
        self.resources = np.clip(self.resources + new_resources, 0, 1)
        
        # Track resource availability
        self.resource_availability.append(np.mean(self.resources))

    def process_prey_actions(self):
        """Process prey actions and handle interactions"""
        for prey in self.prey_agents[:]:
            # Apply base energy decay
            prey.energy *= self.config.energy_decay_rate
            
            # Process actions
            if prey.energy <= self.config.prey_starve_threshold:
                self.prey_agents.remove(prey)
                continue
                
            action = prey.act(
                resources=self.get_resource_positions(),
                predator_positions=[p.position for p in self.predator_agents],
                other_prey=self.prey_agents
            )
            
            # Handle resource consumption
            if action == "forage" and self.resources[prey.position] == 1:
                prey.energy = min(
                    prey.energy + self.config.resource_energy_value,
                    self.config.prey_max_energy
                )
                self.resources[prey.position] = 0
            
            # Handle reproduction
            if (prey.energy >= self.config.prey_reproduction_threshold and 
                len(self.prey_agents) < self.config.max_population):
                self.handle_prey_reproduction(prey)

    def process_predator_actions(self):
        """Process predator actions and track hunting success"""
        successful_hunts = 0
        total_hunts = 0
        
        for predator in self.predator_agents[:]:
            # Apply base energy decay
            predator.energy *= self.config.energy_decay_rate
            
            # Process actions
            if predator.energy <= self.config.predator_starve_threshold:
                self.predator_agents.remove(predator)
                continue
                
            action = predator.act(
                [prey.position for prey in self.prey_agents],
                self.predator_agents,
                self.get_resource_positions(),
                self.territory_map
            )
            
            # Handle hunting
            if action == "hunt":
                total_hunts += 1
                success = self.handle_hunting(predator)
                if success:
                    successful_hunts += 1
            
            # Handle reproduction
            if (predator.energy >= self.config.predator_reproduction_threshold and
                len(self.predator_agents) < self.config.max_population):
                self.handle_predator_reproduction(predator)
        
        # Track hunting success
        if total_hunts > 0:
            self.hunting_success_rate.append(successful_hunts / total_hunts)
        else:
            self.hunting_success_rate.append(0.0)

    def handle_hunting(self, predator: PredatorAgent) -> bool:
        """Handle predator hunting attempt"""
        prey_at_position = [prey for prey in self.prey_agents 
                          if prey.position == predator.position]
        if prey_at_position:
            for prey in prey_at_position:
                self.prey_agents.remove(prey)
            predator.energy = min(
                predator.energy + self.config.predator_hunt_reward,
                self.config.predator_max_energy
            )
            predator.adapt_strategy(True, 'chase')
            return True
        else:
            predator.adapt_strategy(False, 'chase')
            return False

    def handle_prey_reproduction(self, prey: PreyAgent):
        """Handle prey reproduction"""
        if (random.random() < 0.5 and  # 50% chance to reproduce
            prey.energy >= self.config.prey_reproduction_threshold * 1.5):  # Ensure enough energy
            offspring_energy = prey.energy * 0.5
            prey.energy -= offspring_energy + self.config.prey_reproduction_cost
            
            new_prey = PreyAgent(
                name=f"Prey{len(self.prey_agents)}",
                position=prey.position,
                grid_size=self.grid_size,
                config=self.config,  # Pass config
                energy=offspring_energy,
                vision_range=self.config.prey_vision_range,
                camouflage_tiles=prey.camouflage_tiles,
                flock_id=prey.flock_id
            )
            self.prey_agents.append(new_prey)
            logger.info(f"{prey.name} reproduced at {prey.position}. New prey: {new_prey.name}")

    def handle_predator_reproduction(self, predator: PredatorAgent):
        """Handle predator reproduction"""
        if (random.random() < 0.4 and  # 40% chance to reproduce
            predator.energy >= self.config.predator_reproduction_threshold * 1.5):  # Ensure enough energy
            offspring_energy = predator.energy * 0.5
            predator.energy -= offspring_energy + self.config.predator_reproduction_cost
            
            new_predator = PredatorAgent(
                name=f"Predator{len(self.predator_agents)}",
                position=predator.position,
                grid_size=self.grid_size,
                config=self.config,  # Pass config
                energy=offspring_energy,
                vision_range=self.config.predator_vision_range,
                group_id=predator.group_id
            )
            self.predator_agents.append(new_predator)
            logger.info(f"{predator.name} reproduced at {predator.position}. New predator: {new_predator.name}")


    def get_resource_positions(self) -> List[tuple]:
        """Get positions of available resources"""
        return [(x, y) 
                for x in range(self.grid_size) 
                for y in range(self.grid_size) 
                if self.resources[x, y] == 1]

    def step(self) -> bool:
        """Execute one simulation step"""
        # Determine season
        current_season = self.get_current_season()
        
        # Process environmental changes
        self.process_resources(current_season)
        self.pheromone_map *= self.config.pheromone_decay
        self.update_territory_map()
        
        # Process agent actions
        self.process_prey_actions()
        self.process_predator_actions()
        
        # Update metrics
        self.update_metrics()
        
        # Check end conditions
        if (len(self.prey_agents) < self.config.min_population or 
            len(self.predator_agents) < self.config.min_population):
            return False
        
        return True

    def get_current_season(self) -> str:
        """Determine current season based on step count"""
        seasons = ["Spring", "Summer", "Fall", "Winter"]
        return seasons[(len(self.frames) // self.config.season_length) % len(seasons)]

    def update_metrics(self):
        """Update simulation metrics"""
        self.frames.append({
            "prey": [prey.position for prey in self.prey_agents],
            "predators": [predator.position for predator in self.predator_agents],
            "resources": self.resources.copy(),
            "territory": self.territory_map.copy(),
            "season": self.get_current_season()
        })
        
        self.prey_population.append(len(self.prey_agents))
        self.predator_population.append(len(self.predator_agents))
        
        # Track energy distribution
        self.energy_distribution.append({
            "prey": [prey.energy for prey in self.prey_agents],
            "predators": [pred.energy for pred in self.predator_agents]
        })

    def create_animation(self, filename="ecosystem_simulation.mp4"):
        """Create animation of simulation"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Setup plots
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(-1, self.grid_size)
            ax.set_ylim(-1, self.grid_size)
            ax.grid(True)
        
        ax1.set_title("Resources")
        ax2.set_title("Agents")
        ax3.set_title("Territories")

        # Initialize plots
        resource_plot = ax1.imshow(
            self.resources, 
            cmap="YlGn",
            extent=(-0.5, self.grid_size - 0.5, -0.5, self.grid_size - 0.5),
            alpha=0.7
        )
        territory_plot = ax3.imshow(
            self.territory_map,
            cmap="viridis",
            extent=(-0.5, self.grid_size - 0.5, -0.5, self.grid_size - 0.5),
            alpha=0.5
        )
        prey_scatter = ax2.scatter([], [], color="green", label="Prey", s=100)
        predator_scatter = ax2.scatter([], [], color="red", label="Predators", s=100)
        
        plt.colorbar(resource_plot, ax=ax1, label="Resource Level")
        plt.colorbar(territory_plot, ax=ax3, label="Territory")
        ax2.legend()

        def update(frame):
            # Update resources
            resource_plot.set_array(frame["resources"])
            territory_plot.set_array(frame["territory"])
            
            # Update agents
            prey_positions = frame["prey"]
            predator_positions = frame["predators"]
            
            # Handle prey positions
            if prey_positions:
                prey_x, prey_y = zip(*prey_positions)
                prey_coords = np.column_stack((prey_x, prey_y))
            else:
                prey_coords = np.empty((0, 2))
                
            # Handle predator positions
            if predator_positions:
                pred_x, pred_y = zip(*predator_positions)
                pred_coords = np.column_stack((pred_x, pred_y))
            else:
                pred_coords = np.empty((0, 2))
                
            prey_scatter.set_offsets(prey_coords)
            predator_scatter.set_offsets(pred_coords)
            
            # Update titles
            frame_idx = self.frames.index(frame)
            season = self.get_current_season()
            ax1.set_title(f"Resources ({season})")
            ax2.set_title(f"Step {frame_idx}: {len(prey_positions)} Prey, {len(predator_positions)} Pred.")
            ax3.set_title(f"Territories")
            
            return resource_plot, prey_scatter, predator_scatter, territory_plot

        ani = FuncAnimation(
            fig, 
            update, 
            frames=self.frames,
            interval=200,
            blit=True
        )
        
        ani.save(filename, writer='ffmpeg')
        plt.close()

    def plot_population_metrics(self, filename="population_metrics.png"):
        """
        Plots and saves the population metrics of prey and predator over the simulation steps.
        
        Args:
            filename (str): Name of the output file for the population plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot populations
        steps = range(len(self.prey_population))
        plt.plot(steps, self.prey_population, label="Prey Population", 
                color="green", linewidth=2)
        plt.plot(steps, self.predator_population, label="Predator Population", 
                color="red", linewidth=2)
        
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

    def plot_advanced_metrics(self, filename="advanced_metrics.png"):
        """
        Plot advanced simulation metrics including hunting success and territory changes.
        
        Args:
            filename (str): Name of the output file for the advanced metrics plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        steps = range(len(self.prey_population))
        
        # Population dynamics
        ax1.plot(steps, self.prey_population, 'g-', label='Prey')
        ax1.plot(steps, self.predator_population, 'r-', label='Predators')
        ax1.set_title('Population Dynamics')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Population Count')
        ax1.legend()
        ax1.grid(True)
        
        # Hunting success rate
        if self.hunting_success_rate:
            hunt_steps = range(len(self.hunting_success_rate))
            ax2.plot(hunt_steps, self.hunting_success_rate, 'b-')
            ax2.set_title('Hunting Success Rate')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Success Rate')
            ax2.grid(True)
        
        # Territory changes
        if self.territory_changes:
            territory_steps = range(len(self.territory_changes))
            ax3.plot(territory_steps, self.territory_changes, 'purple')
            ax3.set_title('Territory Changes Over Time')
            ax3.set_xlabel('Steps')
            ax3.set_ylabel('Change Magnitude')
            ax3.grid(True)
        
        # Final territory map
        territory_plot = ax4.imshow(self.territory_map, cmap='viridis')
        ax4.set_title('Final Territory Distribution')
        plt.colorbar(territory_plot, ax=ax4)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def save_metrics(self, filename="simulation_metrics.json"):
        """
        Save simulation metrics to a JSON file.
        
        Args:
            filename (str): Name of the output metrics file
        """
        metrics = {
            "prey_population": self.prey_population,
            "predator_population": self.predator_population,
            "hunting_success_rate": self.hunting_success_rate,
            "territory_changes": self.territory_changes,
            "final_populations": {
                "prey": len(self.prey_agents),
                "predators": len(self.predator_agents)
            },
            "average_hunting_success": np.mean(self.hunting_success_rate) if self.hunting_success_rate else 0,
            "simulation_length": len(self.frames)
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=4)

    def save_state(self, filename="simulation_state.json"):
        """Save the current simulation state"""
        state = {
            "prey": [{"name": p.name, "position": p.position, "energy": p.energy} 
                    for p in self.prey_agents],
            "predators": [{"name": p.name, "position": p.position, "energy": p.energy} 
                         for p in self.predator_agents],
            "resources": self.resources.tolist(),
            "obstacles": self.obstacles.tolist(),
            "territory_map": self.territory_map.tolist(),
            "pheromone_map": self.pheromone_map.tolist()
        }
        with open(filename, "w") as f:
            json.dump(state, f)
        logger.info(f"Simulation state saved to {filename}")

    def load_state(self, filename="simulation_state.json"):
        """Load a saved simulation state"""
        with open(filename, "r") as f:
            state = json.load(f)
        
        self.resources = np.array(state["resources"])
        self.obstacles = np.array(state["obstacles"])
        self.territory_map = np.array(state["territory_map"])
        self.pheromone_map = np.array(state["pheromone_map"])
        
        # Add config when creating agents
        self.prey_agents = [
            PreyAgent(**{**p, "config": self.config, "grid_size": self.grid_size}) 
            for p in state["prey"]
        ]
        self.predator_agents = [
            PredatorAgent(**{**p, "config": self.config, "grid_size": self.grid_size}) 
            for p in state["predators"]
        ]
        
        logger.info(f"Simulation state loaded from {filename}")

    import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from agents.prey import PreyAgent
from agents.predator import PredatorAgent
import logging
import json
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from typing import List, Dict, Any, Optional
from config import SimulationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EcosystemSimulation:
    def __init__(self, config: SimulationConfig):
        """
        Initialize the ecosystem simulation with configuration.
        
        Args:
            config (SimulationConfig): Configuration object with simulation parameters
        """
        self.config = config
        self.grid_size = config.grid_size
        
        # Initialize environmental maps
        self.resources = np.random.randint(0, 2, size=(self.grid_size, self.grid_size))
        self.obstacles = np.random.choice(
            [0, 2], 
            size=(self.grid_size, self.grid_size), 
            p=[1 - config.obstacle_density, config.obstacle_density]
        )
        self.territory_map = np.zeros((self.grid_size, self.grid_size))
        self.pheromone_map = np.zeros((self.grid_size, self.grid_size))
        
        # Initialize metrics tracking
        self.frames: List[Dict[str, Any]] = []
        self.prey_population: List[int] = []
        self.predator_population: List[int] = []
        self.hunting_success_rate: List[float] = []
        self.territory_changes: List[float] = []
        self.energy_distribution: List[Dict[str, List[float]]] = []
        self.resource_availability: List[float] = []
        
        # Initialize agents
        self.initialize_agents()

    def initialize_agents(self):
        """Initialize prey and predator agents with configuration parameters"""
        # Initialize prey agents
        self.prey_agents = [
            PreyAgent(
                name=f"Prey{i}",
                position=self.get_random_empty_position(),
                grid_size=self.grid_size,
                config=self.config,
                energy=self.config.prey_initial_energy,
                vision_range=self.config.prey_vision_range,
                camouflage_tiles=self.generate_camouflage_tiles(),
                flock_id=random.randint(1, 3)
            ) for i in range(self.config.num_prey)
        ]

        # Initialize predator agents
        self.predator_agents = [
            PredatorAgent(
                name=f"Predator{i}",
                position=self.get_random_empty_position(),
                grid_size=self.grid_size,
                config=self.config,
                energy=self.config.predator_initial_energy,
                vision_range=self.config.predator_vision_range,
                group_id=random.randint(1, 3)
            ) for i in range(self.config.num_predators)
        ]

        # Initialize territories
        self.update_territory_map(initial=True)

    def get_random_empty_position(self) -> tuple:
        """Get a random position that's not occupied by an obstacle"""
        while True:
            pos = (random.randint(0, self.grid_size - 1), 
                  random.randint(0, self.grid_size - 1))
            if self.obstacles[pos[0], pos[1]] == 0:
                return pos

    def generate_camouflage_tiles(self) -> List[tuple]:
        """Generate camouflage tiles based on configuration"""
        return [(x, y) 
                for x in range(self.grid_size) 
                for y in range(self.grid_size) 
                if random.random() < self.config.prey_camouflage_chance 
                and self.obstacles[x, y] == 0]

    def update_territory_map(self, initial: bool = False):
        """Update territory map with predator influences"""
        if not initial:
            self.territory_map *= self.config.territory_decay_rate
        
        for predator in self.predator_agents:
            self.territory_map = predator.update_territory(self.territory_map)
            
        if not initial:
            self.territory_changes.append(
                np.sum(np.abs(np.diff(self.territory_map)))
            )

    def process_resources(self, season: str):
        """Process resource regeneration based on season"""
        regen_rate = (self.config.summer_resource_rate 
                     if season in ["Spring", "Summer"] 
                     else self.config.winter_resource_rate)
        
        # Regenerate resources
        new_resources = np.random.choice(
            [0, 1], 
            size=self.resources.shape, 
            p=[1 - regen_rate, regen_rate]
        )
        self.resources = np.clip(self.resources + new_resources, 0, 1)
        
        # Track resource availability
        self.resource_availability.append(np.mean(self.resources))

    def process_prey_actions(self):
        """Process prey actions and handle interactions"""
        for prey in self.prey_agents[:]:
            # Apply base energy decay
            prey.energy *= self.config.energy_decay_rate
            
            # Process actions
            if prey.energy <= self.config.prey_starve_threshold:
                self.prey_agents.remove(prey)
                continue
                
            action = prey.act(
                resources=self.get_resource_positions(),
                predator_positions=[p.position for p in self.predator_agents],
                other_prey=self.prey_agents
            )
            
            # Handle resource consumption
            if action == "forage" and self.resources[prey.position] == 1:
                prey.energy = min(
                    prey.energy + self.config.resource_energy_value,
                    self.config.prey_max_energy
                )
                self.resources[prey.position] = 0
            
            # Handle reproduction
            if (prey.energy >= self.config.prey_reproduction_threshold and 
                len(self.prey_agents) < self.config.max_population):
                self.handle_prey_reproduction(prey)

    def process_predator_actions(self):
        """Process predator actions and track hunting success"""
        successful_hunts = 0
        total_hunts = 0
        
        for predator in self.predator_agents[:]:
            # Apply base energy decay
            predator.energy *= self.config.energy_decay_rate
            
            # Process actions
            if predator.energy <= self.config.predator_starve_threshold:
                self.predator_agents.remove(predator)
                continue
                
            action = predator.act(
                [prey.position for prey in self.prey_agents],
                self.predator_agents,
                self.get_resource_positions(),
                self.territory_map
            )
            
            # Handle hunting
            if action == "hunt":
                total_hunts += 1
                success = self.handle_hunting(predator)
                if success:
                    successful_hunts += 1
            
            # Handle reproduction
            if (predator.energy >= self.config.predator_reproduction_threshold and
                len(self.predator_agents) < self.config.max_population):
                self.handle_predator_reproduction(predator)
        
        # Track hunting success
        if total_hunts > 0:
            self.hunting_success_rate.append(successful_hunts / total_hunts)
        else:
            self.hunting_success_rate.append(0.0)

    def handle_hunting(self, predator: PredatorAgent) -> bool:
        """Handle predator hunting attempt"""
        prey_at_position = [prey for prey in self.prey_agents 
                          if prey.position == predator.position]
        if prey_at_position:
            for prey in prey_at_position:
                self.prey_agents.remove(prey)
            predator.energy = min(
                predator.energy + self.config.predator_hunt_reward,
                self.config.predator_max_energy
            )
            predator.adapt_strategy(True, 'chase')
            return True
        else:
            predator.adapt_strategy(False, 'chase')
            return False

    def handle_prey_reproduction(self, prey: PreyAgent):
        """Handle prey reproduction"""
        if (random.random() < 0.5 and  # 50% chance to reproduce
            prey.energy >= self.config.prey_reproduction_threshold * 1.5):  # Ensure enough energy
            offspring_energy = prey.energy * 0.5
            prey.energy -= offspring_energy + self.config.prey_reproduction_cost
            
            new_prey = PreyAgent(
                name=f"Prey{len(self.prey_agents)}",
                position=prey.position,
                grid_size=self.grid_size,
                config=self.config,  # Pass config
                energy=offspring_energy,
                vision_range=self.config.prey_vision_range,
                camouflage_tiles=prey.camouflage_tiles,
                flock_id=prey.flock_id
            )
            self.prey_agents.append(new_prey)
            logger.info(f"{prey.name} reproduced at {prey.position}. New prey: {new_prey.name}")

    def handle_predator_reproduction(self, predator: PredatorAgent):
        """Handle predator reproduction"""
        if (random.random() < 0.4 and  # 40% chance to reproduce
            predator.energy >= self.config.predator_reproduction_threshold * 1.5):  # Ensure enough energy
            offspring_energy = predator.energy * 0.5
            predator.energy -= offspring_energy + self.config.predator_reproduction_cost
            
            new_predator = PredatorAgent(
                name=f"Predator{len(self.predator_agents)}",
                position=predator.position,
                grid_size=self.grid_size,
                config=self.config,  # Pass config
                energy=offspring_energy,
                vision_range=self.config.predator_vision_range,
                group_id=predator.group_id
            )
            self.predator_agents.append(new_predator)
            logger.info(f"{predator.name} reproduced at {predator.position}. New predator: {new_predator.name}")


    def get_resource_positions(self) -> List[tuple]:
        """Get positions of available resources"""
        return [(x, y) 
                for x in range(self.grid_size) 
                for y in range(self.grid_size) 
                if self.resources[x, y] == 1]

    def step(self) -> bool:
        """Execute one simulation step"""
        # Determine season
        current_season = self.get_current_season()
        
        # Process environmental changes
        self.process_resources(current_season)
        self.pheromone_map *= self.config.pheromone_decay
        self.update_territory_map()
        
        # Process agent actions
        self.process_prey_actions()
        self.process_predator_actions()
        
        # Update metrics
        self.update_metrics()
        
        # Check end conditions
        if (len(self.prey_agents) < self.config.min_population or 
            len(self.predator_agents) < self.config.min_population):
            return False
        
        return True
    
    def run_simulation(self, steps: int = 100):
        """
        Run the simulation for a specified number of steps.
        
        Args:
            steps (int): Number of steps to run the simulation
        """
        logger.info(f"Starting simulation for {steps} steps")
        
        for step_num in range(steps):
            # Run one step of simulation
            if not self.step():
                logger.info(f"Simulation ended early at step {step_num}")
                break
                
            # Log progress every 10 steps
            if step_num % 10 == 0:
                logger.info(f"Step {step_num}: Prey={len(self.prey_agents)}, "
                        f"Predators={len(self.predator_agents)}")
        
        logger.info("Simulation completed")
        logger.info(f"Final populations - Prey: {len(self.prey_agents)}, "
                f"Predators: {len(self.predator_agents)}")

    def get_current_season(self) -> str:
        """Determine current season based on step count"""
        seasons = ["Spring", "Summer", "Fall", "Winter"]
        return seasons[(len(self.frames) // self.config.season_length) % len(seasons)]

    def update_metrics(self):
        """Update simulation metrics"""
        self.frames.append({
            "prey": [prey.position for prey in self.prey_agents],
            "predators": [predator.position for predator in self.predator_agents],
            "resources": self.resources.copy(),
            "territory": self.territory_map.copy(),
            "season": self.get_current_season()
        })
        
        self.prey_population.append(len(self.prey_agents))
        self.predator_population.append(len(self.predator_agents))
        
        # Track energy distribution
        self.energy_distribution.append({
            "prey": [prey.energy for prey in self.prey_agents],
            "predators": [pred.energy for pred in self.predator_agents]
        })

    def create_animation(self, filename="ecosystem_simulation.mp4"):
        """Create animation of simulation"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Setup plots
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(-1, self.grid_size)
            ax.set_ylim(-1, self.grid_size)
            ax.grid(True)
        
        ax1.set_title("Resources")
        ax2.set_title("Agents")
        ax3.set_title("Territories")

        # Initialize plots
        resource_plot = ax1.imshow(
            self.resources, 
            cmap="YlGn",
            extent=(-0.5, self.grid_size - 0.5, -0.5, self.grid_size - 0.5),
            alpha=0.7
        )
        territory_plot = ax3.imshow(
            self.territory_map,
            cmap="viridis",
            extent=(-0.5, self.grid_size - 0.5, -0.5, self.grid_size - 0.5),
            alpha=0.5
        )
        prey_scatter = ax2.scatter([], [], color="green", label="Prey", s=100)
        predator_scatter = ax2.scatter([], [], color="red", label="Predators", s=100)
        
        plt.colorbar(resource_plot, ax=ax1, label="Resource Level")
        plt.colorbar(territory_plot, ax=ax3, label="Territory")
        ax2.legend()

        def update(frame):
            # Update resources
            resource_plot.set_array(frame["resources"])
            territory_plot.set_array(frame["territory"])
            
            # Update agents
            prey_positions = frame["prey"]
            predator_positions = frame["predators"]
            
            # Handle prey positions
            if prey_positions:
                prey_x, prey_y = zip(*prey_positions)
                prey_coords = np.column_stack((prey_x, prey_y))
            else:
                prey_coords = np.empty((0, 2))
                
            # Handle predator positions
            if predator_positions:
                pred_x, pred_y = zip(*predator_positions)
                pred_coords = np.column_stack((pred_x, pred_y))
            else:
                pred_coords = np.empty((0, 2))
                
            prey_scatter.set_offsets(prey_coords)
            predator_scatter.set_offsets(pred_coords)
            
            # Update titles
            frame_idx = self.frames.index(frame)
            season = self.get_current_season()
            ax1.set_title(f"Resources ({season})")
            ax2.set_title(f"Step {frame_idx}: {len(prey_positions)} Prey, {len(predator_positions)} Pred.")
            ax3.set_title(f"Territories")
            
            return resource_plot, prey_scatter, predator_scatter, territory_plot

        ani = FuncAnimation(
            fig, 
            update, 
            frames=self.frames,
            interval=200,
            blit=True
        )
        
        ani.save(filename, writer='ffmpeg')
        plt.close()

    def plot_population_metrics(self, filename="population_metrics.png"):
        """
        Plots and saves the population metrics of prey and predator over the simulation steps.
        
        Args:
            filename (str): Name of the output file for the population plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot populations
        steps = range(len(self.prey_population))
        plt.plot(steps, self.prey_population, label="Prey Population", 
                color="green", linewidth=2)
        plt.plot(steps, self.predator_population, label="Predator Population", 
                color="red", linewidth=2)
        
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

    def plot_advanced_metrics(self, filename="advanced_metrics.png"):
        """
        Plot advanced simulation metrics including hunting success and territory changes.
        
        Args:
            filename (str): Name of the output file for the advanced metrics plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        steps = range(len(self.prey_population))
        
        # Population dynamics
        ax1.plot(steps, self.prey_population, 'g-', label='Prey')
        ax1.plot(steps, self.predator_population, 'r-', label='Predators')
        ax1.set_title('Population Dynamics')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Population Count')
        ax1.legend()
        ax1.grid(True)
        
        # Hunting success rate
        if self.hunting_success_rate:
            hunt_steps = range(len(self.hunting_success_rate))
            ax2.plot(hunt_steps, self.hunting_success_rate, 'b-')
            ax2.set_title('Hunting Success Rate')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Success Rate')
            ax2.grid(True)
        
        # Territory changes
        if self.territory_changes:
            territory_steps = range(len(self.territory_changes))
            ax3.plot(territory_steps, self.territory_changes, 'purple')
            ax3.set_title('Territory Changes Over Time')
            ax3.set_xlabel('Steps')
            ax3.set_ylabel('Change Magnitude')
            ax3.grid(True)
        
        # Final territory map
        territory_plot = ax4.imshow(self.territory_map, cmap='viridis')
        ax4.set_title('Final Territory Distribution')
        plt.colorbar(territory_plot, ax=ax4)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def save_metrics(self, filename="simulation_metrics.json"):
        """
        Save simulation metrics to a JSON file.
        
        Args:
            filename (str): Name of the output metrics file
        """
        metrics = {
            "prey_population": self.prey_population,
            "predator_population": self.predator_population,
            "hunting_success_rate": self.hunting_success_rate,
            "territory_changes": self.territory_changes,
            "final_populations": {
                "prey": len(self.prey_agents),
                "predators": len(self.predator_agents)
            },
            "average_hunting_success": np.mean(self.hunting_success_rate) if self.hunting_success_rate else 0,
            "simulation_length": len(self.frames)
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=4)

    def save_state(self, filename="simulation_state.json"):
        """Save the current simulation state"""
        state = {
            "prey": [{"name": p.name, "position": p.position, "energy": p.energy} 
                    for p in self.prey_agents],
            "predators": [{"name": p.name, "position": p.position, "energy": p.energy} 
                         for p in self.predator_agents],
            "resources": self.resources.tolist(),
            "obstacles": self.obstacles.tolist(),
            "territory_map": self.territory_map.tolist(),
            "pheromone_map": self.pheromone_map.tolist()
        }
        with open(filename, "w") as f:
            json.dump(state, f)
        logger.info(f"Simulation state saved to {filename}")

    def load_state(self, filename="simulation_state.json"):
        """Load a saved simulation state"""
        with open(filename, "r") as f:
            state = json.load(f)
        
        self.resources = np.array(state["resources"])
        self.obstacles = np.array(state["obstacles"])
        self.territory_map = np.array(state["territory_map"])
        self.pheromone_map = np.array(state["pheromone_map"])
        
        # Add config when creating agents
        self.prey_agents = [
            PreyAgent(**{**p, "config": self.config, "grid_size": self.grid_size}) 
            for p in state["prey"]
        ]
        self.predator_agents = [
            PredatorAgent(**{**p, "config": self.config, "grid_size": self.grid_size}) 
            for p in state["predators"]
        ]
        
        logger.info(f"Simulation state loaded from {filename}")

    

def run_simulation_with_config(config: SimulationConfig, steps: int = 100) -> EcosystemSimulation:
    """Run simulation with comprehensive analysis using provided configuration"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting simulation with configuration: {config}")
    
    try:
        # Initialize simulation with configuration
        simulation = EcosystemSimulation(config=config)
        
        # Run simulation
        logger.info(f"Running simulation for {steps} steps")
        simulation.run_simulation(steps=steps)
        
        # Save results
        simulation.save_state(f"final_state_{config.num_prey}_{config.num_predators}.json")
        simulation.save_metrics(f"metrics_{config.num_prey}_{config.num_predators}.json")
        
        # Generate visualizations
        simulation.create_animation(f"simulation_{config.num_prey}_{config.num_predators}.mp4")
        simulation.plot_population_metrics(f"population_{config.num_prey}_{config.num_predators}.png")
        simulation.plot_advanced_metrics(f"advanced_{config.num_prey}_{config.num_predators}.png")
        
        return simulation
        
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise