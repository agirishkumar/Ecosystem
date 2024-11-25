# main.py
from simulation.eco_simulation import EcosystemSimulation, run_simulation_with_config
from config import SimulationConfig
import logging
import matplotlib.pyplot as plt
import numpy as np

def configure_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulation.log'),
            logging.StreamHandler()
        ]
    )

def create_test_configs():
    """Create test configurations"""
    return [
        # Dynamic ecosystem
        SimulationConfig(
            # Environment
            grid_size=30,
            resource_regen_rate=0.15,
            resource_energy_value=4,
            
            # Prey
            num_prey=25,
            prey_initial_energy=10.0,
            prey_movement_cost=0.3,
            prey_reproduction_threshold=15.0,
            prey_max_energy=25.0,
            prey_starve_threshold=1.5,
            
            # Predator
            num_predators=5,
            predator_initial_energy=15.0,
            predator_movement_cost=0.5,
            predator_hunt_reward=12.0,
            predator_reproduction_threshold=25.0,
            predator_max_energy=35.0,
            predator_starve_threshold=2.0,
            
            # Seasonal
            season_length=20,
            summer_resource_rate=0.25,
            winter_resource_rate=0.08,
            
            # Control
            max_steps=200,
            min_population=1,
            max_population=150,
            
            # Advanced
            energy_decay_rate=0.99,
            territory_decay_rate=0.95,
            learning_rate=0.15
        ),
        # Harsh environment
        SimulationConfig(
            grid_size=25,
            resource_regen_rate=0.1,
            num_prey=15,
            num_predators=3,
            prey_movement_cost=0.4,
            predator_movement_cost=0.6,
            winter_resource_rate=0.03,
            energy_decay_rate=0.97
        )
    ]

def main():
    configure_logging()
    logger = logging.getLogger(__name__)
    
    configs = create_test_configs()
    results = []
    
    for i, config in enumerate(configs, 1):
        logger.info(f"\nRunning simulation {i}")
        logger.info(f"Configuration:")
        logger.info(f"Grid: {config.grid_size}x{config.grid_size}")
        logger.info(f"Initial populations - Prey: {config.num_prey}, Predators: {config.num_predators}")
        
        try:
            simulation = run_simulation_with_config(config, steps=200)
            results.append(simulation)
            
            # Log results
            logger.info("Simulation completed:")
            logger.info(f"Final Prey: {len(simulation.prey_agents)}")
            logger.info(f"Final Predators: {len(simulation.predator_agents)}")
            if simulation.hunting_success_rate:
                avg_success = np.mean(simulation.hunting_success_rate)
                logger.info(f"Avg Hunting Success: {avg_success:.2f}")
                
        except Exception as e:
            logger.error(f"Simulation {i} failed: {str(e)}")
            continue
    
    if results:
        compare_results(results, configs)
    else:
        logger.error("No successful simulations to analyze")

def compare_results(results, configs):
    """Compare results from multiple simulations"""
    fig = plt.figure(figsize=(15, 10))
    
    # Population dynamics
    ax1 = plt.subplot(211)
    for i, (sim, config) in enumerate(zip(results, configs)):
        ax1.plot(sim.prey_population, 'g-', alpha=0.5, 
                label=f'Prey (Config {i+1})')
        ax1.plot(sim.predator_population, 'r-', alpha=0.5, 
                label=f'Predators (Config {i+1})')
    ax1.set_title('Population Dynamics')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Population')
    ax1.legend()
    ax1.grid(True)
    
    # Success rates
    ax2 = plt.subplot(212)
    for i, sim in enumerate(results):
        if sim.hunting_success_rate:
            ax2.plot(sim.hunting_success_rate, 
                    label=f'Config {i+1}')
    ax2.set_title('Hunting Success Rates')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Success Rate')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
