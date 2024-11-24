from simulation.eco_simulation import EcosystemSimulation

def main():
    # Initialize the simulation
    simulation = EcosystemSimulation(
        grid_size=20,
        num_prey=15,
        num_predators=3,
        resource_regen_rate=0.3
    )

    # Run the simulation
    simulation.run_simulation(steps=100)

    # Save the current state
    simulation.save_state("final_state.json")

    # Generate and save animation
    simulation.create_animation(filename="ecosystem_simulation.mp4")

    # Plot population metrics
    simulation.plot_population_metrics()

    print("Simulation complete. Results saved.")

if __name__ == "__main__":
    main()
