#!/usr/bin/env python3
"""Script for production run OpenMM simulations.

   Author: Dominic Phillips (dominicp6)
"""
from utils.openmm_utils import OpenMMSimulation


if __name__ == "__main__":
    MD_simulation = OpenMMSimulation()

    # Read and process commandline arguments
    MD_simulation.parse_args()

    # Check arguments are valid and then create the system
    MD_simulation.setup_system()

    # Initialise the simulation environment and data reporters
    MD_simulation.setup_simulation()

    # Run the simulation until completion
    MD_simulation.run_simulation()

    # Save final checkpoint and state
    MD_simulation.save_checkpoint()

    # Visualise simulation properties
    MD_simulation.make_graphs()
