#!/usr/bin/env python3
"""Script for production run OpenMM simulations.

   Author: Dominic Phillips (dominicp6)
"""
from utils.openmm_utils import OpenMMSimulation


if __name__ == "__main__":
    MD_simulation = OpenMMSimulation()
    MD_simulation.run()
