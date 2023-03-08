import os
import time
from copy import deepcopy

import openmm
import openmm.app as app
import pandas as pd

from utils.openmm_utils import parse_quantity, SystemArgs, cutoff_method, add_barostat, \
    get_integrator, make_graphs


class EquilibrationProtocol:
    def __init__(self,
                 production_ensemble: str,
                 system: openmm.System,
                 modeller: app.Modeller,
                 args: SystemArgs,
                 output_dir: str,
                 fixed_density: bool = True):

        self.output_dir = output_dir
        self.modeller = modeller
        self.args = args
        self.system = system
        self.simulation_sequence = []
        equilibration_length = self.args.equilibration_length

        if production_ensemble == "NVE":
            # 1. Short NVT simulation to relax to temperature of interest
            # 2. Short NVE equilibration
            self.simulation_sequence = (("NVT", equilibration_length), ("NVE", equilibration_length))
        elif production_ensemble == "NVT":
            if fixed_density:
                # 1. Short NVT simulation to relax to temperature of interest
                self.simulation_sequence = (("NVT", equilibration_length),)
            else:
                # 1. Short NVT simulation to relax to temperature of interest
                # 2. Short NPT simulation to relax to density of interest
                # 3. Short NPT simulation to calculate average box size
                # 4. Short NVT equilibration
                self.simulation_sequence = (("NVT", equilibration_length), ("NPT", equilibration_length), ("NPT", equilibration_length), ("NVT", equilibration_length))
        elif production_ensemble == "NPT":
            # 1. Short NVT simulation to relax to temperature of interest
            # 2. Short NPT simulation to relax to density of interest
            self.simulation_sequence = (("NVT", equilibration_length), ("NPT", equilibration_length))
        else:
            raise ValueError(f"Invalid production ensemble: {production_ensemble}")

    def run(self):
        for simulation_id, equilibration_simulation in enumerate(self.simulation_sequence):
            simulation_type = equilibration_simulation[0]
            duration = equilibration_simulation[1]
            print(f"[✓] Running {duration} {simulation_type} equilibration simulation")
            self._run_simulation(simulation_type, duration, simulation_id, len(self.simulation_sequence)-1)
            print(f"[✓] Finished equilibration simulation {simulation_id+1}/{len(self.simulation_sequence)}")

    def _run_simulation(self, simulation_type: str, duration: str, simulation_id: int = 0, max_simulation_id: int = 0):
        duration = parse_quantity(duration)
        total_steps = int(duration / self.args.stepsize)
        system = deepcopy(self.system)

        # Set up simulation
        unit_cell_dims = self.modeller.getTopology().getUnitCellDimensions()

        if simulation_type == "NVE":
            raise NotImplementedError
        elif simulation_type == "NVT":
            pass
        elif simulation_type == "NPT":
            assert (unit_cell_dims is not None), "PDB file missing unit cell dimensions - cannot run NPT simulation."
            add_barostat(barostat_type="MonteCarloBarostat",
                         system=system,
                         pressure=self.args.pressure,
                         temperature=self.args.temperature)
        else:
            raise ValueError(f"Invalid simulation type: {simulation_type}")

        integrator = get_integrator(args=self.args, integrator_type="LangevinBAOAB")
        properties = {}

        if self.args.gpu is not "":
            platform = openmm.Platform.getPlatformByName("CUDA")
            properties["CudaDeviceIndex"] = self.args.gpu
            properties["Precision"] = self.args.precision
        else:
            platform = openmm.Platform.getPlatformByName("CPU")

        # Create simulation object using the specified integrator
        simulation = app.Simulation(
            self.modeller.topology,
            system,
            integrator,
            platform,
            properties,
        )

        # Add reporter
        simulation.reporters.append(
            app.StateDataReporter(
                os.path.join(self.output_dir, f"equilibration_{simulation_id}.log"),
                100, # steps per save
                step=True,
                time=True,
                speed=True,
                temperature=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                volume=True
                if self.args.pressure
                else False,  # record volume and density for NPT simulations
                density=True if self.args.pressure else False,
                append=True if self.args.resume else False,
            )
        )

        # Run simulation
        if simulation_id == 0:
            assert simulation_type == "NVT", "First equilibration simulation must be NVT, regardless of production ensemble."
            simulation.context.setPositions(self.modeller.positions)
            simulation.step(total_steps)
        else:
            # If this is not the first equilibration simulation, we can just continue from the previous checkpoint
            simulation.loadCheckpoint(os.path.join(self.output_dir, f"equilibration_{simulation_id - 1}.chk"))
            simulation.step(total_steps)

        # Make graphs
        report = pd.read_csv(os.path.join(self.output_dir, f"equilibration_{simulation_id}.log"))
        make_graphs(report, self.args.stepsize, self.output_dir, name=f"equilibration_{simulation_id}")

        # Save state
        if simulation_id < max_simulation_id:
            simulation.saveCheckpoint(f"{self.output_dir}/equilibration_{simulation_id}.chk")
        else:
            simulation.saveCheckpoint(f"{self.output_dir}/equilibration_final.chk")

