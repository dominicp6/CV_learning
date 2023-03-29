from abc import abstractmethod
import os
import argparse
import sys
import warnings
from datetime import datetime
import json
import copy
from typing import Union

import numpy as np
import openmm
import openmm.app as app
from openmmml import MLPotential
import pandas as pd

import mdtraj as md
from openff.toolkit.topology import Molecule, Topology
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

from utils.trajectory_utils import clean_and_align_trajectory
from utils.openmm_utils import parse_quantity, get_system_args, cutoff_method, add_barostat, \
    get_integrator, SystemObjs, SimulationProps, stringify_named_tuple, check_fields_unchanged, \
    update_numerical_fields, make_graphs
from utils.general_utils import printlog
from EquilibrationProtocol import EquilibrationProtocol


# Turns a dictionary into a class
class Dict2Class(object):

    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

    def __str__(self):
        return str(self.__dict__)


class OpenMMSimulation:

    def __init__(self):
        self.CHECKPOINT_FL, self.TRAJECTORY_FL, self.STATE_DATA_FL, self.METADATA_FL, self.LOG_FL = \
            "checkpoint.chk", "trajectory.dcd", "state_data.csv", "metadata.json", "output.log"
        self.required_files = {self.CHECKPOINT_FL, self.TRAJECTORY_FL, self.STATE_DATA_FL, self.METADATA_FL, self.LOG_FL}
        self.valid_force_fields = ["amber14", "charmm36"]
        self.valid_precisions = ["single", "mixed", "double"]
        self.valid_water_models = ["tip3p", "tip3pfb", "spce", "tip4pew", "tip4pfb", "tip5p"]
        self.valid_ensembles = ["NVT", "NPT", None]
        self.forcefield_files = {
            "amber14": ["amber14-all.xml", "amber14/tip3p.xml"],
            "charmm36": ["charmm36.xml", "charmm36/water.xml"]
        }

        # Properties that must be preserved when resuming a pre-existing simulation
        self.preserved_properties = {'savefreq', 'stepsize', 'steps_per_save', 'temperature', 'pressure',
                                     'frictioncoeff', 'solventpadding', 'nonbondedcutoff', 'cutoffmethod',
                                     'periodic', 'precision', 'watermodel', 'integrator', 'plumed'}
        # Properties that cumulate upon resuming a simulation
        self.cumulative_properties = {'duration', 'total_steps'}

        # CLASS PROPERTIES
        self.system = None
        self.simulation = None
        self.output_dir = None
        self.force_field = None

        # BOOLEAN ARGUMENTS
        self.BOOLEAN_ARGS = {
            'periodic': {'flag': '--periodic'},
            'minimise': {'flag': '--minimise'},
            'state_data': {'flag': '--state_data'},
        }

        # Parser and arg check
        self.parser = self._init_parser()

        # try:
        #     args = self.parser.parse_args()
        #     self.args = get_system_args(args)
        # except SystemExit:
        #     # No arguments were passed
        self.args = {}
        # else:
        #     # Check that the arguments are valid
        #     self._check_args()

    def from_args(self, args: dict):
        """
        Initialises the simulation object from a dictionary of arguments and checks them.
        """
        arg_list = [action.dest for action in self.parser._actions]
        for field in arg_list:
            if field in args:
                self.args[field] = args[field]
            else:
                self.args[field] = None
        self.args = Dict2Class(self.args)
        self.args = get_system_args(self.args)
        self._check_args()

        return self

    def run(self):
        """
        Run the complete simulation pipeline.
        """
        self.setup_system()
        self.setup_simulation()
        if self.args.equilibrate is not None:
            self.run_equilibration()
        self.run_simulation()
        self.post_process_simulation()

    def setup_system(self):
        """
        Set up the system.
        STEPS: Initialise directories, read in the model (PDB/MOL2), set up the force field, set up the system model including
        any solvent (optional), save a PDB of the entire system model, and create the system object.
        """
        self._init_output_dir()
        printlog("[✓] Created output directory", os.path.join(self.args.directory, self.LOG_FL))
        model = self._init_model()
        model = self._set_periodicity(model)
        printlog("[✓] Initialised model file(s)", os.path.join(self.args.directory, self.LOG_FL))
        self._init_forcefield(model)
        printlog("[✓] Initialised force field", os.path.join(self.args.directory, self.LOG_FL))
        modeller = self._init_modeller(model)
        printlog("[✓] Initialised modeller", os.path.join(self.args.directory, self.LOG_FL))
        self._write_updated_model(model, modeller)
        system = self._init_system(modeller)
        printlog("[✓] Created system", os.path.join(self.args.directory, self.LOG_FL))
        self.system = SystemObjs(model, modeller, system)

        return self.system

    @abstractmethod
    def _init_model(self) -> Union[app.PDBFile, Dict2Class]:
        """
        Initialises the model (either from PDB or MOL2) and returns the model object.
        """
        raise NotImplementedError

    @abstractmethod
    def _init_forcefield(self, model: Union[app.PDBFile, Dict2Class]) -> app.ForceField:
        """
        Initialises the forcefield and returns it.
        """
        raise NotImplementedError

    def _init_modeller(self, model: Union[app.PDBFile, Dict2Class]) -> app.Modeller:
        """
        Initialises the Modeller.
        """
        modeller = app.Modeller(model.topology, model.positions)
        if self.args.resume:
            # Resuming simulation, skipping modeller modifications
            pass
        else:
            # Check if the loaded topology file already has water
            if np.any([atom.residue.name == 'HOH' for atom in model.topology.atoms()]):
                printlog("Water found in PBD file: Not changing solvation properties; solvent padding ignored!", os.path.join(self.args.directory, self.LOG_FL))
                pass
            else:
                # If no water initially present but the user has specified a water model...
                if self.args.watermodel:
                    # ... then automatically add water to the modeller
                    modeller.addSolvent(
                        self.force_field,
                        model=self.args.watermodel,
                        padding=self.args.solventpadding,
                        numAdded=self.args.num_water,
                        ionicStrength=self.args.ionic_strength
                    )
                else:
                    # Else do not add water
                    pass

        return modeller

    @abstractmethod
    def _write_updated_model(self, model: Union[app.PDBFile, Dict2Class], modeller: app.Modeller,
                             topology_name: str = 'top.pdb', no_water_topology_name: str = 'top_no_water.pdb') -> None:
        raise NotImplementedError

    def setup_simulation(self):
        """
        Set up the simulation.
        STEPS: Create the integrator, barostat (optional), simulation object, initialise particle positions,
        minimise the system energy, set particle velocities, save all simulation metadata, and set up the reporters.
        """
        self._init_simulation()
        printlog("[✓] Initialised simulation", os.path.join(self.args.directory, self.LOG_FL))
        self._minimise_system_energy()
        self.simulation.simulation.context.setVelocitiesToTemperature(
            self.args.temperature
        )
        printlog("[✓] Finished minimisation", os.path.join(self.args.directory, self.LOG_FL))
        self._save_simulation_metadata()
        printlog("[✓] Saved simulation metadata", os.path.join(self.args.directory, self.LOG_FL))
        self._setup_reporters()
        printlog("[✓] Setup reporters", os.path.join(self.args.directory, self.LOG_FL))

    def run_equilibration(self):
        """
        Run the equilibration.
        """
        eq_obj = EquilibrationProtocol(production_ensemble=self.args.equilibrate,
                                       system=self.system.system,
                                       modeller=self.system.modeller,
                                       args=self.args,
                                       fixed_density=True,
                                       output_dir=self.args.directory)
        eq_obj.run()

    def run_simulation(self):
        """
        Run the simulation.
        STEPS: Run the simulation for the specified duration, and save the final checkpoint.
        """
        printlog(f"Running {self.args.duration} production...", os.path.join(self.args.directory, self.LOG_FL))
        if self.args.equilibrate:
            # If equilibration was run, resume from the checkpoint file
            self.simulation.simulation.loadCheckpoint(os.path.join(self.args.directory, f"equilibration_final.chk"))
            printlog("[✓] Loaded checkpoint file", os.path.join(self.args.directory, self.LOG_FL))
        self.simulation.simulation.step(self.args.total_steps)
        printlog("[✓] Finished production", os.path.join(self.args.directory, self.LOG_FL))
        self._save_final_checkpoint_and_endstate()
        printlog("[✓] Saved final checkpoint", os.path.join(self.args.directory, self.LOG_FL))

    def post_process_simulation(self):
        """
        Post-process the simulation.
        STEPS: Centre and align the trajectory, removing solvent, and save the modified trajectory. Read in the
        state chemicals and create plots of energy, temperature etc. Finally, analyse basic statistics of the state
        chemicals and save these to a JSON file.
        """
        clean_and_align_trajectory(working_dir=self.args.directory,
                                   traj_name='trajectory.dcd',
                                   top_name='top.pdb',
                                   save_name='trajectory_processed')
        printlog("[✓] Cleaned and aligned trajectory", os.path.join(self.args.directory, self.LOG_FL))
        if self.args.state_data:
            report = pd.read_csv(os.path.join(self.args.directory, self.STATE_DATA_FL))
            make_graphs(report, self.args.stepsize, self.args.directory, name='simulation_stats')
            printlog("[✓] Made graphs of simulation chemicals", os.path.join(self.args.directory, self.LOG_FL))
            self._make_summary_statistics(report)
            printlog("[✓] Saved summary statistics", os.path.join(self.args.directory, self.LOG_FL))

    def generate_executable_command(self, args: dict):
        """
        Generate a command to run a simulation with the desired arguments.
        :param arg_dict: Dictionary of arguments
        :return: String of command
        """
        command = "python /home/dominic/PycharmProjects/CV_learning/run_openmm.py "

        # Add required arguments
        required_args = [act.dest for act in self.parser._positionals._actions
                         if len(act.option_strings) == 0]
        for arg_name in required_args:
            try:
                command += f"{args[arg_name]} "
            except KeyError:
                raise KeyError(f"Required argument {arg_name} not found in args dict")

        # Add optional arguments
        encountered_args = set()
        for _, arg in self.parser._option_string_actions.items():
            arg_name = arg.dest
            arg_flag = arg.option_strings[-1]
            if arg_name in encountered_args:
                continue
            encountered_args.add(arg_name)

            if arg_name in self.BOOLEAN_ARGS:
                if (arg_name in args) and (args[arg_name] == True):
                    command += f"{self.BOOLEAN_ARGS[arg_name]['flag']} "
                else:
                    pass
                continue

            if arg_name in args and args[arg_name] is not None:
                # Only add argument to command if it exists in args
                command += f"{arg_flag} {args[arg_name]} "

        return command

    #  ================================== HELPER FUNCTIONS ===================================
    def _init_output_dir(self) -> str:
        """
        Initialises the output directory.
        :return: Path to output directory
        """
        if self.args.resume:
            # If resuming, use existing output directory
            output_dir = self.args.resume
        else:
            # Make output directory
            if self.args.pdb:
                filename = os.path.splitext(os.path.basename(self.args.pdb))[0]
            else:
                filename = os.path.splitext(os.path.basename(self.args.mol2))[0]
            exp_name_auto = f"production_{filename}_" \
                            f"{self.args.forcefield}_{datetime.now().strftime('%H%M%S_%d%m%y')}"

            output_dir = None
            # If no output directory specified, use default
            if not self.args.name and not self.args.directory:
                output_dir = os.path.join("../exp/outputs", exp_name_auto)
            # If output name specified, use that
            elif self.args.name and not self.args.directory:
                output_dir = os.path.join("../exp/outputs", self.args.name)
            # If output directory specified, use that
            elif self.args.directory and not self.args.name:
                output_dir = os.path.join(self.args.directory, exp_name_auto)
            # If both name and directory specified, use both
            elif self.args.directory and self.args.name:
                output_dir = os.path.join(self.args.directory, self.args.name)

            try:
                # Make output directory
                os.makedirs(output_dir)
                file = os.path.join(output_dir, self.LOG_FL)
                open(file, 'a').close()
            except FileExistsError:
                files_in_dir = set(os.listdir(output_dir))
                simulation_files = self.required_files.intersection(files_in_dir)
                if len(simulation_files) > 0:
                    printlog(f"Output directory {output_dir} already exists and has simulation files:", os.path.join(self.args.directory, self.LOG_FL))
                    [printlog(file, os.path.join(self.args.directory, self.LOG_FL)) for file in simulation_files]
                    printlog("Please choose a different output directory or delete the existing files.", os.path.join(self.args.directory, self.LOG_FL))
                    sys.exit(1)
                else:
                    pass

        print(output_dir, self.args.directory)
        self.args = self.args._replace(directory=output_dir)
        return self.args.directory

    def _set_periodicity(self, model: Union[app.PDBFile, Dict2Class]) -> Union[app.PDBFile, Dict2Class]:
        if self.args.periodic is False:
            model.topology.setPeriodicBoxVectors(None)

        return model

    @abstractmethod
    def _init_system(self, modeller: app.Modeller):
        """
        Initialises the system object. If using PLUMED, adds the PLUMED force to the system.
        """
        return NotImplementedError

    def _init_simulation(self):
        """
        Initialises the simulation object.
        STEPS: 1) Get unit cell dims 2) Initialise barostat (optional) 3) Initialise integrator 4) Set positions
        5) Load pre-existing checkpoint file (if resuming).
        :return: Simulation object
        """
        properties = {}

        unit_cell_dims = self.system.modeller.getTopology().getUnitCellDimensions()

        if self.args.pressure is None:
            # If no pressure provided, run an NVT simulation
            pass
        else:
            # If pressure provided, run an NPT simulation
            assert (unit_cell_dims is not None), "model file missing unit cell dimensions - cannot run NPT simulation."
            add_barostat(barostat_type="MonteCarloBarostat",
                         system=self.system.system,
                         pressure=self.args.pressure,
                         temperature=self.args.temperature)

        integrator = get_integrator(integrator_type=self.args.integrator, args=self.args)

        if self.args.gpu != "":
            printlog("[✓] Using GPU", os.path.join(self.args.directory, self.LOG_FL))
            platform = openmm.Platform.getPlatformByName("CUDA")
            properties["CudaDeviceIndex"] = self.args.gpu
            properties["Precision"] = self.args.precision
        else:
            printlog("[✓] Using CPU", os.path.join(self.args.directory, self.LOG_FL))
            platform = openmm.Platform.getPlatformByName("CPU")

        # Create simulation object using the specified integrator
        simulation = app.Simulation(
            self.system.modeller.topology,
            self.system.system,
            integrator,
            platform=platform,
            platformProperties=properties,
        )

        if self.args.gpu.count(",") > 0:
            printlog(f"[✓] Multiple GPUs ({self.args.gpu})", os.path.join(self.args.directory, self.LOG_FL))

        # Set initial particle positions
        simulation.context.setPositions(self.system.modeller.positions)

        # If resuming, load the checkpoint file
        if self.args.resume:
            with open(os.path.join(self.args.directory, self.CHECKPOINT_FL), "rb") as f:
                simulation.context.loadCheckpoint(f.read())
                printlog("[✓] Loaded checkpoint for resuming simulation", os.path.join(self.args.directory, self.LOG_FL))

        self.simulation = SimulationProps(integrator, simulation, properties)
        return self.simulation

    def _minimise_system_energy(self):
        """
        Minimises the system energy.
        """
        self.simulation.simulation.minimizeEnergy()
        positions = self.simulation.simulation.context.getState(
            getPositions=True
        ).getPositions()
        self.system.modeller.positions = positions

    def _save_simulation_metadata(self):
        """
        Saves the simulation metadata to a JSON file.
        If the simulation is being resumed, the metadata is loaded from the existing file and updated.
        """
        system_args_dict = stringify_named_tuple(self.args)

        # If resuming, load existing metadata and update
        if self.args.resume:
            # Updating metadata
            system_args_dict = self._update_metadata_file(system_args_dict)
            # Cumulate duration and total_steps from previous runs
            self.args.duration = parse_quantity(system_args_dict['duration'])
            self.args.total_steps = int(self.args.duration / self.args.stepsize)

        # Save metadata
        with open(os.path.join(self.args.directory, self.METADATA_FL), "w") as json_file:
            json.dump(system_args_dict, json_file)

    def _update_metadata_file(self, metadata_update: dict) -> dict:
        """
        Updates the metadata file with the new metadata.
        :param metadata_update: The new metadata to update the file with.
        :return: The updated metadata.
        """
        assert os.path.exists(
            os.path.join(self.args.directory, self.METADATA_FL)), "Cannot resume a simulation without metadata."
        with open(os.path.join(self.args.directory, self.METADATA_FL), "r") as json_file:
            metadata_existing = json.load(json_file)
        check_fields_unchanged(metadata_existing, metadata_update, fields=self.preserved_properties)
        metadata = update_numerical_fields(metadata_existing, metadata_update, fields=self.cumulative_properties)

        return metadata

    def _setup_reporters(self):
        """
        Sets up the reporters for the simulation.
        """
        # Reporter to print info to stdout
        self.simulation.simulation.reporters.append(
            app.StateDataReporter(
                sys.stdout,
                self.args.steps_per_save,
                progress=True,  # Info to print. Add anything you want here.
                remainingTime=True,
                speed=True,
                totalSteps=self.args.total_steps,
            )
        )
        # Reporter to print info to self.LOG_FL
        self.simulation.simulation.reporters.append(
            app.StateDataReporter(
                os.path.join(self.args.directory, self.LOG_FL),
                self.args.steps_per_save,
                progress=True,  # Info to print. Add anything you want here.
                remainingTime=True,
                speed=True,
                totalSteps=self.args.total_steps,
                append=True,
            )
        )
        # Reporter to log info to csv
        if self.args.state_data:
            self.simulation.simulation.reporters.append(
                app.StateDataReporter(
                    os.path.join(self.args.directory, self.STATE_DATA_FL),
                    self.args.steps_per_save,
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
        # Reporter to save trajectory
        # (Saves only a subset of atoms to the trajectory, ignores water)
        self.simulation.simulation.reporters.append(
            app.DCDReporter(
                os.path.join(self.args.directory, self.TRAJECTORY_FL),
                reportInterval=self.args.steps_per_save,
                append=True if self.args.resume else False,
            )
        )

        # Reporter to save regular checkpoints
        self.simulation.simulation.reporters.append(
            app.CheckpointReporter(
                os.path.join(self.args.directory, self.CHECKPOINT_FL),
                self.args.steps_per_save,
            )
        )

    def _save_final_checkpoint_and_endstate(self):
        self.simulation.simulation.saveCheckpoint(
            os.path.join(self.args.directory, self.CHECKPOINT_FL)
        )
        self.simulation.simulation.saveState(
            os.path.join(self.args.directory, "end_state.xml")
        )

    def _make_summary_statistics(self, report):
        statistics = dict()
        statistics["PE"] = report["Potential Energy (kJ/mole)"].mean()
        statistics["dPE"] = report["Potential Energy (kJ/mole)"].std() / np.sqrt(
            self.args.duration / self.args.savefreq
        )
        statistics["KE"] = report["Kinetic Energy (kJ/mole)"].mean()
        statistics["dKE"] = report["Kinetic Energy (kJ/mole)"].std() / np.sqrt(
            self.args.duration / self.args.savefreq
        )
        statistics["TE"] = report["Total Energy (kJ/mole)"].mean()
        statistics["dTE"] = report["Total Energy (kJ/mole)"].std() / np.sqrt(
            self.args.duration / self.args.savefreq
        )
        statistics["T"] = report["Temperature (K)"].mean()
        statistics["dT"] = report["Temperature (K)"].std() / np.sqrt(
            self.args.duration / self.args.savefreq
        )
        statistics["S"] = report["Speed (ns/day)"].mean()
        statistics["dS"] = report["Speed (ns/day)"].std() / np.sqrt(
            self.args.duration / self.args.savefreq
        )
        if self.args.pressure:  # volume and pressure recorded for NPT simulations
            statistics["V"] = report["Box Volume (nm^3)"].mean()
            statistics["dV"] = report["Box Volume (nm^3)"].std() / np.sqrt(
                self.args.duration / self.args.savefreq
            )
            statistics["D"] = report["Density (g/mL)"].mean()
            statistics["dD"] = report["Density (g/mL)"].std() / np.sqrt(
                self.args.duration / self.args.savefreq
            )
        with open(os.path.join(self.args.directory, "summary_statistics.json"), "w") as f:
            json.dump(statistics, f)

    #  ================================== PARSER FUNCTIONS ===================================
    def _init_parser(self):
        parser = argparse.ArgumentParser(
            description="Production run for an equilibrated biomolecule."
        )
        parser.add_argument("forcefield",
                            help=f"Forcefield/Potential to use: {self.valid_force_fields}")
        parser.add_argument("precision",
                            help=f"Precision to use: {self.valid_precisions}")
        parser.add_argument(
            "-pdb",
            "--pdb",
            help="(file) Path to PDB file describing topology and positions.",
        )
        parser.add_argument(
            "-mol2",
            "--mol2",
            help="(file) Path to mol2 file of small molecule (requires -xml).",
        )
        parser.add_argument(
            "-sdf",
            "--sdf",
            help="(file) Path to sdf file of small molecule.",
        )
        parser.add_argument(
            "-xml",
            "--xml",
            help="(file) Path to xml file of small molecule (requires -mol2).",
        )
        parser.add_argument(
            "-ml_res",
            "--ml_residues",
            help="(str) Residues to be treated with an ML forcefield (comma separated).",
        )
        parser.add_argument(
            "-r",
            "--resume",
            help="(dir) Resume simulation from an existing production directory",
        )
        parser.add_argument(
            "-plumed",
            "--plumed",
            help="(PLUMED Script) Path to PLUMED script for enhanced sampling",
        )
        # The CUDA Platform supports parallelizing a simulation across multiple GPUs. \
        # To do that, set this to a comma separated list of values. For example, -g 0,1.
        parser.add_argument(
            "-g",
            "--gpu",
            default="",
            help="(int) Choose CUDA device(s) to target [note - ANI must run on GPU 0]",
        )
        parser.add_argument(
            "-d", "--duration", default="1ns", help="Duration of simulation"
        )
        parser.add_argument(
            "-f",
            "--savefreq",
            default="1ps",
            help="Interval for all reporters to save chemicals",
        )
        parser.add_argument(
            "-s", "--stepsize", default="2fs", help="Integrator step size"
        )
        parser.add_argument(
            "-t",
            "--temperature",
            default="300K",
            help="Temperature for Langevin dynamics",
        )
        parser.add_argument(
            "-p",
            "--pressure",
            default="",
            help="Pressure (bar) for NPT simulation. If blank, an NVT simulation is run instead",
        )
        parser.add_argument(
            "-c",
            "--frictioncoeff",
            default="1ps",
            help="Integrator friction coeff [your value]^-1 ie for 0.1fs^-1 put in 0.1fs. "
                 "The unit but not the value will be converted to its reciprocal.",
        )
        parser.add_argument(
            "-num_water",
            "--num_water",
            default=None,
            help="Number of water molecules to add to the system",
        )
        parser.add_argument(
            "-ionic_strength",
            "--ionic_strength",
            default=None,
            help="The total concentration of ions, both positive and negative, (in M) to add to the system. "
        )
        parser.add_argument(
            "-sp",
            "--solventpadding",
            default="1nm",
            help="Solvent padding distance",
        )
        parser.add_argument(
            "-nbc",
            "--nonbondedcutoff",
            default="1nm",
            help="Non-bonded cutoff distance",
        )
        parser.add_argument(
            "-cm",
            "--cutoffmethod",
            default="PME",
            help="The non-bonded cutoff method to use, one of 'NoCutoff', 'CutoffNonPeriodic', "
                 "'CutoffPeriodic', 'Ewald', or 'PME'",
        )
        parser.add_argument(
            "-pr",
            "--periodic",
            action=argparse.BooleanOptionalAction,
            help="Applies periodic boundary conditions",
        )
        parser.add_argument(
            "-m",
            "--minimise",
            action=argparse.BooleanOptionalAction,
            help="Minimises energy before running the simulation (recommended)",
        )
        parser.add_argument(
            "-w", "--watermodel", default="", help=f"(str) The water model: {self.valid_water_models}"
        )
        parser.add_argument("-seed", "--seed", default='0', help="Random seed")
        parser.add_argument("-name", "--name", default=None, help="(str) Name of simulation. "
                                                                  "If not provided, a name will be generated.")
        parser.add_argument("-dir", "--directory", default=None, help="(str) Directory to save simulation in. "
                                                                      "If not provided, a directory will be generated.")
        parser.add_argument("-equilibrate", "--equilibrate", default=None,
                            help="(str) Target production ensemble NVE/NVT/NPT")
        parser.add_argument("-equilibration_length", "--equilibration_length", default='0.1ns',
                            help="Duration of equilibration (e.g. 0.1ns)")
        parser.add_argument("-integrator", "--integrator", default='Langevin',
                            help="(str) The type of numerical integrator to use, either 'LangevinBAOAB', "
                                 "'LangevinMiddle' or 'Verlet'.")
        parser.add_argument("-state_data", "--state_data", action=argparse.BooleanOptionalAction,
                            help="Whether to save state data in addition to the trajectory.")
        return parser

    @abstractmethod
    def _additional_checks(self):
        raise NotImplementedError

    def _check_args(self):
        """
        Check that the simulation arguments are valid.
        """
        if self.args.forcefield not in self.valid_force_fields:
            printlog(
                f"Error: Invalid forcefield: {self.args.forcefield}, must be {self.valid_force_fields}", os.path.join(self.args.directory, self.LOG_FL))
            quit()

        if (self.args.watermodel is not None) and (self.args.watermodel not in self.valid_water_models):
            printlog(
                f"Error: Invalid water model: {self.args.watermodel}, must be {self.valid_water_models}.", os.path.join(self.args.directory, self.LOG_FL))
            quit()

        if (self.args.watermodel is not None) and (self.args.watermodel != "tip3p"):
            raise NotImplementedError(f"Unsupported water model: {self.args.watermodel}, only tip3p supported for now.")

        if self.args.resume is not None and not os.path.isdir(
                self.args.resume
        ):
            printlog(
                f"Error: Production directory to resume is not a directory: {self.args.resume}.", os.path.join(self.args.directory, self.LOG_FL))
            quit()

        if self.args.resume is not None and self.args.plumed is not None:
            warnings.warn("Using a PLUMED script with a resumed simulation is experimental and may be buggy.",
                          UserWarning)

        if ((self.args.num_water is not None) or (self.args.ionic_strength is not None)
            or (self.args.solventpadding is not None)) and (self.args.watermodel is None):
            printlog(
                f"Error: Must specify a water model to add water and ions or introduce solvent padding.", os.path.join(self.args.directory, self.LOG_FL))
            quit()

        if self.args.num_water is not None and self.args.solventpadding is not None:
            printlog(
                f"Error: Cannot specify both number of water molecules and solvent padding.", os.path.join(self.args.directory, self.LOG_FL))
            quit()

        if self.args.equilibrate not in self.valid_ensembles:
            printlog(
                f"Error: Invalid ensemble: {self.args.equilibrate}, must be {self.valid_ensembles}", os.path.join(self.args.directory, self.LOG_FL))
            quit()

        if self.args.equilibrate == "NPT" and self.args.pressure is None:
            printlog(
                f"Error: Invalid ensemble: {self.args.equilibrate}, must specify pressure.", os.path.join(self.args.directory, self.LOG_FL))
            quit()

        if self.args.resume:
            resume_contains = os.listdir(self.args.resume)
            resume_requires = (
                self.CHECKPOINT_FL,
                self.TRAJECTORY_FL,
                self.STATE_DATA_FL,
                self.METADATA_FL,
            )

            if not all(filename in resume_contains for filename in resume_requires):
                printlog(
                    f"Error: Production directory to resume must contain files with the following names: {resume_requires}", os.path.join(self.args.directory, self.LOG_FL))
                quit()

        self._additional_checks()
        printlog("[✓] Checked arguments", os.path.join(self.args.directory, self.LOG_FL))


class PDBSimulation(OpenMMSimulation):

    def _init_model(self):
        model = app.PDBFile(self.args.pdb)

        return model

    def _init_forcefield(self, model):
        files = self.forcefield_files[self.args.forcefield]
        self.force_field = app.ForceField(*files)

    def _init_system(self, modeller: app.Modeller):
        system = self.force_field.createSystem(
            modeller.topology,
            nonbondedMethod=cutoff_method[self.args.cutoffmethod],
            nonbondedCutoff=self.args.nonbondedcutoff,  # constraints can be added here
        )

        return system

    def _write_updated_model(self, model: Union[app.PDBFile, Dict2Class], modeller: app.Modeller,
                             topology_name: str = 'top.pdb', no_water_topology_name: str = 'top_no_water.pdb'):
        """
        Writes hydrated and non-hydrated PDB object(s) to the output directory.
        :param pdb: PDB object
        :param modeller: Modeller object
        """
        # Create a topology file of system (+ water), as it is saved in the dcd:
        model.writeFile(
            modeller.getTopology(),
            modeller.getPositions(),
            open(os.path.join(self.args.directory, f"{topology_name}"), "w"),
        )
        # If water in the system, then save an additional topology file with without water:
        if self.args.watermodel:
            modeller_copy = copy.deepcopy(modeller)
            modeller_copy.deleteWater()
            model.writeFile(
                modeller_copy.getTopology(),
                modeller_copy.getPositions(),
                open(os.path.join(self.args.directory, f"{no_water_topology_name}"), "w"),
            )
            del modeller_copy

    def _additional_checks(self):
        if self.args.pdb is None:
            printlog("Error: Must provide a PDB file.", os.path.join(self.args.directory, self.LOG_FL))
            quit()


class MOL2Simulation(OpenMMSimulation):

    def _init_model(self) -> Dict2Class:
        ligand_traj = md.load(self.args.mol2)
        ligand_traj.center_coordinates()
        ligand_xyz = ligand_traj.openmm_positions(0)
        ligand_top = ligand_traj.top.to_openmm()
        model = {"topology": ligand_top, "positions": ligand_xyz}
        model = Dict2Class(model)

        return model

    def _init_forcefield(self, model):
        # Includes the forcefield XML file specified by the user:
        files = self.forcefield_files[self.args.forcefield] + [self.args.xml]
        self.force_field = app.ForceField(*files)

    def _init_system(self, modeller: app.Modeller):
        system = self.force_field.createSystem(
            modeller.topology,
            nonbondedMethod=cutoff_method[self.args.cutoffmethod],
            nonbondedCutoff=self.args.nonbondedcutoff,  # constraints can be added here
        )

        return system

    def _write_updated_model(self, model: Union[app.PDBFile, Dict2Class], modeller: app.Modeller,
                             topology_name: str = 'top.pdb', no_water_topology_name: str = 'top_no_water.pdb') -> None:
        pass

    def _additional_checks(self):
        if bool(self.args.mol2) != bool(self.args.xml):  # exclusive-or
            assert self.args.xml is not None, "Specifying a mol2 file requires an xml file (and vice versa)."

        if self.args.equilibrate == "NPT":
            printlog("Error: NPT equilibration ensemble is not currently supported for MOL2 simulations.", os.path.join(self.args.directory, self.LOG_FL))
            quit()

        if self.args.pressure not in ["", None]:
            printlog("Error: Specifying a pressure is not currently supported for MOL2 simulations.", os.path.join(self.args.directory, self.LOG_FL))
            quit()

        if self.args.cutoffmethod != "NoCutoff":
            printlog(
                "Error: Specifying a cutoff method other than 'NoCutoff' is not currently supported for MOL2 simulations.", os.path.join(self.args.directory, self.LOG_FL))
            quit()

# TODO: Implement without requirement to specify a PDB file
class MLSimulation(OpenMMSimulation):

    def _init_model(self):
        model = app.PDBFile(self.args.pdb)

        return model

    def _init_forcefield(self, model):
        files = self.forcefield_files[self.args.forcefield]
        self.force_field = app.ForceField(*files)
        topology = Topology.from_openmm(model.topology, unique_molecules=[Molecule.from_file(self.args.sdf)])
        molecule = Molecule.from_topology(topology)
        smirnoff = SMIRNOFFTemplateGenerator(molecule)
        self.force_field.registerTemplateGenerator(smirnoff.generator)

    def _init_system(self, modeller: app.Modeller):
        # Initialise the system with the standard forcefield
        mm_system = self.force_field.createSystem(
            modeller.topology,
            nonbondedMethod=cutoff_method[self.args.cutoffmethod],
            nonbondedCutoff=self.args.nonbondedcutoff,  # constraints can be added here
        )

        # Check that each ml_residue is present in the topology
        for residue in self.args.ml_residues:
            if residue not in [atom.residue.name for atom in modeller.topology.atoms()]:
                printlog(f"Error: Residue {residue} not present in the topology.", os.path.join(self.args.directory, self.LOG_FL))
                quit()

        # Identify the subset of atoms to be modelled with the ML (ani2x) potential
        ml_atoms = [atom.index for atom in modeller.topology.atoms() if atom.residue.name in self.args.ml_residues]
        printlog(f"ML atoms: {ml_atoms}", os.path.join(self.args.directory, self.LOG_FL))

        # Initialise the ML potential
        potential = MLPotential('ani2x')

        # Add the ML system to the standard forcefield system
        system = potential.createMixedSystem(modeller.topology, mm_system, ml_atoms)

        return system

    def _additional_checks(self):
        if self.args.ml_residues is None:
            printlog("Error: Must specify the residues to be modelled with the ML potential.", os.path.join(self.args.directory, self.LOG_FL))
            quit()

        if self.args.sdf is None:
            printlog("Error: Must specify the sdf file.", os.path.join(self.args.directory, self.LOG_FL))
            quit()

        if self.args.pdb is None:
            printlog("Error: Must provide a PDB file.", os.path.join(self.args.directory, self.LOG_FL))
            quit()

    def _write_updated_model(self, model: Union[app.PDBFile, Dict2Class], modeller: app.Modeller,
                             topology_name: str = 'top.pdb', no_water_topology_name: str = 'top_no_water.pdb') -> None:
        # Create a topology file of system (+ water), as it is saved in the dcd:
        model.writeFile(
            modeller.getTopology(),
            modeller.getPositions(),
            open(os.path.join(self.args.directory, f"{topology_name}"), "w"),
        )
        # If water in the system, then save an additional topology file with without water:
        if self.args.watermodel:
            modeller_copy = copy.deepcopy(modeller)
            modeller_copy.deleteWater()
            model.writeFile(
                modeller_copy.getTopology(),
                modeller_copy.getPositions(),
                open(os.path.join(self.args.directory, f"{no_water_topology_name}"), "w"),
            )
            del modeller_copy
