from abc import abstractmethod
import os
import argparse
import sys
import warnings
from datetime import datetime
import json
import copy
from pathlib import Path
from typing import Union

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
from openmmplumed import PlumedForce
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import seaborn as sns
import mdtraj as md

from utils.trajectory_utils import clean_and_align_trajectory
from utils.openmm_utils import parse_quantity, get_system_args, cutoff_method, get_unit_cell_dims, add_barostat, \
    get_integrator, SystemObjs, SimulationProps, stringify_named_tuple, check_fields_unchanged, \
    update_numerical_fields
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
        self.CHECKPOINT_FN, self.TRAJECTORY_FN, self.STATE_DATA_FN, self.METADATA_FN = \
            "checkpoint.chk", "trajectory.dcd", "state_data.csv", "metadata.json"
        self.required_files = {self.CHECKPOINT_FN, self.TRAJECTORY_FN, self.STATE_DATA_FN, self.METADATA_FN}
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

        # PROPERTIES
        self.system = None
        self.simulation = None
        self.output_dir = None
        self.force_field = None

        # ARGUMENTS
        self.BOOLEAN_ARGS = {
        'periodic': {'flag': '--periodic'},
        'minimise': {'flag': '--minimise'},
        }

        # Parser and arg check
        self.parser = self._init_parser()

        try:
            args = self.parser.parse_args()
            self.args = get_system_args(args)
        except SystemExit:
            # No arguments were passed
            self.args = {}
        else:
            # Check that the arguments are valid
            self._check_args()

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
        STEPS: Initialise directories, read in the PDB, set up the force field, set up the system model including
        any solvent (optional), save a PDB of the entire system model, and create the system object.
        """
        self._init_output_dir()
        print("[✓] Created output directory")
        model = self._init_model()
        print("[✓] Initialised model file(s)")
        self._init_forcefield()
        print("[✓] Initialised force field")
        modeller = self._init_modeller(model)
        print("[✓] Initialised modeller")
        self._write_updated_model(model, modeller)
        system = self._init_system(modeller)
        print("[✓] Created system")
        self.system = SystemObjs(model, modeller, system)

        return self.system

    @abstractmethod
    def _init_model(self):
        raise NotImplementedError

    @abstractmethod
    def _init_forcefield(self) -> app.ForceField:
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
                print("Water found in PBD file: Not changing solvation properties; solvent padding ignored!")
                pass
            else:
                # If no water present and the use has specified using a water model
                if self.args.watermodel:
                    # Automatically add water to the modeller
                    modeller.addSolvent(
                        self.force_field,
                        model=self.args.watermodel,
                        padding=self.args.solventpadding,
                    )
                else:
                    # Do not add water to the modeller
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
        print("[✓] Initialised simulation")
        # if self.args.minimise:
        self._minimise_system_energy()
        print("[✓] Finished minimisation")
        self.simulation.simulation.context.setVelocitiesToTemperature(
            self.args.temperature
        )
        print("[✓] Initialised particle velocities")
        self._save_simulation_metadata()
        print("[✓] Saved simulation metadata")
        self._setup_reporters()
        print("[✓] Setup reporters")

    def run_equilibration(self):
        """
        Run the equilibration.
        """
        eq_obj = EquilibrationProtocol(production_ensemble=self.args.equilibrate,
                                       force_field=self.force_field,
                                       modeller=self.system.modeller,
                                       args=self.args,
                                       fixed_density=True,
                                       output_dir=self.output_dir,
                                       simulation=self.simulation.simulation)
        eq_obj.run()

    def run_simulation(self):
        """
        Run the simulation.
        STEPS: Run the simulation for the specified duration, and save the final checkpoint.
        """
        print(f"Running {self.args.duration} production...")
        if self.args.equilibrate:
            # If equilibration was run, resume from the checkpoint file
            self.simulation.simulation.loadCheckpoint(os.path.join(self.output_dir, f"equilibration_final.chk"))
        self.simulation.simulation.step(self.args.total_steps)
        print("[✓] Finished production")
        self._save_checkpoint()
        print("[✓] Saved final checkpoint")

    def post_process_simulation(self):
        """
        Post-process the simulation.
        STEPS: Centre and align the trajectory, removing solvent, and save the modified trajectory. Read in the
        state chemicals and create plots of energy, temperature etc. Finally, analyse basic statistics of the state
        chemicals and save these to a JSON file.
        """
        clean_and_align_trajectory(working_dir=self.output_dir,
                                   traj_name='trajectory.dcd',
                                   top_name='top.pdb',
                                   save_name='trajectory_processed')
        print("[✓] Cleaned and aligned trajectory")
        report = pd.read_csv(os.path.join(self.output_dir, self.STATE_DATA_FN))
        self._make_graphs(report)
        print("[✓] Made graphs of simulation chemicals")
        self._make_summary_statistics(report)
        print("[✓] Saved summary statistics")

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
            except FileExistsError:
                files_in_dir = set(os.listdir(output_dir))
                simulation_files = self.required_files.intersection(files_in_dir)
                if len(simulation_files) > 0:
                    print(f"Output directory {output_dir} already exists and has simulation files:")
                    [print(file) for file in simulation_files]
                    print("Please choose a different output directory or delete the existing files.")
                    sys.exit(1)
                else:
                    pass

        self.output_dir = output_dir
        return self.output_dir

    def _init_system(self, modeller: app.Modeller):
        """
        Initialises the system object. If using PLUMED, adds the PLUMED force to the system.
        :param modeller: Modeller object
        :return: System object
        """
        system = self.force_field.createSystem(
            modeller.topology,
            nonbondedMethod=cutoff_method[self.args.cutoffmethod],
            nonbondedCutoff=self.args.nonbondedcutoff,   # constraints can be added here
        )
        if self.args.plumed:
            with open(self.args.plumed, "r") as plumed_file:
                plumed_script = plumed_file.read()
            path = Path(self.args.plumed)
            os.chdir(path.parent.absolute())
            system.addForce(PlumedForce(plumed_script))
            print("[✓] Added PLUMED forces")   

        return system

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
            assert (unit_cell_dims is not None), "PDB file missing unit cell dimensions - cannot run NPT simulation."
            add_barostat(barostat_type="MonteCarloBarostat",
                         system=self.system.system,
                         pressure=self.args.pressure,
                         temperature=self.args.temperature)

        integrator = get_integrator(integrator_type=self.args.integrator, args=self.args)

        if self.args.gpu != "":
            print("[✓] Using GPU")
            platform = openmm.Platform.getPlatformByName("CUDA")
            properties["CudaDeviceIndex"] = self.args.gpu
            properties["Precision"] = self.args.precision
        else:
            print("[✓] Using CPU")
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
            print(f"[✓] Multiple GPUs ({self.args.gpu})")

        # Set initial particle positions
        simulation.context.setPositions(self.system.modeller.positions)

        # If resuming, load the checkpoint file
        if self.args.resume:
            with open(os.path.join(self.output_dir, self.CHECKPOINT_FN), "rb") as f:
                simulation.context.loadCheckpoint(f.read())
                print("[✓] Loaded checkpoint for resuming simulation")

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
        # TODO: fix to allow writing of minimised PDB
        # self.system.pdb.writeModel(
        #     self.system.modeller.topology,
        #     positions,
        #     open(os.path.join(self.output_dir, "minimised.pdb"), "w"),
        # )
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
        with open(os.path.join(self.output_dir, self.METADATA_FN), "w") as json_file:
            json.dump(system_args_dict, json_file)

    def _update_metadata_file(self, metadata_update: dict) -> dict:
        """
        Updates the metadata file with the new metadata.
        :param metadata_update: The new metadata to update the file with.
        :return: The updated metadata.
        """
        assert os.path.exists(
            os.path.join(self.output_dir, self.METADATA_FN)), "Cannot resume a simulation without metadata."
        with open(os.path.join(self.output_dir, self.METADATA_FN), "r") as json_file:
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
        # Reporter to log lots of info to csv
        self.simulation.simulation.reporters.append(
            app.StateDataReporter(
                os.path.join(self.output_dir, self.STATE_DATA_FN),
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
                os.path.join(self.output_dir, self.TRAJECTORY_FN),
                reportInterval=self.args.steps_per_save,
                append=True if self.args.resume else False,
            )
        )

        # Reporter to save regular checkpoints
        self.simulation.simulation.reporters.append(
            app.CheckpointReporter(
                os.path.join(self.output_dir, self.CHECKPOINT_FN),
                self.args.steps_per_save,
            )
        )

    def _save_checkpoint(self):
        # Save final checkpoint and state
        self.simulation.simulation.saveCheckpoint(
            os.path.join(self.output_dir, self.CHECKPOINT_FN)
        )
        self.simulation.simulation.saveState(
            os.path.join(self.output_dir, "end_state.xml")
        )

    def _make_graphs(self, report):
        report = report.melt()
        with sns.plotting_context("paper"):
            g = sns.FacetGrid(data=report, row="variable", sharey=False)
            g.map(plt.plot, "value")
            # format the labels with f-strings
            for ax in g.axes.flat:
                ax.xaxis.set_major_formatter(
                    tkr.FuncFormatter(
                        lambda x, p: f"{(x * self.args.stepsize).value_in_unit(unit.nanoseconds):.1f}ns"
                    )
                )
            plt.savefig(
                os.path.join(self.output_dir, "graphs.png"), bbox_inches="tight"
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
        with open(os.path.join(self.output_dir, "summary_statistics.json"), "w") as f:
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
            "-xml",
            "--xml",
            help="(file) Path to xml file of small molecule (requires -mol2).",
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
        parser.add_argument("-integrator", "--integrator", default='Langevin',
                            help="(str) The type of numerical integrator to use, either 'LangevinBAOAB', 'LangevinMiddle' or 'Verlet'.")

        return parser

    # def _check_argument_dict(self, argdict: dict):
    #     self.required_args = [
    #         action.dest
    #         for action in self.parser._actions
    #         if action.required
    #     ]
    #     for key in self.required_args:
    #         assert key in argdict, f"Missing required argument: {key}"
    #     for key in argdict.keys():
    #         assert key in self.parser._option_string_actions.keys() or key in self.required_args, f"Invalid argument: {key}"

    @abstractmethod
    def _additional_checks(self):
        raise NotImplementedError

    def _check_args(self):
        """
        Check that the simulation arguments are valid.
        """
        if self.args.forcefield not in self.valid_force_fields:
            print(
                f"Invalid forcefield: {self.args.forcefield}, must be {self.valid_force_fields}"
            )
            quit()

        if (self.args.watermodel is not None) and (self.args.watermodel not in self.valid_water_models):
            print(
                f"Invalid water model: {self.args.watermodel}, must be {self.valid_water_models}."
            )
            quit()

        if (self.args.watermodel is not None) and (self.args.watermodel != "tip3p"):
            raise NotImplementedError(f"Unsupported water model: {self.args.watermodel}, only tip3p supported for now.")

        if self.args.resume is not None and not os.path.isdir(
                self.args.resume
        ):
            print(
                f"Production directory to resume is not a directory: {self.args.resume}."
            )
            quit()

        if self.args.resume is not None and self.args.plumed is not None:
            warnings.warn("Using a PLUMED script with a resumed simulation is experimental and may be buggy.",
                          UserWarning)

        if self.args.equilibrate not in self.valid_ensembles:
            print(
                f"Invalid ensemble: {self.args.equilibrate}, must be {self.valid_ensembles}"
            )
            quit()

        if self.args.equilibrate == "NPT" and self.args.pressure is None:
            print(
                f"Invalid ensemble: {self.args.equilibrate}, must specify pressure."
            )
            quit()

        if self.args.resume:
            resume_contains = os.listdir(self.args.resume)
            resume_requires = (
                self.CHECKPOINT_FN,
                self.TRAJECTORY_FN,
                self.STATE_DATA_FN,
                self.METADATA_FN,
            )

            if not all(filename in resume_contains for filename in resume_requires):
                print(
                    f"Production directory to resume must contain files with the following names: {resume_requires}"
                )
                quit()

        self._additional_checks()

        print("[✓] Checked arguments")


class PDBSimulation(OpenMMSimulation):

    def _init_model(self):
        """
        Initialises the model from a PDB file.
        Specifies in the PDB object whether periodic boundary conditions are used.
        :return: PDB object
        """
        model = app.PDBFile(self.args.pdb)
        if self.args.periodic is False:
            # print("Setting non-periodic boundary conditions")
            model.topology.setPeriodicBoxVectors(None)

        return model
    
    def _init_forcefield(self):
        """
        Initialises the forcefield object.
        :return: Forcefield object
        """
        files = self.forcefield_files[self.args.forcefield]
        self.force_field = app.ForceField(*files)

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
            open(os.path.join(self.output_dir, f"{topology_name}"), "w"),
        )
        # If water in the system, then save an additional topology file with without water:
        if self.args.watermodel:
            modeller_copy = copy.deepcopy(modeller)
            modeller_copy.deleteWater()
            model.writeFile(
                modeller_copy.getTopology(),
                modeller_copy.getPositions(),
                open(os.path.join(self.output_dir, f"{no_water_topology_name}"), "w"),
            )
            del modeller_copy

    def _additional_checks(self):
        pass


class MOL2Simulation(OpenMMSimulation):

    def _init_model(self) -> Dict2Class:
        """
        Initialises the model from a MOL2 file.
        """
        ligand_traj = md.load(self.args.mol2)
        ligand_traj.center_coordinates()
        ligand_xyz = ligand_traj.openmm_positions(0)
        ligand_top = ligand_traj.top.to_openmm()
        model = {"topology": ligand_top, "positions": ligand_xyz}
        model = Dict2Class(model)

        return model

    def _init_forcefield(self):
        """
        Initialises the forcefield object.
        :return: Forcefield object
        """
        files = self.forcefield_files[self.args.forcefield] + [self.args.xml]
        self.force_field = app.ForceField(*files)

    def _write_updated_model(self, model: Union[app.PDBFile, Dict2Class], modeller: app.Modeller,
                    topology_name: str = 'top.pdb', no_water_topology_name: str = 'top_no_water.pdb') -> None:
        pass

    def _additional_checks(self):
        if bool(self.args.mol2) != bool(self.args.xml):  # exclusive-or
            assert self.args.xml is not None, "Specifying a mol2 file requires an xml file (and vice versa)."

        if self.args.equilibrate == "NPT":
            print("NPT equilibration ensemble is not currently supported for MOL2 simulations.")
            quit()

        if self.args.pressure not in ["", None]:
            print("Specifying a pressure is not currently supported for MOL2 simulations.")
            quit()

        if self.args.cutoffmethod != "NoCutoff":
            print("Specifying a cutoff method other than 'NoCutoff' is not currently supported for MOL2 simulations.")
            quit()
