import os
import argparse
import sys
import warnings
from datetime import datetime
import json
import copy
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
from openmmplumed import PlumedForce
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import seaborn as sns

from utils.trajectory_utils import clean_and_align_trajectory
from utils.openmm_utils import parse_args, parse_quantity, cutoff_method, get_unit_cell_dims, add_barostat, \
    get_integrator, SystemObjs, get_flag, SimulationProps, stringify_named_tuple, check_fields_unchanged, \
    update_numerical_fields
from EquilibrationProtocol import EquilibrationProtocol


class OpenMMSimulation:

    def __init__(self):
        self.CHECKPOINT_FN, self.TRAJECTORY_FN, self.STATE_DATA_FN, self.METADATA_FN = \
            "checkpoint.chk", "trajectory.dcd", "state_data.csv", "metadata.json"
        self.required_files = {self.CHECKPOINT_FN, self.TRAJECTORY_FN, self.STATE_DATA_FN, self.METADATA_FN}
        self.valid_force_fields = ["amber14", "charmm36"]
        self.valid_precisions = ["single", "mixed", "double"]
        self.valid_water_models = ["tip3p", "tip3pfb", "spce", "tip4pew", "tip4pfb", "tip5p"]
        self.valid_ensembles = ["NVT", "NPT"]

        # Properties that must be preserved when resuming a pre-existing simulation
        self.preserved_properties = {'savefreq', 'stepsize', 'steps_per_save', 'temperature', 'pressure',
                                     'frictioncoeff', 'solventpadding', 'nonbondedcutoff', 'cutoffmethod',
                                     'periodic', 'precision', 'watermodel'}
        # Properties that cumulate upon resuming a simulation
        self.cumulative_properties = {'duration', 'total_steps'}

        # PROPERTIES
        self.system = None
        self.simulation = None
        self.output_dir = None
        self.force_field = None

        # Parser and arg check
        self.parser = self._init_parser()
        self.args = None

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
        self.args = parse_args(self.parser)
        self._check_args()
        print("[✓] Checked arguments")
        self._init_output_dir()
        print("[✓] Created output directory")
        pdb = self._init_pdb()
        print("[✓] Initialised PDB file")
        self._init_forcefield()
        print("[✓] Initialised force field")
        modeller = self._init_modeller(pdb)
        print("[✓] Initialised modeller")
        self._write_pdbs(pdb, modeller)
        system = self._init_system(modeller)
        print("[✓] Created system")
        self.system = SystemObjs(pdb, modeller, system)

        return self.system

    def setup_simulation(self):
        """
        Set up the simulation.
        STEPS: Create the integrator, barostat (optional), simulation object, initialise particle positions,
        minimise the system energy, set particle velocities, save all simulation metadata, and set up the reporters.
        """
        self._init_simulation()
        print("[✓] Initialised simulation")
        if self.args.minimise:
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
        state data and create plots of energy, temperature etc. Finally, analyse basic statistics of the state
        data and save these to a JSON file.
        """
        clean_and_align_trajectory(working_dir=self.output_dir,
                                   traj_name='trajectory.dcd',
                                   top_name='top.pdb',
                                   save_name='trajectory_processed')
        print("[✓] Cleaned and aligned trajectory")
        report = pd.read_csv(os.path.join(self.output_dir, self.STATE_DATA_FN))
        self._make_graphs(report)
        print("[✓] Made graphs of simulation data")
        self._make_summary_statistics(report)
        print("[✓] Saved summary statistics")

    def generate_executable_command(self, arg_dict: dict):
        """
        Generate a command to run a simulation with the desired arguments.
        :param arg_dict: Dictionary of arguments
        :return: String of command
        """
        # self._check_argument_dict(arg_dict)

        # Generate command
        command = "python /home/dominic/PycharmProjects/CV_learning/run_openmm.py "

        # Add required arguments
        command += arg_dict['pdb'] + " "
        command += arg_dict['forcefield'] + " "
        command += arg_dict['precision'] + " "

        # Add optional arguments
        try:
            command += arg_dict['resume'] + " "
        except KeyError:
            pass
        command += f"{get_flag(self.parser, '--PLUMED')} {arg_dict['--PLUMED']} "
        try:
            command += f"{get_flag(self.parser, '--gpu')} {arg_dict['--gpu']} "
        except KeyError:
            pass
        command += f"{get_flag(self.parser, '--duration')} {arg_dict['--duration']} "
        command += f"{get_flag(self.parser, '--savefreq')} {arg_dict['--savefreq']} "
        command += f"{get_flag(self.parser, '--stepsize')} {arg_dict['--stepsize']} "
        command += f"{get_flag(self.parser, '--temperature')} {arg_dict['--temperature']} "
        if arg_dict['--pressure'] != "":
            command += f"{get_flag(self.parser, '--pressure')} {arg_dict['--pressure']} "
        command += f"{get_flag(self.parser, '--frictioncoeff')} {arg_dict['--frictioncoeff']} "
        command += f"{get_flag(self.parser, '--solventpadding')} {arg_dict['--solventpadding']} "
        command += f"{get_flag(self.parser, '--nonbondedcutoff')} {arg_dict['--nonbondedcutoff']} "
        command += f"{get_flag(self.parser, '--cutoffmethod')} {arg_dict['--cutoffmethod']} "
        if arg_dict['--periodic'] is True:
            command += f"{get_flag(self.parser, '--periodic')} "
        command += f"{get_flag(self.parser, '--minimise')} "
        command += f"{get_flag(self.parser, '--water')} {arg_dict['--water']} "
        command += f"{get_flag(self.parser, '--seed')} {arg_dict['--seed']} "
        command += f"{get_flag(self.parser, '--name')} {arg_dict['--name']} "
        command += f"{get_flag(self.parser, '--directory')} {arg_dict['--directory']} "
        command += f"{get_flag(self.parser, '--equilibrate')} {arg_dict['--equilibrate']} "

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
            pdb_filename = os.path.splitext(os.path.basename(self.args.pdb))[0]
            exp_name_auto = f"production_{pdb_filename}_" \
                            f"{self.args.forcefield}_{datetime.now().strftime('%H%M%S_%d%m%y')}"

            output_dir = None
            # If no output directory specified, use default
            if not self.args.name and not self.args.dir:
                output_dir = os.path.join("../exp/outputs", exp_name_auto)
            # If output name specified, use that
            elif self.args.name and not self.args.dir:
                output_dir = os.path.join("../exp/outputs", self.args.name)
            # If output directory specified, use that
            elif self.args.dir and not self.args.name:
                output_dir = os.path.join(self.args.dir, exp_name_auto)
            # If both name and directory specified, use both
            elif self.args.dir and self.args.name:
                output_dir = os.path.join(self.args.dir, self.args.name)

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

    def _init_pdb(self) -> app.PDBFile:
        """
        Initialises the PDB object.
        Specifies in the PDB object whether periodic boundary conditions are used.
        :return: PDB object
        """
        pdb = app.PDBFile(self.args.pdb)
        if self.args.periodic is False:
            # print("Setting non-periodic boundary conditions")
            pdb.topology.setPeriodicBoxVectors(None)

        return pdb

    def _init_forcefield(self) -> app.ForceField:
        """
        Initialises the forcefield object.
        :return: Forcefield object
        """
        if self.args.forcefield == "amber14":  # Create AMBER system
            self.force_field = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
        elif self.args.forcefield == "charmm36":  # Create CHARMM system
            self.force_field = app.ForceField("charmm36.xml", "charmm36/water.xml")
        else:
            raise ValueError(f"Force field {self.args.forcefield} not supported.")

        return self.force_field

    def _init_modeller(self, pdb: app.PDBFile) -> app.Modeller:
        """
        Initialises the Modeller object.
        :param pdb: PDB object
        :return: Modeller object
        """
        modeller = app.Modeller(pdb.topology, pdb.positions)
        if self.args.resume:
            # print("Resuming simulation, skipping modeller modifications")
            pass
        else:
            # Check if the loaded topology file already has water
            if np.any([atom.residue.name == 'HOH' for atom in pdb.topology.atoms()]):
                # print("Water found in PBD file: Not changing solvation properties; solvent padding ignored!")
                pass
            else:
                # If no water present and the use has specified using a water model
                if self.args.watermodel:
                    # Automatically add water to the modeller
                    # print("No water found in PDB file: Adding water...")
                    modeller.addSolvent(
                        self.force_field,
                        model=self.args.watermodel,
                        padding=self.args.solventpadding,
                    )
                else:
                    # Do not add water to the modeller
                    pass

        return modeller

    def _write_pdbs(self, pdb: app.PDBFile, modeller: app.Modeller,
                    topology_name: str = 'top.pdb', no_water_topology_name: str = 'top_no_water.pdb') -> None:
        """
        Writes hydrated and non-hydrated PDB object(s) to the output directory.
        :param pdb: PDB object
        :param modeller: Modeller object
        """
        # Create a topology file of system (+ water), as it is saved in the dcd:
        pdb.writeFile(
            modeller.getTopology(),
            modeller.getPositions(),
            open(os.path.join(self.output_dir, f"{topology_name}"), "w"),
        )
        # If water in the system, then save an additional topology file with without water:
        if self.args.watermodel != "":
            modeller_copy = copy.deepcopy(modeller)
            modeller_copy.deleteWater()
            pdb.writeFile(
                modeller_copy.getTopology(),
                modeller_copy.getPositions(),
                open(os.path.join(self.output_dir, f"{no_water_topology_name}"), "w"),
            )
            del modeller_copy

    def _init_system(self, modeller: app.Modeller):
        """
        Initialises the system object. If using PLUMED, adds the PLUMED force to the system.
        :param modeller: Modeller object
        :return: System object
        """
        # print(f"System size: {modeller.topology.getNumAtoms()}")
        system = self.force_field.createSystem(
            modeller.topology,
            nonbondedMethod=cutoff_method[self.args.cutoffmethod],
            nonbondedCutoff=self.args.nonbondedcutoff,
            # constraints = app.AllBonds,
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

        unit_cell_dims = get_unit_cell_dims(modeller=self.system.modeller,
                                            periodic=self.args.periodic,
                                            nonbondedcutoff=self.args.nonbondedcutoff)

        if self.args.pressure is None:
            # If no pressure provided, run an NVT simulation
            # print(
            #     f"Running NVT simulation at temperature {self.args.temperature}."
            # )
            pass
        else:
            # If pressure provided, run an NPT simulation
            assert (unit_cell_dims is not None), "PDB file missing unit cell dimensions - cannot run NPT simulation."
            add_barostat(barostat_type="MonteCarloBarostat",
                         system=self.system.system,
                         pressure=self.args.pressure,
                         temperature=self.args.temperature)
            # print(f"Running NPT simulation at temperature {self.args.temperature} and pressure {self.args.pressure}.")

        integrator = get_integrator(integrator_type="LangevinIntegrator", args=self.args)

        if self.args.gpu is not "":
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
            platform,
            properties,
        )

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
        # print("\nInitial system energy")
        # print(
        #    self.simulation.simulation.context.getState(
        #        getEnergy=True
        #    ).getPotentialEnergy()
        # )
        self.simulation.simulation.minimizeEnergy()
        # print("\nAfter minimization")
        # print(
        #    self.simulation.simulation.context.getState(
        #        getEnergy=True
        #    ).getPotentialEnergy()
        # )
        positions = self.simulation.simulation.context.getState(
            getPositions=True
        ).getPositions()
        self.system.pdb.writeModel(
            self.system.modeller.topology,
            positions,
            open(os.path.join(self.output_dir, "minimised.pdb"), "w"),
        )
        self.system.modeller.positions = positions

    def _save_simulation_metadata(self):
        """
        Saves the simulation metadata to a JSON file.
        If the simulation is being resumed, the metadata is loaded from the existing file and updated.
        """
        system_args_dict = stringify_named_tuple(self.args)

        # If resuming, load existing metadata and update
        if self.args.resume:
            # print('Updating metadata')
            system_args_dict = self._update_metadata_file(system_args_dict)
            # Cumulate duration and total_steps from previous runs
            self.args.duration = parse_quantity(system_args_dict['duration'])
            self.args.total_steps = int(self.args.duration / self.args.stepsize)

        # Save metadata
        # print(system_args_dict)
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
        parser.add_argument(
            "pdb",
            help="(file) PDB file describing topology and positions.",
        )
        parser.add_argument("forcefield", help=f"Forcefield/Potential to use: {self.valid_force_fields}")
        parser.add_argument("precision", help=f"Precision to use: {self.valid_precisions}")
        parser.add_argument(
            "-r",
            "--resume",
            help="(dir) Resume simulation from an existing production directory",
        )
        parser.add_argument(
            "-PLUMED",
            "--PLUMED",
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
            help="Interval for all reporters to save data",
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
            "-w", "--water", default="", help=f"(str) The water model: {self.valid_water_models}"
        )
        parser.add_argument("-seed", "--seed", default='0', help="Random seed")
        parser.add_argument("-name", "--name", default=None, help="(str) Name of simulation. "
                                                                  "If not provided, a name will be generated.")
        parser.add_argument("-dir", "--directory", default=None, help="(str) Directory to save simulation in. "
                                                                      "If not provided, a directory will be generated.")
        parser.add_argument("-equilibrate", "--equilibrate", default=None,
                            help="(str) Target production ensemble NVE/NVT/NPT")

        return parser

    def _check_argument_dict(self, argdict: dict):
        self.required_args = [
            action.dest
            for action in self.parser._actions
            if action.required
        ]
        for key in self.required_args:
            assert key in argdict, f"Missing required argument: {key}"
        for key in argdict.keys():
            assert key in self.parser._option_string_actions.keys() or key in self.required_args, f"Invalid argument: {key}"

    def _check_args(self):
        """
        Check that the simulation arguments are valid.
        """
        if self.args.forcefield not in self.valid_force_fields:
            print(
                f"Invalid forcefield: {self.args.forcefield}, must be {self.valid_force_fields}"
            )
            quit()

        if self.args.watermodel not in self.valid_water_models and not "":
            print(
                f"Invalid water model: {self.args.watermodel}, must be {self.valid_water_models}"
            )
            quit()

        if self.args.watermodel != "" and self.args.watermodel != "tip3p":
            raise NotImplementedError(f"Unsupported water model: {self.args.watermodel}, only tip3p supported for now")

        if self.args.resume is not None and not os.path.isdir(
                self.args.resume
        ):
            print(
                f"Production directory to resume is not a directory: {self.args.resume}"
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
