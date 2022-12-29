import os
import argparse
import sys
from collections import namedtuple
from datetime import datetime
import json
import copy

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import seaborn as sns
# from pdbfixer import PDBFixer

from utils.trajectory_utils import clean_and_align_trajectory

SystemArgs = namedtuple(
    "System_Args",
    "pdb forcefield resume duration savefreq stepsize temperature pressure "
    "frictioncoeff solventpadding nonbondedcutoff cutoffmethod total_steps steps_per_save periodic gpu minimise precision watermodel",
)

SystemObjs = namedtuple("System_Objs", "pdb modeller peptide_indices system")

SimulationProps = namedtuple("Simulation_Props", "integrator simulation properties")

# basic quantity string parsing ("1.2ns" -> openmm.Quantity)
# noinspection PyUnresolvedReferences
unit_labels = {
    "us": unit.microseconds,
    "ns": unit.nanoseconds,
    "ps": unit.picoseconds,
    "fs": unit.femtoseconds,
    "nm": unit.nanometers,
    "bar": unit.bar,
    "K": unit.kelvin,
}

cutoff_method = {
    "NoCutoff": app.NoCutoff,
    "CutoffNonPeriodic": app.CutoffNonPeriodic,
    "CutoffPeriodic": app.CutoffPeriodic,
    "Ewald": app.Ewald,
    "PME": app.PME,
}


def stringify_named_tuple(obj: namedtuple):
    dict_of_obj = {}
    for key, value in obj._asdict().items():
        dict_of_obj[key] = str(value).replace(" ", "")

    return dict_of_obj


def check_fields_unchanged(old_dict: dict, update_dict: dict, fields: set[str]):
    for field in fields:
        assert old_dict[field] == update_dict[field], f"Cannot resume experiment because previous experiment had " \
                                                      f"{field}={old_dict[field]}, which is not the same as " \
                                                      f"{field}={update_dict[field]} specified in the new experiment."


def update_numerical_fields(old_dict: dict, update_dict: dict, fields: set[str]):
    for field in fields:
        update_dict[field] = str(parse_quantity(old_dict[field]) + parse_quantity(update_dict[field])).replace(" ", "")

    return update_dict


# def fix_pdb(path_to_file):
#     print("Creating PDBFixer...")
#     fixer = PDBFixer(path_to_file)
#     print("Finding missing residues...")
#     fixer.findMissingResidues()
#
#     chains = list(fixer.topology.chains())
#     keys = fixer.missingResidues.keys()
#     for key in list(keys):
#         chain = chains[key[0]]
#         if key[1] == 0 or key[1] == len(list(chain.residues())):
#             del fixer.missingResidues[key]
#
#     print("Finding nonstandard residues...")
#     fixer.findNonstandardResidues()
#     print("Replacing nonstandard residues...")
#     fixer.replaceNonstandardResidues()
#     print("Removing heterogens...")
#     fixer.removeHeterogens(keepWater=True)
#
#     print("Finding missing atoms...")
#     fixer.findMissingAtoms()
#     print("Adding missing atoms...")
#     fixer.addMissingAtoms()
#     print("Adding missing hydrogens...")
#     fixer.addMissingHydrogens(7)
#     print("Writing PDB file...")
#
#     app.pdbfile.PDBFile.writeFile(
#         fixer.topology,
#         fixer.positions,
#         open(f"{path_to_file[:-4]}_fixed.pdb", "w"),
#         keepIds=True,
#     )


def isnumber(s: str):
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_quantity(s: str):
    if isnumber(s):
        return float(s)
    try:
        u = s.lstrip("0123456789.")
        v = s[: -len(u)]
        return unit.Quantity(float(v), unit_labels[u])
    except Exception:
        raise ValueError(f"Invalid quantity: {s}")


def round_format_unit(unit: unit.Unit, significant_figures: int):
    return f'{round(unit._value, significant_figures)} {unit.unit.get_symbol()}'


def time_to_iteration_conversion(time: str, duration: unit.Unit, num_frames: int):
    time = parse_quantity(time)
    iteration = int(time / duration * num_frames)
    return iteration


class OpenMMSimulation:

    def __init__(self):
        # CONSTANTS
        self.CHECKPOINT_FN = "checkpoint.chk"
        self.TRAJECTORY_FN = "trajectory.dcd"
        self.STATE_DATA_FN = "state_data.csv"
        self.METADATA_FN = "metadata.json"

        self.valid_ffs = ["amber", "charmm"]
        self.valid_precision = ["single", "mixed", "double"]
        self.valid_wms = ["tip3p", "tip3pfb", "spce", "tip4pew", "tip4pfb", "tip5p"]

        # Properties that must be preserved when resuming a pre-existing simulation
        self.preserved_properties = {'savefreq', 'stepsize', 'steps_per_save', 'temperature', 'pressure',
                                     'frictioncoeff', 'solventpadding', 'nonbondedcutoff', 'cutoffmethod',
                                     'periodic', 'precision', 'watermodel'}
        # Properties that cumulate upon resuming a simulation
        self.cumulative_properties = {'duration', 'total_steps'}

        # PROPERTIES
        self.systemargs = None
        self.systemobjs = None
        self.simulationprops = None
        self.output_dir = None
        self.force_field = None

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Production run for an equilibrated biomolecule."
        )
        parser.add_argument(
            "pdb",
            help="(file) PDB file describing topology and positions.",
        )
        parser.add_argument("ff", help=f"Forcefield/Potential to use: {self.valid_ffs}")
        parser.add_argument("pr", help=f"Precision to use: {self.valid_precision}")
        parser.add_argument(
            "-r",
            "--resume",
            help="(dir) Resume simulation from an existing production directory",
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
            help="The non-bonded cutoff method to use, one of 'NoCutoff', 'CutoffNonPeriodic', 'CutoffPeriodic', 'Ewald', or 'PME'",
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
            "-w", "--water", default="", help=f"(str) The water model: {self.valid_wms}"
        )
        args = parser.parse_args()
        pdb = args.pdb
        forcefield = args.ff.lower()
        precision = args.pr.lower()
        resume = args.resume
        gpu = args.gpu
        duration = parse_quantity(args.duration)
        savefreq = parse_quantity(args.savefreq)
        stepsize = parse_quantity(args.stepsize)
        temperature = parse_quantity(args.temperature)
        pressure = parse_quantity(args.pressure) if args.pressure else None
        frictioncoeff = parse_quantity(args.frictioncoeff)
        solventpadding = parse_quantity(args.solventpadding)
        nonbondedcutoff = parse_quantity(args.nonbondedcutoff)
        cutoffmethod = args.cutoffmethod
        frictioncoeff = frictioncoeff._value / frictioncoeff.unit
        total_steps = int(duration / stepsize)
        steps_per_save = int(savefreq / stepsize)
        periodic = args.periodic
        minimise = args.minimise
        watermodel = args.water

        if not periodic:
            assert cutoffmethod in [
                "NoCutoff",
                "CutoffNonPeriodic",
            ], f"You have specified a non-periodic simulation but have given an incompatible cutoff method " \
               f"({cutoffmethod}). Please change the cutoff method to either 'NoCutoff' or 'CutoffNonPeriodic'."

        self.systemargs = SystemArgs(
            pdb,
            forcefield,
            resume,
            duration,
            savefreq,
            stepsize,
            temperature,
            pressure,
            frictioncoeff,
            solventpadding,
            nonbondedcutoff,
            cutoffmethod,
            total_steps,
            steps_per_save,
            periodic,
            gpu,
            minimise,
            precision,
            watermodel,
        )

        return self.systemargs

    def check_args(self):
        if self.systemargs.forcefield not in self.valid_ffs:
            print(
                f"Invalid forcefield: {self.systemargs.forcefield}, must be {self.valid_ffs}"
            )
            quit()

        if self.systemargs.watermodel not in self.valid_wms and not "":
            print(
                f"Invalid water model: {self.systemargs.watermodel}, must be {self.valid_wms}"
            )

        if self.systemargs.resume is not None and not os.path.isdir(
                self.systemargs.resume
        ):
            print(
                f"Production directory to resume is not a directory: {self.systemargs.resume}"
            )
            quit()

        if self.systemargs.resume:
            resume_contains = os.listdir(self.systemargs.resume)
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

    def setup_system(self):
        print("Setting up system...")
        self.check_args()
        print("[x] Checked valid args")
        self.make_output_directory()
        print("[x] Made output dir")
        pdb = self.initialise_pdb()
        peptide_indices = self.get_peptide_indices(pdb)
        print("[x] Initialised PDB")
        self.initialise_forcefield()
        print("[x] Initialised force field")
        modeller = self.initialise_modeller(pdb)
        print("[x] Initialised modeller")
        self.write_pdb(pdb, modeller)
        system = self.create_system(modeller)
        print("Successfully created system")

        self.systemobjs = SystemObjs(pdb, modeller, peptide_indices, system)
        return self.systemobjs

    def make_output_directory(self) -> str:
        if self.systemargs.resume:
            # Use existing output directory
            output_dir = self.systemargs.resume
        else:
            # Make output directory
            pdb_filename = os.path.splitext(os.path.basename(self.systemargs.pdb))[0]
            output_dir = f"production_{pdb_filename}_{self.systemargs.forcefield}_{datetime.now().strftime('%H%M%S_%d%m%y')}"
            output_dir = os.path.join("../exp/outputs", output_dir)
            os.makedirs(output_dir)

        self.output_dir = output_dir
        return self.output_dir

    def save_simulation_metadata(self):
        args_as_dict = stringify_named_tuple(self.systemargs)
        if self.systemargs.resume:
            print('Updating metadata')
            args_as_dict = self._update_existing_metadata(args_as_dict)
        print(args_as_dict)
        with open(os.path.join(self.output_dir, self.METADATA_FN), "w") as json_file:
            json.dump(args_as_dict, json_file)

    def _update_existing_metadata(self, metadata_update: dict) -> dict:
        assert os.path.exists(
            os.path.join(self.output_dir, self.METADATA_FN)), "Cannot resume a simulation without metadata."
        with open(os.path.join(self.output_dir, self.METADATA_FN), "r") as json_file:
            metadata_existing = json.load(json_file)
        check_fields_unchanged(metadata_existing, metadata_update, fields=self.preserved_properties)
        metadata = update_numerical_fields(metadata_existing, metadata_update, fields=self.cumulative_properties)

        return metadata

    def initialise_pdb(self) -> app.PDBFile:
        pdb = app.PDBFile(self.systemargs.pdb)
        if not self.systemargs.periodic:
            print("Setting non-periodic boundary conditions")
            pdb.topology.setPeriodicBoxVectors(None)

        return pdb

    def get_peptide_indices(self, pdb) -> list[int]:
        return [
            atom.index for atom in pdb.topology.atoms() if atom.residue.name != "HOH"
        ]

    def initialise_forcefield(self) -> app.ForceField:
        if self.systemargs.forcefield == "amber":  # Create AMBER system
            self.force_field = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
        elif self.systemargs.forcefield == "charmm": # Create CHARMM system
            self.force_field = app.ForceField("charmm36.xml", "charmm36/water.xml")
        else:
            raise ValueError(f"Force field {self.systemargs.forcefield} not supported.")

        return self.force_field

    def initialise_modeller(self, pdb) -> app.Modeller:
        modeller = app.Modeller(pdb.topology, pdb.positions)
        if self.systemargs.resume:
            # Do not make any modifications to the modeller
            print("Resuming simulation, skipping modeller modifications")
            pass
        else:
            # Check if the loaded topology file already has water
            if np.any([atom.residue.name == 'HOH' for atom in pdb.topology.atoms()]):
                print("Water found in PBD file: Not changing solvation properties; solvent padding ignored!")
            # If no water present
            else:
                # If we are using a water model
                if self.systemargs.watermodel:
                    # Add water to the modeller
                    print("No water found in PDB file: Adding water...")
                    modeller.addSolvent(
                        self.force_field,
                        model=self.systemargs.watermodel,
                        padding=self.systemargs.solventpadding,
                    )
                else:
                    # Do not add water to the modeller
                    pass

        return modeller

    def write_pdb(self, pdb: app.PDBFile, modeller: app.Modeller):
        # for convenience, create "top.pdb" of the raw peptide, as it is saved in the dcd.
        # this is helpful for analysis scripts which rely on it down the line
        pdb.writeFile(
            modeller.getTopology(),
            modeller.getPositions(),
            open(os.path.join(self.output_dir, "top.pdb"), "w"),
        )
        # If there are water molecules in the topology file, then save an additional topology
        # file excluding those water molecules
        if self.systemargs.watermodel != "":
            modeller_copy = copy.deepcopy(modeller)
            modeller_copy.deleteWater()
            pdb.writeFile(
                modeller_copy.getTopology(),
                modeller_copy.getPositions(),
                open(os.path.join(self.output_dir, "topology_nw.pdb"), "w"),
            )
            del modeller_copy

    def create_system(self, modeller: app.Modeller):
        """
        # TODO: research which of these methods is best
        nonbondedMethod - The cutoff method to use for nonbonded interactions.
        nonbondedCutoff - The cutoff distance to use for nonbonded interactions.
        constraints (object=None) – Specifies which bonds and angles should be implemented with constraints.
                                    Allowed values are None, HBonds, AllBonds, or HAngles.
        """
        print(f"System size: {modeller.topology.getNumAtoms()}")
        return self.force_field.createSystem(
            modeller.topology,
            nonbondedMethod=cutoff_method[self.systemargs.cutoffmethod],
            nonbondedCutoff=self.systemargs.nonbondedcutoff,
            # constraints = app.AllBonds,
        )

    def setup_simulation(self):
        print("Setting up simulation...")
        self.initialise_simulation()
        print("[x] Initialised simulation")
        if self.systemargs.minimise:
            print("Running energy minimisation...")
            # initial system energy
            print("\ninitial system energy")
            print(
                self.simulationprops.simulation.context.getState(
                    getEnergy=True
                ).getPotentialEnergy()
            )
            self.simulationprops.simulation.minimizeEnergy()
            print("\nafter minimization")
            print(
                self.simulationprops.simulation.context.getState(
                    getEnergy=True
                ).getPotentialEnergy()
            )
            positions = self.simulationprops.simulation.context.getState(
                getPositions=True
            ).getPositions()
            print("Writing minimised geometry to PDB file")
            self.systemobjs.pdb.writeModel(
                self.systemobjs.modeller.topology,
                positions,
                open(os.path.join(self.output_dir, "minimised.pdb"), "w"),
            )
            print("[x] Finished minimisation")
        self.simulationprops.simulation.context.setVelocitiesToTemperature(
            self.systemargs.temperature
        )
        self.setup_reporters()
        self.save_simulation_metadata()
        print("Successfully setup simulation")

    def initialise_simulation(self):
        properties = {
            "CudaDeviceIndex": self.systemargs.gpu,
            "Precision": self.systemargs.precision,
        }

        unit_cell_dims = self.systemobjs.modeller.getTopology().getUnitCellDimensions()
        if self.systemargs.periodic:
            print(f"Unit cell dimensions: {unit_cell_dims}, Cutoff distance: {self.systemargs.nonbondedcutoff}.")

        if self.systemargs.pressure is None:
            # Run NVT simulation
            print(
                f"Running NVT simulation at temperature {self.systemargs.temperature}."
            )
        else:
            # Run NPT simulation
            assert (
                    unit_cell_dims is not None
            ), "Periodic boundary conditions not found in PDB file - cannot run NPT simulation."
            # default frequency for pressure changes is 25 time steps
            barostat = openmm.MonteCarloBarostat(
                self.systemargs.pressure, self.systemargs.temperature, 25,
            )
            self.systemobjs.system.addForce(barostat)
            print(
                f"Running NPT simulation at temperature {self.systemargs.temperature} "
                f"and pressure {self.systemargs.pressure}."
            )

        # Create constant temp integrator
        integrator = openmm.LangevinMiddleIntegrator(
            self.systemargs.temperature,
            self.systemargs.frictioncoeff,
            self.systemargs.stepsize,
        )
        # Create simulation and set initial positions
        simulation = app.Simulation(
            self.systemobjs.modeller.topology,
            self.systemobjs.system,
            integrator,
            openmm.Platform.getPlatformByName("CUDA"),
            properties,
        )

        # Specify the initial positions to be the positions that were loaded from the PDB file
        simulation.context.setPositions(self.systemobjs.modeller.positions)
        if self.systemargs.resume:
            with open(os.path.join(self.output_dir, self.CHECKPOINT_FN), "rb") as f:
                simulation.context.loadCheckpoint(f.read())
                print("Loaded checkpoint")

        self.simulationprops = SimulationProps(integrator, simulation, properties)
        return self.simulationprops

    def setup_reporters(self):
        # Reporter to print info to stdout
        self.simulationprops.simulation.reporters.append(
            app.StateDataReporter(
                sys.stdout,
                self.systemargs.steps_per_save,
                progress=True,  # Info to print. Add anything you want here.
                remainingTime=True,
                speed=True,
                totalSteps=self.systemargs.total_steps,
            )
        )
        # Reporter to log lots of info to csv
        # TODO: add configurational energy report
        self.simulationprops.simulation.reporters.append(
            app.StateDataReporter(
                os.path.join(self.output_dir, self.STATE_DATA_FN),
                self.systemargs.steps_per_save,
                step=True,
                time=True,
                speed=True,
                temperature=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                volume=True
                if self.systemargs.pressure
                else False,  # record volume and density for NPT simulations
                density=True if self.systemargs.pressure else False,
                append=True if self.systemargs.resume else False,
            )
        )
        # Reporter to save trajectory
        # Save only a subset of atoms to the trajectory, ignore water
        self.simulationprops.simulation.reporters.append(
            app.DCDReporter(
                os.path.join(self.output_dir, self.TRAJECTORY_FN),
                reportInterval=self.systemargs.steps_per_save,
                append=True if self.systemargs.resume else False,
            )
        )

        # Reporter to save regular checkpoints
        self.simulationprops.simulation.reporters.append(
            app.CheckpointReporter(
                os.path.join(self.output_dir, self.CHECKPOINT_FN),
                self.systemargs.steps_per_save,
            )
        )

    def run_simulation(self):
        # input("> Please press any key to confirm this simulation...")
        print("Running production...")
        self.simulationprops.simulation.step(self.systemargs.total_steps)
        print("Done")

    def save_checkpoint(self):
        # Save final checkpoint and state
        self.simulationprops.simulation.saveCheckpoint(
            os.path.join(self.output_dir, self.CHECKPOINT_FN)
        )
        self.simulationprops.simulation.saveState(
            os.path.join(self.output_dir, "end_state.xml")
        )

    def post_processing_and_analysis(self):
        clean_and_align_trajectory(working_dir=self.output_dir,
                                   traj_name='traj_1ps.dcd',
                                   top_name='top_1ps.pdb',
                                   save_name='trajectory_processed')
        report = pd.read_csv(os.path.join(self.output_dir, self.STATE_DATA_FN))
        self.make_graphs(report)
        self.make_summary_statistics(report)

    def make_graphs(self, report):
        report = report.melt()
        with sns.plotting_context("paper"):
            g = sns.FacetGrid(data=report, row="variable", sharey=False)
            g.map(plt.plot, "value")
            # format the labels with f-strings
            for ax in g.axes.flat:
                ax.xaxis.set_major_formatter(
                    tkr.FuncFormatter(
                        lambda x, p: f"{(x * self.systemargs.stepsize).value_in_unit(unit.nanoseconds):.1f}ns"
                    )
                )
            plt.savefig(
                os.path.join(self.output_dir, "graphs.png"), bbox_inches="tight"
            )

    def make_summary_statistics(self, report):
        statistics = dict()
        statistics["PE"] = report["Potential Energy (kJ/mole)"].mean()
        statistics["dPE"] = report["Potential Energy (kJ/mole)"].std() / np.sqrt(
            self.systemargs.duration / self.systemargs.savefreq
        )
        statistics["KE"] = report["Kinetic Energy (kJ/mole)"].mean()
        statistics["dKE"] = report["Kinetic Energy (kJ/mole)"].std() / np.sqrt(
            self.systemargs.duration / self.systemargs.savefreq
        )
        statistics["TE"] = report["Total Energy (kJ/mole)"].mean()
        statistics["dTE"] = report["Total Energy (kJ/mole)"].std() / np.sqrt(
            self.systemargs.duration / self.systemargs.savefreq
        )
        statistics["T"] = report["Temperature (K)"].mean()
        statistics["dT"] = report["Temperature (K)"].std() / np.sqrt(
            self.systemargs.duration / self.systemargs.savefreq
        )
        statistics["S"] = report["Speed (ns/day)"].mean()
        statistics["dS"] = report["Speed (ns/day)"].std() / np.sqrt(
            self.systemargs.duration / self.systemargs.savefreq
        )
        if self.systemargs.pressure:  # volume and pressure recorded for NPT simulations
            statistics["V"] = report["Box Volume (nm^3)"].mean()
            statistics["dV"] = report["Box Volume (nm^3)"].std() / np.sqrt(
                self.systemargs.duration / self.systemargs.savefreq
            )
            statistics["D"] = report["Density (g/mL)"].mean()
            statistics["dD"] = report["Density (g/mL)"].std() / np.sqrt(
                self.systemargs.duration / self.systemargs.savefreq
            )
        with open(os.path.join(self.output_dir, "summary_statistics.json"), "w") as f:
            json.dump(statistics, f)

# Next Steps
# print a trajectory of the aaa dihedrals, counting the flips
# heatmap of phi and psi would be a good first analysis, use mdanalysis
# aiming for https://docs.mdanalysis.org/1.1.0/documentation_pages/analysis/dihedrals.html
# number of events going between minima states
# "timetrace" - a plot of the dihedral over time (aim for 500ns)
# do this first, shows how often you go back and forth. one plot for each phi/psi angle
# four plots - for each set of pairs
# this gives two heatmap plots like in the documentation
