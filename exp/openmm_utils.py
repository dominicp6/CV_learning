import os
import argparse
import sys
from collections import namedtuple
from datetime import datetime
import json

import openmm
import openmm.app as app
import openmm.unit as unit
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import seaborn as sns

SystemArgs = namedtuple("System_Args",
                        "pdb forcefield resume duration savefreq stepsize "
                        "frictioncoeff total_steps steps_per_save nonperiodic gpu minimise precision watermodel")

SystemObjs = namedtuple("System_Objs",
                        "pdb modeller peptide_indices system")

SimulationProps = namedtuple("Simulation_Props", "integrator simulation properties")


def stringify_named_tuple(obj: namedtuple):
    dict_of_obj = {}
    for key, value in obj.items():
        dict_of_obj[key] = str(value)

    return dict_of_obj


class OpenMMSimulation:

    def __init__(self):
        # CONSTANTS
        self.CHECKPOINT_FN = "checkpoint.chk"
        self.TRAJECTORY_FN = "trajectory.dcd"
        self.STATE_DATA_FN = "state_data.csv"
        self.METADATA_FN = "metadata.json"

        self.valid_ffs = ['ani2x', 'ani1ccx', 'amber', "ani2x_mixed", "ani1ccx_mixed"]
        self.valid_precision = ['single', 'mixed', 'double']
        self.valid_wms = ['tip3p', 'tip3pfb', 'spce', 'tip4pew', 'tip4pfb', 'tip5p']

        # basic quantity string parsing ("1.2ns" -> openmm.Quantity)
        self.unit_labels = {
            "us": unit.microseconds,
            "ns": unit.nanoseconds,
            "ps": unit.picoseconds,
            "fs": unit.femtoseconds
        }

        # PROPERTIES
        self.systemargs = None
        self.systemobjs = None
        self.simulationprops = None
        self.output_dir = None
        self.force_field = None

    def parse_quantity(self, s: str):
        try:
            u = s.lstrip('0123456789.')
            v = s[:-len(u)]
            return unit.Quantity(
                float(v),
                self.unit_labels[u]
            )
        except Exception:
            raise ValueError(f"Invalid quantity: {s}")

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Production run for an equilibrated biomolecule.')
        parser.add_argument("pdb",
                            help="(file) PDB file describing topology and positions. Should be solvated and equilibrated")
        parser.add_argument("ff", help=f"Forcefield/Potential to use: {self.valid_ffs}")
        parser.add_argument("pr", help=f"Precision to use: {self.valid_precision}")
        parser.add_argument("-r", "--resume", help="(dir) Resume simulation from an existing production directory")
        # The CUDA Platform supports parallelizing a simulation across multiple GPUs. \
        # To do that, set this to a comma separated list of values. For example, -g 0,1.
        parser.add_argument("-g", "--gpu", default="",
                            help="(int) Choose CUDA device(s) to target [note - ANI must run on GPU 0]")
        parser.add_argument("-d", "--duration", default="1ns", help="Duration of simulation")
        parser.add_argument("-f", "--savefreq", default="1ps", help="Interval for all reporters to save data")
        parser.add_argument("-s", "--stepsize", default="2fs", help="Integrator step size")
        parser.add_argument("-c", "--frictioncoeff", default="1ps",
                            help="Integrator friction coeff [your value]^-1 ie for 0.1fs^-1 put in 0.1fs. "
                                 "The unit but not the value will be converted to its reciprocal.")
        parser.add_argument("-np", "--nonperiodic", action=argparse.BooleanOptionalAction,
                            help="Prevent periodic boundary conditions from being applied")
        parser.add_argument("-m", "--minimise", action=argparse.BooleanOptionalAction,
                            help="Minimises energy before running the simulation (recommended)")
        parser.add_argument("-w", "--water", default="", help=f"(str) The water model: {self.valid_wms}")
        args = parser.parse_args()
        pdb = args.pdb
        forcefield = args.ff.lower()
        precision = args.pr.lower()
        resume = args.resume
        gpu = args.gpu
        duration = self.parse_quantity(args.duration)
        savefreq = self.parse_quantity(args.savefreq)
        stepsize = self.parse_quantity(args.stepsize)
        frictioncoeff = self.parse_quantity(args.frictioncoeff)
        frictioncoeff = frictioncoeff._value / frictioncoeff.unit
        total_steps = int(duration / stepsize)
        steps_per_save = int(savefreq / stepsize)
        nonperiodic = args.nonperiodic
        minimise = args.minimise
        watermodel = args.water

        self.systemargs = SystemArgs(pdb, forcefield, resume, duration, savefreq, stepsize, frictioncoeff, total_steps,
                                     steps_per_save, nonperiodic, gpu, minimise, precision, watermodel)

        return self.systemargs

    def check_args(self):
        if self.systemargs.forcefield not in self.valid_ffs:
            print(f"Invalid forcefield: {self.systemargs.forcefield}, must be {self.valid_ffs}")
            quit()

        if self.systemargs.watermodel not in self.valid_wms and not None:
            print(f"Invalid water model: {self.systemargs.watermodel}, must be {self.valid_wms}")

        if self.systemargs.resume is not None and not os.path.isdir(self.systemargs.resume):
            print(f"Production directory to resume is not a directory: {self.systemargs.resume}")
            quit()

        if self.systemargs.resume:
            resume_contains = os.listdir(self.systemargs.resume)
            resume_requires = (
                self.CHECKPOINT_FN,
                self.TRAJECTORY_FN,
                self.STATE_DATA_FN
            )

            if not all(filename in resume_contains for filename in resume_requires):
                print(f"Production directory to resume must contain files with the following names: {resume_requires}")
                quit()

    def setup_system(self):
        self.check_args()
        self.make_output_directory()
        pdb = self.initialise_pdb()
        peptide_indices = self.get_peptide_indices(pdb)
        self.initialise_forcefield()
        modeller = self.initialise_modeller(pdb)
        self.write_pdb(pdb, modeller)
        system = self.create_system(pdb)

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
            output_dir = os.path.join("outputs", output_dir)
            os.makedirs(output_dir)

        self.output_dir = output_dir
        return self.output_dir

    # TODO: updating metadata
    def save_simulation_metadata(self):
        with open(os.path.join(self.output_dir, self.METADATA_FN+'.json'), 'w') as json_file:
            json.dump(stringify_named_tuple(self.systemargs), json_file)

    def initialise_pdb(self) -> app.PDBFile:
        pdb = app.PDBFile(self.systemargs.pdb)
        if self.systemargs.nonperiodic:
            pdb.topology.setPeriodicBoxVectors(None)

        return pdb

    def get_peptide_indices(self, pdb) -> list[int]:
        return [atom.index for atom in pdb.topology.atoms() if atom.residue.name != "HOH"]

    def initialise_forcefield(self) -> app.ForceField:
        if self.systemargs.forcefield == "amber":  # Create AMBER system
            self.force_field = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
        else:
            raise ValueError(f'Force field {self.systemargs.forcefield} not supported.')

        return self.force_field

    def initialise_modeller(self, pdb) -> app.Modeller:
        modeller = app.Modeller(
            pdb.topology,
            pdb.positions
        )
        if not self.systemargs.watermodel:
            modeller.deleteWater()
        else:
            modeller.addSolvent(self.force_field, model=self.systemargs.watermodel)

        return modeller

    def write_pdb(self, pdb: app.PDBFile, modeller: app.Modeller):
        # for convenience, create "topology.pdb" of the raw peptide, as it is saved in the dcd.
        # this is helpful for analysis scripts which rely on it down the line
        pdb.writeFile(
            modeller.getTopology(),
            modeller.getPositions(),
            open(os.path.join(self.output_dir, "topology.pdb"), "w")
        )

    def create_system(self, pdb: app.PDBFile):
        """
        nonbondedMethod - The method to use for nonbonded interactions.
                          Allowed values are NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, or PME.
        nonbondedCutoff - The cutoff distance to use for nonbonded interactions.
        constraints (object=None) – Specifies which bonds and angles should be implemented with constraints.
                                    Allowed values are None, HBonds, AllBonds, or HAngles.
        """
        return self.force_field.createSystem(
            pdb.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1 * unit.nanometer,
            # constraints = app.AllBonds,
        )

    def setup_simulation(self):
        self.initialise_simulation()
        self.save_simulation_metadata()
        if self.systemargs.minimise:
            # initial system energy
            print("\ninitial system energy")
            print(self.simulationprops.simulation.context.getState(getEnergy=True).getPotentialEnergy())
            self.simulationprops.simulation.minimizeEnergy()
            print("\nafter minimization")
            print(self.simulationprops.simulation.context.getState(getEnergy=True).getPotentialEnergy())
        self.setup_reporters()

    def initialise_simulation(self):
        print("Initialising production run...")

        properties = {'CudaDeviceIndex': self.systemargs.gpu, 'Precision': self.systemargs.precision}

        # TODO: decide whether and which barostat to add

        # Create constant temp integrator
        integrator = openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin,
            self.systemargs.frictioncoeff,
            self.systemargs.stepsize
        )
        # Create simulation and set initial positions
        simulation = app.Simulation(
            self.systemobjs.pdb.topology,
            self.systemobjs.system,
            integrator,
            openmm.Platform.getPlatformByName("CUDA"),
            properties
        )

        # TODO: what does this do?
        simulation.context.setPositions(self.systemobjs.pdb.positions)
        if self.systemargs.resume:
            with open(os.path.join(self.output_dir, self.CHECKPOINT_FN), "rb") as f:
                simulation.context.loadCheckpoint(f.read())
                print("Loaded checkpoint")

        self.simulationprops = SimulationProps(integrator, simulation, properties)
        return self.simulationprops

    def setup_reporters(self):
        # Reporter to print info to stdout
        self.simulationprops.simulation.reporters.append(app.StateDataReporter(
            sys.stdout,
            self.systemargs.steps_per_save,
            progress=True,  # Info to print. Add anything you want here.
            remainingTime=True,
            speed=True,
            totalSteps=self.systemargs.total_steps,
        ))
        # Reporter to log lots of info to csv
        self.simulationprops.simulation.reporters.append(app.StateDataReporter(
            os.path.join(self.output_dir, self.STATE_DATA_FN),
            self.systemargs.steps_per_save,
            step=True,
            time=True,
            speed=True,
            temperature=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            append=True if self.systemargs.resume else False
        ))
        # Reporter to save trajectory
        # Save only a subset of atoms to the trajectory, ignore water
        self.simulationprops.simulation.reporters.append(app.DCDReporter(
            os.path.join(self.output_dir, self.TRAJECTORY_FN),
            reportInterval=self.systemargs.steps_per_save,
            append=True if self.systemargs.resume else False))

        # Reporter to save regular checkpoints
        self.simulationprops.simulation.reporters.append(app.CheckpointReporter(
            os.path.join(self.output_dir, self.CHECKPOINT_FN),
            self.systemargs.steps_per_save
        ))

    def run_simulation(self):
        print("Running production...")
        self.simulationprops.simulation.step(self.systemargs.total_steps)
        print("Done")

    def save_checkpoint(self):
        # Save final checkpoint and state
        self.simulationprops.simulation.saveCheckpoint(os.path.join(self.output_dir, self.CHECKPOINT_FN))
        self.simulationprops.simulation.saveState(os.path.join(self.output_dir, 'end_state.xml'))

    def make_graphs(self):
        # Make some graphs
        report = pd.read_csv(os.path.join(self.output_dir, self.STATE_DATA_FN))
        report = report.melt()

        with sns.plotting_context('paper'):
            g = sns.FacetGrid(data=report, row='variable', sharey=False)
            g.map(plt.plot, 'value')
            # format the labels with f-strings
            for ax in g.axes.flat:
                ax.xaxis.set_major_formatter(
                    tkr.FuncFormatter(
                        lambda x, p: f'{(x * self.systemargs.stepsize).value_in_unit(unit.nanoseconds):.1f}ns'))
            plt.savefig(os.path.join(self.output_dir, 'graphs.png'), bbox_inches='tight')

# Next Steps
# print a trajectory of the aaa dihedrals, counting the flips
# heatmap of phi and psi would be a good first analysis, use mdanalysis
# aiming for https://docs.mdanalysis.org/1.1.0/documentation_pages/analysis/dihedrals.html
# number of events going between minima states
# "timetrace" - a plot of the dihedral over time (aim for 500ns)
# do this first, shows how often you go back and forth. one plot for each phi/psi angle
# four plots - for each set of pairs
# this gives two heatmap plots like in the documentation
