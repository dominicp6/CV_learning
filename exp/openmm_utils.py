import os
import argparse
from collections import namedtuple
from datetime import datetime

import openmm
import openmm.app as app
import openmm.unit as unit

##############################################
#   CONSTANTS
##############################################

CHECKPOINT_FN = "checkpoint.chk"
TRAJECTORY_FN = "trajectory.dcd"
STATE_DATA_FN = "state_data.csv"

valid_ffs = ['ani2x', 'ani1ccx', 'amber', "ani2x_mixed", "ani1ccx_mixed"]

# basic quantity string parsing ("1.2ns" -> openmm.Quantity)
unit_labels = {
    "us": unit.microseconds,
    "ns": unit.nanoseconds,
    "ps": unit.picoseconds,
    "fs": unit.femtoseconds
}


def parse_quantity(s):
    try:
        u = s.lstrip('0123456789.')
        v = s[:-len(u)]
        return unit.Quantity(
            float(v),
            unit_labels[u]
        )
    except Exception:
        raise ValueError(f"Invalid quantity: {s}")


SystemArgs = namedtuple("System Args",
                        "pdb forcefield resume duration savefreq stepsize "
                        "frictioncoeff total_steps steps_per_save nonperiodic gpu")

SystemObjs = namedtuple("System Objs",
                        "pdb modeller peptide_indices system")

SimulationProps = namedtuple("Simulation Props", "integrator simulation properties")

def parse_args():
    parser = argparse.ArgumentParser(description='Production run for an equilibrated biomolecule.')
    parser.add_argument("pdb", help="PDB file describing topology and positions. Should be solvated and equilibrated")
    parser.add_argument("ff", help=f"Forcefield/Potential to use: {valid_ffs}")
    parser.add_argument("-r", "--resume", help="Resume simulation from an existing production directory")
    parser.add_argument("-g", "--gpu", default="",
                        help="Choose CUDA device(s) to target [note - ANI must run on GPU 0]")
    parser.add_argument("-d", "--duration", default="1ns", help="Duration of simulation")
    parser.add_argument("-f", "--savefreq", default="1ps", help="Interval for all reporters to save data")
    parser.add_argument("-s", "--stepsize", default="2fs", help="Integrator step size")
    parser.add_argument("-c", "--frictioncoeff", default="1ps",
                        help="Integrator friction coeff [your value]^-1 ie for 0.1fs^-1 put in 0.1fs. "
                             "The unit but not the value will be converted to its reciprocal.")
    parser.add_argument("-np", "--nonperiodic", action=argparse.BooleanOptionalAction,
                        help="Prevent periodic boundary conditions from being applied")
    args = parser.parse_args()
    pdb = args.pdb
    forcefield = args.ff.lower()
    resume = args.resume
    duration = parse_quantity(args.duration)
    savefreq = parse_quantity(args.savefreq)
    stepsize = parse_quantity(args.stepsize)
    frictioncoeff = parse_quantity(args.frictioncoeff)
    frictioncoeff = frictioncoeff._value / frictioncoeff.unit
    total_steps = int(duration / stepsize)
    steps_per_save = int(savefreq / stepsize)
    nonperiodic = args.nonperiodic
    gpu = args.gpu

    return SystemArgs(pdb, forcefield, resume, duration, savefreq, stepsize, frictioncoeff, total_steps, steps_per_save, nonperiodic, gpu)


def check_args(SystemArgs: SystemArgs):
    if SystemArgs.forcefield not in valid_ffs:
        print(f"Invalid forcefield: {SystemArgs.forcefield}, must be {valid_ffs}")
        quit()

    if not os.path.isdir(SystemArgs.resume):
        print(f"Production directory to resume is not a directory: {SystemArgs.resume}")
        quit()

    if SystemArgs.resume:
        resume_contains = os.listdir(SystemArgs.resume)
        resume_requires = (
            CHECKPOINT_FN,
            TRAJECTORY_FN,
            STATE_DATA_FN
        )

        if not all(filename in resume_contains for filename in resume_requires):
            print(f"Production directory to resume must contain files with the following names: {resume_requires}")
            quit()

        # # Use existing output directory
        # output_dir = resume


def make_output_directory(SystemArgs: SystemArgs):
    if SystemArgs.resume:
        output_dir = SystemArgs.resume
    else:
        # Make output directory
        pdb_filename = os.path.splitext(os.path.basename(SystemArgs.pdb))[0]
        output_dir = f"production_{pdb_filename}_{SystemArgs.forcefield}_{datetime.datetime.now().strftime('%H%M%S_%d%m%y')}"
        output_dir = os.path.join("outputs", output_dir)
        os.makedirs(output_dir)

    return output_dir


def initialise_pdb(SystemArgs) -> app.PDBFile:
    pdb = app.PDBFile(SystemArgs.pdb)
    if SystemArgs.nonperiodic:
        pdb.topology.setPeriodicBoxVectors(None)

    return pdb


def get_peptide_indices(pdb) -> list[int]:
    return [atom.index for atom in pdb.topology.atoms() if atom.residue.name != "HOH"]


def initialise_modeller(pdb) -> app.Modeller:
    modeller = app.Modeller(
        pdb.topology,
        pdb.positions
    )
    modeller.deleteWater()

    return modeller


def write_pdb(pdb: app.PDBFile, modeller: app.Modeller, output_dir: str):
    # for convenience, create "topology.pdb" of the raw peptide, as it is saved in the dcd.
    # this is helpful for analysis scripts which rely on it down the line
    pdb.writeFile(
        modeller.getTopology(),
        modeller.getPositions(),
        open(os.path.join(output_dir, "topology.pdb"), "w")
    )


def create_system(SystemArgs: SystemArgs, pdb: app.PDBFile):
    """
    nonbondedMethod - The method to use for nonbonded interactions.
                      Allowed values are NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, or PME.
    nonbondedCutoff - The cutoff distance to use for nonbonded interactions.
    constraints (object=None) – Specifies which bonds and angles should be implemented with constraints.
                                Allowed values are None, HBonds, AllBonds, or HAngles.
    """

    if SystemArgs.forcefield is "amber":  # Create AMBER system
        ff = app.ForceField('amber14-all.xml','amber14/tip3p.xml')
    else:
        raise ValueError(f'Force field {SystemArgs.forcefield} not supported.')

    return ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=1 * unit.nanometer,
        # constraints = app.AllBonds,
    )


def setup_system(SystemArgs: SystemArgs, output_dir: str):
    check_args(SystemArgs)
    make_output_directory(SystemArgs)
    pdb = initialise_pdb(SystemArgs)
    peptide_indices = get_peptide_indices(pdb)
    modeller = initialise_modeller(pdb)
    write_pdb(pdb, modeller, output_dir)
    system = create_system(SystemArgs, pdb)

    return SystemObjs(pdb, modeller, peptide_indices, system)


def initialise_simulation(SystemArgs: SystemArgs, SystemObjs: SystemObjs, output_dir: str):
    print("Initialising production run...")

    properties = {'CudaDeviceIndex': SystemArgs.gpu}

    # Create constant temp integrator
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        SystemArgs.frictioncoeff,
        SystemArgs.stepsize
    )
    # Create simulation and set initial positions
    simulation = app.Simulation(
        SystemObjs.pdb.topology,
        SystemObjs.system,
        integrator,
        openmm.Platform.getPlatformByName("CUDA"),
        properties
    )

    simulation.context.setPositions(SystemObjs.pdb.positions)
    if SystemArgs.resume:
        with open(os.path.join(output_dir, CHECKPOINT_FN), "rb") as f:
            simulation.context.loadCheckpoint(f.read())
            print("Loaded checkpoint")

    return SimulationProps(integrator, simulation, properties)









