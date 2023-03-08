import argparse
import os
from collections import namedtuple
from typing import Union

import openmm
import openmm.app as app
import openmm.unit as unit
from openmmtools import integrators
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import seaborn as sns



SystemArgs = namedtuple(
    "System_Args",
    "forcefield precision pdb mol2 sdf xml ml_residues resume plumed duration savefreq stepsize temperature pressure "
    "frictioncoeff num_water ionic_strength solventpadding nonbondedcutoff cutoffmethod total_steps steps_per_save "
    "periodic gpu minimise watermodel seed name directory equilibrate equilibration_length integrator state_data",
)

SystemObjs = namedtuple("System_Objs", "model modeller system")

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
    "mol": unit.molar,
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


def isnumber(s: str):
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_quantity(s: Union[int, float, str, None]):
    if isinstance(s, (int, float)):
        # Already a number
        return s
    elif s is None:
        # Empty quantity 
        return None
    elif isnumber(s):
        # Pure numeric
        return float(s)
    try:
        # Unit conversion
        u = s.lstrip("0123456789.")
        v = s[: -len(u)]
        return unit.Quantity(float(v), unit_labels[u])
    except Exception:
        # Return original string
        return s


def round_format_quantity(quantity: unit.Quantity, significant_figures: int):
    return f'{round(quantity._value, significant_figures)} {quantity.unit.get_symbol()}'


def time_to_iteration_conversion(time: str, duration: unit.Unit, num_frames: int):
    time = parse_quantity(time)
    iteration = int(time / duration * num_frames)
    return iteration


def get_flag(parser: argparse.ArgumentParser, argument: str):
    return parser._option_string_actions[argument].option_strings[0]


def get_unit_cell_dims(modeller: app.Modeller):
    unit_cell_dims = modeller.getTopology().getUnitCellDimensions()
    print(f"Unit cell dimensions: {unit_cell_dims}")

    return unit_cell_dims


def add_barostat(pressure, temperature, system, barostat_type="MonteCarloBarostat"):
    if barostat_type == "MonteCarloBarostat":
        # default frequency for pressure changes is 25 time steps
        barostat = openmm.MonteCarloBarostat(pressure, temperature, 25)
    else:
        raise NotImplementedError(f"Barostat type {barostat_type} not implemented.")

    system.addForce(barostat)


def get_integrator(args: Union[SystemArgs, dict], integrator_type="LangevinBAOAB"):
    if integrator_type == "LangevinBAOAB":
        integrator = integrators.LangevinIntegrator(
            args.temperature,
            args.frictioncoeff,
            args.stepsize,
        )
    elif integrator_type == "LangevinMiddle":
        integrator = openmm.LangevinMiddleIntegrator(
            args.temperature,
            args.frictioncoeff,
            args.stepsize,
        )
    elif integrator_type == "Verlet":
        integrator = openmm.VerletIntegrator(
            args.stepsize,
        )
    else:
        raise NotImplementedError(f"Integrator type {integrator_type} not implemented.")

    return integrator


def make_graphs(report, stepsize, output_dir, name='graphs'):
    report = report.melt()
    with sns.plotting_context("paper"):
        g = sns.FacetGrid(data=report, row="variable", sharey=False)
        g.map(plt.plot, "value")
        # format the labels with f-strings
        for ax in g.axes.flat:
            ax.xaxis.set_major_formatter(
                tkr.FuncFormatter(
                    lambda x, p: f"{(x * stepsize).value_in_unit(unit.nanoseconds):.1f}ns"
                )
            )
        plt.savefig(
            os.path.join(output_dir, f"{name}.png"), bbox_inches="tight"
        )


def get_system_args(args):
    """
    Parse command line arguments.

    :return: Dictionary of arguments
    """
    duration = parse_quantity(args.duration)
    savefreq = parse_quantity(args.savefreq)
    stepsize = parse_quantity(args.stepsize)
    frictioncoeff = parse_quantity(args.frictioncoeff)
    frictioncoeff = frictioncoeff._value / frictioncoeff.unit
    cutoffmethod = args.cutoffmethod
    total_steps = int(duration / stepsize)
    steps_per_save = int(savefreq / stepsize)
    periodic = args.periodic

    if not periodic:
        assert cutoffmethod in [
            "NoCutoff",
            "CutoffNonPeriodic",
        ], f"You have specified a non-periodic simulation but have given an incompatible cutoff method " \
           f"({cutoffmethod}). Please change the cutoff method to either 'NoCutoff' or 'CutoffNonPeriodic'."

    systemargs = SystemArgs(
        args.forcefield.lower(),
        args.precision.lower(),
        args.pdb,
        args.mol2,
        args.sdf,
        args.xml,
        args.ml_residues.split(",") if args.ml_residues else None,
        args.resume,
        args.plumed,
        duration,
        savefreq,
        stepsize,
        parse_quantity(args.temperature),
        parse_quantity(args.pressure) if args.pressure else None,
        frictioncoeff,
        int(args.num_water) if args.num_water else None,
        parse_quantity(args.ionic_strength) if args.ionic_strength else None,
        parse_quantity(args.solventpadding) if args.solventpadding else None,
        parse_quantity(args.nonbondedcutoff),
        cutoffmethod,
        total_steps,
        steps_per_save,
        periodic,
        args.gpu,
        args.minimise,
        args.watermodel,
        int(args.seed) if args.seed else None,
        args.name,
        args.directory,
        args.equilibrate,
        args.equilibration_length,
        args.integrator,
        args.state_data
    )

    return systemargs




