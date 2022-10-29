#!/usr/bin/env python3
import openmm
import openmm.app as app
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import datetime
import os
import sys

from openmm_utils import parse_quantity, valid_ffs



##############################################
#   DATA REPORTERS
##############################################

# Reporter to print info to stdout
simulation.reporters.append(app.StateDataReporter(
    sys.stdout,
    steps_per_save,
    progress=True,  # Info to print. Add anything you want here.
    remainingTime=True,
    speed=True,
    totalSteps=total_steps,
))
# Reporter to log lots of info to csv
simulation.reporters.append(app.StateDataReporter(
    os.path.join(output_dir, STATE_DATA_FN),
    steps_per_save,
    step=True,
    time=True,
    speed=True,
    temperature=True,
    potentialEnergy=True,
    kineticEnergy=True,
    totalEnergy=True,
    append=True if resume else False
))
# Reporter to save trajectory
# Save only a subset of atoms to the trajectory, ignore water
simulation.reporters.append(app.DCDReporter(
    os.path.join(output_dir, TRAJECTORY_FN),
    reportInterval=steps_per_save,
    append=True if resume else False))

# Reporter to save regular checkpoints
simulation.reporters.append(app.CheckpointReporter(
    os.path.join(output_dir, CHECKPOINT_FN),
    steps_per_save
))

##############################################
#   PRODUCTION RUN
##############################################

print("Running production...")
simulation.step(total_steps)
print("Done")

# Save final checkpoint and state
simulation.saveCheckpoint(os.path.join(output_dir, CHECKPOINT_FN))
simulation.saveState(os.path.join(output_dir, 'end_state.xml'))

# Make some graphs
report = pd.read_csv(os.path.join(output_dir, STATE_DATA_FN))
report = report.melt()

with sns.plotting_context('paper'):
    g = sns.FacetGrid(data=report, row='variable', sharey=False)
    g.map(plt.plot, 'value')
    # format the labels with f-strings
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(
            tkr.FuncFormatter(lambda x, p: f'{(x * stepsize).value_in_unit(unit.nanoseconds):.1f}ns'))
    plt.savefig(os.path.join(output_dir, 'graphs.png'), bbox_inches='tight')

# print a trajectory of the aaa dihedrals, counting the flips
# heatmap of phi and psi would be a good first analysis, use mdanalysis
# aiming for https://docs.mdanalysis.org/1.1.0/documentation_pages/analysis/dihedrals.html
# number of events going between minima states
# "timetrace" - a plot of the dihedral over time (aim for 500ns)
# do this first, shows how often you go back and forth. one plot for each phi/psi angle
# four plots - for each set of pairs
# this gives two heatmap plots like in the documentation