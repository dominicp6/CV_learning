import subprocess

SCRIPT_DIR = '../..'
DATA_DIR = '../data/deca-alanine'
DURATION = '0.62us'
SAVE_FRQ = '400ps'
STEP_SIZE = '2fs'
FRIC_COEFF = '1ps'
PRECISION = 'mixed'
WATER = 'tip3p'
TEMP = '300K'
PRESSURE = '1bar'
CUTOFF_DIST = '1nm'
SOLV_PADDING = '1nm'
CUTOFF_METHOD = 'CutoffPeriodic'
periodic = True


# TODO: Saving water in trajectory
# TODO: Check water model correctly saved to metadata?
# TODO: Fix progress bar when resuming a previous experiment

# from exp.openmm_utils import fix_pdb
#
# fix_pdb(f"{DATA_DIR}/deca-ala.pdb")
# for i in range(20):
#     if i == 0:
pr = '-pr' if periodic else ''
subprocess.call(
    f"python {SCRIPT_DIR}/run_openmm.py {DATA_DIR}/deca-alanine-processed.pdb amber -r /home/dominic/PycharmProjects/CV_learning/exp/outputs/production_deca-alanine-processed_amber_114508_261122 {PRECISION} -d {DURATION} -c {FRIC_COEFF} -f {SAVE_FRQ} -s {STEP_SIZE} -t {TEMP} -p {PRESSURE} -sp {SOLV_PADDING} -nbc {CUTOFF_DIST} -cm {CUTOFF_METHOD} {pr} -w {WATER} -m",
    shell=True,
)

# # Big and how
# TEMP = '1100K'
# subprocess.call(
#     f"python {SCRIPT_DIR}/run_openmm.py {DATA_DIR}/chignolin_1uao-big-unit-cell.pdb amber {PRECISION} -d {DURATION} -c {FRIC_COEFF} -f {SAVE_FRQ} -s {STEP_SIZE} -w {WATER} -t {TEMP} -p {PRESSURE} -sp {SOLV_PADDING} -nbc {CUTOFF_DIST} -cm {CUTOFF_METHOD} {pr} -m",
#     shell=True,
# )
#
# # Big and different cutoff
# TEMP = '800K'
# CUTOFF_METHOD = 'CutoffPeriodic'
# subprocess.call(
#     f"python {SCRIPT_DIR}/run_openmm.py {DATA_DIR}/chignolin_1uao-big-unit-cell.pdb amber {PRECISION} -d {DURATION} -c {FRIC_COEFF} -f {SAVE_FRQ} -s {STEP_SIZE} -w {WATER} -t {TEMP} -p {PRESSURE} -sp {SOLV_PADDING} -nbc {CUTOFF_DIST} -cm {CUTOFF_METHOD} {pr} -m",
#     shell=True,
# )
#
# CUTOFF_METHOD = 'CutoffPeriodic'
# SOLV_PADDING = '0nm'
# subprocess.call(
#     f"python {SCRIPT_DIR}/run_openmm.py {DATA_DIR}/chignolin_1uao-big-unit-cell.pdb amber {PRECISION} -d {DURATION} -c {FRIC_COEFF} -f {SAVE_FRQ} -s {STEP_SIZE} -w {WATER} -t {TEMP} -p {PRESSURE} -sp {SOLV_PADDING} -nbc {CUTOFF_DIST} -cm {CUTOFF_METHOD} {pr} -m",
#     shell=True,
# )
