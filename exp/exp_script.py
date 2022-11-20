import subprocess

SCRIPT_DIR = '../..'
DATA_DIR = '../data'
DURATION = '5us'
SAVE_FRQ = '4ps'
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


# from exp.openmm_utils import fix_pdb
#
# fix_pdb(f"{DATA_DIR}/deca-ala.pdb")
# for i in range(20):
#     if i == 0:
pr = '-pr' if periodic else ''
subprocess.call(
    f"python {SCRIPT_DIR}/run_openmm.py {DATA_DIR}/chignolin_1uao-processed.pdb amber {PRECISION} -d {DURATION} -c {FRIC_COEFF} -f {SAVE_FRQ} -s {STEP_SIZE} -w {WATER} -t {TEMP} -p {PRESSURE} -sp {SOLV_PADDING} -nbc {CUTOFF_DIST} -cm {CUTOFF_METHOD} {pr} -m",
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
