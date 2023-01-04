import subprocess

SCRIPT_DIR = '../..'
DATA_DIR = '../data/alanine'
DURATION = '50ns'
SAVE_FRQ = '10ps'
STEP_SIZE = '2fs'
FRIC_COEFF = '1ps'
PRECISION = 'mixed'
WATER = 'tip3p'
TEMP = '800K'
PRESSURE = '1.0bar'
CUTOFF_DIST = '0.80nm'
SOLV_PADDING = '1nm'
CUTOFF_METHOD = 'CutoffPeriodic'
FORCE_FIELD = "amber"
SEED = "0"
NAME = "50ns_NPT_800K_alanine"
DIR = None
periodic = True



# TODO: Saving water in trajectory
# TODO: Check water model correctly saved to metadata?

pr = '-pr' if periodic else ''
subprocess.call(
    f"python {SCRIPT_DIR}/run_openmm.py {DATA_DIR}/alanine-processed.pdb {FORCE_FIELD} {PRECISION} -d {DURATION} "
    f"-c {FRIC_COEFF} -f {SAVE_FRQ} -s {STEP_SIZE} -t {TEMP} -sp {SOLV_PADDING} -nbc {CUTOFF_DIST} "
    f"-cm {CUTOFF_METHOD} {pr} -m -w {WATER} -name {NAME}",
    shell=True,
)


