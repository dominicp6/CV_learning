import subprocess

SCRIPT_DIR = '../..'
DATA_DIR = '../chemicals/chignolin'
DURATION = '2ns'
SAVE_FRQ = '10ps'
STEP_SIZE = '2fs'
FRIC_COEFF = '1ps'
PRECISION = 'mixed'
WATER = 'tip3p'
TEMP = '300K'
PRESSURE = '1.0bar'
CUTOFF_DIST = '1.0nm'
SOLV_PADDING = '1nm'
CUTOFF_METHOD = 'CutoffPeriodic'
FORCE_FIELD = "charmm36"
SEED = "0"
NAME = "chignolin_equilibration_test"
DIR = None
periodic = True
EQUIL = "NPT"


# TODO: Saving water in trajectory
# TODO: Check water model correctly saved to metadata?

pr = '-pr' if periodic else ''
subprocess.call(
    f"python {SCRIPT_DIR}/run_openmm.py {DATA_DIR}/minimised.pdb {FORCE_FIELD} {PRECISION} -d {DURATION} "
    f"-c {FRIC_COEFF} -f {SAVE_FRQ} -s {STEP_SIZE} -t {TEMP} -p {PRESSURE} -sp {SOLV_PADDING} -nbc {CUTOFF_DIST} "
    f"-cm {CUTOFF_METHOD} {pr} -m -w {WATER} -name {NAME} -equilibrate {EQUIL}",
    shell=True,
)


