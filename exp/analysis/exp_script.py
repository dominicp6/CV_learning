import subprocess

SCRIPT_DIR = '..'
DATA_DIR = '../data/chignolin_open'
DURATION = '20ns'
SAVE_FRQ = '1ps'
STEP_SIZE = '2fs'
FRIC_COEFF = '1ps'
PRECISION = 'mixed'
WATER = 'tip3p'
TEMP = '300K'


for i in range(20):
    subprocess.call(
        f"python {SCRIPT_DIR}/run_openmm.py {DATA_DIR}/structure{i}.pdb amber {PRECISION} -d {DURATION} -c {FRIC_COEFF} -f {SAVE_FRQ} -s {STEP_SIZE} -w {WATER} -t {TEMP} -m",
        shell=True,
    )