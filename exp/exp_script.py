import subprocess

SCRIPT_DIR = '../..'
DATA_DIR = '../data'
DURATION = '2us'
SAVE_FRQ = '2ps'
STEP_SIZE = '2fs'
FRIC_COEFF = '1ps'
PRECISION = 'mixed'
WATER = 'tip3p'
TEMP = '300K'


# from exp.openmm_utils import fix_pdb
#
# fix_pdb(f"{DATA_DIR}/deca-ala.pdb")
# for i in range(20):
#     if i == 0:
subprocess.call(
    f"python {SCRIPT_DIR}/run_openmm.py {DATA_DIR}/alanine.pdb amber {PRECISION} -d {DURATION} -c {FRIC_COEFF} -f {SAVE_FRQ} -s {STEP_SIZE} -w {WATER} -t {TEMP} -m",
    shell=True,
)
