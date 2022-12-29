import subprocess

SCRIPT_DIR = '../..'
DATA_DIR = '../data/chignolin'
DURATION = '0.2222us'
SAVE_FRQ = '200ps'
STEP_SIZE = '2fs'
FRIC_COEFF = '1ps'
PRECISION = 'mixed'
WATER = 'tip3p'
TEMP = '340K'
PRESSURE = ''
CUTOFF_DIST = '1nm'
SOLV_PADDING = '1nm'
CUTOFF_METHOD = 'CutoffPeriodic'
FORCE_FIELD = "charmm"
periodic = True


# TODO: Saving water in trajectory
# TODO: Check water model correctly saved to metadata?

pr = '-pr' if periodic else ''
subprocess.call(
    f"python {SCRIPT_DIR}/run_openmm.py {DATA_DIR}/minimised.pdb -r /home/dominic/PycharmProjects/CV_learning/exp/exp/outputs/production_chignolin_desres_charmm_162251_161222 {FORCE_FIELD} {PRECISION} -d {DURATION} -c {FRIC_COEFF} -f {SAVE_FRQ} -s {STEP_SIZE} -t {TEMP} -sp {SOLV_PADDING} -nbc {CUTOFF_DIST} -cm {CUTOFF_METHOD} {pr} -w {WATER}",
    shell=True,
)


