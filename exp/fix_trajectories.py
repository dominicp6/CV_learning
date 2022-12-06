from utils.trajectory_utils import clean_and_align_trajectory

if __name__ == "__main__":
    working_dirs = ['5us_NPT_deca_alanine'] #production_chignolin_1uao-big-unit-cell_800K_amber_104833_181122', 'production_chignolin_1uao-big-unit-cell_800K_cutoffperiodic_amber_111956_181122', 'production_chignolin_1uao-big-unit-cell_800K_cutoffperiodic_nosolventpadding_amber_114255_181122', 'production_chignolin_1uao-big-unit-cell_1100K_amber_111527_181122']
    for subdir in working_dirs:
        working_dir = f"/home/dominic/PycharmProjects/CV_learning/exp/outputs/{subdir}"#deca_alanine/{subdir}"
        top = "top_1ps.pdb"
        traj = "trajectory.dcd"
        clean_and_align_trajectory(working_dir, traj_name=traj, top_name=top, save_name="trajectory_processed", stride=1, iterload=False, remove_water=True)

