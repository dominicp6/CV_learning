from utils.trajectory_utils import clean_and_align_trajectory

if __name__ == "__main__":
    working_dirs = ['50ns_NPT_800K_chignolin_1uao'] #production_chignolin_1uao-big-unit-cell_800K_amber_104833_181122', 'production_chignolin_1uao-big-unit-cell_800K_cutoffperiodic_amber_111956_181122', 'production_chignolin_1uao-big-unit-cell_800K_cutoffperiodic_nosolventpadding_amber_114255_181122', 'production_chignolin_1uao-big-unit-cell_1100K_amber_111527_181122']
    for subdir in working_dirs:
        working_dir = f"/home/dominic/PycharmProjects/CV_learning/exp/outputs/chignolin/{subdir}"
        top = "../../../data/chignolin_1uao-big-unit-cell.pdb"
        traj = "trajectory.dcd"
        clean_and_align_trajectory(working_dir, traj_name=traj, top_name=top, save_name="trajectory_processed")

