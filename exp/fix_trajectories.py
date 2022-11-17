from ..utils.trajectory_utils import clean_and_align_trajctory

if __name__ == "__main__":
    working_dir = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/chignolin/production_chignolin_1uao-processed_amber_154659_161122"
    top = "topology.pdb"
    traj = "trajectory.dcd"
    clean_and_align_trajectory(working_dir, traj, top, save_name="trajectory_processed")

