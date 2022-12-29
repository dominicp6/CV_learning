import os

from pymol import cmd

if __name__ == "__main__":
    filename = "CLN025-0-protein.mae"
    directory = "/home/dominic/PycharmProjects/CV_learning/exp/outputs/chignolin/desres/DESRES-Trajectory_CLN025-0-protein/CLN025-0-protein"
    cmd.load(os.path.join(directory, filename))
    cmd.save(os.path.join(directory,filename[:-4] + '.pdb'))
