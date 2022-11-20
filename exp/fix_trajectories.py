import os
from typing import Optional

import mdtraj as md


def clean_and_align_trajectory(working_dir: str, top_name: str, traj_name: str, save_name: Optional[str] = None, remove_water: bool = True, align_protein: bool = True, centre: bool = True, stride: int = 1):
    mtraj = md.load_dcd(os.path.join(working_dir, traj_name), top=os.path.join(working_dir, top_name), stride=stride)

    if remove_water:
        ## Remove waters, ions, and lipid.
        print('Removing water')
        mtraj = mtraj.atom_slice(mtraj.top.select('not resname HOH POPC CL NA'))

    if align_protein:
        ## Centre and align protein; useful for removing artifacts from PBCs
        print('Aligning protein')
        prot = mtraj.top.select('protein')
        mtraj.superpose(mtraj, atom_indices=prot)

    if centre:
        print('Centering')
        mtraj.center_coordinates()

    if save_name:
        print('Saving modified trajectory')
        mtraj.save(os.path.join(working_dir, f'{save_name}.dcd'))


if __name__ == "__main__":
    working_dirs = ['production_chignolin_1uao-processed_amber_154659_161122'] #production_chignolin_1uao-big-unit-cell_800K_amber_104833_181122', 'production_chignolin_1uao-big-unit-cell_800K_cutoffperiodic_amber_111956_181122', 'production_chignolin_1uao-big-unit-cell_800K_cutoffperiodic_nosolventpadding_amber_114255_181122', 'production_chignolin_1uao-big-unit-cell_1100K_amber_111527_181122']
    for subdir in working_dirs:
        working_dir = f"/home/dominic/PycharmProjects/CV_learning/exp/outputs/chignolin/{subdir}"
        top = "../../../data/chignolin_1uao-big-unit-cell.pdb"
        traj = "trajectory.dcd"
        clean_and_align_trajectory(working_dir, traj_name=traj, top_name=top, save_name="trajectory_processed")

