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
        print('Saving modified PDB file')
        pdb = mtraj.slice(0)
        pdb.save_pdb(os.path.join(working_dir, f'{save_name}.pdb'))
