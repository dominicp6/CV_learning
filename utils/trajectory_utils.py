import os
from typing import Optional

import mdtraj as md


def merge_dcd_trajectories_in_dir(
    working_dir: str,
    top_name: str,
    save_name: str,
):
    top_path = os.path.join(working_dir, top_name)

    traj_names = get_filenames_with_suffix(working_dir, suffix='.dcd')
    print("The trajectories are being merged in alphabetical order:")
    print(traj_names)

    trajs = [md.load_dcd(os.path.join(working_dir, traj_name), top=top_path) for traj_name in traj_names]
    merged_traj = md.join(trajs)
    save_traj(merged_traj, working_dir, save_name)


def get_filenames_with_suffix(directory, suffix):
    filenames = os.listdir(directory)
    filenames_with_suffix = [filename for filename in filenames if filename.endswith(suffix)]
    sorted_filenames = sorted(filenames_with_suffix)

    return sorted_filenames


def clean_and_align_trajectory(
    working_dir: str,
    top_name: str,
    traj_name: str,
    save_name: Optional[str] = None,
    remove_water: bool = True,
    align_protein: bool = True,
    centre: bool = True,
    stride: int = 1,
    iterload: bool = False,
    chunk: int = 1000,
) -> md.Trajectory:

    traj_path = os.path.join(working_dir, traj_name)
    top_path = os.path.join(working_dir, top_name)

    if iterload:
        # Load the trajectory in memory chunks
        traj = md.iterload(traj_path, top=top_path, stride=stride, chunk=chunk)
        # Process each chunk separately
        trajectory_chunks = [
            process_trajectory(chunk, remove_water, align_protein, centre)
            for chunk in traj
        ]
        # Merge chunks into a contiguous, processed trajectory
        traj = md.join(trajectory_chunks)
    else:
        # Load the whole trajectory into memory
        traj = md.load_dcd(traj_path, top=top_path, stride=stride)
        # Process the whole trajectory simultaneously
        traj = process_trajectory(traj, remove_water, align_protein, centre)

    # Save the trajectory to file
    traj = save_traj(traj, working_dir, save_name)

    return traj


def process_trajectory(traj: md.Trajectory, remove_water: bool, align_protein: bool, centre: bool) -> md.Trajectory:
    if remove_water:
        traj = do_remove_water(traj)

    if align_protein:
        traj = do_align_protein(traj)

    if centre:
        traj = do_centre_coordinates(traj)

    return traj


def do_remove_water(traj: md.Trajectory) -> md.Trajectory:
    # Remove waters, ions, and lipid.
    print("Removing water")
    traj = traj.atom_slice(traj.top.select("not resname HOH POPC CL NA"))

    return traj


def do_align_protein(traj: md.Trajectory) -> md.Trajectory:
    # Centre and align protein; useful for removing artifacts from PBCs
    print("Aligning protein")
    prot = traj.top.select("protein")
    traj.superpose(traj, atom_indices=prot)

    return traj


def do_centre_coordinates(traj: md.Trajectory) -> md.Trajectory:
    print("Centering")
    traj.center_coordinates()

    return traj


def save_traj(traj: md.Trajectory, working_dir: str, save_name: Optional[str]) -> md.Trajectory:
    if save_name:
        print("Saving modified trajectory")
        traj.save(os.path.join(working_dir, f"{save_name}.dcd"))
        print("Saving modified PDB file")
        pdb = traj.slice(0)
        pdb.save_pdb(os.path.join(working_dir, f"{save_name}.pdb"))

    return traj
