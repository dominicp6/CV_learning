import shutil
import os
import subprocess
import math
import openmm.unit as unit
import concurrent.futures
import numpy as np
from collections import namedtuple
from utils.plotting_functions import init_plot
import matplotlib.pyplot as plt
from utils.general_utils import remove_nans
from scipy.interpolate import SmoothBivariateSpline, griddata

from utils.plotting_functions import init_multiplot

BiasTrajectory = namedtuple("BiasTrajectory", "feat1 feat2 free_energy")

def get_fe_trajs(data, reweight=False, file_type="fes"):
    """
    Get feature and free energy trajectories from the data array.
    """
    # Different column orders for reweighted and non-reweighted data
    if reweight:
        delta_idx = 1
    else:
        delta_idx = 0
    feature1_traj = data[:, 0+delta_idx]
    feature2_traj = data[:, 1+delta_idx]
    if file_type is "fes":
        fe = data[:, 2+delta_idx]
        fe = fe - np.min(fe)
    elif file_type is "COLVAR":
        fe = data[:, 2+delta_idx]
        fe = np.max(fe) - fe

    return feature1_traj, feature2_traj, fe

def read_HILLS(path_to_HILLS: str):
    """
    Read HILLS file and return a numpy array of the data and a list of the row strings
    """
    HILLS_arr = []
    HILLS_strs = []
    HILLS_header = []
    with open(path_to_HILLS, "r") as f:
        for idx, line in enumerate(f.readlines()):
            if "#" in line:
                HILLS_header.append(line)
            else:
                entries = [float(segment.strip()) for segment in line.split()]
                HILLS_arr.append(entries)
                HILLS_strs.append(line)

    return np.array(HILLS_arr), HILLS_strs, HILLS_header


def construct_partial_HILLS(directory: str, number_of_files: int):
    """
    Construct partial HILLS files containing various fractions of the total HILLS data
    """
    path_to_HILLS = f"{directory}/HILLS"
    _, HILLS_strs, HILLS_header = read_HILLS(path_to_HILLS)
    lines_in_arr = len(HILLS_strs)
    fractions = np.linspace(0, 1, number_of_files + 1)

    if os.path.exists(os.path.join(directory, "partial")):
        # remove all files in the partial directory
        shutil.rmtree(os.path.join(directory, "partial"))
    
    os.mkdir(os.path.join(directory, "partial"))

    for idx, fraction in enumerate(fractions):
        if idx == 0:
            continue
        lines_to_keep = int(fraction * lines_in_arr)
        trimmed_HILLS_strs = HILLS_strs[:lines_to_keep]

        new_HILLS = os.path.join(directory, "partial", f"HILLS_{idx}")
        with open(new_HILLS, "w") as f:
            # re-write header
            for line in HILLS_header:
                f.write(line)
            # write trimmed HILLS
            for line in trimmed_HILLS_strs:
                f.write(line)

    print(f"Created {number_of_files} partial HILLS files in {os.path.join(directory, 'partial')}")


def create_partial_fes_files(directory: str):
    """
    Create partial FES files from partial HILLS files by calling plumed sum_hills
    """
    files = [os.path.join(directory, "partial", file) for file in os.listdir(os.path.join(directory, "partial")) if file.startswith("HILLS")]
    number_of_files = len(files)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for file in files:
            index = os.path.basename(file).split("_")[1]
            fes_file = os.path.join(directory, "partial", f"FES_{index}")
            future = executor.submit(subprocess.call, f"plumed sum_hills --hills {file} --outfile {fes_file}", shell=True, stdout=subprocess.DEVNULL)
            futures.append(future)

    completed_futures = concurrent.futures.as_completed(futures)

    print(f"Created {number_of_files} FES files in {os.path.join(directory, 'partial')}")


def compute_fes_evolution(directory: str):
    """
    Compute FES bezier surface snapshots and trajectories from partial FES files
    """
    fes_files = []
    for file in os.listdir(os.path.join(directory, "partial")):
        if file.startswith("FES"):
            fes_files.append(os.path.join(directory, "partial", file))

    fes_files.sort()
    fes_surfaces = []
    fe_trajs = []
    for fes_file in fes_files:
        fes_arr = np.loadtxt(fes_file)
        feat1, feat2, fe = get_fe_trajs(fes_arr)
        traj = BiasTrajectory(feat1, feat2, fe)
        xyz = remove_nans(np.column_stack((feat1, feat2, fe)))
        x = xyz[:, 0]
        y = xyz[:, 1]
        free_energy = xyz[:, 2] 
        # create a 2D spline interpolation of the free energy surface
        bz = SmoothBivariateSpline(x, y, free_energy, kx=3, ky=3)
        fes_surfaces.append(bz)
        fe_trajs.append(traj)

    return fes_surfaces, fe_trajs

def slice_fes(beta: unit.Quantity, ax, bias_traj: BiasTrajectory, dim=0, label=None, color=None):
    """
    Plots a slice of the free energy surface in the specified dimension.
    """
    xyz = remove_nans(np.column_stack((bias_traj.feat1, bias_traj.feat2, bias_traj.free_energy)))
    x = xyz[:, 0]
    y = xyz[:, 1]
    free_energy = xyz[:, 2] 
    X,Y = np.meshgrid(np.linspace(x.min(),x.max(),100),np.linspace(y.min(),y.max(),100))
    Z = griddata((x,y),free_energy,(X,Y),method='cubic')

    if dim == 0:
        z = - beta * np.log(np.sum(np.exp(-beta*Z), axis=0)/np.sum(np.exp(-beta*Z)))
        im = ax.plot(Y[:,0], z, label=label, c=color)
    elif dim == 1:
        z = - beta * np.log(np.sum(np.exp(-beta*Z), axis=1)/np.sum(np.exp(-beta*Z)))
        im = ax.plot(X[0,:], z, label=label, c=color)
    else:
        raise ValueError("dim must be either 0 or 1")

    return ax, im


def plot_fes_slice_evolution(beta, fe_trajs):
    """
    Plot the evolution of slices of the FES surface
    """
    fig, panel_axes = init_multiplot(nrows=1, ncols=2, title="Free Energy Convergence", panels=['0,0', '0,1'])
    colors = plt.cm.get_cmap('Blues', len(fe_trajs))
    for i, traj in enumerate(fe_trajs):
        color = colors(i)
        #if i == 0:
        #    label = stride * self.savefreq
        #elif i == len(directory) - 1:
        #    label = stride * self.savefreq * len(directory)
        #else:
        #    label = None
        ax1, im1 = slice_fes(beta, panel_axes[0], traj, dim=0, color=color) #label=label)
        ax2, im2 = slice_fes(beta, panel_axes[1], traj, dim=1, color=color) #label=label)
            
    ax1.set_xlabel('CV 1')
    ax2.set_xlabel('CV 2')
    ax1.legend()
    ax2.legend()


def plot_fes_evolution(fes_surfaces, fig_size=(20, 20), xmin=-np.pi, xmax=np.pi, ymin=-np.pi, ymax=np.pi):
    """
    Plot the free energy surface snapshots 
    """
    number_of_plots = len(fes_surfaces)
    num_rows = math.ceil(number_of_plots / 5) 
    fig, axs = plt.subplots(num_rows, 5, figsize=fig_size)

    for idx, fes_surface in enumerate(fes_surfaces):
        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)
        X, Y = np.meshgrid(x, y)
        Z = fes_surface.ev(X.flatten(), Y.flatten())
        Z = Z.reshape(X.shape)
        idx1 = idx // 5
        idx2 = idx % 5
        axs[idx1][idx2].imshow(Z, extent=[x.min(),x.max(),y.min(),y.max()], origin='lower', cmap='RdBu_r')
        axs[idx1][idx2].set_title(f"Time {idx}")
        axs[idx1][idx2].axis('off')

    plt.tight_layout()


def plot_fe_point_evolution(fes_surfaces, points: np.array):
    """
    Plot the free energy evolution of a point in CV space as a function of simulation time
    """
    fig, ax = init_plot(title="Free energy evolution", xlabel="Time", ylabel="Free energy")
    for point in points:
        fe = []
        for fes_surface in fes_surfaces:
            fe.append(fes_surface.ev(point[0], point[1]))
        ax.plot(fe, label=f"Point {point}")
        ax.set_xscale("log")