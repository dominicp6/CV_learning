#!/usr/bin/env python

"""Class to systematically run OpenMM metadynamics enhanced sampling experiments using PLUMED script integration.
   Allows for computation of point-wise free energy convergence.

   Author: Dominic Phillips (dominicp6)
"""
import contextlib
import os
import subprocess
import time as time
import json
from typing import Optional, Union
from datetime import timedelta

import numpy as np

from Experiment import Experiment
from OpenMMSimulation import PDBSimulation
from utils.general_utils import count_files, list_files
from utils.feature_utils import get_cv_type_and_dim, get_features_and_coefficients


class EnhancedSamplingExperiments:
    def __init__(
            self,
            output_dir: str,
            unbiased_exp: str,
            CVs: list[str],
            starting_structures: str,
            number_of_repeats: int,
            openmm_parameters: dict,
            meta_d_parameters: dict,
            cos_sin: bool = False,
            features: Optional[Union[dict, list[str], np.array]] = None,
            num_cv_features: int = None,
            subtract_feature_means: bool = False,
            **kwargs,
    ):
        """

        :param output_dir: Directory to store all simulation outputs.
        :param unbiased_exp: Path to the unbiased experiment to use as a template for the metadynamics experiments
        (for loading topology files).
        :param CVs:  List of CVs to use in the metadynamics experiments. Format: [CV1, CV2, ...].
        :param starting_structures: Path to directory containing starting structures for the metadynamics experiments.
        :param number_of_repeats: Number of times to repeat each metadynamics experiment (for each starting structure).
        :param openmm_parameters: Dictionary of parameters to pass to the openmm simulation.
        :param meta_d_parameters: Dictionary of parameters to pass to the PLUMED metadynamics simulation.
        :param features: Features to use with the unbiased simulation before learning the CVs.
        :param cos_sin: Whether to use cos and sin features for the CVs.
        :param num_cv_features: Number of features to use for each CV.
        """

        self.initialised = False

        if features is None:
            print("Warning: No features specified. Using default cartesian features. (Not recommended)")
        self.exp = Experiment(location=unbiased_exp, features=features, cos_sin=cos_sin)
        self.number_of_repeats = number_of_repeats
        self.starting_structures = starting_structures
        self.num_starting_structures = count_files(self.starting_structures, 'pdb')
        self.num_total_features = len(self.exp.featurizer.describe())
        self.num_cv_features = num_cv_features
        self.subtract_feature_means = subtract_feature_means

        self.output_dir = output_dir
        self.CVs = CVs
        for cv in self.CVs:
            traditional_cv, cv_type, cv_dim = get_cv_type_and_dim(cv)
            if traditional_cv:
                if cv_type == "PCA":
                    if self.exp.CVs[cv_type] is None:
                        self.exp.compute_cv(cv_type, 3)
                elif cv_type == "TICA":
                    if self.exp.CVs[cv_type] is None:
                        self.exp.compute_cv(cv_type, 3, **kwargs)
                elif cv_type == "VAMP":
                    if self.exp.CVs[cv_type] is None:
                        self.exp.compute_cv(cv_type, 3, **kwargs)
                else:
                    raise ValueError(f"CV type {cv_type} not recognised.")

        os.chdir(self.output_dir)
        # OpenMMSimulation()._check_argument_dict(openmm_parameters)
        self.openmm_params = openmm_parameters
        self.openmm_params["directory"] = self.output_dir
        self.meta_d_params = meta_d_parameters

        self.eta = None
        self.exps_ran = 0
        self.cumulative_time = 0
        self.total_exps = self.num_starting_structures * number_of_repeats

        with open(os.path.join(self.output_dir, 'meta_d_params.json'), 'w') as f:
            json.dump(self.meta_d_params, f)

    def progress_logger(self, ti: float, tf: float):
        if self.exps_ran == 0:
            print("Running first experiment. ETA will be computed after completion.")
        else:
            self.cumulative_time += tf - ti
            exps_remaining = self.total_exps - self.exps_ran
            average_time_per_exp = self.cumulative_time / self.exps_ran
            eta = average_time_per_exp * exps_remaining
            eta_str = str(timedelta(seconds=eta))
            print(
                f"ETA: {eta_str} Completed {self.exps_ran}/{self.total_exps} experiments."
            )

    # 1)
    def initialise_hills_and_PLUMED(self):
        features, coefficients = get_features_and_coefficients(self.exp, self.CVs, self.num_cv_features)
        for pdb_file in list_files(self.starting_structures, 'pdb'):
            for repeat in range(self.number_of_repeats):
                exp_name = self.get_exp_name(self.CVs, pdb_file, repeat)
                os.mkdir(f"{self.output_dir}/{exp_name}")
                open(f"{self.output_dir}/{exp_name}/HILLS", "w")
                open(f"{self.output_dir}/{exp_name}/COLVAR", "w")
                self.exp.create_plumed_metadynamics_script(CVs=self.CVs,
                                                           features=features,
                                                           coefficients=coefficients,
                                                           filename=f"{self.output_dir}/{exp_name}/plumed.dat",
                                                           exp_name=exp_name,
                                                           gaussian_height=self.meta_d_params['gaussian_height'],
                                                           gaussian_pace=self.meta_d_params['gaussian_pace'],
                                                           well_tempered=self.meta_d_params['well_tempered'],
                                                           bias_factor=self.meta_d_params['bias_factor'],
                                                           temperature=self.meta_d_params['temperature'],
                                                           sigma_list=self.meta_d_params['sigma_list'],
                                                           normalised=self.meta_d_params['normalised'],
                                                           subtract_feature_means=self.subtract_feature_means,
                                                           print_to_terminal=False)
        self.initialised = True

    def run_openmm_experiments(self):
        ti = None
        tf = None
        assert self.initialised is True
        for pdb_file in list_files(self.starting_structures, 'pdb'):
            for repeat in range(self.number_of_repeats):
                self.progress_logger(ti, tf)
                ti = time.time()
                exp_name = self.get_exp_name(self.CVs, pdb_file, repeat)
                self.openmm_params["seed"] = repeat
                self.openmm_params["name"] = exp_name
                self.openmm_params["plumed"] = f"{self.output_dir}/{exp_name}/plumed.dat"
                simulation = PDBSimulation().from_args(self.openmm_params)
                with open(f"{self.output_dir}/{exp_name}/output.log", "w") as f:
                    with contextlib.redirect_stdout(f):
                        try:
                            simulation.run()
                        except Exception:
                            print(f"Error in experiment {exp_name}, terminated early.")
                tf = time.time()
                self.exps_ran += 1

    def get_exp_name(self, CVs: list[str], pdb_file: str, repeat: int):
        CVs = [cv.replace(" ", "_") for cv in CVs]
        return f"{'_'.join(CVs)}_{pdb_file[:-4]}_repeat_{repeat}_total_features_{self.num_total_features}_feature_dimensions_{self.num_cv_features}"

    # def read_HILLS_file(file, skipinitial=3):
    #     HILLS_arr = []
    #     HILLS_strs = []
    #     HILLS_header = []
    #     with open(file, "r") as f:
    #         for idx, line in enumerate(f.readlines()):
    #             if idx < skipinitial:
    #                 HILLS_header.append(line)
    #             else:
    #                 entries = [float(segment.strip()) for segment in line.split()]
    #                 HILLS_arr.append(entries)
    #                 HILLS_strs.append(line)
    #
    #     return np.array(HILLS_arr), HILLS_strs, HILLS_header

    #                    method              fraction            repeat           grid matrix
    # self.raw_grid_data = defaultdict(
    #    lambda: defaultdict(lambda: defaultdict(lambda: np.zeros(0)))
    # )
    #              method              critical point      fraction            repeat              value
    # self.fe_data = defaultdict(
    #    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    # )
    #                  method              critical point       [[mean, std], ...]
    # self.avg_fe_data = defaultdict(lambda: defaultdict(lambda: []))

    # self.conver_x_data = [frac * self.simulation_length for frac in self.fractions]

    # def set_points_on_convergence_plot(self, points_on_convergence_plot: int):
    #     self._points_on_convergence_plot = points_on_convergence_plot
    #     self.fractions = [
    #         round((idx + 1) / self._points_on_convergence_plot, 3)
    #         for idx in range(self._points_on_convergence_plot)
    #     ]

    # 3)
    # def run_analysis(self):
    #     print("Loading FES...")
    #     self.load_FESs()
    #     print("Extracting FE values...")
    #     self.extract_fe_vals()
    #     print("Summarising FE chemicals...")
    #     self.summarise_fe_data()
    #     print("Making convergence plots...")
    #     # self.make_convergence_plots()

    # a)
    # def load_FESs(self):
    #     with tqdm(
    #             total=len(self.methods * self.number_of_repeats * len(self.fractions))
    #     ) as pbar:
    #         assert self.simulations_complete is True
    #         for method in self.methods:
    #             for repeat in range(self.number_of_repeats):
    #                 for fraction in self.fractions:
    #                     self.load_exp_FES(method, repeat, fraction)
    #                     pbar.update(1)
    #         self.FES_constructed = True

    # def load_exp_FES(self, method: str, repeat: int, fraction: float):
    #     new_HILLS_file_name = self.construct_trimmed_HILLS_file(
    #         method, repeat, fraction
    #     )
    #     exp_name = self.run_reweighting(method, repeat, fraction)
    #     # os.remove(new_HILLS_file_name)
    #     self.raw_grid_data[method][fraction][repeat] = np.genfromtxt(
    #         f"{self.dir}/ff_{exp_name}_{fraction}.dat", delimiter=" ", autostrip=True
    #     )

    # @staticmethod
    # def read_HILLS_file(file: str, skipinitial: int = 3):
    #     HILLS_arr = []
    #     HILLS_strs = []
    #     HILLS_header = []
    #     with open(file, "r") as f:
    #         for idx, line in enumerate(f.readlines()):
    #             if idx < skipinitial:
    #                 HILLS_header.append(line)
    #             else:
    #                 entries = [float(segment.strip()) for segment in line.split()]
    #                 HILLS_arr.append(entries)
    #                 HILLS_strs.append(line)
    #
    #     return np.array(HILLS_arr), HILLS_strs, HILLS_header

    # def construct_trimmed_HILLS_file(self, method: str, repeat: int, fraction: float):
    #     HILLS_file = f"{self.dir}/HILLS_{method}{repeat}"
    #     HILLS_arr, HILLS_strs, HILLS_header = self.read_HILLS_file(
    #         HILLS_file, skipinitial=3
    #     )
    #     lines_in_arr = len(HILLS_strs)
    #     lines_to_keep = int(fraction * lines_in_arr)
    #     trimmed_HILLS_strs = HILLS_strs[:lines_to_keep]
    #
    #     new_HILLS_file = HILLS_file + "_" + str(fraction)
    #     with open(new_HILLS_file, "w") as f:
    #         for line in HILLS_header:
    #             f.write(line)
    #         for line in trimmed_HILLS_strs:
    #             f.write(line)
    #
    #     # filename
    #     return new_HILLS_file

    # def run_reweighting(self, method: str, repeat: int, fraction: float):
    #     exp_name = self.get_exp_name(method, repeat)
    #     with open(f"{self.dir}/plumed_reweight_{exp_name}_{fraction}.dat", "w") as f:
    #         PLUMED_reweight_string = self.REWEIGHT_PLUMED(method, repeat, fraction)
    #         f.write(PLUMED_reweight_string)
    #     subprocess.call(
    #         f"plumed driver --mf_xtc {self.dir}/{exp_name}/trajectory.xtc --plumed {self.dir}/plumed_reweight_{exp_name}_{fraction}.dat",
    #         shell=True,
    #     )
    #
    #     return exp_name

    # @staticmethod
    # def get_header(file: str, num_lines: int):
    #     with open(file, "r") as f:
    #         head = [next(f) for _ in range(num_lines)]
    #
    #     return head

    # b)
    # def extract_fe_vals(self):
    #     assert self.FES_constructed == True
    #     for method in self.methods:
    #         for repeat in range(self.number_of_repeats):
    #             for fraction in self.fractions:
    #                 phi, psi, fe = self.get_fe_grid_of_exp(method, repeat, fraction)
    #                 fe_grid = self.convert_xyz_to_numpy_grid(phi, psi, fe)
    #                 self.compute_critical_point_free_energys(
    #                     fe_grid, method, repeat, fraction
    #                 )
    #     self.fe_vals_computed = True

    # def get_fe_grid_of_exp(self, method: str, repeat: int, fraction: float):
    #     chemicals = self.raw_grid_data[method][fraction][repeat]
    #     phi = chemicals[:, 0]
    #     psi = chemicals[:, 1]
    #     fe = chemicals[:, 2] - np.min(chemicals[:, 2])
    #     return phi, psi, fe

    # def convert_xyz_to_numpy_grid(self, x: np.array, y: np.array, z: np.array) -> np.array:
    #     grid_x, grid_y = np.mgrid[
    #                      -np.pi: np.pi: complex(0, self.fe_grid_size),
    #                      -np.pi: np.pi: complex(0, self.fe_grid_size),
    #                      ]
    #     points = np.column_stack((x, y))
    #     return griddata(points, z, (grid_x, grid_y), method="linear")

    # c)
    # def summarise_fe_data(self):
    #     assert self.fe_vals_computed == True
    #     for method in self.methods:
    #         for critical_point in self.critical_point_labels:
    #             for fraction in self.fractions:
    #                 running_list = []
    #                 for repeat in range(self.number_of_repeats):
    #                     running_list.append(
    #                         self.fe_data[method][critical_point][fraction][repeat]
    #                     )
    #                 mean = np.mean(running_list)
    #                 std = np.std(running_list)
    #                 self.avg_fe_data[method][critical_point].append([mean, std])
    #
    #     self.fe_vals_summarised = True

    # def compute_critical_point_free_energys(self, fe_grid, method: str, repeat: int, fraction: float):
    #     for critical_point in self.critical_points:
    #         phi_crit = critical_point[1]
    #         psi_crit = critical_point[2]
    #         estimated_fe = self.grid_interpolate(fe_grid, phi_crit, psi_crit)
    #         self.save_fe_energy_estimate(
    #             method, repeat, fraction, critical_point[0], estimated_fe
    #         )

    # def grid_interpolate(self, grid, x: np.array, y: np.array):
    #     x_cell = ((x + np.pi) / (2 * np.pi)) * (self.fe_grid_size - 1)
    #     y_cell = ((y + np.pi) / (2 * np.pi)) * (self.fe_grid_size - 1)
    #
    #     lower_interpolant = (x_cell - int(np.floor(x_cell))) * grid[
    #         int(np.floor(x_cell)) + 1, int(np.floor(y_cell))
    #     ] + (1 - x_cell + int(np.floor(x_cell))) * grid[
    #                             int(np.floor(x_cell)), int(np.floor(y_cell))
    #                         ]
    #     upper_interpolant = (x_cell - int(np.floor(x_cell))) * grid[
    #         int(np.floor(x_cell)) + 1, int(np.floor(y_cell)) + 1
    #     ] + (1 - x_cell + int(np.floor(x_cell))) * grid[
    #                             int(np.floor(x_cell)), int(np.floor(y_cell)) + 1
    #                         ]
    #
    #     final_interpolant = (y_cell - int(np.floor(y_cell))) * upper_interpolant + (
    #             1 - y_cell + int(np.floor(y_cell))
    #     ) * lower_interpolant
    #
    #     return final_interpolant

    # def save_fe_energy_estimate(self, method: str, repeat: int, fraction: float, critical_point, value):
    #     self.fe_data[method][critical_point][fraction][repeat] = value

    # def make_convergence_plots(self):
    #     assert self.fe_vals_summarised == True
    #     for method in self.methods:
    #         self.convergence_plot(method)

    # def convergence_plot(self, method: str):
    #     for critical_point in self.critical_point_labels:
    #         plt.plot(
    #             [val.value_in_unit(unit.nanoseconds) for val in self.conver_x_data],
    #             [val[0] for val in self.avg_fe_data[method][critical_point]],
    #             label=critical_point,
    #         )
    #
    #     # TODO: fill between
    #     plt.xlabel("Elapsed Time (ns)", fontsize=18)
    #     plt.ylabel(r"$\log\left(F-\hat{F}\right)$", fontsize=18)
    #     plt.yscale("log")
    #     plt.legend()
    #     plt.show()
    #     # TODO: save plot

#     def PLUMED(self, method: str, repeat: int):
#         prefix = f"RESTART \n\
# TORSION ATOMS=5,7,9,15 LABEL=5_7_9_15 \n\
# TORSION ATOMS=7,9,15,17 LABEL=7_9_15_17 \n\
# TORSION ATOMS=2,5,7,9 LABEL=2_5_7_9 \n\
# TORSION ATOMS=9,15,17,19 LABEL=9_15_17_19 \n\
# MATHEVAL ARG=5_7_9_15 FUNC=cos(x)--0.018100298941135406 LABEL=cos_5_7_9_15 PERIODIC=NO \n\
# MATHEVAL ARG=5_7_9_15 FUNC=sin(x)--0.7750170826911926 LABEL=sin_5_7_9_15 PERIODIC=NO \n\
# MATHEVAL ARG=7_9_15_17 FUNC=cos(x)-0.11455978453159332 LABEL=cos_7_9_15_17 PERIODIC=NO \n\
# MATHEVAL ARG=7_9_15_17 FUNC=sin(x)-0.5776147246360779 LABEL=sin_7_9_15_17 PERIODIC=NO \n\
# MATHEVAL ARG=2_5_7_9 FUNC=cos(x)--0.9876720905303955 LABEL=cos_2_5_7_9 PERIODIC=NO \n\
# MATHEVAL ARG=2_5_7_9 FUNC=sin(x)--0.0018686018884181976 LABEL=sin_2_5_7_9 PERIODIC=NO \n\
# MATHEVAL ARG=9_15_17_19 FUNC=cos(x)--0.9883973598480225 LABEL=cos_9_15_17_19 PERIODIC=NO \n\
# MATHEVAL ARG=9_15_17_19 FUNC=sin(x)--0.004300905857235193 LABEL=sin_9_15_17_19 PERIODIC=NO \n\
# "
#         if method == "TICA":
#             return (
#                     prefix
#                     + f"COMBINE LABEL=TICA_0 ARG=cos_5_7_9_15,sin_5_7_9_15,cos_7_9_15_17,sin_7_9_15_17,cos_2_5_7_9,sin_2_5_7_9,cos_9_15_17_19,sin_9_15_17_19 COEFFICIENTS=0.34064384071734,0.92966430112022,0.0161593856858182,-0.026732070218086,0.124684184179194,-0.054801458726274,-0.0042902856957667,0.0119406643788291 PERIODIC=NO \n\
# METAD ARG=TICA_0 SIGMA=0.1 HEIGHT=1.2 BIASFACTOR=8 TEMP=300 FILE=HILLS_TICA{repeat} PACE=500 LABEL=metad \n\
# PRINT ARG=TICA_0,metad.bias STRIDE={self.stride} FILE=COLVAR_TICA{repeat} \n"
#             )
#         elif method == "PCA":
#             return (
#                     prefix
#                     + f"COMBINE LABEL=PCA_0 ARG=cos_5_7_9_15,sin_5_7_9_15,cos_7_9_15_17,sin_7_9_15_17,cos_2_5_7_9,sin_2_5_7_9,cos_9_15_17_19,sin_9_15_17_19 COEFFICIENTS=-0.5233437836034278,0.14863186024723976,-0.8323598980242899,-0.10014450000828198,2.8964499312778003e-05,0.03174602265864699,-0.00016289310381917277,0.01265291173498418 PERIODIC=NO \n\
# METAD ARG=PCA_0 SIGMA=0.1 HEIGHT=1.2 BIASFACTOR=8 TEMP=300 FILE=HILLS_PCA{repeat} PACE=500 LABEL=metad \n\
# PRINT ARG=PCA_0,metad.bias STRIDE={self.stride} FILE=COLVAR_PCA{repeat} \n"
#             )
#         elif method == "VAMP":
#             return (
#                     prefix
#                     + f"COMBINE LABEL=VAMP_0 ARG=cos_5_7_9_15,sin_5_7_9_15,cos_7_9_15_17,sin_7_9_15_17,cos_2_5_7_9,sin_2_5_7_9,cos_9_15_17_19,sin_9_15_17_19 COEFFICIENTS=0.339774471816777,0.930293229662793,0.01633229139417103,-0.0254281296973736,0.1227416304730610,-0.0544009983962363,-0.00522093244037105,0.01192673508843518 PERIODIC=NO \n\
# METAD ARG=VAMP_0 SIGMA=0.1 HEIGHT=1.2 BIASFACTOR=8 TEMP=300 FILE=HILLS_VAMP{repeat} PACE=500 LABEL=metad \n\
# PRINT ARG=VAMP_0,metad.bias STRIDE={self.stride} FILE=COLVAR_VAMP{repeat} \n"
#             )
#         elif method == "DMAP":
#             return (
#                     prefix
#                     + f"COMBINE LABEL=DMAP_0 ARG=cos_5_7_9_15,sin_5_7_9_15,cos_7_9_15_17,sin_7_9_15_17,cos_2_5_7_9,sin_2_5_7_9,cos_9_15_17_19,sin_9_15_17_19 COEFFICIENTS=-0.341182884310888,-0.93079433857876,-0.015982887915466,0.026217148677748,-0.11462269900346,0.0539851347818776,-0.0088330576090523,-0.01194011395056 PERIODIC=NO \n\
# METAD ARG=DMAP_0 SIGMA=0.1 HEIGHT=1.2 BIASFACTOR=8 TEMP=300 FILE=HILLS_DMAP{repeat} PACE=500 LABEL=metad \n\
# PRINT ARG=DMAP_0,metad.bias STRIDE={self.stride} FILE=COLVAR_DMAP{repeat} \n"
            #)

#     def REWEIGHT_PLUMED(self, method: str, repeat: int, fraction: float):
#         prefix = f"RESTART \n\
# TORSION ATOMS=5,7,9,15 LABEL=5_7_9_15 \n\
# TORSION ATOMS=7,9,15,17 LABEL=7_9_15_17 \n\
# TORSION ATOMS=2,5,7,9 LABEL=2_5_7_9 \n\
# TORSION ATOMS=9,15,17,19 LABEL=9_15_17_19 \n\
# MATHEVAL ARG=5_7_9_15 FUNC=cos(x)--0.018100298941135406 LABEL=cos_5_7_9_15 PERIODIC=NO \n\
# MATHEVAL ARG=5_7_9_15 FUNC=sin(x)--0.7750170826911926 LABEL=sin_5_7_9_15 PERIODIC=NO \n\
# MATHEVAL ARG=7_9_15_17 FUNC=cos(x)-0.11455978453159332 LABEL=cos_7_9_15_17 PERIODIC=NO \n\
# MATHEVAL ARG=7_9_15_17 FUNC=sin(x)-0.5776147246360779 LABEL=sin_7_9_15_17 PERIODIC=NO \n\
# MATHEVAL ARG=2_5_7_9 FUNC=cos(x)--0.9876720905303955 LABEL=cos_2_5_7_9 PERIODIC=NO \n\
# MATHEVAL ARG=2_5_7_9 FUNC=sin(x)--0.0018686018884181976 LABEL=sin_2_5_7_9 PERIODIC=NO \n\
# MATHEVAL ARG=9_15_17_19 FUNC=cos(x)--0.9883973598480225 LABEL=cos_9_15_17_19 PERIODIC=NO \n\
# MATHEVAL ARG=9_15_17_19 FUNC=sin(x)--0.004300905857235193 LABEL=sin_9_15_17_19 PERIODIC=NO \n\
# "
#         if method == "TICA":
#             return (
#                     prefix
#                     + f"COMBINE LABEL=TICA_0 ARG=cos_5_7_9_15,sin_5_7_9_15,cos_7_9_15_17,sin_7_9_15_17,cos_2_5_7_9,sin_2_5_7_9,cos_9_15_17_19,sin_9_15_17_19 COEFFICIENTS=0.34064384071734,0.92966430112022,0.0161593856858182,-0.026732070218086,0.124684184179194,-0.054801458726274,-0.0042902856957667,0.0119406643788291 PERIODIC=NO \n\
# METAD ARG=TICA_0 SIGMA=0.1 HEIGHT=0.0 FILE=HILLS_TICA{repeat}_{fraction} PACE=10000000 LABEL=metad RESTART=YES \n\
# PRINT ARG=TICA_0,metad.bias STRIDE=1 FILE=COLVAR_TICA{repeat}_{fraction}_REWEIGHT \n\
# as: REWEIGHT_BIAS ARG=TICA_0  TEMP=300 \n\
# hh: HISTOGRAM ARG=5_7_9_15,7_9_15_17 GRID_MIN=-3.14,-3.14 GRID_MAX=3.14,3.14 GRID_BIN={self.fe_grid_size},{self.fe_grid_size} BANDWIDTH=0.05,0.05 LOGWEIGHTS=as \n\
# ff: CONVERT_TO_FES GRID=hh TEMP=300 \n\
# DUMPGRID GRID=ff FILE=ff_TICA{repeat}_{fraction}.dat"
#             )
#         elif method == "PCA":
#             return (
#                     prefix
#                     + f"COMBINE LABEL=PCA_0 ARG=cos_5_7_9_15,sin_5_7_9_15,cos_7_9_15_17,sin_7_9_15_17,cos_2_5_7_9,sin_2_5_7_9,cos_9_15_17_19,sin_9_15_17_19 COEFFICIENTS=-0.5233437836034278,0.14863186024723976,-0.8323598980242899,-0.10014450000828198,2.8964499312778003e-05,0.03174602265864699,-0.00016289310381917277,0.01265291173498418 PERIODIC=NO \n\
# METAD ARG=PCA_0 SIGMA=0.1 HEIGHT=0.0 FILE=HILLS_PCA{repeat}_{fraction} PACE=10000000 LABEL=metad RESTART=YES \n\
# PRINT ARG=PCA_0,metad.bias STRIDE=1 FILE=COLVAR_PCA{repeat}_{fraction}_REWEIGHT \n\
# as: REWEIGHT_BIAS ARG=PCA_0 TEMP=300 \n\
# hh: HISTOGRAM ARG=5_7_9_15,7_9_15_17 GRID_MIN=-3.14,-3.14 GRID_MAX=3.14,3.14 GRID_BIN={self.fe_grid_size},{self.fe_grid_size} BANDWIDTH=0.05,0.05 LOGWEIGHTS=as \n\
# ff: CONVERT_TO_FES GRID=hh TEMP=300 \n\
# DUMPGRID GRID=ff FILE=ff_PCA{repeat}_{fraction}.dat"
#             )
#         elif method == "VAMP":
#             return (
#                     prefix
#                     + f"COMBINE LABEL=VAMP_0 ARG=cos_5_7_9_15,sin_5_7_9_15,cos_7_9_15_17,sin_7_9_15_17,cos_2_5_7_9,sin_2_5_7_9,cos_9_15_17_19,sin_9_15_17_19 COEFFICIENTS=0.339774471816777,0.930293229662793,0.01633229139417103,-0.0254281296973736,0.1227416304730610,-0.0544009983962363,-0.00522093244037105,0.01192673508843518 PERIODIC=NO \n\
# METAD ARG=VAMP_0 SIGMA=0.1 HEIGHT=0.0 FILE=HILLS_VAMP{repeat}_{fraction} PACE=10000000 LABEL=metad RESTART=YES \n\
# PRINT ARG=VAMP_0,metad.bias STRIDE=1 FILE=COLVAR_VAMP{repeat}_{fraction}_REWEIGHT \n\
# as: REWEIGHT_BIAS ARG=VAMP_0 TEMP=300 \n\
# hh: HISTOGRAM ARG=5_7_9_15,7_9_15_17 GRID_MIN=-3.14,-3.14 GRID_MAX=3.14,3.14 GRID_BIN={self.fe_grid_size},{self.fe_grid_size} BANDWIDTH=0.05,0.05 LOGWEIGHTS=as \n\
# ff: CONVERT_TO_FES GRID=hh TEMP=300 \n\
# DUMPGRID GRID=ff FILE=ff_VAMP{repeat}_{fraction}.dat"
#             )
#         elif method == "DMAP":
#             return (
#                     prefix
#                     + f"COMBINE LABEL=DMAP_0 ARG=cos_5_7_9_15,sin_5_7_9_15,cos_7_9_15_17,sin_7_9_15_17,cos_2_5_7_9,sin_2_5_7_9,cos_9_15_17_19,sin_9_15_17_19 COEFFICIENTS=-0.341182884310888,-0.93079433857876,-0.015982887915466,0.026217148677748,-0.11462269900346,0.0539851347818776,-0.0088330576090523,-0.01194011395056 PERIODIC=NO \n\
# METAD ARG=DMAP_0 SIGMA=0.1 HEIGHT=0.0 FILE=HILLS_DMAP{repeat}_{fraction} PACE=10000000 LABEL=metad RESTART=YES \n\
# PRINT ARG=DMAP_0,metad.bias STRIDE=1 FILE=COLVAR_DMAP{repeat}_{fraction}_REWEIGHT \n\
# as: REWEIGHT_BIAS ARG=DMAP_0 TEMP=300 \n\
# hh: HISTOGRAM ARG=5_7_9_15,7_9_15_17 GRID_MIN=-3.14,-3.14 GRID_MAX=3.14,3.14 GRID_BIN={self.fe_grid_size},{self.fe_grid_size} BANDWIDTH=0.05,0.05 LOGWEIGHTS=as \n\
# ff: CONVERT_TO_FES GRID=hh TEMP=300 \n\
# DUMPGRID GRID=ff FILE=ff_DMAP{repeat}_{fraction}.dat"
#             )
