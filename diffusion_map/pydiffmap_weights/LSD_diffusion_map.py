from .diffusion_map import DiffusionMap
from openmm.app import PDBReporter
import numpy as np
import os
import subprocess


class LSDMap(DiffusionMap):
    # TODO fix header text
    """
       Diffusion Map object to be used in data analysis for fun and profit.

       Parameters
       ----------
       alpha : scalar, optional
           Exponent to be used for the left normalization in constructing the diffusion map.
       k : int, optional
           Number of nearest neighbors over which to construct the kernel.
       kernel_type : string, optional
           Type of kernel to construct. Currently the only option is 'gaussian', but more will be implemented.
       epsilon: scalar, optional
           Method for choosing the epsilon.  Currently, the only options are to provide a scalar (epsilon is set to the provided scalar) or 'bgh' (Berry, Giannakis and Harlim).
       n_evecs : int, optional
           Number of diffusion map eigenvectors to return
       neighbor_params : dict or None, optional
           Optional parameters for the nearest Neighbor search. See scikit-learn NearestNeighbors class for details.
       metric : string, optional
           Metric for distances in the kernel. Default is 'euclidean'. The callable should take two arrays as input and return one value indicating the distance between them.
       metric_params : dict or None, optional
           Optional parameters required for the metric given.

       Examples
       --------
       # setup neighbor_params list with as many jobs as CPU cores and kd_tree neighbor search.
       >>> neighbor_params = {'n_jobs': -1, 'algorithm': 'kd_tree'}
       # initialize diffusion map object with the top two eigenvalues being computed, epsilon set to 0.1
       # and alpha set to 1.0.
       >>> mydmap = DiffusionMap(n_evecs = 2, epsilon = .1, alpha = 1.0, neighbor_params = neighbor_params)

    """
    def __init__(self, alpha=0.5, k=64, epsilon=0.05, metric='euclidean', metric_params=None, status="constant"):
        """
        Initializes Diffusion Map, sets parameters.
        """
        super().__init__(alpha, k, epsilon, metric, metric_params)
        assert status in ['constant', 'kneighbor', 'kneighbor_mean', 'user'], f"Status '{status}' is invalid"
        assert metric in ['euclidean', 'rmsd', 'cmd', 'contact_matrix_distance', 'dihedral'], f"Metric '{metric}' is invalid"
        assert isinstance(epsilon, float), f"Epsilon must be a float, not {type(epsilon)}"
        self.status = status

    def _convert_PDB_trajectory_to_gro_trajectory(self):
        pdb_to_gro_command = f'gmx pdb2gmx -f {self.trajectory_file}.pdb  -o {self.trajectory_file}.gro'
        print(pdb_to_gro_command)
        subprocess.call(pdb_to_gro_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)

    def _run_LSDmap(self, config_file):
        LSDmap_command = f'bash -c "conda activate Py2diffusion; cd /home/dominic/PycharmProjects/mdfeature/notebooks; lsdmap -f {config_file} -c {self.trajectory_file}.gro"'
        #todo: check to make sure that lsdmap exists
        print(LSDmap_command)
        subprocess.call(LSDmap_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)

    def _construct_config_file(self, file_name):
        with open(file_name, "w") as f:
            if self.metric == "euclidean" or self.metric == "rmsd":
                LSDmap_metric = "rmsd"
            elif self.metric == "cmd" or self.metric == "contact_matrix_distance":
                LSDmap_metric = "cmd"
            elif self.metric == "dihedral":
                LSDmap_metric = "dihedral"
            else:
                raise

            LSDmap_r0 = 3.5
            if self.metric_params is not None:
                assert 'r0' in self.metric_params.keys(), "The metric parameter r0 was not supplied"
                LSDmap_r0 = self.metric_params['r0']

            print("[LSDMAP]", file=f)
            print(";metric used to compute the distance matrix(rmsd, cmd, dihedral)", file=f)
            print(f"metric = {LSDmap_metric}", file=f)
            print(";constant r0 used with cmd metric ( in nm)")
            print(f"r0 = {LSDmap_r0}", file=f)
            print("[LOCALSCALE]", file=f)
            print(";status (constant, kneighbor, kneighbor_mean, user)", file=f)
            print(f"status = {self.status}", file=f)
            print(";constant epsilon used in case status is constant( in nm)", file=f)
            print(f"epsilon = {self.epsilon}", file=f)
            print(";value of k in case status is kneighbor or kneighbor_mean", file=f)
            print(f"k = {self.k}", file=f)

    def _make_diffusion_coords(self):
        self._construct_config_file(file_name='config.ini')
        self._run_LSDmap(config_file='config.ini')
        evals = np.genfromtxt(f'{self.trajectory_file}.eg')
        evecs = np.genfromtxt(f'{self.trajectory_file}.ev')
        dmap = np.dot(evecs, np.diag(evals))

        return dmap, evecs, evals

    def _check_pdb_file(self, pdb_file):
        with open(pdb_file) as file:
            num_lines = len(file.readlines())

        assert num_lines > 2, f"Check your PDB experiment file ({self.trajectory_file}) - it appears to be empty!"

    def _make_pdb_compatible_with_gromacs(self, pdb_trajectory):
        f = open(pdb_trajectory, 'r')
        filedata = f.read()
        f.close()

        # todo add check to make sure that this doesn't add recursively by mistake
        newdata = filedata.replace(" H1 ", "HH31")
        newdata = newdata.replace(" H2 ", "HH32")
        newdata = newdata.replace(" H3 ", "HH33")

        f = open(pdb_trajectory, 'w')
        f.write(filedata)
        f.close()

    def _pdb_to_gro_converter(self, pdb_path, gro_path):
        with open(pdb_path, "r") as pdb_file:
            first_pass = True
            start_idx = 0
            end_idx = 0
            for idx, line in enumerate(pdb_file.readlines()):
                if "MODEL" in line:
                    if first_pass:
                        first_pass = False
                        start_idx = idx
                    else:
                        end_idx = idx
                        break
        number_of_atoms = end_idx - start_idx - 3

        with open(pdb_path, "r") as pdb_file:
            with open(gro_path, "w") as gro_file:
                for line in pdb_file.readlines():
                    if "REMARK" in line or "ENDMDL" in line:
                        continue
                    if "MODEL" in line:
                        print("Protein t=   0.00000", file=gro_file)
                        print(f"   {number_of_atoms}", file=gro_file)
                        continue
                    if "TER" in line:
                        print("  50.00000  50.00000  50.00000", file=gro_file)
                        continue
                    else:
                        pdb_cols = [entry.strip() for entry in line.split()]
                        print(pdb_cols)
                        gro_col_1 = pdb_cols[5]+pdb_cols[3]
                        gro_col_2 = pdb_cols[2]
                        gro_col_3 = pdb_cols[1]
                        gro_col_4 = pdb_cols[6]
                        gro_col_5 = pdb_cols[7]
                        gro_col_6 = pdb_cols[8]
                        gro_cols_7 = "0"
                        gro_cols_8 = "0"
                        gro_cols_9 = "0"
                        gro_cols = "    "+gro_col_1+" "*(6-len(gro_col_2))+gro_col_2+" "*(5-len(gro_col_3))+gro_col_3\
                                   +" "*(7-len(gro_col_4))+gro_col_4+" "*(7-len(gro_col_5))+gro_col_5+" "*(7-len(gro_col_6))+gro_col_6+"  "\
                                   +gro_cols_7+"  "+gro_cols_8+"  "+gro_cols_9
                        print(gro_cols, file=gro_file)
                        continue


    def fit(self, pdb_trajectory):
        self.trajectory_file = pdb_trajectory.strip('.pdb')
        self._check_pdb_file(pdb_trajectory)
        self._pdb_to_gro_converter(pdb_trajectory, self.trajectory_file+'.gro')
        dmap, evecs, evals = self._make_diffusion_coords()

        self.evals = evals
        self.evecs = evecs
        self.dmap = dmap

        return self

    def fit_transform(self, traj):
        self.fit(traj)
        return self.dmap




