import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from deeptime.clustering import KMeans
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.markov.tools.analysis import mfpt
from tqdm import tqdm

from utils.openmm_utils import round_format_unit
from utils.plot_utils import my_draw_networkx_edge_labels
from utils.plotting_functions import init_plot

# TODO: tidy up into a class


def msm_clustering(samples: np.array, n_clusters: int, show_inertia: bool = True):
    """
    Clusters samples into n_clusters and returns the discrete trajectory.
    """
    estimator = KMeans(n_clusters=n_clusters, init_strategy='kmeans++', fixed_seed=13, n_jobs=os.cpu_count() - 1,
                       progress=tqdm)
    clustering = estimator.fit(samples).fetch_model()
    if show_inertia:
        """
        Inertia measures how well a dataset was clustered by K-Means. It is calculated by measuring the distance 
        between each data point and its centroid, squaring this distance, and summing these squares across one cluster.
        """
        fig, ax = init_plot("Inertia of KMeans Training", "iteration", "inertia", xscale="log")
        ax.plot(clustering.inertias)
        plt.show()

    return clustering.transform(samples)


def msm_discrete_traj(clustering, samples):
    print("Getting discrete trajectory")
    discrete_traj = clustering.transform(samples)
    print("Cluster centres", clustering.cluster_centers)

    return discrete_traj

def msm_transition_counts(discrete_traj: np.array, lagstep: int, count_mode="sliding"):
    estimator = TransitionCountEstimator(lagtime=lagstep, count_mode=count_mode)
    counts = estimator.fit(discrete_traj).fetch_model()
    print("Weakly connected sets:", counts.connected_sets(directed=False))
    print("Strongly connected sets:", counts.connected_sets(directed=True))

    return counts


def maximum_likelihood_msm(counts: np.array):
    estimator = MaximumLikelihoodMSM(reversible=True, stationary_distribution_constraint=None)
    msm = estimator.fit(counts).fetch_model()

    return msm


def plot_transition_graph(msm, lagstep, savefreq):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    threshold = 1e-2
    title = f"Transition matrix with connectivity threshold {threshold:.0e}"
    G = nx.DiGraph()
    ax.set_title(title)
    edge_labels = {}
    for i in range(msm.n_states):
        G.add_node(i, title=f"{i + 1}")
    for i in range(msm.n_states):
        for j in range(msm.n_states):
            if msm.transition_matrix[i, j] > threshold:
                transition_time = mfpt(msm.transition_matrix, target=j, origin=i, tau=lagstep)
                G.add_edge(i, j)
                edge_labels[(i,
                             j)] = f"{msm.transition_matrix[i, j]:.3e} " \
                                   f"({round_format_unit(transition_time * savefreq, 3)})"

    # edge_labels = nx.get_edge_attributes(G, 'title')
    pos = nx.fruchterman_reingold_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=nx.get_node_attributes(G, 'title'))
    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='-|>',
                           connectionstyle='arc3, rad=0.3')
    my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, rotate=False, rad=0.25)

