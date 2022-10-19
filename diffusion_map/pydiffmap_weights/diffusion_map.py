# -*- coding: utf-8 -*-
"""
Routines and Class definitions for the diffusion maps algorithm.
"""
from __future__ import absolute_import
from abc import ABC, abstractmethod

class DiffusionMap(ABC):
    """
    Diffusion Map base class.

    Parameters
    ----------
    alpha : scalar, optional
        Exponent to be used for the left normalization in constructing the diffusion map.
    k : int, optional
        Number of nearest neighbors over which to construct the kernel.
    epsilon: string or scalar, optional
        Method for choosing the epsilon.  Currently, the only options are to provide a scalar (epsilon is set to the provided scalar) or 'bgh' (Berry, Giannakis and Harlim).
    metric : string, optional
        Metric for distances in the kernel. Default is 'euclidean'. The callable should take two arrays as input and return one value indicating the distance between them.
    metric_params : dict or None, optional
        Optional parameters required for the metric given.
    """

    def __init__(self, alpha=0.5, k=64, epsilon='bgh', metric='euclidean', metric_params=None):
        """
        Initializes Diffusion Map, sets parameters.
        """
        self.alpha = alpha
        self.k = k
        self.epsilon = epsilon
        self.metric = metric
        self.metric_params = metric_params

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def fit_transform(self, X):
        pass


