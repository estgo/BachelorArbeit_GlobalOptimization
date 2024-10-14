import os
import math
import warnings
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

import numpy as np

from abc import ABC, abstractmethod

class Function(ABC):

    '''@property
    @abstractmethod
    def dim(self):
        """This is an abstract property that must be overridden in a subclass."""
        pass

    @property
    @abstractmethod
    def bounds(self):
        """This is an abstract property that must be overridden in a subclass."""
        pass'''


    @abstractmethod
    def fun(self, x):
        """This is an abstract property that must be overridden in a subclass."""
        pass

    def eval_objective(self, x):
        return self.fun(unnormalize(x, self.bounds()))


class Ackley20D(Function):
    def __init__(self):
        self.dim = 20
        self.bounds = np.zeros((2, self.dim))
        self.bounds[0, :].fill(-5)
        self.bounds[1, :].fill(10)
        self.fun = Ackley(dim=self.dim, negate=False)  # Initialize Ackley function

    def fun(self, x):
        return self.fun(x)

class Branin(Function):
    def __init__(self):
        self.dim = 2
        self.bounds = np.zeros((2, self.dim))
        self.bounds[0, :].fill(-5)
        self.bounds[1, :].fill(10)

    def fun(self, x):
        a = 1.0
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        return a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s

    '''@dim.setter
    def dim(self, value):
        self._dim = value

    @bounds.setter
    def bounds(self, value):
        self._bounds = value'''

class Michalewicz10D(Function):
    def __init__(self):
        self.dim = 10
        self.bounds = np.zeros((2, self.dim))
        #self.bounds[0, :].fill(0) #already done
        self.bounds[1, :].fill(np.pi)

    def fun(self, x, m=10):
        if len(x) != 10:
            raise ValueError("Input vector must have 10 dimensions.")

        term = 0
        for i in range(10):
            term += np.sin(x[i]) * (np.sin(((i + 1) * x[i] ** 2) / np.pi)) ** (2 * m)

        return -term

class Himmelblau2D(Function):
    def __init__(self):
        self.dim = 2
        self.bounds = np.zeros((2, self.dim))
        self.bounds[0, :].fill(-5)
        self.bounds[1, :].fill(5)

    def fun(self, x):
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2