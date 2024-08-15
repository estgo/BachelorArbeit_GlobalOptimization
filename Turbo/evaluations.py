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

    @property
    @abstractmethod
    def dim(self):
        """This is an abstract property that must be overridden in a subclass."""
        pass

    @property
    @abstractmethod
    def bounds(self):
        """This is an abstract property that must be overridden in a subclass."""
        pass


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

