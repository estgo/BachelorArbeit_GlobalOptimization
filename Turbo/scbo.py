import os
import math
import warnings
from dataclasses import dataclass

import torch
from torch import Tensor
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
from botorch.test_functions import SyntheticTestFunction
from botorch.settings import debug

from abc import ABC
from typing import List, Optional, Tuple, Union
from botorch.exceptions.errors import InputDataError
from botorch.test_functions.base import BaseTestProblem, ConstrainedBaseTestProblem
from botorch.test_functions.utils import round_nearest



# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize


import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.settings import debug

# Globally enable debugging mode
debug._set_state(True)


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Change to 'cuda' if using GPU
}

print(device)
print(tkwargs)


def f(X, negate = False):
    result = (1.0 - X[0]) ** 2 + 100.0 * ((X[1] - X[0] ** 2) ** 2)
    return result if not negate else -result


bounds = torch.tensor([
    [-1.5, -0.5],  # Lower bounds for x1 and x2
    [1.5, 2.5]  # Upper bounds for x1 and x2
], dtype=torch.float, device=device)

dim = 2
lb, ub = bounds

batch_size = 10
n_init = 10
max_cholesky_size = float("inf")  # Always use Cholesky


def eval_objective(x):
    """This is a helper function we use to unnormalize and evalaute a point"""

    return f(unnormalize(x.to(device), bounds), True)

def c1(x):  # Equivalent to enforcing that x[1] - x[0]^2 > 0
    return x[0]**2 - x[1]


def c2(x):  # Equivalent to enforcing that  (x[0] - 1)^3 - x[1] + 0.7 > 0
    return x[1] - 0.7 - (x[0] - 1)**3



#TODO
# We assume c1, c2 have same bounds as the Ackley function above
def eval_c1(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return c1(unnormalize(x, bounds))


def eval_c2(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return c2(unnormalize(x, bounds))


@dataclass
class ScboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    best_constraint_values: Tensor = torch.ones(2, **tkwargs) * torch.inf
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))


def get_best_index_for_batch(Y: Tensor, C: Tensor):
    """Return the index for the best point."""
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():  # Choose best feasible candidate
        score = Y.clone()
        score[~is_feas] = -float("inf")
        return score.argmax()
    return C.clamp(min=0).sum(dim=-1).argmin()


def update_tr_length(state: ScboState):
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state


def update_state(state, Y_next, C_next):
    """Method used to update the TuRBO state after each step of optimization.

    Success and failure counters are updated according to the objective values
    (Y_next) and constraint values (C_next) of the batch of candidate points
    evaluated on the optimization step.

    As in the original TuRBO paper, a success is counted whenver any one of the
    new candidate points improves upon the incumbent best point. The key difference
    for SCBO is that we only compare points by their objective values when both points
    are valid (meet all constraints). If exactly one of the two points being compared
    violates a constraint, the other valid point is automatically considered to be better.
    If both points violate some constraints, we compare them inated by their constraint values.
    The better point in this case is the one with minimum total constraint violation
    (the minimum sum of constraint values)"""

    # Pick the best point from the batch
    best_ind = get_best_index_for_batch(Y=Y_next, C=C_next)
    y_next, c_next = Y_next[best_ind], C_next[best_ind]

    if (c_next <= 0).all():
        # At least one new candidate is feasible
        improvement_threshold = state.best_value + 1e-3 * math.fabs(state.best_value)
        if y_next > improvement_threshold or (state.best_constraint_values > 0).any():
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:
        # No new candidate is feasible
        total_violation_next = c_next.clamp(min=0).sum(dim=-1)
        total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
        if total_violation_next < total_violation_center:
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1

    # Update the length of the trust region according to the success and failure counters
    state = update_tr_length(state)
    return state


#Überprüfe ob random
def get_initial_points(dim, n_pts, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    C,  # Constraint values
    batch_size,
    n_candidates,  # Number of candidates for Thompson sampling
    constraint_model,
    sobol: SobolEngine,
):
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    # Create the TR bounds
    best_ind = get_best_index_for_batch(Y=Y, C=C)
    x_center = X[best_ind, :].clone()
    tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)

    # Thompson Sampling w/ Constraints (SCBO)
    dim = X.shape[-1]
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

    # Create candidate points from the perturbations and the mask
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Sample on the candidate points using Constrained Max Posterior Sampling
    constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
        model=model, constraint_model=constraint_model, replacement=False
    )
    with torch.no_grad():
        X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

    return X_next

gpytorch.settings.show_progress_bars = True


def get_fitted_model(X, Y):
    assert not torch.isnan(X).any(), "Training data contains NaN values"
    assert not torch.isnan(Y).any(), "Training labels contain NaN values"
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=1.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
    )
    model = SingleTaskGP(
        X,
        Y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        fit_gpytorch_mll(mll)

    return model


def search_optimum_constrained(
        n_init,
        generate_points_fun,
        eval_objective_fun,
        eval_constrain_functions,
        state,
        generate_batch_fun,
        fitted_model_fun
):
    # Generate initial data
    dim = state.dim
    batch_size = state.batch_size
    N_CANDIDATES = 30 if not SMOKE_TEST else 4
    sobol = SobolEngine(dim, scramble=True, seed=1)

    train_X = generate_points_fun(dim, n_init)
    train_Y = torch.tensor([eval_objective_fun(x) for x in train_X], **tkwargs).unsqueeze(-1)

    '''for eval_constrain_fun in eval_constrain_functions:
        if(i == 0):
            C = torch.tensor([eval_constrain_fun(x) for x in train_X], **tkwargs).unsqueeze(-1)
            i += 1
            continue
        C_i = torch.tensor([eval_constrain_fun(x) for x in train_X], **tkwargs).unsqueeze(-1)
        C = torch.cat([C, C_i], dim=-1)'''

    C = [torch.tensor([eval_constrain_fun(x) for x in train_X], **tkwargs).unsqueeze(-1) for eval_constrain_fun in
         eval_constrain_functions]

    debugging = False
    while not state.restart_triggered and not debugging:  # Run until TuRBO converges
        # Fit GP models for objective and constraints
        model = fitted_model_fun(train_X, train_Y)
        C_model = [get_fitted_model(train_X, C_i) for C_i in C]

        # Generate a batch of candidates
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            X_next = generate_batch_fun(
                state=state,
                model=model,
                X=train_X,
                Y=train_Y,
                C=torch.cat(C, dim=-1),
                batch_size=batch_size,
                n_candidates=N_CANDIDATES,
                constraint_model=ModelListGP(*C_model),
                sobol=sobol,
            )
        # Evaluate both the objective and constraints for the selected candidaates
        Y_next = torch.tensor([eval_objective(x) for x in X_next], dtype=dtype, device=device).unsqueeze(-1)
        C_next = [torch.tensor([eval_constrain_fun(x) for x in X_next], **tkwargs).unsqueeze(-1) for eval_constrain_fun
                  in eval_constrain_functions]

        # Update TuRBO state
        state = update_state(state=state, Y_next=Y_next, C_next=torch.cat(C_next, dim=-1))
        # Append data. Note that we append all data, even points that violate
        # the constraints. This is so our constraint models can learn more
        # about the constraint functions and gain confidence in where violations occur.
        train_X = torch.cat((train_X, X_next), dim=0)
        train_Y = torch.cat((train_Y, Y_next), dim=0)
        for i in range(len(C_next)):
            C[i] = torch.cat((C[i], C_next[i]), dim=0)

        # Print current status. Note that state.best_value is always the best
        # objective value found so far which meets the constraints, or in the case
        # that no points have been found yet which meet the constraints, it is the
        # objective value of the point with the minimum constraint violation.
        if (state.best_constraint_values <= 0).all():
            print(f"{len(train_X)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}")
        else:
            violation = state.best_constraint_values.clamp(min=0).sum()
            print(
                f"{len(train_X)}) No feasible point yet! Smallest total violation: "
                f"{violation:.2e}, TR length: {state.length:.2e}"
            )
    print("finished")
    return train_X, train_Y, torch.cat(C, dim=-1)


def testOne():
    # testing initial:
    state = ScboState(dim, batch_size=5)
    return search_optimum_constrained(10, get_initial_points, eval_objective, [eval_c1, eval_c2], state, generate_batch,
                                         get_fitted_model)

def testTwo():
    # testing initial:
    state = ScboState(dim, batch_size=5)
    return search_optimum_constrained(10, get_initial_points, eval_objective, [eval_c1, eval_c2], state, generate_batch,
                                         get_fitted_model)