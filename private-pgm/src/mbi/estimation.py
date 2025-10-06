"""Algorithms for estimating graphical models from marginal-based loss functions.

This module provides a flexible set of optimization algorithms, each sharing the
the same API.  The supported algorithms are:
    1. Mirror Descent [our recommended algorithm]
    2. L-BFGS (using back-belief propagation)
    3. Regularized Dual Averaging
    4. Interior Gradient

Each algorithm can be given an initial set of potentials, or can automatically
intialize the potentials to zero for you.  Any CliqueVector of potentials that
support the cliques of the marginal-based loss function can be used here.
"""

import numpy as np
from mbi import Domain, CliqueVector, Factor, LinearMeasurement
from mbi import marginal_oracles, marginal_loss, synthetic_data
from typing import Callable
import jax
import jax.numpy as jnp
import chex
import attr
import optax

import sys, os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONSTRAINT_DIR = os.path.join(THIS_DIR, '..', '..', '..', 'utils')
sys.path.append(CONSTRAINT_DIR)
from Constraints import *

_DEFAULT_CALLBACK = lambda t, loss: print(loss) if t % 50 == 0 else None
jax.config.update("jax_enable_x64", True)

# API may change, we'll see
@attr.dataclass(frozen=True)
class GraphicalModel:
    potentials: CliqueVector
    marginals: CliqueVector
    total: chex.Numeric = 1

    def project(self, attrs: tuple[str, ...]) -> Factor:
        try:
            return self.marginals.project(attrs)
        except:
            return marginal_oracles.variable_elimination(
                self.potentials, attrs, self.total
            )

    def synthetic_data(self, rows: int | None = None):
        return synthetic_data.from_marginals(self, rows or self.total)

    @property
    def domain(self):
        return self.potentials.domain

    @property
    def cliques(self):
        return self.potentials.cliques


def minimum_variance_unbiased_total(measurements: list[LinearMeasurement]) -> float:
    # find the minimum variance estimate of the total given the measurements
    estimates, variances = [], []
    for M in measurements:
        y = M.noisy_measurement
        try:
            # TODO: generalize to support any linear measurement that supports total query
            if np.allclose(M.query(y), y):  # query = Identity
                estimates.append(y.sum())
                variances.append(M.stddev**2 * y.size)
        except:
            continue
    estimates, variances = np.array(estimates), np.array(variances)
    if len(estimates) == 0:
        return 1
    else:
        variance = 1.0 / np.sum(1.0 / variances)
        estimate = variance * np.sum(estimates / variances)
        return max(1, estimate)


def _initialize(domain, loss_fn, known_total, potentials):
    if isinstance(loss_fn, list):
        if known_total is None:
            known_total = minimum_variance_unbiased_total(loss_fn)
        loss_fn = marginal_loss.from_linear_measurements(loss_fn)
    elif known_total is None:
        raise ValueError("Must set known_total is giving a custom MarginalLossFn")

    if potentials is None:
        potentials = CliqueVector.zeros(domain, loss_fn.cliques)

    if not all(potentials.supports(cl) for cl in loss_fn.cliques):
        potentials = potentials.expand(loss_fn.cliques)

    return loss_fn, known_total, potentials

def mirror_descent(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    *,
    known_total: float | None = None,
    potentials: CliqueVector | None = None,
    marginal_oracle=marginal_oracles.message_passing_fast,
    stateful: bool = False,
    iters: int = 1000,
    stepsize: float | None = None,
    callback_fn: Callable[[CliqueVector], None] = lambda _: None,
):
    """Optimization using the Mirror Descent algorithm.

    This is a first-order proximal optimization algorithm for solving
    a (possibly nonsmooth) convex optimization problem over the marginal polytope.
    This is an  implementation of Algorithm 1 from the paper
    ["Graphical-model based estimation and inference for differential privacy"]
    (https://arxiv.org/pdf/1901.09136).  If stepsize is not provided, this algorithm
    uses a line search to automatically choose appropriate step sizes that satisfy
    the Armijo condition.

    Args:
        domain: The domain over which the model should be defined.
        loss_fn: A MarginalLossFn or a list of Linear Measurements.
        known_total: The known or estimated number of records in the data.
        potentials: The initial potentials.  Must be defind over a set of cliques
            that supports the cliques in the loss_fn.
        marginal_oracle: The function to use to compute marginals from potentials.
        stateful: flag specifying whether the marginal_oracle is stateful or not
            (e.g., whether messages should be preserved from one call to the next).
        iters: The maximum number of optimization iterations.
        stepsize: The step size for the optimization.  If not provided, this algorithm
            will use a line search to automatically choose appropriate step sizes.
        callback_fn: A function to call at each iteration with the iteration number

    Returns:
        A GraphicalModel object with the estimated potentials and marginals.
    """
    loss_fn, known_total, potentials = _initialize(
        domain, loss_fn, known_total, potentials
    )

    if not stateful:
        stateless_oracle = marginal_oracle
        marginal_oracle = lambda theta, total, state: (stateless_oracle(theta, total), state)
    elif stepsize is None:
        raise ValueError('Stepsize should be manually tuned when using a stateful oracle.')

    @jax.jit
    def update(theta, alpha, state = None):
        mu, state = marginal_oracle(theta, known_total, state)
        loss, dL = jax.value_and_grad(loss_fn)(mu)

        theta2 = theta - alpha * dL
        if stepsize is not None:
            return theta2, loss, alpha, mu, state

        mu2, _ = marginal_oracle(theta2, known_total, state)
        loss2 = loss_fn(mu2)

        sufficient_decrease = loss - loss2 >= 0.5 * alpha * dL.dot(mu - mu2)
        alpha = jax.lax.select(sufficient_decrease, 1.01 * alpha, 0.5 * alpha)
        theta = jax.lax.cond(sufficient_decrease, lambda: theta2, lambda: theta)
        loss = jax.lax.select(sufficient_decrease, loss2, loss)

        return theta, loss, alpha, mu, state

    # A reasonable initial learning rate seems to be 2.0 L / known_total,
    # where L is the Lipschitz constant.  Starting from a value too high
    # can be fine in some cases, but lead to incorrect behavior in others.
    # We don't currently take L as an argument, but for the most common case,
    # where our loss function is || mu - y ||_2^2, we have L = 1.
    alpha = 2.0 / known_total if stepsize is None else stepsize
    mu, state = marginal_oracle(potentials, known_total, state=None)
    for t in range(iters):
        potentials, loss, alpha, mu, state = update(potentials, alpha, state)
        # print(f"Iteratoin: {t} -- Marginal loss: {loss}")
        callback_fn(mu)

    marginals, _ = marginal_oracle(potentials, known_total, state)
    # print(f"[INFO] Marg loss: {loss} | Reg loss: {calculate_regularizer_loss(potentials, known_total)} |")
    return GraphicalModel(potentials, marginals, known_total)


def _optimize(loss_and_grad_fn, params, iters=250, known_total=None, callback_fn=lambda _: None):
    loss_fn = lambda theta: loss_and_grad_fn(theta)[0]

    @jax.jit
    def update(params, opt_state):
        # 1) compute loss & grad
        loss, grad = loss_and_grad_fn(params)
        grad_norm = jnp.sqrt(grad.dot(grad))

        # 2) one L-BFGS step
        updates, new_opt_state = optimizer.update(
            grad, opt_state, params,
            value=loss, grad=grad, value_fn=loss_fn
        )
        new_params = optax.apply_updates(params, updates)

        # 3) grab the line-search state as the last field of new_opt_state
        ls_state = new_opt_state[-1]

        # 4) extract the step_size
        lr = ls_state.linesearch_state.step_size

        return new_params, new_opt_state, loss, grad_norm, lr

    optimizer = optax.lbfgs(
        memory_size=1,
        linesearch=optax.scale_by_zoom_linesearch(128, max_learning_rate=1),
    )
    state = optimizer.init(params)
    prev_loss = float("inf")
    for t in range(iters):
        params, state, loss, grad_norm, lr = update(params, state)
        # print(f"Iter {t}: loss = {loss}, grad norm = {grad_norm}, lr = {lr}")
        # print(f"the Reg loss: {calculate_regularizer_loss(params, known_total)}.")
        callback_fn(params)
        # if loss == prev_loss: break
        prev_loss = loss
    return params


def lbfgs(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    known_total: float | None = None,
    potentials: CliqueVector | None = None,
    marginal_oracle=marginal_oracles.message_passing_stable,
    iters: int = 1000,
    callback_fn: Callable[[CliqueVector], None] = lambda _: None,
):
    """Gradient-based optimization on the potentials (theta) via L-BFGS.

    This optimizer works by calculating the gradients with respect to the
    potentials by back-propagting through the marginal inference oracle.

    This is a standard approach for fitting the parameters of a graphical model
    without noise (i.e., when you know the exact marginals).  In this case,
    the loss function with respect to theta is convex, and therefore this approach
    enjoys convergence guarantees.  With generic marginal loss functions that arise
    for instance ith noisy marginals, the loss function is typically convex with
    respect to mu, but not with respect to theta.  Therefore, this optimizer is not
    guaranteed to converge to the global optimum in all cases.  In practice, it
    tends to work well in these settings despite non-convexities.  This approach
    appeared in the paper ["Learning Graphical Model Parameters with Approximate
    Marginal Inference"](https://arxiv.org/abs/1301.3193).

    Args:
      domain: The domain over which the model should be defined.
      loss_fn: A MarginalLossFn or a list of Linear Measurements.
      known_total: The known or estimated number of records in the data.
        If loss_fn is provided as a list of LinearMeasurements, this argument
        is optional.  Otherwise, it is required.
      potentials: The initial potentials.  Must be defined over a set of cliques
        that supports the cliques in the loss_fn.
      marginal_oracle: The function to use to compute marginals from potentials.
      iters: The maximum number of optimization iterations.
      callback_fn
    """
    loss_fn, known_total, potentials = _initialize(
        domain, loss_fn, known_total, potentials
    )
    lambda_reg = 0.05
    def total_loss(theta):
        mu = marginal_oracle(theta, known_total)

        main_loss = loss_fn(mu)
        reg_loss = calculate_regularizer_loss(theta, known_total)

        # Compute individual gradients
        grad_main = jax.grad(lambda t: loss_fn(marginal_oracle(t, known_total)))(theta)
        # grad_reg = jax.grad(lambda t: calculate_regularizer_loss(t, known_total))(theta)

        # grad_main_norm = jnp.sqrt(grad_main.dot(grad_main))
        # grad_reg_norm = jnp.sqrt(grad_reg.dot(grad_reg)) + 1e-12  # avoid divide by zero

        # dynamic_lambda = lambda_reg * (grad_main_norm / grad_reg_norm)
        # dynamic_lambda = jnp.clip(dynamic_lambda, 0.0, 100.0)  # to avoid exploding


        return main_loss
        # return main_loss + dynamic_lambda * reg_loss

    # theta_loss = lambda theta: loss_fn(marginal_oracle(theta, known_total))
    # theta_loss_and_grad = jax.value_and_grad(theta_loss)
    theta_loss_and_grad = jax.value_and_grad(total_loss)
    theta_callback_fn = lambda theta: callback_fn(marginal_oracle(theta, known_total))
    potentials = _optimize(
        theta_loss_and_grad, potentials, iters=iters, known_total=known_total, callback_fn=theta_callback_fn)

    # print(f"the final Reg loss: {calculate_regularizer_loss(potentials, known_total)}.")
    
    return GraphicalModel(
        potentials, marginal_oracle(potentials, known_total), known_total
    ), (1, 1, 1, 1)


def mle_from_marginals(
    marginals: CliqueVector,
    known_total: float,
    iters: int = 250,
    marginal_oracle=marginal_oracles.message_passing_stable,
    callback_fn=lambda *_: None,
) -> GraphicalModel:
    """Compute the MLE Graphical Model from the marginals.

    Args:
        marginals: The marginal probabilities.
        known_total: The known or estimated number of records in the data.

    Returns:
        A GraphicalModel object with the final potentials and marginals.
    """

    def loss_and_grad_fn(theta):
        mu = marginal_oracle(theta, known_total)
        return -marginals.dot(mu.log()), mu - marginals

    potentials = CliqueVector.zeros(marginals.domain, marginals.cliques)
    potentials = _optimize(loss_and_grad_fn, potentials, iters=iters)
    return GraphicalModel(
        potentials, marginal_oracle(potentials, known_total), known_total
    )


def dual_averaging(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    lipschitz: float,
    known_total: float | None = None,
    potentials: CliqueVector | None = None,
    marginal_oracle=marginal_oracles.message_passing_stable,
    iters: int = 1000,
    callback_fn: Callable[[CliqueVector], None] = lambda _: None,
) -> GraphicalModel:
    """Optimization using the Regularized Dual Averaging (RDA) algorithm.

    RDA is an accelerated proximal algorithm for solving a smooth convex optimization
    problem over the marginal polytope.  This algorithm requires knowledge of
    the Lipschitz constant of the gradient of the loss function.

    Args:
        domain: The domain over which the model should be defined.
        loss_fn: A MarginalLossFn or a list of Linear Measurements.
        lipschitz: The Lipschitz constant of the gradient of the loss function.
        known_total: The known or estimated number of records in the data.
        potentials: The initial potentials.  Must be defind over a set of cliques
            that supports the cliques in the loss_fn.
        marginal_oracle: The function to use to compute marginals from potentials.
        iters: The maximum number of optimization iterations.
        callback_fn: A function to call with intermediate solution at each iteration.

    Returns:
        A GraphicalModel object with the final potentials and marginals.
    """
    loss_fn, known_total, potentials = _initialize(
        domain, loss_fn, known_total, potentials
    )
    D = np.sqrt(domain.size() * np.log(domain.size()))  # upper bound on entropy
    Q = 0  # upper bound on variance of stochastic gradients
    gamma = Q / D

    L = lipschitz / known_total

    @jax.jit
    def update(w, v, gbar, c, beta):
        u = (1 - c) * w + c * v
        g = jax.grad(loss_fn)(u) / known_total
        gbar = (1 - c) * gbar + c * g
        theta = -t * (t + 1) / (4 * L + beta) * gbar
        v = marginal_oracle(theta, known_total)
        w = (1 - c) * w + c * v
        return w, v, gbar

    w = v = marginal_oracle(potentials, known_total)
    gbar = CliqueVector.zeros(domain, loss_fn.cliques)
    for t in range(1, iters + 1):
        c = 2.0 / (t + 1)
        beta = gamma * (t + 1) ** 1.5 / 2
        w, v, gbar = update(w, v, gbar, c, beta)
        callback_fn(w)

    return mle_from_marginals(w, known_total)


def interior_gradient(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    lipschitz: float | None = None,
    known_total: float | None = None,
    potentials: CliqueVector | None = None,
    marginal_oracle=marginal_oracles.message_passing_stable,
    iters: int = 1000,
    stepsize: float | None = None,
    callback_fn: Callable[[CliqueVector], None] = lambda _: None,
):
    """Optimization using the Interior Point Gradient Descent algorithm.

    Interior Gradient is an accelerated proximal algorithm for solving a smooth
    convex optimization problem over the marginal polytope.  This algorithm
    requires knowledge of the Lipschitz constant of the gradient of the loss function.
    This algorithm is based on the paper titled
    ["Interior Gradient and Proximal Methods for Convex and Conic Optimization"](https://epubs.siam.org/doi/abs/10.1137/S1052623403427823?journalCode=sjope8).

    Args:
        domain: The domain over which the model should be defined.
        loss_fn: A MarginalLossFn or a list of Linear Measurements.
        lipschitz: The Lipschitz constant of the gradient of the loss function.
        known_total: The known or estimated number of records in the data.
        potentials: The initial potentials.  Must be defind over a set of cliques
            that supports the cliques in the loss_fn.
        marginal_oracle: The function to use to compute marginals from potentials.
        iters: The maximum number of optimization iterations.
        callback_fn: A function to call at each iteration with the iteration number

    Returns:
        A GraphicalModel object with the optimized potentials and marginals.
    """
    loss_fn, known_total, potentials = _initialize(
        domain, loss_fn, known_total, potentials
    )

    # Algorithm parameters
    c = 1
    sigma = 1
    l = sigma / lipschitz

    @jax.jit
    def update(theta, c, x, y, z):
        a = (((c * l) ** 2 + 4 * c * l) ** 0.5 - l * c) / 2
        y = (1 - a) * x + a * z
        c = c * (1 - a)
        g = jax.grad(loss_fn)(y)
        theta = theta - a / c / known_total * g
        z = marginal_oracle(theta, known_total)
        x = (1 - a) * x + a * z
        return theta, c, x, y, z

    x = y = z = marginal_oracle(potentials, known_total)
    gbar = CliqueVector.zeros(domain, loss_fn.cliques)
    theta = potentials
    for t in range(1, iters + 1):
        theta, c, x, y, z = update(theta, c, x, y, z)
        callback_fn(x)

    return mle_from_marginals(x, known_total)


def proximal_mirror_descent(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    *,
    known_total: float | None = None,
    potentials: CliqueVector | None = None,
    marginal_oracle=marginal_oracles.message_passing_fast,
    stateful: bool = False,
    iters: int = 1000,
    stepsize: float | None = None,
    callback_fn: Callable[[CliqueVector], None] = lambda _: None,
    cmi_ratio: float = 0.1,
    degree: int = 20
):
    """Optimization using the Mirror Descent algorithm.

    This is a first-order proximal optimization algorithm for solving
    a (possibly nonsmooth) convex optimization problem over the marginal polytope.
    This is an  implementation of Algorithm 1 from the paper
    ["Graphical-model based estimation and inference for differential privacy"]
    (https://arxiv.org/pdf/1901.09136).  If stepsize is not provided, this algorithm
    uses a line search to automatically choose appropriate step sizes that satisfy
    the Armijo condition.

    Args:
        domain: The domain over which the model should be defined.
        loss_fn: A MarginalLossFn or a list of Linear Measurements.
        known_total: The known or estimated number of records in the data.
        potentials: The initial potentials.  Must be defind over a set of cliques
            that supports the cliques in the loss_fn.
        marginal_oracle: The function to use to compute marginals from potentials.
        stateful: flag specifying whether the marginal_oracle is stateful or not
            (e.g., whether messages should be preserved from one call to the next).
        iters: The maximum number of optimization iterations.
        stepsize: The step size for the optimization.  If not provided, this algorithm
            will use a line search to automatically choose appropriate step sizes.
        callback_fn: A function to call at each iteration with the iteration number

    Returns:
        A GraphicalModel object with the estimated potentials and marginals.
    """
    loss_fn, known_total, potentials = _initialize(
        domain, loss_fn, known_total, potentials
    )

    if not stateful:
        stateless_oracle = marginal_oracle
        marginal_oracle = lambda theta, total, state: (stateless_oracle(theta, total), state)
    elif stepsize is None:
        raise ValueError('Stepsize should be manually tuned when using a stateful oracle.')

    @jax.jit
    def update_6(
        theta,
        alpha,
        state=None,
        iteration: int = 0,
        reg_loss_max: float = 0.0,
        reg_started: bool = False,
        lambda_reg_prev: float = 0.0,
        should_stop_prev: bool = False,
        prev_total_grad_norm: float = 1e9,
        # tolerances:
        grad_tol: float = 1.02,    # allow up to 2% increase
        reg_min_ratio: float = 0.2,
    ):
        # 1) Marginal step proposal
        mu, state = marginal_oracle(theta, known_total, state)
        loss_marg, dL_marg = jax.value_and_grad(loss_fn)(mu)
        marg_norm = jnp.sqrt(dL_marg.dot(dL_marg))
        dL_marg = dL_marg / marg_norm
        theta_marg = theta - alpha * dL_marg

        # 2) Regularizer pass on the half-step
        mu2, state = marginal_oracle(theta_marg, known_total, state)
        loss_reg, dL_reg = jax.value_and_grad(calculate_regularizer_loss, argnums=0)(
            theta_marg, known_total
        )

        # update reg-loss peak & check CMI stop
        new_reg_loss_max = jnp.maximum(reg_loss_max, loss_reg)
        new_reg_started  = reg_started | (loss_reg < new_reg_loss_max - 1e-8)
        stop_reg         = new_reg_started & (loss_reg < reg_min_ratio * new_reg_loss_max)

        # schedule lambda_reg
        lambda_reg_candidate = iteration / degree
        lambda_reg = jax.lax.select(
            should_stop_prev | stop_reg,
            lambda_reg_prev,
            lambda_reg_candidate
        )

        # 3) Compute total gradient norm at this proposal
        total_grad = dL_marg + lambda_reg * dL_reg
        total_grad_norm = jnp.sqrt(total_grad.dot(total_grad))

        # 4) Blowup check (only once we’ve entered Phase 2)
        blowup = new_reg_started & (total_grad_norm > grad_tol * prev_total_grad_norm)

        # 5) Final stop flag
        should_stop = should_stop_prev | stop_reg | blowup

        # 6) Conditionally perform the full update
        def do_full_update(_):
            return theta_marg - alpha * (lambda_reg * dL_reg)

        # If should_stop, return original theta; else apply both steps
        theta_next = jax.lax.cond(
            should_stop,
            lambda th: th,
            do_full_update,
            theta
        )

        return (
            theta_next,
            loss_marg,
            loss_reg,
            alpha,
            mu2,
            state,
            loss_marg + lambda_reg * loss_reg,
            total_grad_norm,
            marg_norm,
            jnp.sqrt(dL_reg.dot(dL_reg)),
            lambda_reg,
            new_reg_loss_max,
            new_reg_started,
            should_stop,
            total_grad_norm,   # for the next prev_total_grad_norm
        )

    alpha = 0.01
    lambda_reg = (1/2000)**2  # Start small
    prev_reg_loss = None
    prev_marginal_loss = None
    prev_marginal_grad = 1
    prev_reg_grad = 1
    prev_total_grad_norm = 1.0
    theta = potentials
    reg_loss_max = 0.0
    reg_started_decreasing = False
    stop_flag = False
    theta             = potentials
    reg_loss_max      = 0.0
    reg_started       = False
    plateau_count     = 0
    prev_reg_loss     = 0.0
    prev_lambda_reg = None

    mu, state = marginal_oracle(potentials, known_total, state=None)


    # ——— setup before the loop ———
    alpha                = 0.01
    theta                = potentials
    reg_loss_max         = 0.0
    reg_started          = False
    # lambda_reg_prev      = (1/2000)**2
    lambda_reg_prev      = 0.0
    should_stop          = False
    prev_total_grad_norm = 1.0

    mu, state = marginal_oracle(theta, known_total, state=None)

    x = None
    for t in range(iters):
        
        # call update_6 and unpack everything it returns
        (
            theta,
            loss_marg,         # current marginal loss
            loss_reg,          # current regularizer loss
            alpha,             # stepsize (unchanged here)
            mu,                # new marginals
            state,             # new oracle state
            total_loss,        # combined loss
            total_grad_norm,   # ‖∇ₘₐᵣg + λ∙∇ᵣₑg‖
            marg_norm,         # ‖∇ₘₐᵣg‖
            reg_norm,          # ‖∇ᵣₑg‖
            lambda_reg,        # updated λ
            reg_loss_max,      # tracked peak reg_loss
            reg_started,       # Boolean: have we entered phase‐2?
            should_stop,       # Boolean: CMI‐bound or grad‐blowup triggered
            prev_total_grad_norm,  # for next iteration’s blowup check
        ) = update_6(
            theta,
            alpha,
            state=state,
            iteration=t,
            reg_loss_max=reg_loss_max,
            reg_started=reg_started,
            lambda_reg_prev=lambda_reg_prev,
            should_stop_prev=should_stop,
            prev_total_grad_norm=prev_total_grad_norm,
            # grad_tol=1.02,         # e.g. 2% blowup tolerance
            grad_tol=1000,         # e.g. 40% blowup tolerance
            reg_min_ratio=cmi_ratio,
        )
        print(f"Iteration: {t} \nCurrent Marginal loss: {loss_marg} \nalpha value: {lambda_reg} \nCurrent Reg Loss: {loss_reg} \nTotal loss: {total_loss}")
        print(f"Total Grad: {total_grad_norm} \nMarginal Grad: {marg_norm} \nReg Grad: {reg_norm} \nlambda_reg: {lambda_reg}")
        print(f"status: {reg_started_decreasing}, reg_loss_max: {reg_loss_max}")

        callback_fn(mu)

        # if we ever latch `should_stop`, exit
        lambda_reg_prev = lambda_reg
        # lambda_reg_prev = 0.0
        if should_stop:
            should_stop = False
            if x == None:
                x = t
            pass

        # prepare for next step

    # ——— after the loop ———
    marginals, _ = marginal_oracle(theta, known_total, state)
    # print(f"[INFO] Final total_grad_norm: {total_grad_norm} | lambda_reg={lambda_reg}")
    return GraphicalModel(theta, marginals, known_total), \
        (loss_marg, loss_reg, total_grad_norm, lambda_reg)

def calculate_regularizer_loss(theta, total):
    """
    Compute the regularizer loss (e.g., Total Variation Distance).
    
    Parameters:
    - theta: JAX array for model parameters.
    - mu: Current marginals computed from `theta`.
    - constraint: A list defining (X, Y, Z) relationships.
    - total: Total number of samples.
    - method: 'CMI' or 'TVD'.
    - tv_method: 'L1' or 'L2' (for TV distance calculation).

    Returns:
    - A JAX scalar for the regularizer loss.
    """
    def marginal_inference(theta, attrs, total):
        """Compute marginal inference, using either `mu.project` or fallback."""
        return marginal_oracles.variable_elimination(theta, attrs, total)
    
    method, tv_method = 'CMI', 'infer' # You can change the method to other options, but don't change tv_method!!

    ## Adult
    # target_attr = 'income'
    # protected_attribute = 'sex'
    # protected_attributes = ['sex']
    # inadmissibles = ['marital-status']
    # admissibles = ['occupation', 'education-num', 'hours-per-week', 'age']
    # binary_columns = ['sex', 'income']

    ## Compas
    # target_attr = 'is-recid'
    # protected_attribute = 'race'
    # inadmissibles = ['age-cat', 'priors-count']
    # admissibles = ['c-charge-degree']
    # binary_columns = ['is-recid', 'race', 'c-charge-degree']

    ## Dutch
    # target_attr = 'occupation'
    # protected_attribute = 'sex'
    # admissibles = ['economic_status', 'household_position', 'household_size']
    # inadmissibles = ['edu_level', 'age', 'marital_status', 'country_birth', 'citizenship', 'prev_residence_place']

    ## Car
    # target_attr = 'class'
    # protected_attribute = 'doors'
    # admissibles = ['buying', 'maint', 'safety','persons']
    # binary_columns = []

    ### Car i
    # target_attr = 'class'
    # protected_attribute = 'doors'
    # inadmissibles = ['safety']
    # admissibles = ['buying', 'maint', 'persons', 'lug_boot']

    
    ## Boston
    # target_attr = 'MEDV'
    # protected_attribute = 'AGE'
    # # admissibles = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'RAD', 'TAX', 'PTRATIO', 'LSTAT'] 
    # admissibles = ['CRIM', 'B', 'CHAS', 'RM', 'ZN', 'PTRATIO']
    # admissibles = ['CRIM', 'CHAS', 'RM', 'LSTAT']
    # binary_columns = []


    ## how CI constraint should look like
    print(CONSTRAINT_DIR)
    target_attr = TARGET_ATTR
    protected_attribute = PROTECTED_ATTR
    inadmissibles = INADMISSIBLE_ATTRS
    admissibles = ADMISSIBLE_ATTRS
    # constraint = [[protected_attribute],inadmissibles , admissibles]
    # constraint = [[protected_attribute]+inadmissibles, [target_attr] , admissibles]
    constraint = [[protected_attribute], [target_attr] , admissibles]
    # constraint = [protected_attributes, [target_attr], admissibles]
    

    assert len(constraint) == 3, "constraint must have three elements (e.g., [X, Y, Z])"
    
    _count_XYZ = marginal_inference(theta, tuple([attr for proj in constraint for attr in proj]), total)
    # _count_Z_xyz = _count_XYZ.project(constraint[-1], total)
    _count_Z = marginal_inference(theta, constraint[-1], total)
    _count_XZ = marginal_inference(theta, tuple(constraint[0] + constraint[-1]), total)
    # _count_XZ_xyz = _count_XYZ.project(tuple(constraint[0] + constraint[-1]), total)
    _count_YZ = marginal_inference(theta, tuple(constraint[1] + constraint[-1]), total)
    # _count_YZ_xyz = _count_XYZ.project(tuple(constraint[1] + constraint[-1]), total)

    _count_XY = marginal_inference(theta, tuple(constraint[0] + constraint[1]), total)
    _count_X = marginal_inference(theta, constraint[0], total)
    _count_Y = marginal_inference(theta, constraint[1], total)


    # Convert to JAX arrays
    prob_Z = None
    prob_XZ = None
    prob_YZ = None
    prob_XYZ = None
    prob_XY = None
    prob_X = None
    prob_Y = None
    select_infer = True
    # if tv_method in ['infer', 'L1', 'L2']:
    if select_infer:
        prob_Z = jnp.array(_count_Z.dataprobs(False), dtype=jnp.float64)
        prob_XZ = jnp.array(_count_XZ.dataprobs(False), dtype=jnp.float64)
        prob_YZ = jnp.array(_count_YZ.dataprobs(False), dtype=jnp.float64)
        prob_XYZ = jnp.array(_count_XYZ.dataprobs(False), dtype=jnp.float64)
        prob_XY = jnp.array(_count_XY.dataprobs(False), dtype=jnp.float64)
        prob_X = jnp.array(_count_X.dataprobs(False), dtype=jnp.float64)
        prob_Y = jnp.array(_count_Y.dataprobs(False), dtype=jnp.float64)

    else:
        prob_Z = jnp.array(_count_Z_xyz.dataprobs(False), dtype=jnp.float64)
        prob_XZ = jnp.array(_count_XZ_xyz.dataprobs(False), dtype=jnp.float64)
         
        prob_YZ = jnp.array(_count_YZ_xyz.dataprobs(False), dtype=jnp.float64)
        prob_XYZ = jnp.array(_count_XYZ.dataprobs(False), dtype=jnp.float64)

    # Ensure shapes match
    prob_Z_b, prob_XZ_b, prob_YZ_b = align_shapes(prob_Z, prob_XZ, prob_YZ, prob_XYZ)
    prob_X_b, prob_Y_b = align_shapes_3(prob_X, prob_Y, prob_XY)


    # jax.debug.print("method {}", method)
    if method == 'MI':
        return calculate_mutual_information_loss(prob_XY, prob_X_b, prob_Y_b)
    if method == 'CMI':
        return calculate_conditional_mutual_information(prob_XYZ, prob_XZ_b, prob_YZ_b, prob_Z_b)
    elif method == 'TVD':
        # Compute TVD loss
        return calculate_total_variation_distance(prob_XYZ, prob_XZ_b, prob_YZ_b, prob_Z_b, method=tv_method)
    elif method == 'HLD':
        return calculate_hellinger_distance(prob_XYZ, prob_XZ_b, prob_YZ_b, prob_Z_b)
    else:
        raise ValueError("Method must be either 'CMI' or 'TVD'.")

def calculate_mutual_information_loss(prob_XY, prob_X_b, prob_Y_b):
    """
    Calculate mutual information loss using JAX.
    
    Parameters:
    - prob_XY: JAX array for P(X, Y).
    - prob_X_b: JAX array for P(X).
    - prob_Y_b: JAX array for P(Y).

    Returns:
    - JAX scalar: Mutual information loss.
    """

    prob_X_reshaped = prob_X_b.reshape(-1, 1)    # shape (n_x, 1)
    prob_Y_reshaped = prob_Y_b.reshape(1, -1)    # shape (1, n_y)

    # Compute mutual information values
    mi_values = prob_XY * jnp.log2(
        (prob_XY + 1e-12) / (prob_X_reshaped * prob_Y_reshaped + 1e-12)
    )
    # Sum over all values
    mi = jnp.sum(mi_values)
    
    return mi  # Returns a JAX scalar, which works with JIT & gradients

def calculate_conditional_mutual_information(prob_XYZ, prob_XZ_b, prob_YZ_b, prob_Z_b):
    """
    Calculate conditional mutual information using JAX.
    
    Parameters:
    - prob_XYZ: JAX array for P(X, Y, Z).
    - prob_XZ_b: JAX array for P(X, Z).
    - prob_YZ_b: JAX array for P(Y, Z).
    - prob_Z_b: JAX array for P(Z).

    Returns:
    - JAX scalar: Conditional mutual information (CMI).
    """
    # Compute conditional mutual information values
    cmi_values = prob_XYZ * jnp.log2(
        (prob_XYZ * prob_Z_b + 1e-12) / (prob_XZ_b * prob_YZ_b + 1e-12)
    )
    # Sum over all values
    cmi = jnp.sum(cmi_values)
    # Ensure non-negative CMI using `jax.lax.max()`
    # regularizer_loss = jnp.maximum(cmi, 0.0) 
    regularizer_loss = cmi 

    return regularizer_loss  # Returns a JAX scalar, which works with JIT & gradients


def calculate_total_variation_distance(prob_XYZ, prob_XZ_b, prob_YZ_b, prob_Z_b, method='L1'):
    """
    Calculate the Total Variation (TV) distance between two distributions.

    Parameters:
    - prob_XYZ: JAX array for P_XYZ.
    - prob_XZ_b: JAX array for P_XZ.
    - prob_YZ_b: JAX array for P(Y|Z).
    - prob_Z_b: JAX array for P_Z.
    - method: 'L1' or 'L2'.

    Returns:
    - A JAX scalar for the TV distance.
    """
    # Avoid division by zero
    denom = jnp.clip(prob_Z_b, 1e-10, None)
    P_X_given_Z = prob_XZ_b / denom
    diff = (prob_XYZ) - (P_X_given_Z * prob_YZ_b)


    if method == 'L1':
        tv_distance = 0.5 * jnp.sum(jnp.abs(diff))
    elif method == 'L2':
        tv_distance = jnp.sqrt(jnp.sum(diff ** 2) + 1e-12)
    else:
        raise ValueError("Method must be either 'L1' or 'L2'.")

    return tv_distance


def calculate_hellinger_distance(prob_XYZ, prob_XZ_b, prob_YZ_b, prob_Z_b):
    """
    Compute the Hellinger distance between P(X,Y|Z) and P(X|Z)P(Y|Z) with JAX.

    Parameters:
    - prob_XYZ: JAX array for P(X, Y, Z)
    - prob_XZ_b: JAX array for P(X, Z)
    - prob_YZ_b: JAX array for P(Y, Z)
    - prob_Z_b: JAX array for P(Z)

    Returns:
    - Hellinger distance (JAX scalar)
    """
    # Prevent division by zero
    prob_Z_b = jnp.clip(prob_Z_b, 1e-10, None)  # Ensures stability

    # Compute conditional probabilities safely
    P_XY_Z = jnp.clip(prob_XYZ, 1e-10, None)
    P_X_Z = jnp.clip(prob_XZ_b, 1e-10, None)
    P_Y_Z = jnp.clip(prob_YZ_b / prob_Z_b, 1e-10, None)
    P_product = P_Y_Z * P_X_Z

    # Compute the Hellinger distance
    sqrt_diff = jnp.sqrt(P_XY_Z) - jnp.sqrt(P_product)
    hellinger_distance = jnp.sqrt(jnp.sum(sqrt_diff ** 2) + 1e-12) / jnp.sqrt(2)

    return hellinger_distance


def align_shapes(prob_Z, prob_XZ, prob_YZ, prob_XYZ):
    """
    Align the shapes of prob_Z, prob_XZ, and prob_YZ to match prob_XYZ for broadcasting.

    Parameters:
    - prob_Z: JAX array representing P(Z).
    - prob_XZ: JAX array representing P(X, Z).
    - prob_YZ: JAX array representing P(Y, Z).
    - prob_XYZ: JAX array representing P(X, Y, Z).

    Returns:
    - Aligned versions of prob_Z, prob_XZ, and prob_YZ for broadcasting with prob_XYZ.
    """

    shape_Z = prob_Z.shape
    shape_XZ = prob_XZ.shape
    shape_YZ = prob_YZ.shape
    shape_XYZ = prob_XYZ.shape

    # Determine the number of extra dimensions each probability distribution needs
    num_X = len(shape_XYZ) - len(shape_YZ)  # X dimensions
    num_Y = len(shape_YZ) - len(shape_Z)    # Y dimensions
    num_Z = len(shape_Z)                    # Z dimensions

    # Create new shapes for broadcasting
    new_shape_Z = (1,) * num_X + (1,) * num_Y + shape_Z  # Broadcast Z over X and Y
    new_shape_XZ = shape_XZ[:num_X] + (1,) * num_Y + shape_XZ[num_X:]  # Broadcast XZ over Y
    new_shape_YZ = (1,) * num_X + shape_YZ[:num_Y] + shape_YZ[num_Y:]  # Broadcast YZ over X

    # Reshape using JAX (no in-place modification)
    aligned_prob_Z = prob_Z.reshape(new_shape_Z)
    aligned_prob_XZ = prob_XZ.reshape(new_shape_XZ)
    aligned_prob_YZ = prob_YZ.reshape(new_shape_YZ)

    return aligned_prob_Z, aligned_prob_XZ, aligned_prob_YZ

def align_shapes_3(prob_X, prob_Y, prob_XY):
    """
    Align the shapes of prob_X and prob_Y to broadcast with prob_XY.

    Parameters:
    - prob_X: array with shape corresponding to marginal P(X)
    - prob_Y: array with shape corresponding to marginal P(Y)
    - prob_XY: array with shape corresponding to joint P(X, Y)

    Returns:
    - aligned_X: reshaped P(X) with singleton dimensions for broadcasting
    - aligned_Y: reshaped P(Y) with singleton dimensions for broadcasting
    """
    shape_X = prob_X.shape
    shape_Y = prob_Y.shape
    shape_XY = prob_XY.shape

    len_X = len(shape_X)
    len_Y = len(shape_Y)
    len_XY = len(shape_XY)

    num_X = len_X
    num_Y = len_Y

    # Assume prob_X has shape (d1, d2, ..., dx)
    # and prob_Y has shape (d1', d2', ..., dy')
    # and prob_XY has shape (d1, d2, ..., dx, d1', ..., dy')

    aligned_X = prob_X.reshape(shape_X + (1,) * num_Y)
    aligned_Y = prob_Y.reshape((1,) * num_X + shape_Y)

    return aligned_X, aligned_Y
