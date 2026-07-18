"""From-scratch Gaussian-process regression for sequential experiment design.

Two pieces:

* :class:`GP`, a plain exact GP with RBF or Matern-3/2 kernel and
  marginal-likelihood hyperparameter fitting (Cholesky solves, L-BFGS-B over
  log-parameters).
* :class:`SequentialGP`, the same posterior maintained incrementally over a
  fixed candidate set while measurements arrive one at a time. Adding one
  observation is O(candidates x n) via the rank-one posterior update, so a
  full active-scanning run stays cheap; the incremental state is unit-tested
  against the batch posterior.

Everything is written against normalised coordinates in [0, 1]^2; the
lengthscale is in the same units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

KERNELS = ("rbf", "matern32")

_JITTER = 1e-10
_VAR_FLOOR = 1e-12


@dataclass(frozen=True)
class GPHyperparams:
    """Kernel and noise hyperparameters.

    Attributes:
        lengthscale: Correlation length in normalised coordinate units.
        signal_var: Kernel variance (prior variance of the latent field).
        noise_var: Observation noise variance.
    """

    lengthscale: float
    signal_var: float
    noise_var: float


def kernel_matrix(
    kind: str, x1: np.ndarray, x2: np.ndarray, lengthscale: float, signal_var: float
) -> np.ndarray:
    """Evaluate the covariance matrix between two coordinate sets.

    Args:
        kind: "rbf" or "matern32".
        x1: (n1, 2) coordinates.
        x2: (n2, 2) coordinates.
        lengthscale: Kernel lengthscale.
        signal_var: Kernel variance.

    Returns:
        (n1, n2) covariance matrix.
    """
    if kind not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kind!r}")
    d = cdist(np.atleast_2d(x1), np.atleast_2d(x2))
    if kind == "rbf":
        return signal_var * np.exp(-0.5 * (d / lengthscale) ** 2)
    s = np.sqrt(3.0) * d / lengthscale
    return signal_var * (1.0 + s) * np.exp(-s)


def _log_marginal_likelihood(
    kind: str, x: np.ndarray, y: np.ndarray, log_params: np.ndarray
) -> float:
    ls, sv, nv = np.exp(log_params)
    k = kernel_matrix(kind, x, x, ls, sv)
    k[np.diag_indices_from(k)] += nv + _JITTER
    try:
        low = cho_factor(k, lower=True)
    except np.linalg.LinAlgError:
        return -1e12
    alpha = cho_solve(low, y)
    log_det = 2.0 * np.sum(np.log(np.diag(low[0])))
    n = len(y)
    return float(-0.5 * y @ alpha - 0.5 * log_det - 0.5 * n * np.log(2.0 * np.pi))


class GP:
    """Exact GP regression with a constant (empirical-mean) prior mean.

    Args:
        kernel: "rbf" or "matern32".
        hyper: Hyperparameters; if None they must be fitted before predicting.
    """

    def __init__(self, kernel: str = "rbf", hyper: GPHyperparams | None = None) -> None:
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
        self.kernel = kernel
        self.hyper = hyper
        self._x: np.ndarray | None = None
        self._resid: np.ndarray | None = None
        self._mean: float = 0.0
        self._low = None

    def fit_hyperparams(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_restarts: int = 2,
        lengthscale_bounds: tuple[float, float] = (0.01, 1.0),
        fixed_lengthscale: float | None = None,
    ) -> GPHyperparams:
        """Fit hyperparameters by maximising the log marginal likelihood.

        Args:
            x: (n, 2) observed coordinates.
            y: (n,) observed values.
            n_restarts: Extra random restarts beyond the default start.
            lengthscale_bounds: Box bounds for the lengthscale.
            fixed_lengthscale: If given, the lengthscale is clamped to this
                value and only signal and noise variances are fitted (used
                for the surrogate-misspecification benchmarks).

        Returns:
            The fitted hyperparameters (also stored on the instance).
        """
        y0 = y - y.mean()
        y_var = max(float(y0.var()), 1e-6)
        rng = np.random.default_rng(0)
        if fixed_lengthscale is not None:
            ls_bounds = (fixed_lengthscale, fixed_lengthscale)
            ls_starts = [fixed_lengthscale] * (1 + n_restarts)
        else:
            ls_bounds = lengthscale_bounds
            ls_starts = [0.1] + list(
                np.exp(rng.uniform(np.log(ls_bounds[0]), np.log(ls_bounds[1]), size=n_restarts))
            )
        bounds = [np.log(ls_bounds), np.log((1e-3 * y_var, 1e3 * y_var)), np.log((1e-8, 10.0))]
        best_val, best_p = -np.inf, None
        for ls0 in ls_starts:
            p0 = np.log([ls0, y_var, 0.1 * y_var])
            p0 = np.clip(p0, [b[0] for b in bounds], [b[1] for b in bounds])
            res = minimize(
                lambda p: -_log_marginal_likelihood(self.kernel, x, y0, p),
                p0,
                method="L-BFGS-B",
                bounds=bounds,
            )
            if -res.fun > best_val:
                best_val, best_p = -res.fun, res.x
        if best_p is None:
            raise RuntimeError("hyperparameter optimisation failed for every start")
        ls, sv, nv = np.exp(best_p)
        self.hyper = GPHyperparams(float(ls), float(sv), float(nv))
        return self.hyper

    def fit(self, x: np.ndarray, y: np.ndarray) -> GP:
        """Condition the posterior on observations (hyperparameters fixed).

        Args:
            x: (n, 2) observed coordinates.
            y: (n,) observed values.

        Returns:
            self, for chaining.
        """
        if self.hyper is None:
            raise RuntimeError("set or fit hyperparameters before calling fit()")
        self._x = np.atleast_2d(np.asarray(x, dtype=float))
        self._mean = float(np.mean(y))
        self._resid = np.asarray(y, dtype=float) - self._mean
        k = kernel_matrix(
            self.kernel, self._x, self._x, self.hyper.lengthscale, self.hyper.signal_var
        )
        k[np.diag_indices_from(k)] += self.hyper.noise_var + _JITTER
        self._low = cho_factor(k, lower=True)
        return self

    def predict(self, x_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and variance at query coordinates.

        Args:
            x_star: (m, 2) query coordinates.

        Returns:
            Tuple (mean, variance), each of shape (m,). Variance is the
            latent-function variance (no observation noise added).
        """
        if self._x is None or self._low is None or self.hyper is None:
            raise RuntimeError("call fit() before predict()")
        k_star = kernel_matrix(
            self.kernel, x_star, self._x, self.hyper.lengthscale, self.hyper.signal_var
        )
        alpha = cho_solve(self._low, self._resid)
        mean = self._mean + k_star @ alpha
        v = solve_triangular(self._low[0], k_star.T, lower=True)
        var = self.hyper.signal_var - np.einsum("ij,ij->j", v, v)
        return mean, np.maximum(var, _VAR_FLOOR)


class SequentialGP:
    """GP posterior over a fixed candidate set, updated one point at a time.

    The class caches the cross-covariance between candidates and observed
    points, so each :meth:`add` costs O(candidates x n) instead of a full
    refit, and :meth:`refit` re-optimises hyperparameters from scratch on
    the accumulated data.

    Args:
        candidates: (c, 2) coordinates of every allowed probe position.
        kernel: Kernel name shared with :class:`GP`.
        hyper: Initial hyperparameters.
    """

    def __init__(self, candidates: np.ndarray, kernel: str, hyper: GPHyperparams) -> None:
        self.candidates = np.atleast_2d(np.asarray(candidates, dtype=float))
        self.kernel = kernel
        self.hyper = hyper
        self.obs_idx: list[int] = []
        self.obs_y: list[float] = []
        self._rebuild()

    @property
    def n_obs(self) -> int:
        return len(self.obs_idx)

    def _rebuild(self) -> None:
        """Recompute the full posterior state from the stored observations."""
        c = len(self.candidates)
        self._mean_const = float(np.mean(self.obs_y)) if self.obs_y else 0.0
        if not self.obs_idx:
            self.mu = np.full(c, self._mean_const)
            self.var = np.full(c, self.hyper.signal_var)
            self._chol = np.zeros((0, 0))
            self._kcx = np.zeros((c, 0))
            self._resid = np.zeros(0)
            return
        x = self.candidates[self.obs_idx]
        self._resid = np.asarray(self.obs_y) - self._mean_const
        k = kernel_matrix(self.kernel, x, x, self.hyper.lengthscale, self.hyper.signal_var)
        k[np.diag_indices_from(k)] += self.hyper.noise_var + _JITTER
        self._chol = np.linalg.cholesky(k)
        self._kcx = kernel_matrix(
            self.kernel, self.candidates, x, self.hyper.lengthscale, self.hyper.signal_var
        )
        w = solve_triangular(self._chol, self._kcx.T, lower=True)
        alpha = solve_triangular(
            self._chol.T, solve_triangular(self._chol, self._resid, lower=True)
        )
        self.mu = self._mean_const + self._kcx @ alpha
        self.var = np.maximum(self.hyper.signal_var - np.einsum("ij,ij->j", w, w), _VAR_FLOOR)

    def add(self, index: int, value: float) -> None:
        """Incorporate one new measurement at a candidate position.

        Args:
            index: Flat index into the candidate set.
            value: Measured value at that position.
        """
        x_new = self.candidates[index : index + 1]
        n = self.n_obs
        resid_new = value - self._mean_const
        if n == 0:
            self.obs_idx.append(int(index))
            self.obs_y.append(float(value))
            # keep the zero prior mean fixed until the first refit; the
            # posterior update below handles the offset through the residual
            k_cand = self._kernel_to(x_new)
            denom = self.hyper.signal_var + self.hyper.noise_var + _JITTER
            self.mu = self.mu + k_cand * (resid_new / denom)
            self.var = np.maximum(self.var - k_cand**2 / denom, _VAR_FLOOR)
            self._chol = np.array([[np.sqrt(denom)]])
            self._kcx = k_cand[:, None]
            self._resid = np.array([resid_new])
            return
        k_obs = kernel_matrix(
            self.kernel,
            self.candidates[self.obs_idx],
            x_new,
            self.hyper.lengthscale,
            self.hyper.signal_var,
        ).ravel()
        w = solve_triangular(self._chol, k_obs, lower=True)
        beta = solve_triangular(self._chol.T, w)
        k_cand = self._kernel_to(x_new)
        k_post = k_cand - self._kcx @ beta
        var_at_new = self.var[index]
        denom = var_at_new + self.hyper.noise_var + _JITTER
        gain = (resid_new - (self.mu[index] - self._mean_const)) / denom
        self.mu = self.mu + k_post * gain
        self.var = np.maximum(self.var - k_post**2 / denom, _VAR_FLOOR)
        d = np.sqrt(max(self.hyper.signal_var + self.hyper.noise_var + _JITTER - w @ w, _JITTER))
        chol_new = np.zeros((n + 1, n + 1))
        chol_new[:n, :n] = self._chol
        chol_new[n, :n] = w
        chol_new[n, n] = d
        self._chol = chol_new
        self._kcx = np.concatenate([self._kcx, k_cand[:, None]], axis=1)
        self._resid = np.append(self._resid, resid_new)
        self.obs_idx.append(int(index))
        self.obs_y.append(float(value))

    def _kernel_to(self, x_new: np.ndarray) -> np.ndarray:
        return kernel_matrix(
            self.kernel, self.candidates, x_new, self.hyper.lengthscale, self.hyper.signal_var
        ).ravel()

    def refit(self, fixed_lengthscale: float | None = None) -> GPHyperparams:
        """Re-optimise hyperparameters on the data so far and rebuild.

        Args:
            fixed_lengthscale: Forwarded to :meth:`GP.fit_hyperparams`; used
                by the misspecification benchmarks to pin the lengthscale.

        Returns:
            The new hyperparameters.
        """
        gp = GP(self.kernel)
        x = self.candidates[self.obs_idx]
        y = np.asarray(self.obs_y)
        self.hyper = gp.fit_hyperparams(x, y, fixed_lengthscale=fixed_lengthscale)
        self._rebuild()
        return self.hyper
