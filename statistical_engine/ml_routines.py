import warnings

import numpy as np
import scipy.optimize as opt
from scipy.special import expit
from typing import Tuple, List, Optional


class FastBetaModel:
    """Optimized Beta Model implementation with fast likelihood computation"""

    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.in_params = np.zeros(n_nodes)
        self.out_params = np.zeros(n_nodes)

    def log_likelihood(self, adjacency_matrix: np.ndarray,
                       in_params: np.ndarray, out_params: np.ndarray) -> float:
        """Directed Bernoulli likelihood; theta_in indexes targets (columns)."""
        logits = out_params[:, None] + in_params[None, :]
        mask = ~np.eye(self.n_nodes, dtype=bool)
        y = adjacency_matrix[mask]
        z = logits[mask]
        return float(np.sum(y * z - np.logaddexp(0, z)))

    def _compute_prob_matrix(self, in_params: np.ndarray, out_params: np.ndarray) -> np.ndarray:
        probabilities = expit(out_params[:, None] + in_params[None, :])
        np.fill_diagonal(probabilities, 0.)
        return probabilities

    def compute_score(self, adjacency_matrix: np.ndarray,
                      in_params: np.ndarray, out_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        residual = adjacency_matrix - self._compute_prob_matrix(in_params, out_params)
        residual = residual.copy()
        np.fill_diagonal(residual, 0.)
        return residual.sum(axis=0), residual.sum(axis=1)

    def _mle_objective(self, params: np.ndarray, adjacency_matrix: np.ndarray) -> float:
        """Objective function for MLE optimization"""
        n_half = len(params) // 2
        in_p = params[:n_half]
        out_p = params[n_half:]
        return -self.log_likelihood(adjacency_matrix, in_p, out_p)

    def mle_estimate(self, adjacency_matrix: np.ndarray,
                     initial_params: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                     maxiter: int = 1000, gtol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Maximum likelihood estimation using optimized routines"""
        if initial_params is None:
            # Initialize with degree-based estimates
            in_degrees = np.sum(adjacency_matrix, axis=0)
            out_degrees = np.sum(adjacency_matrix, axis=1)
            in_params_init = np.log(in_degrees + 1) - np.mean(np.log(in_degrees + 1))
            out_params_init = np.log(out_degrees + 1) - np.mean(np.log(out_degrees + 1))
        else:
            in_params_init, out_params_init = initial_params

        # Fix the last incoming coefficient to zero. The transformation leaves
        # every edge logit unchanged and removes the additive identification ridge.
        offset = in_params_init[-1]
        params_init = np.r_[in_params_init[:-1] - offset, out_params_init + offset]

        def unpack(params):
            return np.r_[params[:self.n_nodes - 1], 0.], params[self.n_nodes - 1:]

        result = opt.minimize(
            lambda x: -self.log_likelihood(adjacency_matrix, *unpack(x)),
            params_init, method='BFGS', options={'maxiter': maxiter, 'gtol': gtol}
        )
        if not result.success:
            warnings.warn(f"MLE optimization did not converge: {result.message}")
        return unpack(result.x)


class FastPseudoLikelihood:
    """
    Optimized pseudo-likelihood computation for general ERGMs, with
    drop-in compatibility to the FastBetaModel API.

    Public API (keeps names as they are):
      - log_pseudo_likelihood(adjacency_matrix, theta)
      - compute_pseudo_score(adjacency_matrix, theta)

    Beta-model-compatible aliases (same signatures as FastBetaModel):
      - log_likelihood(adjacency_matrix, in_params, out_params)
      - compute_score(adjacency_matrix, in_params, out_params)
      - mle_estimate(adjacency_matrix, initial_params=None)
    """

    def __init__(self, statistics: Optional[List] = None, n_nodes: Optional[int] = None):
        """
        statistics : list of NetworkStatistic objects used for ERGM pseudo-likelihood (optional).
        n_nodes : required if you plan to use the beta-model-compatible methods (in/out params).
        """
        self.statistics = statistics or []
        self.n_stats = len(self.statistics)
        self.n_nodes = n_nodes

    # -------------------------
    # ERGM pseudo-likelihood API
    # -------------------------
    def log_pseudo_likelihood(self, adjacency_matrix: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute pseudo log-likelihood for general ERGM:
          theta: parameter vector of length n_stats
        """
        n = adjacency_matrix.shape[0]
        if theta.shape[0] != self.n_stats:
            # allow theta length 2*n_nodes as a fallback (beta-like), but prefer explicit call
            if self.n_stats == 0:
                raise ValueError("No statistics provided and theta length doesn't match.")
            else:
                raise ValueError("theta length must equal number of statistics (self.n_stats).")

        log_pl = 0.0
        eps = 1e-10
        # iterate over ordered pairs (i != j)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # compute change statistics vector for the dyad (i,j)
                change_stats = np.array([(1.0 - 2.0 * adjacency_matrix[i, j]) * stat.change_statistic(adjacency_matrix, i, j) for stat in self.statistics])
                linear_pred = float(np.dot(theta, change_stats))
                p = expit(linear_pred)
                log_pl += adjacency_matrix[i, j] * linear_pred - np.logaddexp(0, linear_pred)
        return float(log_pl)

    def compute_pseudo_score(self, adjacency_matrix: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Pseudo-score (gradient) for general ERGM: returns a vector of length n_stats.
        """
        n = adjacency_matrix.shape[0]
        if theta.shape[0] != self.n_stats:
            raise ValueError("theta length must equal number of statistics (self.n_stats).")

        score = np.zeros(self.n_stats)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                change_stats = np.array([(1.0 - 2.0 * adjacency_matrix[i, j]) * stat.change_statistic(adjacency_matrix, i, j) for stat in self.statistics])
                linear_pred = float(np.dot(theta, change_stats))
                p = expit(linear_pred)
                score += change_stats * (adjacency_matrix[i, j] - p)
        return score

    # ------------------------------------------------
    # Beta-model-compatible API (drop-in compatibility)
    # ------------------------------------------------
    def _compute_prob_matrix_beta(self, in_params: np.ndarray, out_params: np.ndarray) -> np.ndarray:
        """Vectorized probability matrix for Beta model (sigmoid(in_i + out_j)), zero diagonal."""
        if self.n_nodes is None:
            self.n_nodes = in_params.shape[0]
        n = self.n_nodes
        L = out_params.reshape((n, 1)) + in_params.reshape((1, n))
        P = expit(L)
        np.fill_diagonal(P, 0.0)
        return P

    def log_likelihood(self, adjacency_matrix: np.ndarray,
                       in_params: np.ndarray, out_params: np.ndarray) -> float:
        """
        Alias to compute a pseudo-likelihood using in/out parameters (same signature as FastBetaModel.log_likelihood).
        """
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("adjacency_matrix must be square")
        n = adjacency_matrix.shape[0]
        if in_params.shape[0] != n or out_params.shape[0] != n:
            raise ValueError("in_params and out_params must each have length n_nodes")

        return FastBetaModel(n).log_likelihood(adjacency_matrix, in_params, out_params)

    def compute_score(self, adjacency_matrix: np.ndarray,
                      in_params: np.ndarray, out_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alias to compute pseudo-score (in_score, out_score) with same semantics as FastBetaModel.compute_score.
        in_score[i] = sum_j (Y[j,i] - p[j,i]), out_score[j] = sum_i (Y[i,j] - p[i,j])
        """
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("adjacency_matrix must be square")
        n = adjacency_matrix.shape[0]
        if in_params.shape[0] != n or out_params.shape[0] != n:
            raise ValueError("in_params and out_params must each have length n_nodes")

        P = self._compute_prob_matrix_beta(in_params, out_params)
        mask = ~np.eye(n, dtype=bool)
        residuals = (adjacency_matrix - P) * mask  # zero diagonal
        # out_score: sum over columns for each row (outgoing from i) -> sum over j adjacency[i,j]-p[i,j]
        out_score = np.sum(residuals, axis=1)
        # in_score: sum over rows for each column (incoming to j)
        in_score = np.sum(residuals, axis=0)
        return in_score, out_score

    def _mle_objective_beta(self, params: np.ndarray, adjacency_matrix: np.ndarray) -> float:
        n_half = len(params) // 2
        in_p = params[:n_half]
        out_p = params[n_half:]
        return -self.log_likelihood(adjacency_matrix, in_p, out_p)

    def mle_estimate(self, adjacency_matrix: np.ndarray,
                     initial_params: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                     maxiter: int = 1000, gtol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits parameters by maximizing the pseudo-likelihood in the Beta-model parameterization
        (returns in_params, out_params). This is a drop-in for FastBetaModel.mle_estimate.
        If you prefer to estimate a general ERGM theta, use scipy.optimize on log_pseudo_likelihood directly.
        """
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("adjacency_matrix must be square")
        n = adjacency_matrix.shape[0]
        self.n_nodes = n

        if initial_params is None:
            in_degrees = np.sum(adjacency_matrix, axis=0)
            out_degrees = np.sum(adjacency_matrix, axis=1)
            in_init = np.log(in_degrees + 1) - np.mean(np.log(in_degrees + 1))
            out_init = np.log(out_degrees + 1) - np.mean(np.log(out_degrees + 1))
        else:
            in_init, out_init = initial_params
            if in_init.shape[0] != n or out_init.shape[0] != n:
                raise ValueError("initial_params must be two arrays of length n_nodes")

        return FastBetaModel(n).mle_estimate(adjacency_matrix, (in_init, out_init), maxiter=maxiter, gtol=gtol)
