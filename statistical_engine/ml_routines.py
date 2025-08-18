import warnings

import numpy as np
from typing import Tuple, List, Optional
from scipy.optimize import minimize
from data_handler.data_structures import NetworkStatistic


class FastBetaModel:
    """Optimized Beta Model implementation with fast likelihood computation"""

    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.in_params = np.zeros(n_nodes)
        self.out_params = np.zeros(n_nodes)

    def log_likelihood(self, adjacency_matrix: np.ndarray,
                       in_params: np.ndarray, out_params: np.ndarray) -> float:
        """Fast log-likelihood computation for beta model"""
        # Apply identification constraint: sum of in_params = sum of out_params
        in_params = in_params - np.mean(in_params) + np.mean(out_params)

        # Compute probability matrix
        prob_matrix = self._compute_prob_matrix(in_params, out_params)

        # Log-likelihood computation
        log_lik = 0.0
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:  # No self-loops
                    if adjacency_matrix[i, j] == 1:
                        log_lik += np.log(prob_matrix[i, j] + 1e-10)
                    else:
                        log_lik += np.log(1 - prob_matrix[i, j] + 1e-10)

        return log_lik

    def _compute_prob_matrix(self, in_params: np.ndarray, out_params: np.ndarray) -> np.ndarray:
        """Compute edge probability matrix"""
        prob_matrix = np.zeros((self.n_nodes, self.n_nodes))

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    linear_predictor = in_params[i] + out_params[j]
                    prob_matrix[i, j] = 1.0 / (1.0 + np.exp(-linear_predictor))

        return prob_matrix

    def compute_score(self, adjacency_matrix: np.ndarray,
                      in_params: np.ndarray, out_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute score (gradient) of log-likelihood"""
        prob_matrix = self._compute_prob_matrix(in_params, out_params)

        in_score = np.zeros(self.n_nodes)
        out_score = np.zeros(self.n_nodes)

        # Compute in-degree scores
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    in_score[i] += adjacency_matrix[j, i] - prob_matrix[j, i]

        # Compute out-degree scores
        for j in range(self.n_nodes):
            for i in range(self.n_nodes):
                if i != j:
                    out_score[j] += adjacency_matrix[i, j] - prob_matrix[i, j]

        return in_score, out_score

    def _mle_objective(self, params: np.ndarray, adjacency_matrix: np.ndarray) -> float:
        """Objective function for MLE optimization"""
        n_half = len(params) // 2
        in_p = params[:n_half]
        out_p = params[n_half:]
        return -self.log_likelihood(adjacency_matrix, in_p, out_p)

    def mle_estimate(self, adjacency_matrix: np.ndarray,
                     initial_params: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Maximum likelihood estimation using optimized routines"""
        if initial_params is None:
            # Initialize with degree-based estimates
            in_degrees = np.sum(adjacency_matrix, axis=0)
            out_degrees = np.sum(adjacency_matrix, axis=1)
            in_params_init = np.log(in_degrees + 1) - np.mean(np.log(in_degrees + 1))
            out_params_init = np.log(out_degrees + 1) - np.mean(np.log(out_degrees + 1))
        else:
            in_params_init, out_params_init = initial_params

        # Combine parameters for optimization
        params_init = np.concatenate([in_params_init, out_params_init])

        # Optimize using BFGS
        result = opt.minimize(
            self._mle_objective,
            params_init,
            args=(adjacency_matrix,),
            method='BFGS',
            options={'maxiter': 1000, 'gtol': 1e-6}
        )

        if not result.success:
            warnings.warn("MLE optimization did not converge")

        n_half = len(result.x) // 2
        return result.x[:n_half], result.x[n_half:]


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

    def __init__(self, statistics: Optional[List[NetworkStatistic]] = None, n_nodes: Optional[int] = None):
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
                change_stats = np.array([stat.change_statistic(adjacency_matrix, i, j) for stat in self.statistics])
                linear_pred = float(np.dot(theta, change_stats))
                p = 1.0 / (1.0 + np.exp(-linear_pred))
                if adjacency_matrix[i, j] == 1:
                    log_pl += np.log(p + eps)
                else:
                    log_pl += np.log(1.0 - p + eps)
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
                change_stats = np.array([stat.change_statistic(adjacency_matrix, i, j) for stat in self.statistics])
                linear_pred = float(np.dot(theta, change_stats))
                p = 1.0 / (1.0 + np.exp(-linear_pred))
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
        L = in_params.reshape((n, 1)) + out_params.reshape((1, n))
        P = 1.0 / (1.0 + np.exp(-L))
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

        # compute probabilities and vectorized log-likelihood
        P = self._compute_prob_matrix_beta(in_params, out_params)
        mask = ~np.eye(n, dtype=bool)
        eps = 1e-10
        adj_vals = adjacency_matrix[mask]
        prob_vals = P[mask]
        log_lik = np.sum(adj_vals * np.log(prob_vals + eps) + (1 - adj_vals) * np.log(1 - prob_vals + eps))
        return float(log_lik)

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

        params_init = np.concatenate([in_init, out_init])

        result = opt.minimize(
            self._mle_objective_beta,
            params_init,
            args=(adjacency_matrix,),
            method='BFGS',
            options={'maxiter': maxiter, 'gtol': gtol}
        )

        if not result.success:
            warnings.warn(f"MLE optimization did not converge: {result.message}")

        n_half = len(result.x) // 2
        in_hat = result.x[:n_half]
        out_hat = result.x[n_half:]
        return in_hat, out_hat