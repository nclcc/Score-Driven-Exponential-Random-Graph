import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from abc import ABC, abstractmethod


# =============================================================================
# DATA GENERATION PROCESSES
# =============================================================================
from statistical_engine.score_random_network import ScoreDrivenERGM


class DataGenerator:
    """Generate synthetic temporal networks according to various DGPs"""

    @staticmethod
    def constant_dgp(theta_0: np.ndarray, T: int) -> np.ndarray:
        """Constant parameters DGP"""
        n_params = len(theta_0)
        theta_sequence = np.tile(theta_0, (T, 1))
        return theta_sequence

    @staticmethod
    def sinusoidal_dgp(theta_0: np.ndarray, amplitude: float, T: int) -> np.ndarray:
        """Sinusoidal variation DGP"""
        n_params = len(theta_0)
        theta_sequence = np.zeros((T, n_params))

        for t in range(T):
            for i in range(n_params):
                if i % 2 == 1:  # Vary odd-indexed parameters
                    phase = np.random.uniform(0, 2 * np.pi)  # Random phase per parameter
                    theta_sequence[t, i] = theta_0[i] + amplitude * np.sin(4 * np.pi * t / T + phase)
                else:
                    theta_sequence[t, i] = theta_0[i]

        return theta_sequence

    @staticmethod
    def step_dgp(theta_0: np.ndarray, step_size: float, T: int) -> np.ndarray:
        """Step function DGP with change at T/2"""
        n_params = len(theta_0)
        theta_sequence = np.zeros((T, n_params))

        for t in range(T):
            for i in range(n_params):
                if i % 2 == 1:  # Vary odd-indexed parameters
                    if t < T // 2:
                        theta_sequence[t, i] = theta_0[i]
                    else:
                        theta_sequence[t, i] = theta_0[i] + step_size
                else:
                    theta_sequence[t, i] = theta_0[i]

        return theta_sequence

    @staticmethod
    def ar1_dgp(theta_0: np.ndarray, phi: float, sigma: float, T: int) -> np.ndarray:
        """AR(1) DGP: theta_t = phi_0 + phi_1 * theta_{t-1} + epsilon_t"""
        n_params = len(theta_0)
        theta_sequence = np.zeros((T, n_params))
        theta_sequence[0] = theta_0

        for t in range(1, T):
            for i in range(n_params):
                if i % 2 == 1:  # Vary odd-indexed parameters
                    phi_0 = (1 - phi) * theta_0[i]  # Ensure mean = theta_0[i]
                    epsilon = np.random.normal(0, sigma)
                    theta_sequence[t, i] = phi_0 + phi * theta_sequence[t - 1, i] + epsilon
                else:
                    theta_sequence[t, i] = theta_0[i]

        return theta_sequence

    @classmethod
    def generate_temporal_network_data(cls, dgp_type: str, model: ScoreDrivenERGM,
                                       theta_0: np.ndarray, T: int,
                                       **dgp_kwargs) -> Tuple[List[np.ndarray], np.ndarray]:
        """Generate complete temporal network dataset"""
        # Generate true parameter sequence
        if dgp_type == 'constant':
            theta_true = cls.constant_dgp(theta_0, T)
        elif dgp_type == 'sinusoidal':
            amplitude = dgp_kwargs.get('amplitude', 0.2)
            theta_true = cls.sinusoidal_dgp(theta_0, amplitude, T)
        elif dgp_type == 'step':
            step_size = dgp_kwargs.get('step_size', 0.3)
            theta_true = cls.step_dgp(theta_0, step_size, T)
        elif dgp_type == 'ar1':
            phi = dgp_kwargs.get('phi', 0.9)
            sigma = dgp_kwargs.get('sigma', 0.1)
            theta_true = cls.ar1_dgp(theta_0, phi, sigma, T)
        else:
            raise ValueError(f"Unknown DGP type: {dgp_type}")

        # Generate network sequence
        adjacency_sequence = []
        for t in range(T):
            network = model.simulate_network(theta_true[t])
            adjacency_sequence.append(network)

        return adjacency_sequence, theta_true


# =============================================================================
# CORE DATA STRUCTURES AND BASE CLASSES
# =============================================================================

@dataclass
class SDERGMParams:
    """Parameters for Score-Driven ERGM models"""
    w: np.ndarray  # Intercept parameters
    alpha: np.ndarray  # Score coefficients
    beta: np.ndarray  # Autoregressive coefficients


class NetworkStatistic(ABC):
    """Abstract base class for network statistics"""

    @abstractmethod
    def compute(self, adjacency_matrix: np.ndarray) -> float:
        """Compute the statistic for given adjacency matrix"""
        pass

    @abstractmethod
    def change_statistic(self, adjacency_matrix: np.ndarray, i: int, j: int) -> float:
        """Compute change in statistic when toggling edge (i,j)"""
        pass


class EdgesStatistic(NetworkStatistic):
    """Number of edges statistic"""

    def compute(self, adjacency_matrix: np.ndarray) -> float:
        return np.sum(adjacency_matrix)

    def change_statistic(self, adjacency_matrix: np.ndarray, i: int, j: int) -> float:
        return 1.0 if adjacency_matrix[i, j] == 0 else -1.0


class GWESPStatistic(NetworkStatistic):
    """Geometrically Weighted Edgewise Shared Partners statistic"""

    def __init__(self, decay: float = 0.5):
        self.decay = decay

    def compute(self, adjacency_matrix: np.ndarray) -> float:
        """Compute GWESP statistic"""
        n = adjacency_matrix.shape[0]
        gwesp = 0.0

        # For each edge, count shared partners
        for i in range(n):
            for j in range(n):
                if i != j and adjacency_matrix[i, j] == 1:
                    # Count shared partners
                    shared_partners = 0
                    for k in range(n):
                        if k != i and k != j:
                            if adjacency_matrix[i, k] == 1 and adjacency_matrix[j, k] == 1:
                                shared_partners += 1

                    # Add to GWESP with geometric weighting
                    if shared_partners > 0:
                        gwesp += np.exp(self.decay) * (1 - (1 - np.exp(-self.decay)) ** shared_partners)

        return gwesp

    def change_statistic(self, adjacency_matrix: np.ndarray, i: int, j: int) -> float:
        """Compute change in GWESP when toggling edge (i,j)"""
        current_gwesp = self.compute(adjacency_matrix)

        # Toggle edge
        adjacency_matrix[i, j] = 1 - adjacency_matrix[i, j]
        new_gwesp = self.compute(adjacency_matrix)

        # Toggle back
        adjacency_matrix[i, j] = 1 - adjacency_matrix[i, j]

        return new_gwesp - current_gwesp


# Additional specialized network statistics for completeness
class TriangleStatistic(NetworkStatistic):
    """Triangle count statistic"""

    def compute(self, adjacency_matrix: np.ndarray) -> float:
        # Count directed triangles
        A = adjacency_matrix
        return np.trace(A @ A @ A) / 3.0

    def change_statistic(self, adjacency_matrix: np.ndarray, i: int, j: int) -> float:
        current = self.compute(adjacency_matrix)
        adjacency_matrix[i, j] = 1 - adjacency_matrix[i, j]
        new = self.compute(adjacency_matrix)
        adjacency_matrix[i, j] = 1 - adjacency_matrix[i, j]
        return new - current


class TwoStarsStatistic(NetworkStatistic):
    """Two-stars (2-paths) statistic"""

    def compute(self, adjacency_matrix: np.ndarray) -> float:
        out_degrees = np.sum(adjacency_matrix, axis=1)
        return np.sum(out_degrees * (out_degrees - 1))

    def change_statistic(self, adjacency_matrix: np.ndarray, i: int, j: int) -> float:
        current_out_i = np.sum(adjacency_matrix[i, :])

        if adjacency_matrix[i, j] == 0:  # Adding edge
            new_out_i = current_out_i + 1
            return 2 * current_out_i  # Change in i's contribution
        else:  # Removing edge
            new_out_i = current_out_i - 1
            return -2 * new_out_i  # Change in i's contribution


# Additional specialized statistics for financial networks
class ReciprocityStatistic(NetworkStatistic):
    """Reciprocity statistic for mutual connections"""

    def compute(self, adjacency_matrix: np.ndarray) -> float:
        return np.sum(adjacency_matrix * adjacency_matrix.T)

    def change_statistic(self, adjacency_matrix: np.ndarray, i: int, j: int) -> float:
        if adjacency_matrix[i, j] == 0:  # Adding edge
            return 1.0 if adjacency_matrix[j, i] == 1 else 0.0
        else:  # Removing edge
            return -1.0 if adjacency_matrix[j, i] == 1 else 0.0


class ConcentrationStatistic(NetworkStatistic):
    """Market concentration statistic"""

    def compute(self, adjacency_matrix: np.ndarray) -> float:
        out_degrees = np.sum(adjacency_matrix, axis=1)
        total_degree = np.sum(out_degrees)
        if total_degree == 0:
            return 0.0
        shares = out_degrees / total_degree
        return np.sum(shares ** 2)  # Herfindahl index

    def change_statistic(self, adjacency_matrix: np.ndarray, i: int, j: int) -> float:
        current = self.compute(adjacency_matrix)
        adjacency_matrix[i, j] = 1 - adjacency_matrix[i, j]
        new = self.compute(adjacency_matrix)
        adjacency_matrix[i, j] = 1 - adjacency_matrix[i, j]
        return new - current


class SizeHeterogeneityStatistic(NetworkStatistic):
    """Bank size heterogeneity statistic"""

    def compute(self, adjacency_matrix: np.ndarray) -> float:
        degrees = np.sum(adjacency_matrix, axis=1) + np.sum(adjacency_matrix, axis=0)
        return np.var(degrees)

    def change_statistic(self, adjacency_matrix: np.ndarray, i: int, j: int) -> float:
        current = self.compute(adjacency_matrix)
        adjacency_matrix[i, j] = 1 - adjacency_matrix[i, j]
        new = self.compute(adjacency_matrix)
        adjacency_matrix[i, j] = 1 - adjacency_matrix[i, j]
        return new - current
