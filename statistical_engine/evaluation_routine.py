import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
from typing import List, Dict
from statistical_engine.score_random_network import ScoreDrivenERGM

# =============================================================================
# EVALUATION AND TESTING FUNCTIONS
# =============================================================================


class ModelEvaluator:
    """Evaluation metrics and testing procedures"""

    @staticmethod
    def compute_rmse(theta_true: np.ndarray, theta_estimated: np.ndarray) -> float:
        """Compute Root Mean Square Error"""
        return np.sqrt(np.mean((theta_true - theta_estimated) ** 2))

    @staticmethod
    def cross_sectional_estimates(adjacency_sequence: List[np.ndarray],
                                  model: ScoreDrivenERGM) -> np.ndarray:
        """Compute cross-sectional ERGM estimates for comparison"""
        T = len(adjacency_sequence)
        n_params = model.n_params

        theta_cs = np.zeros((T, n_params))

        for t in range(T):
            if model.model_type == 'beta':
                in_params, out_params = model.model.mle_estimate(adjacency_sequence[t])
                theta_cs[t] = np.concatenate([in_params, out_params])
            elif model.model_type == 'pseudo':
                # Simple pseudo-likelihood MLE for single observation
                def objective(theta):
                    return -model.model.log_pseudo_likelihood(adjacency_sequence[t], theta)

                result = opt.minimize(objective, np.zeros(n_params), method='BFGS')
                theta_cs[t] = result.x

        return theta_cs

    @staticmethod
    def lm_test_time_variation(adjacency_sequence: List[np.ndarray],
                               model: ScoreDrivenERGM) -> Dict[str, float]:
        """Reserved for a calibrated score test with an information matrix.

        Raw score variances do not have the previously assumed chi-square law.
        Returning numerical p-values here would misrepresent the procedure.
        """
        raise NotImplementedError("A calibrated LM test is not implemented; raw score variance is not an LM statistic")
