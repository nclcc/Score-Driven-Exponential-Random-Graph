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
        """Lagrange Multiplier test for parameter time variation"""
        # This is a simplified implementation of the LM test
        # Full implementation would require more sophisticated econometric procedures

        T = len(adjacency_sequence)

        # Estimate static model (pooled)
        if model.model_type == 'beta':
            # Pool all adjacency matrices
            pooled_adj = np.mean(adjacency_sequence, axis=0)
            in_params_static, out_params_static = model.model.mle_estimate(pooled_adj)
            theta_static = np.concatenate([in_params_static, out_params_static])
        else:
            # For pseudo-likelihood models
            def pooled_objective(theta):
                log_pl = 0.0
                for adj in adjacency_sequence:
                    log_pl += model.model.log_pseudo_likelihood(adj, theta)
                return -log_pl

            result = opt.minimize(pooled_objective, np.zeros(model.n_params), method='BFGS')
            theta_static = result.x

        # Compute score sequence under static model
        scores = []
        for t in range(T):
            if model.model_type == 'beta':
                n_half = len(theta_static) // 2
                in_score, out_score = model.model.compute_score(
                    adjacency_sequence[t], theta_static[:n_half], theta_static[n_half:])
                score = np.concatenate([in_score, out_score])
            else:
                score = model.model.compute_pseudo_score(adjacency_sequence[t], theta_static)
            scores.append(score)

        scores = np.array(scores)

        # Simple test: variance of scores (high variance suggests time variation)
        # In practice, this would use proper LM test statistics
        score_variance = np.var(scores, axis=0)
        p_values = 1 - stats.chi2.cdf(score_variance * T, df=1)  # Approximate

        return {'test_statistics': score_variance, 'p_values': p_values}
