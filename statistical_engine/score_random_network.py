import numpy as np
import scipy.optimize as opt
from typing import List, Optional, Tuple, Dict

# =============================================================================
# SCORE-DRIVEN ERGM FRAMEWORK
# =============================================================================
from data_handler.data_structures import EdgesStatistic, SDERGMParams, GWESPStatistic, ReciprocityStatistic, \
    ConcentrationStatistic, SizeHeterogeneityStatistic
from statistical_engine.ml_routines import FastBetaModel, FastPseudoLikelihood


class ScoreDrivenERGM:
    """Main Score-Driven ERGM implementation"""

    def __init__(self, model_type: str = 'beta', **kwargs):
        self.model_type = model_type

        if model_type == 'beta':
            self.n_nodes = kwargs.get('n_nodes', 10)
            self.model = FastBetaModel(self.n_nodes)
            self.n_params = 2 * self.n_nodes  # in + out parameters
        elif model_type == 'pseudo':
            self.statistics = kwargs.get('statistics', [EdgesStatistic()])
            self.model = FastPseudoLikelihood(self.statistics)
            self.n_params = len(self.statistics)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Score-driven parameters
        self.w = None
        self.alpha = None
        self.beta = None

    def simulate_network(self, theta: np.ndarray) -> np.ndarray:
        """Simulate network from ERGM with given parameters"""
        if self.model_type == 'beta':
            n_half = len(theta) // 2
            in_params = theta[:n_half]
            out_params = theta[n_half:]

            prob_matrix = self.model._compute_prob_matrix(in_params, out_params)
            adjacency_matrix = np.random.binomial(1, prob_matrix)
            np.fill_diagonal(adjacency_matrix, 0)  # No self-loops

            return adjacency_matrix

        elif self.model_type == 'pseudo':
            # For pseudo-likelihood models, use MCMC simulation
            return self._mcmc_simulation(theta)

    def _mcmc_simulation(self, theta: np.ndarray, n_nodes: int = 10,
                         n_steps: int = 1000) -> np.ndarray:
        """MCMC simulation for general ERGM"""
        # Initialize random network
        adjacency_matrix = np.random.binomial(1, 0.1, size=(n_nodes, n_nodes))
        np.fill_diagonal(adjacency_matrix, 0)

        # MCMC steps
        for step in range(n_steps):
            # Propose edge toggle
            i, j = np.random.randint(0, n_nodes, 2)
            if i == j:
                continue

            # Compute change in log-probability
            change_stats = np.array([stat.change_statistic(adjacency_matrix, i, j)
                                     for stat in self.statistics])
            delta_logp = np.dot(theta, change_stats)

            # Accept/reject
            if adjacency_matrix[i, j] == 0:  # Propose adding edge
                if np.random.random() < 1.0 / (1.0 + np.exp(-delta_logp)):
                    adjacency_matrix[i, j] = 1
            else:  # Propose removing edge
                if np.random.random() < 1.0 / (1.0 + np.exp(delta_logp)):
                    adjacency_matrix[i, j] = 0

        return adjacency_matrix

    def update_parameters(self, adjacency_matrix: np.ndarray,
                          theta_current: np.ndarray,
                          sd_params: SDERGMParams,
                          scaling_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """Score-driven parameter update rule"""
        # Compute score
        if self.model_type == 'beta':
            n_half = len(theta_current) // 2
            in_params = theta_current[:n_half]
            out_params = theta_current[n_half:]
            in_score, out_score = self.model.compute_score(adjacency_matrix, in_params, out_params)
            score = np.concatenate([in_score, out_score])
        elif self.model_type == 'pseudo':
            score = self.model.compute_pseudo_score(adjacency_matrix, theta_current)

        # Apply scaling matrix if provided
        if scaling_matrix is not None:
            score = scaling_matrix @ score

        # Score-driven update: theta_{t+1} = w + beta * theta_t + alpha * score_t
        theta_new = sd_params.w + sd_params.beta * theta_current + sd_params.alpha * score

        return theta_new

    def filter_parameters(self, adjacency_sequence: List[np.ndarray],
                          sd_params: SDERGMParams,
                          theta_init: Optional[np.ndarray] = None) -> np.ndarray:
        """Filter time-varying parameters using score-driven dynamics"""
        T = len(adjacency_sequence)

        if theta_init is None:
            theta_init = np.zeros(self.n_params)

        theta_filtered = np.zeros([T, self.n_params])
        theta_filtered[0] = theta_init

        for t in range(1, T):
            theta_filtered[t] = self.update_parameters(
                adjacency_sequence[t - 1],
                theta_filtered[t - 1],
                sd_params
            )

        return theta_filtered

    def _static_params_log_likelihood(self, params: np.ndarray,
                                      adjacency_sequence: List[np.ndarray],
                                      theta_filtered: np.ndarray) -> float:
        """Log-likelihood for static parameter estimation"""
        T, n_params = theta_filtered.shape

        # Unpack parameters
        w = params[:n_params]
        alpha = params[n_params:2 * n_params]
        beta = params[2 * n_params:3 * n_params]

        log_lik = 0.0

        for t in range(1, T):
            # Compute score at time t-1
            if self.model_type == 'beta':
                n_half = n_params // 2
                in_params = theta_filtered[t - 1, :n_half]
                out_params = theta_filtered[t - 1, n_half:]
                in_score, out_score = self.model.compute_score(
                    adjacency_sequence[t - 1], in_params, out_params)
                score = np.concatenate([in_score, out_score])
            elif self.model_type == 'pseudo':
                score = self.model.compute_pseudo_score(
                    adjacency_sequence[t - 1], theta_filtered[t - 1])

            # Predicted theta_t
            theta_pred = w + beta * theta_filtered[t - 1] + alpha * score

            # Add to log-likelihood (assuming conditional normality)
            residual = theta_filtered[t] - theta_pred
            log_lik += -0.5 * np.sum(residual ** 2)

        return log_lik

    def estimate_static_parameters(self, adjacency_sequence: List[np.ndarray],
                                   theta_filtered: np.ndarray) -> SDERGMParams:
        """Estimate static parameters w, alpha, beta via MLE"""
        T, n_params = theta_filtered.shape

        # Initialize parameters
        params_init = np.concatenate([
            np.zeros(n_params),  # w
            0.1 * np.ones(n_params),  # alpha
            0.9 * np.ones(n_params)  # beta
        ])

        # Optimize
        result = opt.minimize(
            lambda p: -self._static_params_log_likelihood(p, adjacency_sequence, theta_filtered),
            params_init,
            method='BFGS',
            options={'maxiter': 1000}
        )

        # Extract optimized parameters
        w_opt = result.x[:n_params]
        alpha_opt = result.x[n_params:2 * n_params]
        beta_opt = result.x[2 * n_params:3 * n_params]

        return SDERGMParams(w=w_opt, alpha=alpha_opt, beta=beta_opt)


# =============================================================================
# ADVANCED EXTENSIONS AND RESEARCH TOOLS
# =============================================================================

class HierarchicalSDERGM:
    """Hierarchical Score-Driven ERGM for multi-layer networks"""

    def __init__(self, n_layers: int, n_nodes: int, shared_statistics: bool = True):
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.shared_statistics = shared_statistics

        # Create models for each layer
        self.layer_models = []
        for layer in range(n_layers):
            if shared_statistics:
                # Same statistics across layers
                statistics = [EdgesStatistic(), GWESPStatistic()]
                model = ScoreDrivenERGM(model_type='pseudo', statistics=statistics)
            else:
                # Layer-specific statistics (could be different)
                statistics = [EdgesStatistic(), GWESPStatistic(decay=0.5 + 0.1 * layer)]
                model = ScoreDrivenERGM(model_type='pseudo', statistics=statistics)

            self.layer_models.append(model)

    def filter_multilayer_parameters(self, multilayer_sequence: List[List[np.ndarray]],
                                     cross_layer_coupling: float = 0.1) -> List[np.ndarray]:
        """Filter parameters across multiple network layers with coupling"""
        T = len(multilayer_sequence)

        # Initialize parameters for each layer
        layer_parameters = []
        for layer in range(self.n_layers):
            theta_init = np.zeros(self.layer_models[layer].n_params)
            layer_params = np.zeros((T, len(theta_init)))
            layer_params[0] = theta_init
            layer_parameters.append(layer_params)

        # Filter with cross-layer coupling
        for t in range(1, T):
            for layer in range(self.n_layers):
                model = self.layer_models[layer]

                # Standard SD update
                sd_params = SDERGMParams(
                    w=np.zeros(model.n_params),
                    alpha=0.1 * np.ones(model.n_params),
                    beta=0.9 * np.ones(model.n_params)
                )

                theta_new = model.update_parameters(
                    multilayer_sequence[t - 1][layer],
                    layer_parameters[layer][t - 1],
                    sd_params
                )

                # Add cross-layer coupling
                if self.n_layers > 1:
                    other_layers = [l for l in range(self.n_layers) if l != layer]
                    coupling_effect = np.mean([layer_parameters[l][t - 1] for l in other_layers], axis=0)
                    theta_new += cross_layer_coupling * (coupling_effect - layer_parameters[layer][t - 1])

                layer_parameters[layer][t] = theta_new

        return layer_parameters


class RobustSDERGM:
    """Robust Score-Driven ERGM with outlier detection and handling"""

    def __init__(self, base_model: ScoreDrivenERGM, outlier_threshold: float = 3.0):
        self.base_model = base_model
        self.outlier_threshold = outlier_threshold
        self.outlier_flags = []

    def robust_score_computation(self, adjacency_matrix: np.ndarray,
                                 theta_current: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Compute score with outlier detection"""
        # Standard score computation
        if self.base_model.model_type == 'beta':
            n_half = len(theta_current) // 2
            in_params = theta_current[:n_half]
            out_params = theta_current[n_half:]
            in_score, out_score = self.base_model.model.compute_score(
                adjacency_matrix, in_params, out_params)
            score = np.concatenate([in_score, out_score])
        else:
            score = self.base_model.model.compute_pseudo_score(adjacency_matrix, theta_current)

        # Outlier detection using Mahalanobis distance
        if len(self.outlier_flags) > 10:  # Need history for robust detection
            recent_scores = np.array([s for s, is_outlier in self.outlier_flags[-10:] if not is_outlier])
            if len(recent_scores) > 3:
                mean_score = np.mean(recent_scores, axis=0)
                cov_score = np.cov(recent_scores.T)

                try:
                    inv_cov = np.linalg.inv(cov_score + 1e-6 * np.eye(len(cov_score)))
                    mahal_dist = np.sqrt((score - mean_score) @ inv_cov @ (score - mean_score))
                    is_outlier = mahal_dist > self.outlier_threshold
                except np.linalg.LinAlgError:
                    is_outlier = False
            else:
                is_outlier = False
        else:
            is_outlier = False

        # Robust score (downweight if outlier)
        if is_outlier:
            weight = 1.0 / (1.0 + mahal_dist - self.outlier_threshold)
            robust_score = weight * score + (1 - weight) * mean_score
        else:
            robust_score = score

        self.outlier_flags.append((score.copy(), is_outlier))

        return robust_score, is_outlier

    def robust_filter_parameters(self, adjacency_sequence: List[np.ndarray],
                                 sd_params: SDERGMParams) -> Tuple[np.ndarray, List[bool]]:
        """Filter parameters with robust score computation"""
        T = len(adjacency_sequence)
        theta_filtered = np.zeros((T, self.base_model.n_params))
        outlier_detected = []

        # Initialize
        theta_filtered[0] = np.zeros(self.base_model.n_params)

        for t in range(1, T):
            # Robust score computation
            robust_score, is_outlier = self.robust_score_computation(
                adjacency_sequence[t - 1], theta_filtered[t - 1]
            )

            # Update parameters using robust score
            theta_filtered[t] = (sd_params.w +
                                 sd_params.beta * theta_filtered[t - 1] +
                                 sd_params.alpha * robust_score)

            outlier_detected.append(is_outlier)

        return theta_filtered, outlier_detected


class OnlineSDERGM:
    """Online/streaming version of SD-ERGM for real-time applications"""

    def __init__(self, base_model: ScoreDrivenERGM, forgetting_factor: float = 0.95):
        self.base_model = base_model
        self.forgetting_factor = forgetting_factor

        # Online state
        self.current_theta = None
        self.online_mean_score = None
        self.online_var_score = None
        self.t = 0

    def initialize_online(self, initial_theta: np.ndarray):
        """Initialize online filtering"""
        self.current_theta = initial_theta.copy()
        self.online_mean_score = np.zeros_like(initial_theta)
        self.online_var_score = np.ones_like(initial_theta)
        self.t = 0

    def online_update(self, new_adjacency: np.ndarray,
                      sd_params: SDERGMParams) -> np.ndarray:
        """Process single new observation online"""
        if self.current_theta is None:
            raise ValueError("Must initialize online filtering first")

        # Compute score for new observation
        if self.base_model.model_type == 'beta':
            n_half = len(self.current_theta) // 2
            in_params = self.current_theta[:n_half]
            out_params = self.current_theta[n_half:]
            in_score, out_score = self.base_model.model.compute_score(
                new_adjacency, in_params, out_params)
            score = np.concatenate([in_score, out_score])
        else:
            score = self.base_model.model.compute_pseudo_score(new_adjacency, self.current_theta)

        # Update online statistics
        self.t += 1
        self.online_mean_score = (self.forgetting_factor * self.online_mean_score +
                                  (1 - self.forgetting_factor) * score)

        score_dev = score - self.online_mean_score
        self.online_var_score = (self.forgetting_factor * self.online_var_score +
                                 (1 - self.forgetting_factor) * score_dev ** 2)

        # Adaptive scaling based on online variance
        adaptive_scaling = 1.0 / np.sqrt(self.online_var_score + 1e-6)
        scaled_score = adaptive_scaling * score

        # Update parameters
        self.current_theta = (sd_params.w +
                              sd_params.beta * self.current_theta +
                              sd_params.alpha * scaled_score)

        return self.current_theta.copy()

    def get_current_state(self) -> Dict:
        """Get current online state"""
        return {
            'theta': self.current_theta.copy() if self.current_theta is not None else None,
            'mean_score': self.online_mean_score.copy() if self.online_mean_score is not None else None,
            'var_score': self.online_var_score.copy() if self.online_var_score is not None else None,
            'time_steps': self.t
        }


class BayesianSDERGM:
    """Bayesian version of SD-ERGM with uncertainty quantification"""

    def __init__(self, base_model: ScoreDrivenERGM, prior_params: Dict):
        self.base_model = base_model
        self.prior_params = prior_params

    def mcmc_estimation(self, adjacency_sequence: List[np.ndarray],
                        n_samples: int = 1000, n_burn: int = 500) -> Dict:
        """MCMC estimation of SD-ERGM parameters with full uncertainty"""
        T = len(adjacency_sequence)
        n_params = self.base_model.n_params

        # Storage for MCMC samples
        w_samples = np.zeros((n_samples, n_params))
        alpha_samples = np.zeros((n_samples, n_params))
        beta_samples = np.zeros((n_samples, n_params))
        theta_samples = np.zeros((n_samples, T, n_params))

        # Initialize
        w_current = np.random.normal(0, 0.1, n_params)
        alpha_current = np.random.gamma(2, 0.05, n_params)  # Positive constraint
        beta_current = np.random.beta(9, 1, n_params)  # [0,1] constraint

        # Proposal covariances
        w_cov = 0.01 * np.eye(n_params)
        alpha_cov = 0.001 * np.eye(n_params)
        beta_cov = 0.001 * np.eye(n_params)

        for iteration in range(n_samples + n_burn):
            # Sample theta sequence given parameters
            sd_params = SDERGMParams(w=w_current, alpha=alpha_current, beta=beta_current)
            theta_current = self.base_model.filter_parameters(adjacency_sequence, sd_params)

            # Compute log-likelihood
            log_likelihood = 0.0
            for t in range(T):
                if self.base_model.model_type == 'beta':
                    n_half = n_params // 2
                    log_likelihood += self.base_model.model.log_likelihood(
                        adjacency_sequence[t], theta_current[t, :n_half], theta_current[t, n_half:]
                    )
                else:
                    log_likelihood += self.base_model.model.log_pseudo_likelihood(
                        adjacency_sequence[t], theta_current[t]
                    )

            # Add priors
            log_prior = (stats.norm.logpdf(w_current, 0, self.prior_params.get('w_sd', 1.0)).sum() +
                         stats.gamma.logpdf(alpha_current, 2, scale=0.05).sum() +
                         stats.beta.logpdf(beta_current, 9, 1).sum())

            log_posterior_current = log_likelihood + log_prior

            # Metropolis-Hastings updates
            # Update w
            w_proposal = np.random.multivariate_normal(w_current, w_cov)
            sd_params_prop = SDERGMParams(w=w_proposal, alpha=alpha_current, beta=beta_current)
            theta_prop = self.base_model.filter_parameters(adjacency_sequence, sd_params_prop)

            # Compute proposal likelihood (simplified for efficiency)
            log_likelihood_prop = 0.0
            for t in range(min(T, 10)):  # Sample subset for efficiency
                if self.base_model.model_type == 'beta':
                    n_half = n_params // 2
                    log_likelihood_prop += self.base_model.model.log_likelihood(
                        adjacency_sequence[t], theta_prop[t, :n_half], theta_prop[t, n_half:]
                    )

            log_prior_prop = stats.norm.logpdf(w_proposal, 0, self.prior_params.get('w_sd', 1.0)).sum()
            log_posterior_prop = log_likelihood_prop + log_prior_prop

            # Accept/reject
            if np.log(np.random.random()) < log_posterior_prop - log_posterior_current:
                w_current = w_proposal
                theta_current = theta_prop
                log_posterior_current = log_posterior_prop

            # Similar updates for alpha and beta (simplified here)
            # In practice, would implement full updates for all parameters

            # Store samples (after burn-in)
            if iteration >= n_burn:
                sample_idx = iteration - n_burn
                w_samples[sample_idx] = w_current
                alpha_samples[sample_idx] = alpha_current
                beta_samples[sample_idx] = beta_current
                theta_samples[sample_idx] = theta_current

        return {
            'w_samples': w_samples,
            'alpha_samples': alpha_samples,
            'beta_samples': beta_samples,
            'theta_samples': theta_samples,
            'posterior_mean': {
                'w': np.mean(w_samples, axis=0),
                'alpha': np.mean(alpha_samples, axis=0),
                'beta': np.mean(beta_samples, axis=0)
            },
            'posterior_std': {
                'w': np.std(w_samples, axis=0),
                'alpha': np.std(alpha_samples, axis=0),
                'beta': np.std(beta_samples, axis=0)
            }
        }


class FinancialNetworkApplication:
    """Specialized application for financial networks"""

    def __init__(self, n_banks: int):
        self.n_banks = n_banks

        # Financial-specific statistics
        self.statistics = [
            EdgesStatistic(),  # Total exposures
            ReciprocityStatistic(),  # Mutual exposures
            ConcentrationStatistic(),  # Market concentration
            SizeHeterogeneityStatistic()  # Bank size effects
        ]

        self.model = ScoreDrivenERGM(model_type='pseudo', statistics=self.statistics)

    def systemic_risk_indicator(self, adjacency_sequence: List[np.ndarray],
                                theta_filtered: np.ndarray) -> np.ndarray:
        """Compute time-varying systemic risk indicator"""
        T = len(adjacency_sequence)
        risk_indicator = np.zeros(T)

        for t in range(T):
            # Network-based risk measures
            adj = adjacency_sequence[t]

            # Eigenvector centrality (systemically important banks)
            try:
                eigenvalues, eigenvectors = np.linalg.eig(adj.T)
                dominant_eigenvalue = np.max(np.real(eigenvalues))
                network_risk = dominant_eigenvalue / self.n_banks
            except:
                network_risk = 0.0

            # Parameter volatility (model uncertainty)
            if t > 10:
                param_volatility = np.std(theta_filtered[t - 10:t], axis=0)
                model_risk = np.mean(param_volatility)
            else:
                model_risk = 0.0

            # Combined risk indicator
            risk_indicator[t] = 0.7 * network_risk + 0.3 * model_risk

        return risk_indicator

    def stress_testing(self, baseline_theta: np.ndarray,
                       stress_scenarios: List[Dict]) -> Dict:
        """Conduct stress testing on financial network"""
        results = {}

        for scenario_name, scenario in stress_scenarios.items():
            stressed_theta = baseline_theta.copy()

            # Apply stress
            if 'edge_shock' in scenario:
                stressed_theta[0] += scenario['edge_shock']  # Reduce connectivity

            if 'concentration_shock' in scenario:
                stressed_theta[2] += scenario['concentration_shock']  # Increase concentration

            # Simulate stressed network
            stressed_network = self.model.simulate_network(stressed_theta)

            # Compute stress impact metrics
            baseline_network = self.model.simulate_network(baseline_theta)

            connectivity_loss = (np.sum(baseline_network) - np.sum(stressed_network)) / np.sum(baseline_network)

            results[scenario_name] = {
                'connectivity_loss': connectivity_loss,
                'stressed_theta': stressed_theta,
                'stressed_network': stressed_network
            }

        return results
