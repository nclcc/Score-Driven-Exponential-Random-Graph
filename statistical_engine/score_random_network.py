import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
from scipy.special import expit
from typing import List, Optional, Tuple, Dict


# =============================================================================
# SCORE-DRIVEN ERGM FRAMEWORK
# =============================================================================

class ScoreDrivenERGM:
    """Main Score-Driven ERGM implementation"""

    def __init__(self, model_type: str = 'beta', **kwargs):
        self.model_type = model_type
        self.n_nodes = kwargs.get("n_nodes", 10)

        if model_type == 'beta':
            # Import here to avoid circular import
            from statistical_engine.ml_routines import FastBetaModel
            self.n_nodes = kwargs.get('n_nodes', 10)
            self.model = FastBetaModel(self.n_nodes)
            self.n_params = 2 * self.n_nodes  # in + out parameters
        elif model_type == 'pseudo':
            # Import here to avoid circular import
            from statistical_engine.ml_routines import FastPseudoLikelihood
            from data_handler.data_structures import EdgesStatistic
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

    def _mcmc_simulation(self, theta: np.ndarray, n_nodes: Optional[int] = None,
                         n_steps: int = 1000) -> np.ndarray:
        """MCMC simulation for general ERGM"""
        n_nodes = self.n_nodes if n_nodes is None else n_nodes
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

            # Symmetric edge-toggle Barker update: delta_logp already contains
            # the sign for both additions and removals. This obeys detailed balance.
            if np.random.random() < expit(delta_logp):
                adjacency_matrix[i, j] = 1 - adjacency_matrix[i, j]

        return adjacency_matrix

    def update_parameters(self, adjacency_matrix: np.ndarray,
                          theta_current: np.ndarray,
                          sd_params,  # Remove type hint to avoid import
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
                          sd_params,  # Remove type hint to avoid import
                          theta_init: Optional[np.ndarray] = None) -> np.ndarray:
        """Filter time-varying parameters using score-driven dynamics"""
        T = len(adjacency_sequence)
        if T == 0:
            raise ValueError("adjacency_sequence must not be empty")

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
        from data_handler.data_structures import SDERGMParams
        n = self.n_params
        sd_params = SDERGMParams(params[:n], params[n:2*n], params[2*n:])
        filtered = self.filter_parameters(adjacency_sequence, sd_params, theta_filtered[0])
        return self._observation_log_likelihood(adjacency_sequence, filtered)

    def _observation_log_likelihood(self, adjacency_sequence, theta_filtered):
        value = 0.
        for y, theta in zip(adjacency_sequence, theta_filtered):
            if self.model_type == 'beta':
                value += self.model.log_likelihood(y, theta[:self.n_nodes], theta[self.n_nodes:])
            else:
                value += self.model.log_pseudo_likelihood(y, theta)
        return float(value)

    def estimate_static_parameters(self, adjacency_sequence: List[np.ndarray],
                                   theta_filtered: np.ndarray):  # Remove type hint
        """Conditional observation MLE (pseudo-MLE for dependent-edge models)."""
        from data_handler.data_structures import SDERGMParams

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
            method='L-BFGS-B',
            bounds=[(None, None)] * n_params + [(0., None)] * n_params + [(0., 1. - 1e-6)] * n_params,
            options={'maxiter': 1000}
        )

        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Static parameter estimation failed: {result.message}")

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
                # Import here to avoid circular import
                from data_handler.data_structures import EdgesStatistic, GWESPStatistic
                # Same statistics across layers
                statistics = [EdgesStatistic(), GWESPStatistic()]
                model = ScoreDrivenERGM(model_type='pseudo', n_nodes=n_nodes, statistics=statistics)
            else:
                # Import here to avoid circular import
                from data_handler.data_structures import EdgesStatistic, GWESPStatistic
                # Layer-specific statistics (could be different)
                statistics = [EdgesStatistic(), GWESPStatistic(decay=0.5 + 0.1 * layer)]
                model = ScoreDrivenERGM(model_type='pseudo', n_nodes=n_nodes, statistics=statistics)

            self.layer_models.append(model)

    def filter_multilayer_parameters(self, multilayer_sequence: List[List[np.ndarray]],
                                     cross_layer_coupling: float = 0.1) -> List[np.ndarray]:
        """Filter parameters across multiple network layers with coupling"""
        from data_handler.data_structures import SDERGMParams

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
                                 sd_params) -> Tuple[np.ndarray, List[bool]]:  # Remove type hint
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
                      sd_params) -> np.ndarray:  # Remove type hint
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
    """Conditional Bayesian beta-model inference; pseudo-posterior for other ERGMs.

    Uses symmetric block random-walk proposals in the original coordinates.
    Out-of-support proposals are rejected, not redrawn or clipped, so no
    truncation correction or transformation Jacobian is needed.
    """

    def __init__(self, base_model: ScoreDrivenERGM, prior_params: Dict):
        self.base_model = base_model
        self.prior_params = {
            'w_sd': 1., 'alpha_shape': 2., 'alpha_scale': .05,
            'beta_a': 9., 'beta_b': 1., **prior_params
        }
        if any(not np.isfinite(v) or v <= 0 for v in self.prior_params.values()):
            raise ValueError("Prior scales and shapes must be finite and positive")

    def _log_posterior(self, observations, w, alpha, beta):
        from data_handler.data_structures import SDERGMParams
        if (not np.all(np.isfinite(np.r_[w, alpha, beta])) or
                np.any(alpha <= 0) or np.any(beta <= 0) or np.any(beta >= 1)):
            return -np.inf, None
        priors = self.prior_params
        with np.errstate(over='ignore', invalid='ignore'):
            theta = self.base_model.filter_parameters(observations, SDERGMParams(w, alpha, beta))
            if not np.all(np.isfinite(theta)):
                return -np.inf, None
            value = self.base_model._observation_log_likelihood(observations, theta)
            value += stats.norm.logpdf(w, scale=priors['w_sd']).sum()
            value += stats.gamma.logpdf(alpha, a=priors['alpha_shape'], scale=priors['alpha_scale']).sum()
            value += stats.beta.logpdf(beta, a=priors['beta_a'], b=priors['beta_b']).sum()
        return (float(value), theta) if np.isfinite(value) else (-np.inf, None)

    def mcmc_estimation(self, adjacency_sequence: List[np.ndarray],
                        n_samples: int = 1000, n_burn: int = 500,
                        random_state=None, proposal_scales=None) -> Dict:
        """Sample all parameter blocks using one full-data target.

        Acceptance rates and traces are diagnostics, not proof of convergence.
        The initial filtered state is fixed to zero for this conditional target.
        """
        if len(adjacency_sequence) == 0:
            raise ValueError("adjacency_sequence must not be empty")
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        if not isinstance(n_burn, (int, np.integer)) or n_burn < 0:
            raise ValueError("n_burn must be a nonnegative integer")
        rng = np.random.default_rng(random_state)
        n, T = self.base_model.n_params, len(adjacency_sequence)
        p = self.prior_params
        current = dict(w=rng.normal(0, .1, n),
                       alpha=rng.gamma(p['alpha_shape'], p['alpha_scale'], n),
                       beta=rng.beta(p['beta_a'], p['beta_b'], n))
        scales = {'w': .1, 'alpha': .025, 'beta': .025, **(proposal_scales or {})}
        if set(scales) != set(current) or any(not np.isfinite(v) or v <= 0 for v in scales.values()):
            raise ValueError("Proposal scales must be positive finite values for w, alpha and beta")
        traces = {name: np.empty((n_samples, n)) for name in current}
        theta_samples = np.empty((n_samples, T, n))
        posterior_trace = np.empty(n_samples)
        accepted = {name: 0 for name in current}
        posterior, theta = self._log_posterior(adjacency_sequence, **current)
        if not np.isfinite(posterior):
            raise ValueError("Initial posterior is nonfinite; check data and prior settings")
        iterations = n_samples + n_burn
        for iteration in range(iterations):
            for name in current:
                proposal = {key: value.copy() for key, value in current.items()}
                proposal[name] += rng.normal(0, scales[name], n)
                proposed_posterior, proposed_theta = self._log_posterior(adjacency_sequence, **proposal)
                if np.log(rng.random()) < proposed_posterior - posterior:
                    current, posterior, theta = proposal, proposed_posterior, proposed_theta
                    accepted[name] += 1
            if iteration >= n_burn:
                index = iteration - n_burn
                for name in current:
                    traces[name][index] = current[name]
                theta_samples[index] = theta
                posterior_trace[index] = posterior
        return {
            **{name + '_samples': trace for name, trace in traces.items()},
            'theta_samples': theta_samples,
            'posterior_mean': {name: trace.mean(axis=0) for name, trace in traces.items()},
            'posterior_std': {name: trace.std(axis=0) for name, trace in traces.items()},
            'log_posterior': posterior_trace,
            'acceptance_rates': {name: count / iterations for name, count in accepted.items()},
            'target': 'conditional_posterior' if self.base_model.model_type == 'beta' else 'conditional_pseudo_posterior'
        }


class FinancialNetworkApplication:
    """Specialized application for financial networks"""

    def __init__(self, n_banks: int):
        self.n_banks = n_banks

        # Import here to avoid circular import
        from data_handler.data_structures import (EdgesStatistic, ReciprocityStatistic,
                                                  ConcentrationStatistic, SizeHeterogeneityStatistic)

        # Financial-specific statistics
        self.statistics = [
            EdgesStatistic(),  # Total exposures
            ReciprocityStatistic(),  # Mutual exposures
            ConcentrationStatistic(),  # Market concentration
            SizeHeterogeneityStatistic()  # Bank size effects
        ]

        self.model = ScoreDrivenERGM(model_type='pseudo', n_nodes=n_banks, statistics=self.statistics)

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
                       stress_scenarios: Dict) -> Dict:  # Changed from List[Dict]
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
