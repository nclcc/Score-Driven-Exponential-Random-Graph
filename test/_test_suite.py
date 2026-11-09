"""
Comprehensive Test Suite for Score-Driven Exponential Random Graphs
====================================================================
This module provides an organized testing framework for all SD-ERGM functionalities
including basic models, advanced extensions, and specialized applications.

Autor: @nclcc, niciccarone@gmail.com
"""

import time
import warnings
from typing import Dict

import numpy as np
import scipy.optimize as opt
# Import all required modules
from data_handler.data_structures import (
    DataGenerator, SDERGMParams, EdgesStatistic, GWESPStatistic, TriangleStatistic
)
from statistical_engine.evaluation_routine import ModelEvaluator
from statistical_engine.score_random_network import (
    ScoreDrivenERGM, HierarchicalSDERGM, RobustSDERGM,
    OnlineSDERGM, BayesianSDERGM, FinancialNetworkApplication
)




class SDERGMTestSuite:
    """Main test suite class for SD-ERGM models"""

    def __init__(self, verbose: bool = True, save_results: bool = True):
        """
        Initialize the test suite

        Args:
            verbose: Whether to print detailed output
            save_results: Whether to save results to files
        """
        self.verbose = verbose
        self.save_results = save_results
        self.results = {}
        self.test_configs = self._initialize_test_configs()

    def _initialize_test_configs(self) -> Dict:
        """Initialize default test configurations"""
        return {
            'basic': {
                'n_nodes': 10,
                'T': 100,
                'n_replications': 20
            },
            'advanced': {
                'n_nodes': 15,
                'T': 150,
                'n_replications': 10
            },
            'quick': {
                'n_nodes': 8,
                'T': 50,
                'n_replications': 5
            }
        }

    def _print(self, message: str, level: int = 0):
        """Helper for conditional printing with indentation"""
        if self.verbose:
            indent = "  " * level
            print(f"{indent}{message}")

    # =======================
    # BASIC MODEL TESTS
    # =======================

    def test_beta_model(self, config: str = 'quick') -> Dict:
        """
        Test Beta model functionality

        Args:
            config: Configuration to use ('quick', 'basic', 'advanced')

        Returns:
            Dictionary with test results
        """
        self._print("=" * 60)
        self._print("TESTING BETA MODEL")
        self._print("=" * 60)

        cfg = self.test_configs[config]
        results = {
            'config': cfg,
            'dgp_results': {},
            'performance': {}
        }

        # Initialize model
        model = ScoreDrivenERGM(model_type='beta', n_nodes=cfg['n_nodes'])

        # Generate base parameters
        np.random.seed(42)
        theta_0 = np.random.normal(0, 0.5, 2 * cfg['n_nodes'])

        # Test different DGPs
        dgp_types = ['constant', 'sinusoidal', 'step', 'ar1']
        dgp_params = {
            'sinusoidal': {'amplitude': 0.2},
            'step': {'step_size': 0.3},
            'ar1': {'phi': 0.9, 'sigma': 0.1}
        }

        for dgp_type in dgp_types:
            self._print(f"Testing {dgp_type} DGP...", 1)

            # Generate data
            kwargs = dgp_params.get(dgp_type, {})
            adjacency_sequence, theta_true = DataGenerator.generate_temporal_network_data(
                dgp_type, model, theta_0, cfg['T'], **kwargs
            )

            # Filter parameters
            sd_params = SDERGMParams(
                w=0.1 * theta_0,
                alpha=0.1 * np.ones(model.n_params),
                beta=0.9 * np.ones(model.n_params)
            )

            start_time = time.time()
            theta_filtered = model.filter_parameters(adjacency_sequence, sd_params, theta_0)
            filter_time = time.time() - start_time

            # Cross-sectional estimates
            start_time = time.time()
            theta_cs = ModelEvaluator.cross_sectional_estimates(adjacency_sequence, model)
            cs_time = time.time() - start_time

            # Compute metrics
            def edge_logits(path):
                logits = path[:, cfg['n_nodes']:, None] + path[:, None, :cfg['n_nodes']]
                return logits[:, ~np.eye(cfg['n_nodes'], dtype=bool)]
            rmse_sd = ModelEvaluator.compute_rmse(edge_logits(theta_true), edge_logits(theta_filtered))
            rmse_cs = ModelEvaluator.compute_rmse(edge_logits(theta_true), edge_logits(theta_cs))

            results['dgp_results'][dgp_type] = {
                'rmse_sd': rmse_sd,
                'rmse_cs': rmse_cs,
                'improvement': (rmse_cs - rmse_sd) / rmse_cs * 100,
                'filter_time': filter_time,
                'cs_time': cs_time
            }

            self._print(f"  SD-ERGM RMSE: {rmse_sd:.4f}", 2)
            self._print(f"  Cross-sectional RMSE: {rmse_cs:.4f}", 2)
            self._print(f"  Improvement: {results['dgp_results'][dgp_type]['improvement']:.1f}%", 2)

        self.results['beta_model'] = results
        return results

    def test_pseudo_likelihood(self, config: str = 'quick') -> Dict:
        """
        Test pseudo-likelihood model with GWESP statistics

        Args:
            config: Configuration to use

        Returns:
            Dictionary with test results
        """
        self._print("=" * 60)
        self._print("TESTING PSEUDO-LIKELIHOOD MODEL")
        self._print("=" * 60)

        cfg = self.test_configs[config]
        results = {
            'config': cfg,
            'statistics_tested': [],
            'performance': {}
        }

        # Test different statistics combinations
        statistics_sets = [
            ('Basic', [EdgesStatistic()]),
            ('GWESP', [EdgesStatistic(), GWESPStatistic(decay=0.5)]),
            ('Complex', [EdgesStatistic(), GWESPStatistic(), TriangleStatistic()])
        ]

        for stat_name, statistics in statistics_sets:
            self._print(f"Testing {stat_name} statistics...", 1)

            model = ScoreDrivenERGM(model_type='pseudo', statistics=statistics)
            n_params = len(statistics)
            theta_0 = np.array([-2.0] + [0.5] * (n_params - 1))

            # Generate data
            adjacency_sequence, theta_true = DataGenerator.generate_temporal_network_data(
                'ar1', model, theta_0, cfg['T'], phi=0.95, sigma=0.08
            )

            # Filter parameters
            sd_params = SDERGMParams(
                w=0.05 * theta_0,
                alpha=0.05 * np.ones(n_params),
                beta=0.95 * np.ones(n_params)
            )

            start_time = time.time()
            theta_filtered = model.filter_parameters(adjacency_sequence, sd_params, theta_0)
            filter_time = time.time() - start_time

            # Compute RMSE
            rmse = ModelEvaluator.compute_rmse(theta_true, theta_filtered)

            results['statistics_tested'].append({
                'name': stat_name,
                'n_statistics': n_params,
                'rmse': rmse,
                'computation_time': filter_time
            })

            self._print(f"  RMSE: {rmse:.4f}", 2)
            self._print(f"  Time: {filter_time:.3f}s", 2)

        self.results['pseudo_likelihood'] = results
        return results

    # =======================
    # ADVANCED FEATURE TESTS
    # =======================

    def test_robust_filtering(self, config: str = 'quick') -> Dict:
        """Test robust SD-ERGM with outlier detection"""
        self._print("=" * 60)
        self._print("TESTING ROBUST SD-ERGM")
        self._print("=" * 60)

        cfg = self.test_configs[config]
        n_nodes = cfg['n_nodes']
        T = cfg['T']

        # Initialize base model
        base_model = ScoreDrivenERGM(model_type='beta', n_nodes=n_nodes)
        theta_0 = np.random.normal(0, 0.3, 2 * n_nodes)

        # Generate clean data
        adjacency_sequence, theta_true = DataGenerator.generate_temporal_network_data(
            'ar1', base_model, theta_0, T, phi=0.9, sigma=0.1
        )

        # Add outliers
        n_outliers = max(3, T // 20)
        outlier_times = np.random.choice(T, n_outliers, replace=False)

        for t in outlier_times:
            # Create outlier network (much denser)
            outlier_network = np.random.binomial(1, 0.8, (n_nodes, n_nodes))
            np.fill_diagonal(outlier_network, 0)
            adjacency_sequence[t] = outlier_network

        # Standard filtering
        sd_params = SDERGMParams(
            w=0.05 * theta_0,
            alpha=0.1 * np.ones(base_model.n_params),
            beta=0.9 * np.ones(base_model.n_params)
        )

        theta_standard = base_model.filter_parameters(adjacency_sequence, sd_params, theta_0)

        # Robust filtering
        robust_model = RobustSDERGM(base_model, outlier_threshold=2.5)
        theta_robust, outliers_detected = robust_model.robust_filter_parameters(
            adjacency_sequence, sd_params
        )

        # Compute metrics
        rmse_standard = ModelEvaluator.compute_rmse(theta_true, theta_standard)
        rmse_robust = ModelEvaluator.compute_rmse(theta_true, theta_robust)

        # Detection accuracy
        detected_times = [t for t, is_outlier in enumerate(outliers_detected) if is_outlier]
        true_positives = len(set(detected_times) & set(outlier_times))
        precision = true_positives / len(detected_times) if detected_times else 0
        recall = true_positives / len(outlier_times) if outlier_times else 0

        results = {
            'rmse_standard': rmse_standard,
            'rmse_robust': rmse_robust,
            'improvement': (rmse_standard - rmse_robust) / rmse_standard * 100,
            'outliers_added': len(outlier_times),
            'outliers_detected': len(detected_times),
            'precision': precision,
            'recall': recall
        }

        self._print(f"Standard RMSE: {rmse_standard:.4f}", 1)
        self._print(f"Robust RMSE: {rmse_robust:.4f}", 1)
        self._print(f"Improvement: {results['improvement']:.1f}%", 1)
        self._print(f"Detection - Precision: {precision:.2f}, Recall: {recall:.2f}", 1)

        self.results['robust_filtering'] = results
        return results

    def test_online_learning(self, config: str = 'quick') -> Dict:
        """Test online SD-ERGM for streaming data"""
        self._print("=" * 60)
        self._print("TESTING ONLINE SD-ERGM")
        self._print("=" * 60)

        cfg = self.test_configs[config]
        n_nodes = cfg['n_nodes']
        T = cfg['T']

        # Initialize model
        base_model = ScoreDrivenERGM(model_type='beta', n_nodes=n_nodes)
        theta_0 = np.random.normal(0, 0.3, 2 * n_nodes)

        # Generate streaming data
        adjacency_sequence, theta_true = DataGenerator.generate_temporal_network_data(
            'sinusoidal', base_model, theta_0, T, amplitude=0.2
        )

        # Online learning
        online_model = OnlineSDERGM(base_model, forgetting_factor=0.95)
        online_model.initialize_online(theta_0)

        sd_params = SDERGMParams(
            w=0.05 * theta_0,
            alpha=0.1 * np.ones(base_model.n_params),
            beta=0.9 * np.ones(base_model.n_params)
        )

        online_estimates = []
        processing_times = []

        for t in range(T):
            start_time = time.time()
            theta_online = online_model.online_update(adjacency_sequence[t], sd_params)
            processing_times.append(time.time() - start_time)
            online_estimates.append(theta_online.copy())

        online_estimates = np.array(online_estimates)

        # Batch comparison
        theta_batch = base_model.filter_parameters(adjacency_sequence, sd_params, theta_0)

        # Compute metrics
        rmse_online = ModelEvaluator.compute_rmse(theta_true, online_estimates)
        rmse_batch = ModelEvaluator.compute_rmse(theta_true, theta_batch)

        results = {
            'rmse_online': rmse_online,
            'rmse_batch': rmse_batch,
            'avg_processing_time': np.mean(processing_times),
            'max_processing_time': np.max(processing_times),
            'convergence_time': online_model.t,
            'final_variance': np.mean(online_model.online_var_score) if online_model.online_var_score is not None else 0
        }

        self._print(f"Online RMSE: {rmse_online:.4f}", 1)
        self._print(f"Batch RMSE: {rmse_batch:.4f}", 1)
        self._print(f"Avg processing time: {results['avg_processing_time'] * 1000:.2f}ms", 1)

        self.results['online_learning'] = results
        return results

    def test_hierarchical_networks(self, config: str = 'quick') -> Dict:
        """Test hierarchical multi-layer networks"""
        self._print("=" * 60)
        self._print("TESTING HIERARCHICAL MULTI-LAYER NETWORKS")
        self._print("=" * 60)

        cfg = self.test_configs[config]
        n_nodes = min(cfg['n_nodes'], 12)  # Limit size for computational efficiency
        n_layers = 3
        T = cfg['T']

        # Create hierarchical model
        hierarchical_model = HierarchicalSDERGM(n_layers, n_nodes, shared_statistics=False)

        # Generate multi-layer data
        multilayer_sequence = []
        layer_theta_true = []

        layer_base_params = [
            np.array([-1.8, 0.6]),  # High connectivity layer
            np.array([-2.5, 0.4]),  # Medium connectivity layer
            np.array([-3.2, 0.3])  # Low connectivity layer
        ]

        dgp_types = ['ar1', 'sinusoidal', 'step']
        dgp_params = [
            {'phi': 0.95, 'sigma': 0.08},
            {'amplitude': 0.15},
            {'step_size': 0.3}
        ]

        self._print("Generating multi-layer data...", 1)

        for layer in range(n_layers):
            model = hierarchical_model.layer_models[layer]
            base_params = layer_base_params[layer]

            adj_seq, theta_seq = DataGenerator.generate_temporal_network_data(
                dgp_types[layer], model, base_params, T, **dgp_params[layer]
            )

            if layer == 0:
                multilayer_sequence = [[adj] for adj in adj_seq]
                layer_theta_true = [theta_seq]
            else:
                for t in range(T):
                    multilayer_sequence[t].append(adj_seq[t])
                layer_theta_true.append(theta_seq)

        # Filter with cross-layer coupling
        self._print("Filtering with cross-layer coupling...", 1)

        coupling_strengths = [0.0, 0.1, 0.2]
        coupling_results = []

        for coupling in coupling_strengths:
            layer_parameters = hierarchical_model.filter_multilayer_parameters(
                multilayer_sequence, cross_layer_coupling=coupling
            )

            # Compute layer-wise RMSE
            layer_rmse = []
            for layer in range(n_layers):
                rmse = ModelEvaluator.compute_rmse(layer_theta_true[layer], layer_parameters[layer])
                layer_rmse.append(rmse)

            # Compute cross-layer correlations
            cross_correlations = np.zeros((n_layers, n_layers))
            for i in range(n_layers):
                for j in range(n_layers):
                    if i != j:
                        corr = np.corrcoef(layer_parameters[i][:, 0], layer_parameters[j][:, 0])[0, 1]
                        cross_correlations[i, j] = corr

            coupling_results.append({
                'coupling_strength': coupling,
                'layer_rmse': layer_rmse,
                'avg_rmse': np.mean(layer_rmse),
                'avg_correlation': np.mean(np.abs(cross_correlations[cross_correlations != 0]))
            })

        # Select best coupling
        best_coupling = min(coupling_results, key=lambda x: x['avg_rmse'])

        results = {
            'n_layers': n_layers,
            'n_nodes': n_nodes,
            'best_coupling': best_coupling['coupling_strength'],
            'best_avg_rmse': best_coupling['avg_rmse'],
            'layer_performance': best_coupling['layer_rmse'],
            'cross_layer_correlation': best_coupling['avg_correlation'],
            'coupling_comparison': coupling_results
        }

        self._print(f"Best coupling strength: {best_coupling['coupling_strength']}", 1)
        self._print(f"Average RMSE: {best_coupling['avg_rmse']:.4f}", 1)
        self._print(f"Cross-layer correlation: {best_coupling['avg_correlation']:.3f}", 1)

        self.results['hierarchical_networks'] = results
        return results

    def test_bayesian_inference(self, config: str = 'quick') -> Dict:
        """Test Bayesian SD-ERGM with uncertainty quantification"""
        self._print("=" * 60)
        self._print("TESTING BAYESIAN SD-ERGM")
        self._print("=" * 60)

        # Use smaller network for Bayesian methods
        n_nodes = 6
        T = 30

        # Initialize model
        base_model = ScoreDrivenERGM(model_type='beta', n_nodes=n_nodes)
        theta_0 = np.random.normal(0, 0.4, 2 * n_nodes)

        # Generate data
        adjacency_sequence, theta_true = DataGenerator.generate_temporal_network_data(
            'ar1', base_model, theta_0, T, phi=0.9, sigma=0.12
        )

        # Bayesian estimation
        self._print("Running conditional Bayesian MCMC...", 1)

        prior_params = {
            'w_sd': 0.5,
            'alpha_shape': 2.0,
            'alpha_scale': 0.05,
            'beta_a': 8.0,
            'beta_b': 2.0
        }

        bayesian_model = BayesianSDERGM(base_model, prior_params)

        # Run simplified MCMC
        mcmc_results = bayesian_model.mcmc_estimation(
            adjacency_sequence, n_samples=100, n_burn=50, random_state=42
        )

        # Frequentist comparison
        sd_params_freq = SDERGMParams(
            w=0.05 * theta_0,
            alpha=0.1 * np.ones(base_model.n_params),
            beta=0.9 * np.ones(base_model.n_params)
        )

        theta_filtered_freq = base_model.filter_parameters(adjacency_sequence, sd_params_freq, theta_0)

        # Bayesian filtering with posterior mean
        sd_params_bayes = SDERGMParams(
            w=mcmc_results['posterior_mean']['w'],
            alpha=mcmc_results['posterior_mean']['alpha'],
            beta=mcmc_results['posterior_mean']['beta']
        )

        theta_filtered_bayes = base_model.filter_parameters(adjacency_sequence, sd_params_bayes, theta_0)

        # Compute metrics
        rmse_freq = ModelEvaluator.compute_rmse(theta_true, theta_filtered_freq)
        rmse_bayes = ModelEvaluator.compute_rmse(theta_true, theta_filtered_bayes)

        # Coverage analysis
        theta_samples = mcmc_results['theta_samples']
        theta_lower = np.percentile(theta_samples, 2.5, axis=0)
        theta_upper = np.percentile(theta_samples, 97.5, axis=0)
        coverage = np.mean((theta_true >= theta_lower) & (theta_true <= theta_upper))

        results = {
            'rmse_frequentist': rmse_freq,
            'rmse_bayesian': rmse_bayes,
            'coverage_95': coverage,
            'posterior_mean_w': mcmc_results['posterior_mean']['w'][:3].tolist(),
            'posterior_std_w': mcmc_results['posterior_std']['w'][:3].tolist(),
            'n_samples': 100,
            'n_burn': 50
        }

        self._print(f"Frequentist RMSE: {rmse_freq:.4f}", 1)
        self._print(f"Bayesian RMSE: {rmse_bayes:.4f}", 1)
        self._print(f"95% Coverage: {coverage:.1%}", 1)

        self.results['bayesian_inference'] = results
        return results

    def test_financial_application(self, config: str = 'quick') -> Dict:
        """Test financial network application"""
        self._print("=" * 60)
        self._print("TESTING FINANCIAL NETWORK APPLICATION")
        self._print("=" * 60)

        n_banks = 15
        T_total = 60

        # Initialize financial application
        fin_app = FinancialNetworkApplication(n_banks)

        # Generate three-phase data (pre-crisis, crisis, recovery)
        T_pre = T_total // 3
        T_crisis = T_total // 3
        T_post = T_total - T_pre - T_crisis

        # Parameters for different phases
        pre_crisis_theta = np.array([-1.5, 0.8, 0.2, 0.15])
        crisis_theta = np.array([-2.5, 0.3, 0.6, 0.4])
        post_crisis_theta = np.array([-2.0, 0.5, 0.4, 0.25])

        self._print("Generating financial crisis scenario...", 1)

        # Generate data for each phase
        adj_pre, theta_pre = DataGenerator.generate_temporal_network_data(
            'ar1', fin_app.model, pre_crisis_theta, T_pre, phi=0.95, sigma=0.05
        )

        adj_crisis, theta_crisis = DataGenerator.generate_temporal_network_data(
            'step', fin_app.model, crisis_theta, T_crisis, step_size=0.0
        )

        adj_post, theta_post = DataGenerator.generate_temporal_network_data(
            'sinusoidal', fin_app.model, post_crisis_theta, T_post, amplitude=0.1
        )

        # Combine sequences
        adjacency_sequence = adj_pre + adj_crisis + adj_post
        theta_true = np.vstack([theta_pre, theta_crisis, theta_post])

        # Filter parameters
        self._print("Filtering parameters and computing risk indicators...", 1)

        sd_params = SDERGMParams(
            w=0.02 * pre_crisis_theta,
            alpha=0.12 * np.ones(len(pre_crisis_theta)),
            beta=0.88 * np.ones(len(pre_crisis_theta))
        )

        theta_filtered = fin_app.model.filter_parameters(adjacency_sequence, sd_params, pre_crisis_theta)

        # Compute systemic risk
        risk_indicator = fin_app.systemic_risk_indicator(adjacency_sequence, theta_filtered)

        # Stress testing
        self._print("Conducting stress tests...", 1)

        stress_scenarios = {
            'liquidity_freeze': {'edge_shock': -0.6, 'concentration_shock': 0.2},
            'bank_failure': {'edge_shock': -0.4, 'concentration_shock': 0.5}
        }

        stress_results = fin_app.stress_testing(theta_filtered[T_pre // 2], stress_scenarios)

        # Early warning indicators
        window = 5
        param_volatility = np.zeros(T_total - window + 1)
        for t in range(window - 1, T_total):
            param_volatility[t - window + 1] = np.std(theta_filtered[t - window + 1:t + 1, 0])

        # Detect warning signals
        warning_threshold = np.percentile(param_volatility[:T_pre - 5], 90)
        warnings_before_crisis = np.sum(param_volatility[T_pre - 10:T_pre] > warning_threshold)

        results = {
            'n_banks': n_banks,
            'pre_crisis_risk': np.mean(risk_indicator[:T_pre]),
            'crisis_risk': np.mean(risk_indicator[T_pre:T_pre + T_crisis]),
            'post_crisis_risk': np.mean(risk_indicator[T_pre + T_crisis:]),
            'stress_test_results': {
                scenario: {'connectivity_loss': result['connectivity_loss']}
                for scenario, result in stress_results.items()
            },
            'early_warnings': warnings_before_crisis,
            'warning_window': 10,
            'filtering_rmse': ModelEvaluator.compute_rmse(theta_true, theta_filtered)
        }

        self._print(f"Pre-crisis risk: {results['pre_crisis_risk']:.3f}", 1)
        self._print(f"Crisis risk: {results['crisis_risk']:.3f}", 1)
        self._print(f"Post-crisis risk: {results['post_crisis_risk']:.3f}", 1)
        self._print(f"Early warnings issued: {warnings_before_crisis}/10", 1)

        self.results['financial_application'] = results
        return results

    # =======================
    # PERFORMANCE TESTS
    # =======================

    def test_scalability(self) -> Dict:
        """Test computational scalability"""
        self._print("=" * 60)
        self._print("TESTING SCALABILITY")
        self._print("=" * 60)

        network_sizes = [5, 10, 15, 20, 25]
        T = 50

        results = {
            'sizes': network_sizes,
            'beta_model': {'times': [], 'rmse': []},
            'pseudo_likelihood': {'times': [], 'rmse': []},
            'network_density': []
        }

        for n_nodes in network_sizes:
            self._print(f"Testing size {n_nodes}...", 1)

            # Beta model test
            model_beta = ScoreDrivenERGM(model_type='beta', n_nodes=n_nodes)
            theta_0_beta = np.random.normal(0, 0.3, 2 * n_nodes)

            adj_seq, theta_true = DataGenerator.generate_temporal_network_data(
                'ar1', model_beta, theta_0_beta, T, phi=0.9, sigma=0.1
            )

            sd_params = SDERGMParams(
                w=0.1 * theta_0_beta,
                alpha=0.1 * np.ones(model_beta.n_params),
                beta=0.9 * np.ones(model_beta.n_params)
            )

            start_time = time.time()
            theta_filtered = model_beta.filter_parameters(adj_seq, sd_params, theta_0_beta)
            beta_time = time.time() - start_time

            beta_rmse = ModelEvaluator.compute_rmse(theta_true, theta_filtered)

            results['beta_model']['times'].append(beta_time)
            results['beta_model']['rmse'].append(beta_rmse)

            # Pseudo-likelihood test
            if n_nodes <= 15:  # Limit for computational efficiency
                statistics = [EdgesStatistic(), GWESPStatistic()]
                model_pseudo = ScoreDrivenERGM(model_type='pseudo', statistics=statistics)
                theta_0_pseudo = np.array([-2.0, 0.5])

                adj_seq, theta_true = DataGenerator.generate_temporal_network_data(
                    'ar1', model_pseudo, theta_0_pseudo, T, phi=0.95, sigma=0.05
                )

                sd_params_pseudo = SDERGMParams(
                    w=0.05 * theta_0_pseudo,
                    alpha=0.05 * np.ones(len(statistics)),
                    beta=0.95 * np.ones(len(statistics))
                )

                start_time = time.time()
                theta_filtered = model_pseudo.filter_parameters(adj_seq, sd_params_pseudo, theta_0_pseudo)
                pseudo_time = time.time() - start_time

                pseudo_rmse = ModelEvaluator.compute_rmse(theta_true, theta_filtered)

                results['pseudo_likelihood']['times'].append(pseudo_time)
                results['pseudo_likelihood']['rmse'].append(pseudo_rmse)

            # Calculate density
            density = np.mean([np.sum(adj) / (n_nodes * (n_nodes - 1)) for adj in adj_seq])
            results['network_density'].append(density)

            self._print(f"  Beta: {beta_time:.3f}s, RMSE: {beta_rmse:.4f}", 2)
            if n_nodes <= 15:
                self._print(f"  Pseudo: {pseudo_time:.3f}s, RMSE: {pseudo_rmse:.4f}", 2)

        self.results['scalability'] = results
        return results

    def test_convergence(self, config: str = 'quick') -> Dict:
        """Test convergence properties of SD-ERGM"""
        self._print("=" * 60)
        self._print("TESTING CONVERGENCE PROPERTIES")
        self._print("=" * 60)

        cfg = self.test_configs[config]
        n_nodes = cfg['n_nodes']
        T = 200  # Longer sequence for convergence

        # Initialize model
        model = ScoreDrivenERGM(model_type='beta', n_nodes=n_nodes)
        theta_0 = np.random.normal(0, 0.3, 2 * n_nodes)

        # Generate data with sudden change
        adjacency_sequence, theta_true = DataGenerator.generate_temporal_network_data(
            'step', model, theta_0, T, step_size=0.5
        )

        # Test different alpha values
        alpha_values = [0.01, 0.05, 0.1, 0.2]
        convergence_results = []

        for alpha in alpha_values:
            self._print(f"Testing alpha = {alpha}...", 1)

            sd_params = SDERGMParams(
                w=0.1 * theta_0,
                alpha=alpha * np.ones(model.n_params),
                beta=0.9 * np.ones(model.n_params)
            )

            theta_filtered = model.filter_parameters(adjacency_sequence, sd_params, theta_0)

            # Measure convergence after step change
            change_point = T // 2
            post_change = theta_filtered[change_point:]

            # Time to convergence (when variance stabilizes)
            window = 10
            variances = []
            for t in range(window, len(post_change)):
                var = np.var(post_change[t - window:t, 0])
                variances.append(var)

            # Find convergence point (variance below threshold)
            threshold = 0.01
            convergence_time = len(variances)
            for t, var in enumerate(variances):
                if var < threshold:
                    convergence_time = t + window
                    break

            rmse = ModelEvaluator.compute_rmse(theta_true, theta_filtered)

            convergence_results.append({
                'alpha': alpha,
                'convergence_time': convergence_time,
                'rmse': rmse,
                'final_variance': variances[-1] if variances else 0
            })

            self._print(f"  Convergence time: {convergence_time} steps", 2)
            self._print(f"  RMSE: {rmse:.4f}", 2)

        results = {
            'alpha_values': alpha_values,
            'convergence_results': convergence_results,
            'optimal_alpha': min(convergence_results, key=lambda x: x['rmse'])['alpha']
        }

        self._print(f"Optimal alpha: {results['optimal_alpha']}", 1)

        self.results['convergence'] = results
        return results

    # =======================
    # COMPREHENSIVE TEST SUITE
    # =======================

    def test_mle_estimation(self, config: str = 'quick') -> Dict:
        """
        Test MLE estimation for different network statistics

        Args:
            config: Configuration to use

        Returns:
            Dictionary with MLE test results
        """
        self._print("=" * 60)
        self._print("TESTING MLE ESTIMATION FOR DIFFERENT STATISTICS")
        self._print("=" * 60)

        cfg = self.test_configs[config]
        n_nodes = min(cfg['n_nodes'], 10)  # Limit for computational efficiency

        results = {
            'beta_model': {},
            'pseudo_likelihood': {},
            'statistics_comparison': []
        }

        # -----------------------
        # Test Beta Model MLE
        # -----------------------
        self._print("Testing Beta Model MLE...", 1)

        from statistical_engine.ml_routines import FastBetaModel

        beta_model = FastBetaModel(n_nodes)

        # Generate true parameters
        true_in_params = np.random.normal(0, 0.5, n_nodes)
        true_out_params = np.random.normal(0, 0.5, n_nodes)

        # Create adjacency matrix from true parameters (vectorized)
        prob_matrix = beta_model._compute_prob_matrix(true_in_params, true_out_params)
        test_adjacency = np.random.binomial(1, prob_matrix)
        np.fill_diagonal(test_adjacency, 0)

        # Estimate parameters using model's mle_estimate
        start_time = time.time()
        est_in_params, est_out_params = beta_model.mle_estimate(test_adjacency)
        mle_time = time.time() - start_time

        # For fair comparison apply identification shift to true_in (same convention used in models)
        true_in_adj = true_in_params - true_in_params[-1]
        true_out_adj = true_out_params + true_in_params[-1]

        in_error = float(np.mean(np.abs(true_in_adj - est_in_params)))
        out_error = float(np.mean(np.abs(true_out_adj - est_out_params)))

        # Log-likelihood at true and estimated parameters
        ll_true = float(beta_model.log_likelihood(test_adjacency, true_in_params, true_out_params))
        ll_est = float(beta_model.log_likelihood(test_adjacency, est_in_params, est_out_params))

        results['beta_model'] = {
            'estimation_time': mle_time,
            'in_param_error': in_error,
            'out_param_error': out_error,
            'log_likelihood_true': ll_true,
            'log_likelihood_estimated': ll_est,
            'likelihood_improvement': ll_est - ll_true
        }

        self._print(f"  Estimation time: {mle_time:.3f}s", 2)
        self._print(f"  In-param error: {in_error:.4f}", 2)
        self._print(f"  Out-param error: {out_error:.4f}", 2)
        self._print(f"  LL improvement: {ll_est - ll_true:.2f}", 2)

        # ---------------------------------------------
        # Test different statistics combinations for pseudo-likelihood
        # ---------------------------------------------
        self._print("\nTesting Pseudo-Likelihood MLE with different statistics...", 1)

        from data_handler.data_structures import (EdgesStatistic, ReciprocityStatistic,
                                                  GWESPStatistic, TriangleStatistic,
                                                  TwoStarsStatistic)
        from statistical_engine.ml_routines import FastPseudoLikelihood
        import scipy.optimize as opt

        statistics_combinations = [
            ('Edges only', [EdgesStatistic()]),
            ('Edges + Reciprocity', [EdgesStatistic(), ReciprocityStatistic()]),
            ('Edges + GWESP', [EdgesStatistic(), GWESPStatistic(decay=0.5)]),
            ('Edges + Triangles', [EdgesStatistic(), TriangleStatistic()]),
            ('Edges + TwoStars', [EdgesStatistic(), TwoStarsStatistic()]),
            ('Full suite', [
                EdgesStatistic(),
                GWESPStatistic(decay=0.5),
                ReciprocityStatistic(),
                TriangleStatistic()
            ])
        ]

        for stat_name, statistics in statistics_combinations:
            self._print(f"  Testing {stat_name}...", 2)

            # Create ScoreDrivenERGM for simulation
            model = ScoreDrivenERGM(model_type='pseudo', statistics=statistics)
            n_params = len(statistics)

            # True parameters for simulation
            true_theta = np.array([-2.0] + [0.3] * (n_params - 1))

            # Simulate network and ensure no self-loops
            test_network = model.simulate_network(true_theta)
            np.fill_diagonal(test_network, 0)

            # Create FastPseudoLikelihood instance with these statistics
            fpl = FastPseudoLikelihood(statistics=statistics)

            # Define objective function (minimize negative pseudo-likelihood)
            def objective(theta):
                return -fpl.log_pseudo_likelihood(test_network, theta)

            # MLE estimation using scipy.optimize
            start_time = time.time()

            # Initialize with small random values
            theta_init = np.random.normal(0, 0.1, n_params)
            theta_init[0] = -2.0  # Start edges parameter at reasonable value

            try:
                result = opt.minimize(
                    objective,
                    theta_init,
                    method='BFGS',
                    options={'maxiter': 500, 'gtol': 1e-5}
                )
                est_theta = result.x
                optimization_success = result.success
            except Exception as e:
                self._print(f"    Optimization failed: {e}", 3)
                est_theta = theta_init
                optimization_success = False

            mle_time = time.time() - start_time

            # Compute errors
            param_error = float(np.mean(np.abs(true_theta - est_theta)))

            # Log-likelihood comparison
            ll_true = float(fpl.log_pseudo_likelihood(test_network, true_theta))
            ll_est = float(fpl.log_pseudo_likelihood(test_network, est_theta))

            stat_result = {
                'name': stat_name,
                'n_parameters': n_params,
                'estimation_time': mle_time,
                'parameter_error': param_error,
                'log_likelihood_true': ll_true,
                'log_likelihood_estimated': ll_est,
                'likelihood_improvement': ll_est - ll_true,
                'optimization_success': optimization_success
            }

            results['statistics_comparison'].append(stat_result)

            self._print(f"    Param error: {param_error:.4f}", 3)
            self._print(f"    LL improvement: {ll_est - ll_true:.2f}", 3)
            self._print(f"    Time: {mle_time:.3f}s", 3)
            if not optimization_success:
                self._print(f"    Warning: Optimization did not converge", 3)

        # Summary statistics
        successful_tests = [r for r in results['statistics_comparison'] if r['optimization_success']]
        if successful_tests:
            avg_error = np.mean([r['parameter_error'] for r in successful_tests])
            avg_time = np.mean([r['estimation_time'] for r in successful_tests])

            results['pseudo_likelihood'] = {
                'n_successful_tests': len(successful_tests),
                'n_total_tests': len(results['statistics_comparison']),
                'avg_parameter_error': avg_error,
                'avg_estimation_time': avg_time
            }

            self._print(f"\nSummary: {len(successful_tests)}/{len(results['statistics_comparison'])} tests successful",
                        1)
            self._print(f"Average parameter error: {avg_error:.4f}", 1)
            self._print(f"Average estimation time: {avg_time:.3f}s", 1)

        self.results['mle_estimation'] = results
        return results
