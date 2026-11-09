"""
Configuration-based Test Runner for SD-ERGM Test Suite
========================================================
This module provides functions to run the SD-ERGM test suite using
YAML configuration files without modifying the original test code.

Author: @nclcc, niciccarone@gmail.com
"""

import yaml
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import time
from datetime import datetime


def load_config(config_path: str = "config/sdergm_config.yaml") -> Dict:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_config_to_suite(suite, config: Dict):
    """
    Apply configuration to an existing SDERGMTestSuite instance

    Args:
        suite: SDERGMTestSuite instance
        config: Configuration dictionary
    """
    # Apply global settings
    if 'global' in config:
        suite.verbose = config['global'].get('verbose', True)
        suite.save_results = config['global'].get('save_results', False)

        # Set random seed if specified
        if 'random_seed' in config['global']:
            np.random.seed(config['global']['random_seed'])

    # Override test configurations
    if 'test_configs' in config:
        for cfg_name, cfg_params in config['test_configs'].items():
            suite.test_configs[cfg_name] = cfg_params

    return suite


def get_filter_params(config: Dict, model_type: str, theta_0: np.ndarray):
    """
    Get filter parameters from configuration

    Args:
        config: Configuration dictionary
        model_type: 'beta_model' or 'pseudo_likelihood'
        theta_0: Initial parameters

    Returns:
        SDERGMParams object
    """
    from data_handler.data_structures import SDERGMParams

    filter_config = config.get('filter_parameters', {}).get(model_type, {})

    # Check for custom parameters first
    if filter_config.get('custom_w') is not None:
        w = np.array(filter_config['custom_w'])
    else:
        w_mult = filter_config.get('w_multiplier', 0.1)
        w = w_mult * theta_0

    if filter_config.get('custom_alpha') is not None:
        alpha = np.array(filter_config['custom_alpha'])
    else:
        alpha_val = filter_config.get('alpha', 0.1)
        alpha = alpha_val * np.ones(len(theta_0))

    if filter_config.get('custom_beta') is not None:
        beta = np.array(filter_config['custom_beta'])
    else:
        beta_val = filter_config.get('beta', 0.9)
        beta = beta_val * np.ones(len(theta_0))

    return SDERGMParams(w=w, alpha=alpha, beta=beta)


def get_dgp_params(config: Dict, dgp_type: str) -> Dict:
    """
    Get DGP parameters from configuration

    Args:
        config: Configuration dictionary
        dgp_type: Type of DGP

    Returns:
        Dictionary of DGP parameters
    """
    dgp_config = config.get('data_generation', {}).get('dgp_types', {})

    if dgp_type in dgp_config:
        return dgp_config[dgp_type].get('params', {})

    return {}


def run_configured_tests(config_path: str = "config/sdergm_config.yaml",
                         override_tests: Optional[List[str]] = None):
    """
    Run SD-ERGM tests based on YAML configuration

    Args:
        config_path: Path to configuration file
        override_tests: Optional list of tests to run (overrides config)

    Returns:
        Test results dictionary
    """
    # Load configuration
    config = load_config(config_path)

    # Import the test suite (assuming it's in the path)
    from test._test_suite import SDERGMTestSuite  # Adjust import as needed

    # Initialize suite with config settings
    suite = SDERGMTestSuite(
        verbose=config.get('global', {}).get('verbose', True),
        save_results=config.get('global', {}).get('save_results', False)
    )

    # Apply configuration
    suite = apply_config_to_suite(suite, config)

    # Determine which tests to run
    if override_tests:
        tests_to_run = override_tests
    else:
        tests_to_run = [
            test_name for test_name, enabled
            in config.get('tests_to_run', {}).items()
            if enabled
        ]

    # Get default configuration name
    default_cfg = config.get('default_config', 'basic')

    print(f"\n{'=' * 60}")
    print(f"SD-ERGM TEST SUITE - Configuration Based Run")
    print(f"{'=' * 60}")
    print(f"Configuration file: {config_path}")
    print(f"Default config: {default_cfg}")
    print(f"Tests to run: {len(tests_to_run)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    # Run each test with custom parameters
    results = {}
    for i, test_name in enumerate(tests_to_run, 1):
        print(f"\n[{i}/{len(tests_to_run)}] Running: {test_name}")
        print("-" * 40)

        try:
            # Get test method
            test_method = getattr(suite, f'test_{test_name}')

            # Check for test-specific overrides
            test_specific = config.get('test_specific', {}).get(test_name, {})

            # Modify suite parameters based on test-specific config
            if test_name == 'robust_filtering' and 'outlier_threshold' in test_specific:
                # This would need to be passed to the test method
                # Since we can't modify the original code, we store it for reference
                suite._test_specific_config = test_specific

            # Run test
            if test_name == 'scalability':
                # Scalability doesn't use config parameter
                result = test_method()
            else:
                # Check for config override
                cfg_override = test_specific.get('config_override', default_cfg)
                result = test_method(config=cfg_override)

            results[test_name] = result
            print(f"✓ {test_name} completed successfully")

        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            results[test_name] = {'error': str(e)}

    # Save results if configured
    if config.get('global', {}).get('save_results', False):
        save_results(results, config)

    print(f"\n{'=' * 60}")
    print(f"Test suite completed!")
    print(f"Successful tests: {sum(1 for r in results.values() if 'error' not in r)}/{len(results)}")
    print(f"{'=' * 60}\n")

    return results


def save_results(results: Dict, config: Dict):
    """
    Save test results based on configuration

    Args:
        results: Test results dictionary
        config: Configuration dictionary
    """
    output_config = config.get('output', {})
    output_dir = Path(config.get('global', {}).get('output_directory', './results'))
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save based on format
    format_type = output_config.get('format', 'json')

    if format_type in ['json', 'both']:
        json_path = output_dir / f'sdergm_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {json_path}")

    if format_type in ['csv', 'both']:
        # Flatten results for CSV
        import pandas as pd

        flattened = []
        for test_name, test_results in results.items():
            if isinstance(test_results, dict) and 'error' not in test_results:
                flat_result = {'test': test_name}
                flat_result.update(_flatten_dict(test_results))
                flattened.append(flat_result)

        if flattened:
            df = pd.DataFrame(flattened)
            csv_path = output_dir / f'sdergm_results_{timestamp}.csv'
            df.to_csv(csv_path, index=False)
            print(f"Results saved to: {csv_path}")

    # Save summary report if configured
    if output_config.get('summary_report', True):
        report_path = output_dir / f'sdergm_summary_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write(generate_summary_report(results, config))
        print(f"Summary report saved to: {report_path}")


def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """
    Flatten nested dictionary

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            # Convert to string for CSV
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def generate_summary_report(results: Dict, config: Dict) -> str:
    """
    Generate a summary report of test results

    Args:
        results: Test results dictionary
        config: Configuration dictionary

    Returns:
        Summary report as string
    """
    report = []
    report.append("=" * 70)
    report.append("SD-ERGM TEST SUITE SUMMARY REPORT")
    report.append("=" * 70)
    report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Configuration: {config.get('default_config', 'unknown')}")
    report.append("")

    # Test summary
    successful = sum(1 for r in results.values() if isinstance(r, dict) and 'error' not in r)
    failed = len(results) - successful

    report.append(f"Tests Run: {len(results)}")
    report.append(f"Successful: {successful}")
    report.append(f"Failed: {failed}")
    report.append("")

    # Individual test results
    report.append("-" * 70)
    report.append("INDIVIDUAL TEST RESULTS")
    report.append("-" * 70)

    for test_name, test_result in results.items():
        report.append(f"\n{test_name.upper()}")
        report.append("-" * len(test_name))

        if isinstance(test_result, dict) and 'error' in test_result:
            report.append(f"Status: FAILED")
            report.append(f"Error: {test_result['error']}")
        else:
            report.append(f"Status: SUCCESS")

            # Add key metrics based on test type
            if test_name == 'beta_model' and 'dgp_results' in test_result:
                for dgp, metrics in test_result['dgp_results'].items():
                    report.append(f"  {dgp}: RMSE={metrics.get('rmse_sd', 'N/A'):.4f}, "
                                  f"Improvement={metrics.get('improvement', 'N/A'):.1f}%")

            elif test_name == 'robust_filtering':
                report.append(f"  RMSE Improvement: {test_result.get('improvement', 'N/A'):.1f}%")
                report.append(f"  Detection Precision: {test_result.get('precision', 'N/A'):.2f}")
                report.append(f"  Detection Recall: {test_result.get('recall', 'N/A'):.2f}")

            elif test_name == 'scalability' and 'beta_model' in test_result:
                times = test_result['beta_model'].get('times', [])
                if times:
                    report.append(f"  Min time: {min(times):.3f}s")
                    report.append(f"  Max time: {max(times):.3f}s")

    report.append("")
    report.append("=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)

    return "\n".join(report)


# Utility functions for custom configurations
def create_custom_statistics(config: Dict):
    """
    Create custom statistics based on configuration

    Args:
        config: Configuration dictionary

    Returns:
        List of statistic objects
    """
    from data_handler.data_structures import (
        EdgesStatistic, ReciprocityStatistic, GWESPStatistic,
        TriangleStatistic, TwoStarsStatistic
    )

    statistics = []
    stats_config = config.get('test_specific', {}).get('pseudo_likelihood', {}).get('statistics_sets', [])

    for stat_set in stats_config:
        if not stat_set.get('enabled', True):
            continue

        for stat_name in stat_set.get('statistics', []):
            if stat_name == 'edges':
                statistics.append(EdgesStatistic())
            elif stat_name == 'reciprocity':
                statistics.append(ReciprocityStatistic())
            elif stat_name == 'gwesp':
                decay = stat_set.get('gwesp_decay', 0.5)
                statistics.append(GWESPStatistic(decay=decay))
            elif stat_name == 'triangles':
                statistics.append(TriangleStatistic())
            elif stat_name == 'twostars':
                statistics.append(TwoStarsStatistic())

    return statistics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run exploratory SD-ERGM experiments")
    parser.add_argument("config", nargs="?", default="config/sdergm_config.yaml")
    parser.add_argument("--tests", nargs="+", help="Experiment names, e.g. beta_model")
    args = parser.parse_args()
    results = run_configured_tests(args.config, override_tests=args.tests)
    raise SystemExit(int(any('error' in result for result in results.values())))
