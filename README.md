# Score-Driven Exponential Random Graph Models (SD-ERGM)

## Understanding System Evolution Through Network Dynamics

Nature presents us with fascinating examples of self-organization and evolution. From the intricate patterns of neural networks to the complex structures of ecological systems, we observe a fundamental principle: systems naturally evolve from states of order toward greater complexity and interconnectedness.

Drawing inspiration from thermodynamics, we know that entropy serves as a powerful measure of system organization. While entropy typically increases over time as systems move toward greater randomness, this process reveals profound insights about how complex networks emerge and adapt.

## Theoretical Foundation

This work builds upon the pioneering insights of **Aleksandr Bogdanov's tectology theory**, which identified three fundamental types of system organization:

- **Organized systems** (low entropy) - where collective behavior exceeds the sum of individual components
- **Disorganized systems** - where fragmentation reduces overall system effectiveness  
- **Neutral systems** - where organizing and disorganizing forces reach dynamic equilibrium

Rather than viewing entropy increase as purely destructive, we recognize that most natural systems demonstrate remarkable resilience. They evolve toward out-of-equilibrium states with rich, distributed network structures that enable adaptation and growth.

## Research Foundation

This repository implements computational tools inspired by the groundbreaking work of **D. Di Gangi, G. Bormetti, and F. Lillo** (2024) on score-driven exponential random graph models. Their research provides a mathematical framework for tracking both entropy changes and the underlying network dynamics that drive system evolution.

📄 **Reference Paper:** [Score-driven exponential random graph models (Di Gangi, Bormetti & Lillo, 2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3394593)

## What This Repository Offers

These tools enable researchers and practitioners to:
- Monitor entropy evolution in complex network systems
- Understand the relationship between local interactions and global organization
- Model how systems maintain coherence while adapting to change
- Explore the mathematical beauty underlying natural self-organization processes

By combining rigorous mathematical modeling with intuitive understanding of system dynamics, this work opens new pathways for studying everything from social networks to biological systems, financial markets to technological infrastructures.

---

### Execution Pipeline

To execute the test suite for the Score-Driven Exponential Random Graph Models (SD-ERGM), you would typically use the provided `config_runner.py` script. This script acts as the main entry point, allowing for flexible and reproducible testing based on a YAML configuration file.

1.  **Preparation**: Ensure the test suite (`_test_suite.py`) and the runner script (`config_runner.py`) are accessible in your environment. You'll also need a YAML configuration file (e.g., `sdergm_config.yaml`) to specify which tests to run and what parameters to use. The runner script looks for this file by default but allows you to specify a different one.

2.  **Configuration**: Define your testing parameters in the YAML file. This includes:
    * **`global` settings**: Control verbosity, result saving, and the random seed for reproducibility.
    * **`test_configs`**: Define different test scenarios (e.g., `'quick'`, `'basic'`, `'advanced'`) by setting the number of nodes, time steps, and replications for the data generation process.
    * **`tests_to_run`**: A list of booleans to enable or disable specific tests (e.g., `beta_model`, `robust_filtering`, `scalability`).
    * **`output` settings**: Specify where to save the results and in what format (JSON, CSV, or both), along with an option to generate a summary report.

3.  **Execution**: Run the `config_runner.py` script from your command line. The script will parse the YAML configuration and execute the specified tests from `_test_suite.py`.
    * **Default Run**: To run the tests using the default configuration file (`sdergm_config.yaml`), simply execute the script without any arguments: `python config_runner.py`.
    * **Custom Run**: You can override the default configuration file by providing the path as an argument: `python config_runner.py config/custom_config.yaml`.
    * **Specific Tests**: To run a subset of tests, you can pass an `override_tests` list to the `run_configured_tests` function. The script's example usage block demonstrates how to programmatically call it to run specific tests, for example: `run_configured_tests("sdergm_config.yaml", override_tests=['beta_model', 'robust_filtering'])`.

4.  **Results**: The script will print the progress and status of each test to the console. If the `save_results` option is enabled in the configuration, the results will be saved to the specified output directory in the chosen format(s) (JSON, CSV, and a summary text report). The summary report provides an overview of which tests succeeded, which failed, and key performance metrics.

---

### Available Tests and How to Trigger Them

You can trigger a variety of tests available in the SD-ERGM test suite by enabling them in the YAML configuration file or by programmatically overriding the tests to be run.

#### Basic Model Tests
These tests validate the core functionality of the Beta and pseudo-likelihood models.

* **`test_beta_model`**: This test assesses the performance of the Beta model on different data-generating processes (DGPs), such as constant, sinusoidal, step, and AR(1). It compares the performance of the score-driven filter against traditional cross-sectional estimates.
    * **Trigger**: In your `sdergm_config.yaml` file, set `beta_model: True` under the `tests_to_run` section.

* **`test_pseudo_likelihood`**: This test evaluates the pseudo-likelihood model using various network statistics, including a basic set, one with GWESP, and a more complex combination.
    * **Trigger**: In your `sdergm_config.yaml` file, set `pseudo_likelihood: True` under the `tests_to_run` section.

---

#### Advanced Feature Tests
These tests focus on the specialized and extended capabilities of the SD-ERGM models.

* **`test_robust_filtering`**: This test checks the ability of the robust SD-ERGM to detect and mitigate the effect of outliers in a time series of networks.
    * **Trigger**: In your `sdergm_config.yaml` file, set `robust_filtering: True` under the `tests_to_run` section.

* **`test_online_learning`**: This test evaluates the online SD-ERGM model's performance on streaming data by comparing its estimates to those from a batch process.
    * **Trigger**: In your `sdergm_config.yaml` file, set `online_learning: True` under the `tests_to_run` section.

* **`test_hierarchical_networks`**: This test assesses the model's capacity to handle multi-layer hierarchical networks and the effectiveness of cross-layer coupling.
    * **Trigger**: In your `sdergm_config.yaml` file, set `hierarchical_networks: True` under the `tests_to_run` section.

* **`test_bayesian_inference`**: This test validates the Bayesian SD-ERGM, which provides uncertainty quantification and compares its results with a frequentist approach.
    * **Trigger**: In your `sdergm_config.yaml` file, set `bayesian_inference: True` under the `tests_to_run` section.

* **`test_financial_application`**: This test applies the model to a financial network scenario to evaluate its ability to track systemic risk, conduct stress tests, and provide early warning signals.
    * **Trigger**: In your `sdergm_config.yaml` file, set `financial_application: True` under the `tests_to_run` section.

---

#### Performance and Estimation Tests
These tests focus on the computational and statistical properties of the models.

* **`test_scalability`**: This test measures the computational time and performance of the Beta and pseudo-likelihood models as the network size increases.
    * **Trigger**: In your `sdergm_config.yaml` file, set `scalability: True` under the `tests_to_run` section.

* **`test_convergence`**: This test investigates the convergence properties of the model, particularly how different values of the alpha parameter affect the speed of convergence after a sudden change in network dynamics.
    * **Trigger**: In your `sdergm_config.yaml` file, set `convergence: True` under the `tests_to_run` section.

* **`test_mle_estimation`**: This test validates the maximum likelihood estimation (MLE) routines for both the Beta model and pseudo-likelihood models with various statistics.
    * **Trigger**: In your `sdergm_config.yaml` file, set `mle_estimation: True` under the `tests_to_run` section.

---

### Customizing Filtering Parameters

The `config_runner.py` script provides a way to customize the score-driven **filtering parameters** (`w`, `alpha`, and `beta`). This is done through the `filter_parameters` section in the YAML configuration file. You can define these parameters either as a multiplier of the initial parameters (`w_multiplier`, `alpha`, `beta`) or by providing a custom array of values.

For example, to configure parameters for the `beta_model` test, you would add a section like this to your YAML file:

```yaml
filter_parameters:
  beta_model:
    w_multiplier: 0.05
    alpha: 0.1
    beta: 0.9
