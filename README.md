# Score-driven exponential random graph models

Research implementations of directed beta-model and ERGM pseudo-likelihood filters, inspired by [Di Gangi, Bormetti and Lillo](https://arxiv.org/abs/1905.10806). This repository is not a validated reproduction of every result in that paper.

## Install and check

From the repository root, with Python 3.11 or 3.12:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
```

The offline regression suite checks analytic scores against numerical derivatives, edge-statistic differences, Bernoulli probabilities, simulation frequencies, Bayesian targets using more than ten observations, prior recovery, and seeded reproducibility. GitHub Actions runs it on both supported Python versions. The older `test/_test_suite.py` is an exploratory simulation harness, not the assertion-based regression suite. An experiment finishing without an exception does not certify its statistical result.

## Model conventions

For a directed adjacency matrix, `Y[i,j]` is the edge from i to j. Incoming coefficients index columns, outgoing coefficients index rows:

```
p[i,j] = expit(out[i] + in[j]), i != j
score_in  = column_sums(Y - p)
score_out = row_sums(Y - p)
theta[t+1] = w + beta * theta[t] + alpha * score[t]
```

There are no self-loops. The Bernoulli log likelihood uses `logaddexp` for numerical stability. Static beta-model MLE fixes the last incoming coefficient to zero; the opposite shift is applied to outgoing coefficients. This removes the additive identification ridge while preserving edge probabilities. Parameter-recovery comparisons must use the same identification convention; comparing fitted edge probabilities avoids that arbitrary convention entirely.

Network statistics expose signed *toggle* changes. Pseudo-likelihood converts them to `g(Y with edge=1) - g(Y with edge=0)`. The simulator instead uses the signed toggle delta directly in a Barker acceptance probability, for additions and removals alike. Simulations use a finite number of MCMC steps, not exact draws; mixing is not guaranteed for general ERGMs.

`estimate_static_parameters` refilters observations under every candidate `(w, alpha, beta)` and maximizes the conditional observation likelihood, rather than regressing on a supplied filtered path. Its legacy `theta_filtered` argument supplies only the fixed initial state. For dependent-edge models the objective is a pseudo-likelihood, not the normalized ERGM likelihood. Optimization failures are reported. Nonnegative alpha and beta in [0,1) are parameter bounds, not a proof of filter stability.

## Bayesian inference

```python
import numpy as np
from statistical_engine.score_random_network import ScoreDrivenERGM, BayesianSDERGM

model = ScoreDrivenERGM('beta', n_nodes=3)
y = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
result = BayesianSDERGM(model, {}).mcmc_estimation(
    [y] * 12, n_samples=500, n_burn=200, random_state=42
)
print(result['acceptance_rates'])
```

All three parameter blocks are sampled using symmetric random walks in their original coordinates. Proposals outside alpha>0 or 0<beta<1 are rejected without clipping or redrawing. Both states use exactly the same full observation sequence and Normal/Gamma/Beta priors. Prior keys `w_sd`, `alpha_shape`, `alpha_scale`, `beta_a`, and `beta_b` are honored. `proposal_scales` optionally sets positive proposal standard deviations for `w`, `alpha`, and `beta`.

The initial filtered state is fixed at zero; latent paths are deterministic conditional on parameters and observed networks. For general ERGMs this samples a **pseudo-posterior**, not an exact normalized ERGM posterior. Outputs include parameter/path traces, log targets and block acceptance rates. These are research diagnostics: multiple-chain convergence, effective sample sizes, identification, predictive calibration and simulation-based coverage remain necessary for substantive inference. Passing the regression suite is not a convergence certificate.

## Experimental extensions

Hierarchical coupling, robust filtering, online rescaling and financial-network indicators remain experimental heuristics. Their presence does not establish a published estimator or a validated systemic-risk metric. The former variance-based “LM test” returned uncalibrated chi-square p-values; it now raises `NotImplementedError` until a properly derived and calibrated test is implemented.

The configuration runner is optional:

```sh
python -m test.config_runner config/sdergm_config.yaml --tests beta_model
```

Configuration currently controls harness settings and experiment selection. Not every descriptive field in the legacy YAML is consumed by every experiment; inspect the harness before interpreting its output. Historical results produced before the numerical corrections are not comparable without rerunning them.
