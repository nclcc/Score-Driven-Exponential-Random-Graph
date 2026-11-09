import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit
from scipy.stats import norm, gamma, beta

from data_handler.data_structures import EdgesStatistic, ReciprocityStatistic, SDERGMParams
from statistical_engine.ml_routines import FastBetaModel, FastPseudoLikelihood
from statistical_engine.score_random_network import ScoreDrivenERGM, BayesianSDERGM


def network():
    return np.array([[0, 1, 0], [0, 0, 1], [1, 1, 0]])


def test_beta_score_matches_likelihood_gradient():
    model = FastBetaModel(3)
    params = np.array([0.2, -0.4, 0.1, -0.3, 0.5, 0.4])
    numerical = approx_derivative(lambda x: model.log_likelihood(network(), x[:3], x[3:]), params).ravel()
    assert_allclose(np.concatenate(model.compute_score(network(), params[:3], params[3:])), numerical, atol=1e-6)


def test_in_parameters_control_columns_and_out_parameters_control_rows():
    model = FastBetaModel(3)
    probabilities = model._compute_prob_matrix(np.array([1., 0., 0.]), np.zeros(3))
    assert_allclose(probabilities[1:, 0], expit(1.))
    assert_allclose(probabilities[0, 1:], 0.5)


@pytest.mark.parametrize('coefficient', [-1000., 1000.])
def test_extreme_log_likelihood_is_finite(coefficient):
    model = FastBetaModel(3)
    assert np.isfinite(model.log_likelihood(network(), np.full(3, coefficient), np.zeros(3)))


def test_edges_pseudolikelihood_is_bernoulli_likelihood():
    model = FastPseudoLikelihood([EdgesStatistic()])
    y = network()
    theta = np.array([0.8])
    expected = y.sum() * theta[0] - 6 * np.logaddexp(0, theta[0])
    assert_allclose(model.log_pseudo_likelihood(y, theta), expected)
    assert_allclose(model.compute_pseudo_score(y, theta), [y.sum() - 6 * expit(theta[0])])


def test_reciprocity_change_matches_statistic_difference():
    stat = ReciprocityStatistic()
    y = network()
    toggled = y.copy()
    toggled[0, 2] = 1 - toggled[0, 2]
    assert stat.change_statistic(y, 0, 2) == stat.compute(toggled) - stat.compute(y)


def test_pseudo_simulation_respects_node_count_and_edge_probability():
    np.random.seed(123)
    model = ScoreDrivenERGM('pseudo', n_nodes=4, statistics=[EdgesStatistic()])
    assert model.simulate_network(np.array([1.])).shape == (4, 4)
    draws = [model._mcmc_simulation(np.array([np.log(4.)]), n_steps=500) for _ in range(150)]
    assert abs(np.mean([a.sum() / 12 for a in draws]) - 0.8) < 0.04


def test_static_objective_is_observation_likelihood():
    model = ScoreDrivenERGM('beta', n_nodes=3)
    observations = [network(), network().T, network()] 
    params = np.r_[np.zeros(6), np.full(6, .05), np.full(6, .8)]
    initial = np.zeros((3, 6))
    filtered = model.filter_parameters(observations, SDERGMParams(params[:6], params[6:12], params[12:]))
    expected = sum(model.model.log_likelihood(y, t[:3], t[3:]) for y, t in zip(observations, filtered))
    assert_allclose(model._static_params_log_likelihood(params, observations, initial), expected)


def test_empty_filter_has_clear_error():
    model = ScoreDrivenERGM('beta', n_nodes=3)
    with pytest.raises(ValueError, match='empty'):
        model.filter_parameters([], SDERGMParams(np.zeros(6), np.zeros(6), np.zeros(6)))


def test_bayesian_target_uses_every_observation_and_all_configured_priors():
    model = ScoreDrivenERGM('beta', n_nodes=3)
    priors = dict(w_sd=.7, alpha_shape=3., alpha_scale=.1, beta_a=4., beta_b=2.)
    sampler = BayesianSDERGM(model, priors)
    observations = [network()] * 12
    w, a, b = np.full(6, .02), np.full(6, .1), np.full(6, .7)
    value, filtered = sampler._log_posterior(observations, w, a, b)
    likelihood = sum(model.model.log_likelihood(y, t[:3], t[3:]) for y, t in zip(observations, filtered))
    expected = likelihood + norm.logpdf(w, scale=.7).sum() + gamma.logpdf(a, a=3., scale=.1).sum() + beta.logpdf(b, a=4., b=2.).sum()
    assert_allclose(value, expected)
    changed = observations[:-1] + [np.zeros((3, 3))]
    assert sampler._log_posterior(changed, w, a, b)[0] != value


@pytest.mark.parametrize('kind', ['beta', 'pseudo'])
def test_mcmc_updates_all_blocks_and_is_reproducible(kind):
    model = ScoreDrivenERGM(kind, n_nodes=3)
    sampler = BayesianSDERGM(model, {})
    observations = [network()] * 12
    one = sampler.mcmc_estimation(observations, n_samples=35, n_burn=10, random_state=13)
    two = sampler.mcmc_estimation(observations, n_samples=35, n_burn=10, random_state=13)
    for name in ('w', 'alpha', 'beta'):
        trace = one[name + '_samples']
        assert np.any(np.ptp(trace, axis=0) > 0)
        assert_allclose(trace, two[name + '_samples'])
        assert 0 < one['acceptance_rates'][name] <= 1
    assert np.all(one['alpha_samples'] > 0)
    assert np.all((one['beta_samples'] > 0) & (one['beta_samples'] < 1))
    assert np.all(np.isfinite(one['log_posterior']))


def test_unsupported_lm_test_does_not_report_uncalibrated_pvalues():
    from statistical_engine.evaluation_routine import ModelEvaluator
    with pytest.raises(NotImplementedError, match='calibrat'):
        ModelEvaluator.lm_test_time_variation([network()] * 3, ScoreDrivenERGM('beta', n_nodes=3))


@pytest.mark.parametrize('stat_name', ['EdgesStatistic', 'ReciprocityStatistic', 'TriangleStatistic', 'TwoStarsStatistic', 'GWESPStatistic'])
def test_pseudoscore_matches_finite_difference_for_network_statistics(stat_name):
    from data_handler import data_structures
    model = FastPseudoLikelihood([getattr(data_structures, stat_name)()])
    theta = np.array([.3])
    numerical = approx_derivative(lambda x: model.log_pseudo_likelihood(network(), x), theta).ravel()
    assert_allclose(model.compute_pseudo_score(network(), theta), numerical, atol=1e-6)


def test_bayesian_sampler_recovers_known_prior_when_likelihood_is_constant():
    # With one node there are no possible edges, so posterior equals prior.
    model = ScoreDrivenERGM('pseudo', n_nodes=1)
    sampler = BayesianSDERGM(model, dict(w_sd=.2, alpha_shape=2., alpha_scale=.05, beta_a=3., beta_b=3.))
    result = sampler.mcmc_estimation([np.zeros((1, 1))], n_samples=3000, n_burn=300, random_state=71,
                                      proposal_scales={'w': .3, 'alpha': .07, 'beta': .3})
    assert abs(result['posterior_mean']['w'][0]) < .025
    assert abs(result['posterior_mean']['alpha'][0] - .1) < .015
    assert abs(result['posterior_mean']['beta'][0] - .5) < .04


def test_sinusoidal_generator_keeps_the_same_phase_across_time():
    from data_handler.data_structures import DataGenerator
    values = DataGenerator.sinusoidal_dgp(np.zeros(4), .2, 20)
    assert_allclose(values[:10], values[10:], atol=1e-12)


def test_unsigned_adjacency_does_not_wrap_signed_change_statistics():
    model = FastPseudoLikelihood([EdgesStatistic()])
    y = network().astype(np.uint8)
    theta = np.array([.8])
    expected = y.sum() * theta[0] - 6 * np.logaddexp(0, theta[0])
    assert_allclose(model.log_pseudo_likelihood(y, theta), expected)


@pytest.mark.parametrize('alpha,beta_value', [(0., .5), (-.1, .5), (.1, 0.), (.1, 1.), (.1, 1.1)])
def test_sampler_rejects_out_of_support_proposals(alpha, beta_value):
    sampler = BayesianSDERGM(ScoreDrivenERGM('pseudo', n_nodes=3), {})
    value, theta = sampler._log_posterior([network()], np.zeros(1), np.array([alpha]), np.array([beta_value]))
    assert value == -np.inf
    assert theta is None
