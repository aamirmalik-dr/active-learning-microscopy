"""Tests for the from-scratch GP: kernels, batch posterior, incremental updates."""

import numpy as np
import pytest

from activescan import GP, GPHyperparams, SequentialGP, kernel_matrix


@pytest.fixture()
def coords():
    return np.random.default_rng(1).uniform(0, 1, size=(120, 2))


def test_kernel_symmetric_and_diagonal(coords):
    for kind in ("rbf", "matern32"):
        k = kernel_matrix(kind, coords, coords, 0.2, 1.7)
        assert np.allclose(k, k.T)
        assert np.allclose(np.diag(k), 1.7)


def test_kernel_positive_semidefinite(coords):
    for kind in ("rbf", "matern32"):
        k = kernel_matrix(kind, coords, coords, 0.2, 1.0)
        eig = np.linalg.eigvalsh(k)
        assert eig.min() > -1e-8


def test_kernel_decays_with_distance():
    x0 = np.array([[0.0, 0.0]])
    xs = np.array([[0.05, 0.0], [0.2, 0.0], [0.6, 0.0]])
    for kind in ("rbf", "matern32"):
        vals = kernel_matrix(kind, x0, xs, 0.15, 1.0).ravel()
        assert vals[0] > vals[1] > vals[2] > 0


def test_kernel_unknown_name_raises(coords):
    with pytest.raises(ValueError):
        kernel_matrix("laplace", coords, coords, 0.2, 1.0)


def test_gp_interpolates_at_low_noise():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(40, 2))
    y = np.sin(4 * x[:, 0]) + np.cos(3 * x[:, 1])
    gp = GP("rbf", GPHyperparams(0.3, 1.0, 1e-8)).fit(x, y)
    mu, var = gp.predict(x)
    assert np.abs(mu - y).max() < 1e-3
    assert var.max() < 1e-3


def test_gp_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        GP("rbf", GPHyperparams(0.2, 1.0, 0.1)).predict(np.zeros((3, 2)))


def test_gp_variance_positive_and_grows_away_from_data():
    x = np.array([[0.5, 0.5]])
    gp = GP("rbf", GPHyperparams(0.1, 1.0, 0.01)).fit(x, np.array([1.0]))
    _, var = gp.predict(np.array([[0.5, 0.5], [0.9, 0.9]]))
    assert var[0] < var[1]
    assert (var > 0).all()


def test_fit_hyperparams_recovers_lengthscale_scale():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, size=(150, 2))
    k = kernel_matrix("rbf", x, x, 0.2, 1.0) + 1e-8 * np.eye(150)
    y = np.linalg.cholesky(k) @ rng.normal(size=150) + 0.05 * rng.normal(size=150)
    hyper = GP("rbf").fit_hyperparams(x, y)
    assert 0.05 < hyper.lengthscale < 0.8
    assert hyper.noise_var < 0.1


def test_fit_hyperparams_fixed_lengthscale_is_pinned():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, size=(50, 2))
    y = rng.normal(size=50)
    hyper = GP("rbf").fit_hyperparams(x, y, fixed_lengthscale=0.123)
    assert hyper.lengthscale == pytest.approx(0.123)


def test_sequential_matches_zero_mean_batch():
    rng = np.random.default_rng(3)
    cand = np.random.default_rng(1).uniform(0, 1, size=(200, 2))
    hyper = GPHyperparams(0.15, 1.2, 0.05)
    seq = SequentialGP(cand, "rbf", hyper)
    idx = rng.choice(200, size=30, replace=False)
    vals = rng.normal(size=30) + 2.0
    for i, v in zip(idx, vals):
        seq.add(int(i), float(v))
    k = kernel_matrix("rbf", cand[idx], cand[idx], 0.15, 1.2)
    k[np.diag_indices_from(k)] += 0.05 + 1e-10
    ks = kernel_matrix("rbf", cand, cand[idx], 0.15, 1.2)
    mu_ref = ks @ np.linalg.solve(k, vals)
    var_ref = 1.2 - np.einsum("ij,ji->i", ks, np.linalg.solve(k, ks.T))
    assert np.abs(seq.mu - mu_ref).max() < 1e-10
    assert np.abs(seq.var - np.maximum(var_ref, 1e-12)).max() < 1e-10


def test_sequential_rebuild_matches_batch_gp():
    rng = np.random.default_rng(4)
    cand = rng.uniform(0, 1, size=(150, 2))
    hyper = GPHyperparams(0.2, 1.0, 0.02)
    seq = SequentialGP(cand, "matern32", hyper)
    idx = rng.choice(150, size=20, replace=False)
    vals = rng.normal(size=20)
    for i, v in zip(idx, vals):
        seq.add(int(i), float(v))
    seq._rebuild()
    gp = GP("matern32", hyper).fit(cand[idx], vals)
    mu, var = gp.predict(cand)
    assert np.abs(seq.mu - mu).max() < 1e-10
    assert np.abs(seq.var - var).max() < 1e-10


def test_sequential_variance_shrinks_at_measured_point():
    cand = np.random.default_rng(5).uniform(0, 1, size=(100, 2))
    seq = SequentialGP(cand, "rbf", GPHyperparams(0.2, 1.0, 0.01))
    before = seq.var[7]
    seq.add(7, 0.5)
    assert seq.var[7] < before
    assert (seq.var > 0).all()


def test_sequential_refit_updates_hyperparams():
    rng = np.random.default_rng(6)
    cand = rng.uniform(0, 1, size=(150, 2))
    seq = SequentialGP(cand, "rbf", GPHyperparams(0.5, 1.0, 0.5))
    idx = rng.choice(150, size=40, replace=False)
    for i in idx:
        seq.add(int(i), float(np.sin(6 * cand[i, 0])))
    old = seq.hyper
    new = seq.refit()
    assert new is seq.hyper
    assert (new.lengthscale, new.signal_var, new.noise_var) != (
        old.lengthscale,
        old.signal_var,
        old.noise_var,
    )
