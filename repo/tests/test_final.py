import numpy as np

from cyclefuse.final import (
    POLICIES,
    apply_dynamic,
    disagreement_reliability,
    fit_dynamic_clip,
    horizon_bin_ids,
    route_dynamic_increment,
    stable_and_dynamic,
    tune_rho,
)


def test_decomposition_reconstructs_recent():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(5, 96, 3)).astype(np.float32)
    stable = raw + 0.2
    recent = stable - 0.05
    stable_corr, dynamic = stable_and_dynamic(raw, stable, recent)
    np.testing.assert_allclose(raw + stable_corr + dynamic, recent, atol=1e-6)


def test_zero_rho_falls_back_to_stable_not_raw():
    raw = np.zeros((4, 96, 2), dtype=np.float32)
    stable = np.ones_like(raw)
    dynamic = np.full_like(raw, 3.0)
    out = apply_dynamic(stable, dynamic, np.zeros(4), clip=1.0)
    np.testing.assert_array_equal(out, stable)
    assert not np.array_equal(out, raw)


def test_tune_selects_dynamic_when_it_matches_truth():
    stable = np.zeros((12, 96, 2), dtype=np.float32)
    dynamic = np.ones_like(stable)
    truth = 0.5 * np.ones_like(stable)
    np.testing.assert_array_equal(tune_rho(stable, dynamic, truth), np.full(4, 0.5))


def test_route_rejects_harmful_dynamic_increment():
    stable = np.zeros((16, 96, 1), dtype=np.float32)
    truth = np.zeros_like(stable)
    harmful = np.ones_like(stable)
    gated, audit = route_dynamic_increment(stable, harmful, truth, np.ones(4))
    np.testing.assert_array_equal(gated, np.zeros(4))
    assert all(x <= 0 for x in audit["gain_over_stable_pct"])


def test_disagreement_shrink_is_bounded_and_decreases():
    stable = np.ones((3, 96, 1), dtype=np.float32)
    small = np.full_like(stable, 0.1)
    large = np.full_like(stable, 10.0)
    r_small = disagreement_reliability(stable, small)
    r_large = disagreement_reliability(stable, large)
    assert np.all((0 <= r_large) & (r_large <= 1))
    assert np.all(r_large < r_small)


def test_frozen_policy_names_and_bins():
    assert set(POLICIES) == {
        "final_fixed25", "final_val_tuned", "final_route_safe", "final_disagreement_shrink"
    }
    bins = horizon_bin_ids(96)
    assert [int(np.sum(bins == b)) for b in range(4)] == [12, 12, 24, 48]
    assert fit_dynamic_clip(np.asarray([0.0, -2.0, 1.0]), 0.9) > 0
