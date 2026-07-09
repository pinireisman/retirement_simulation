"""PRD_UNDO_MAXIMIZE_DISTRIBUTION.md Phase C: return-distribution preview.

Architect-owned acceptance tests (do not edit) for engine.figures.fig_return_distribution.
"""
import numpy as np
import plotly.graph_objects as go

from engine.figures import fig_return_distribution


def _histogram_values(fig):
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert isinstance(trace, go.Histogram)
    return np.asarray(trace.x, dtype=float)


def test_returns_a_figure_with_one_histogram_trace():
    fig = fig_return_distribution(0.05, 0.15, fat_tails_enabled=False, fat_tails_df=5)
    assert isinstance(fig, go.Figure)
    _histogram_values(fig)  # asserts exactly one Histogram trace


def test_normal_mode_sample_is_centered_near_mu():
    mu, sigma = 0.05, 0.15
    x = _histogram_values(fig_return_distribution(mu, sigma, fat_tails_enabled=False, fat_tails_df=5))
    assert len(x) > 1000
    assert abs(np.mean(x) - mu) < 0.02


def test_fat_tails_mode_has_heavier_tails_than_normal_mode_at_low_df():
    """Student-t with a low df has more probability mass in the extreme
    tails than a Normal with the same target std -- rescaled per
    engine.simulation.sample_real_returns's own comment ('lower df = fatter
    tails'). Compare the 99th percentile of |x - mu| as a cheap proxy."""
    mu, sigma = 0.05, 0.15
    normal_x = _histogram_values(
        fig_return_distribution(mu, sigma, fat_tails_enabled=False, fat_tails_df=5))
    fat_x = _histogram_values(
        fig_return_distribution(mu, sigma, fat_tails_enabled=True, fat_tails_df=3))
    normal_extreme = np.percentile(np.abs(normal_x - mu), 99)
    fat_extreme = np.percentile(np.abs(fat_x - mu), 99)
    assert fat_extreme > normal_extreme


def test_deterministic_across_calls():
    """A fixed internal seed keeps the preview from jittering visually on
    every keystroke that doesn't actually change mu/sigma/fat-tails."""
    fig1 = fig_return_distribution(0.05, 0.15, fat_tails_enabled=True, fat_tails_df=5)
    fig2 = fig_return_distribution(0.05, 0.15, fat_tails_enabled=True, fat_tails_df=5)
    np.testing.assert_array_equal(_histogram_values(fig1), _histogram_values(fig2))


def test_fat_tails_disabled_ignores_df():
    """When fat tails are off, changing fat_tails_df must not change the
    (still-deterministic) sample -- it's a plain Normal draw either way."""
    fig1 = fig_return_distribution(0.05, 0.15, fat_tails_enabled=False, fat_tails_df=3)
    fig2 = fig_return_distribution(0.05, 0.15, fat_tails_enabled=False, fat_tails_df=9)
    np.testing.assert_array_equal(_histogram_values(fig1), _histogram_values(fig2))
