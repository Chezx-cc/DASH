from __future__ import annotations

import numpy as np
import pytest

from cyclefuse.core import (
    Timeline,
    circular_shift_fft,
    physical_periods,
    recent_cycles,
    spectral_periods,
)
from cyclefuse.data import full_normalized_timeline, timeline_from_dataset
from cyclefuse.fusion import EmpiricalBayesScalarFusion, block_gain_lcb, smooth_graph_horizon
from cyclefuse.registry import VARIANT_SPECS, build_variant


def synthetic_problem() -> tuple[Timeline, np.ndarray, np.ndarray, Timeline, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2027)
    length, channels, horizon = 760, 3, 24
    t = np.arange(length, dtype=np.float32)
    values = np.stack(
        [
            1.5 * np.sin(2 * np.pi * t / 24 + phase)
            + 0.4 * np.sin(2 * np.pi * t / 168)
            + 0.001 * t
            + 0.03 * rng.normal(size=length)
            for phase in (0.0, 0.3, -0.5)
        ],
        axis=1,
    ).astype(np.float32)
    timestamp_ns = np.arange(length, dtype=np.int64) * 3600 * 1_000_000_000
    validation_origins = np.arange(420, 480, dtype=np.int64)
    test_origins = np.arange(600, 620, dtype=np.int64)

    def arrays(origins: np.ndarray) -> tuple[Timeline, np.ndarray, np.ndarray]:
        truth = np.stack([values[row + 1 : row + horizon + 1] for row in origins])
        raw = truth + 0.22 * np.stack(
            [np.sin(2 * np.pi * np.arange(1, horizon + 1) / 24 + 0.8)[:, None]] * len(origins)
        )
        raw = raw + 0.05 * rng.normal(size=truth.shape)
        timeline = Timeline(
            values=values,
            timestamp_ns=timestamp_ns,
            origin_rows=origins,
            origin_timestamp_ns=timestamp_ns[origins],
            sampling_step_ns=3600 * 1_000_000_000,
            split="val" if origins[0] < 500 else "test",
        )
        return timeline, raw.astype(np.float32), truth.astype(np.float32)

    val_timeline, val_raw, val_truth = arrays(validation_origins)
    test_timeline, test_raw, test_truth = arrays(test_origins)
    return val_timeline, val_raw, val_truth, test_timeline, test_raw, test_truth


def test_registry_has_exact_ten_variants() -> None:
    assert list(VARIANT_SPECS) == [
        "v1_physical_period",
        "v2_shift_align",
        "v3_tangent_rank2",
        "v4_tangent_warp",
        "v5_robust_tangent",
        "v6_dyadic_state",
        "v7_eb_scalar",
        "v8_eb_graph",
        "v9_wasserstein_expert",
        "v10_cycle_image_router",
    ]


def test_physical_and_spectral_period_detection() -> None:
    assert physical_periods(3600 * 1_000_000_000) == [12, 24, 168]
    assert physical_periods(15 * 60 * 1_000_000_000) == [48, 96, 672]
    assert physical_periods(10 * 60 * 1_000_000_000) == [72, 144, 1008]
    t = np.arange(600)
    series = np.sin(2 * np.pi * t / 24)[:, None]
    found = spectral_periods(series, max_period=100, max_candidates=3)
    assert any(abs(period - 24) <= 1 for period in found)


def test_fft_alignment_recovers_roll() -> None:
    phase = np.arange(48)
    reference = np.sin(2 * np.pi * phase / 48)[:, None]
    shifted = np.roll(reference, 5, axis=0)
    estimated, confidence = circular_shift_fft(shifted, reference, max_shift=10)
    assert np.allclose(np.roll(shifted, estimated, axis=0), reference, atol=1e-6)
    assert confidence > 0.99


def test_recent_cycles_never_read_future() -> None:
    values = np.arange(600, dtype=np.float32)[:, None]
    cycles, max_source = recent_cycles(values, origin_row=400, period=24, count=8)
    assert cycles.shape == (8, 24, 1)
    assert max_source == 400
    assert np.max(cycles) == 400


def test_timeline_rejects_disorder_and_mismatched_inputs() -> None:
    values = np.arange(50, dtype=np.float32)[:, None]
    stamps = np.arange(50, dtype=np.int64)
    timeline = Timeline(values, stamps, np.arange(10, 20), stamps[10:20], 1, "val")
    inputs = np.stack([values[row - 3 : row + 1] for row in range(10, 20)])
    assert timeline.validate(inputs)["origins_unit_stride"]
    bad = Timeline(values, stamps, np.asarray([10, 12]), stamps[[10, 12]], 1, "val")
    with pytest.raises(ValueError, match="unit-stride"):
        bad.validate()


def test_dataset_bridge_uses_global_rows_and_real_timestamps() -> None:
    source = np.arange(30, dtype=np.float32)[:, None]
    stamps = np.arange(30, dtype=np.int64) * 10
    class FakeDataset:
        source_timestamp_ns = stamps
        timestamp_step_ns = 10
        split_row_bounds = {"train": (0, 18), "val": (14, 24), "test": (20, 30)}
        train = source[:18]
        val = source[14:24]
        test = source[20:30]
        seq_len = 4
        pred_len = 2

        def __len__(self) -> int:
            return 5

    dataset = FakeDataset()
    values, timestamp = full_normalized_timeline(dataset)
    assert np.array_equal(values, source)
    assert np.array_equal(timestamp, stamps)
    timeline = timeline_from_dataset(dataset, "test", origin_count=5)
    assert np.array_equal(timeline.origin_rows, np.arange(23, 28))


def test_eb_weight_and_route_are_finite() -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(40, 12, 2)).astype(np.float32)
    truth = np.zeros_like(raw)
    prior = 0.5 * raw
    confidence = np.ones(40, dtype=np.float32)
    model = EmpiricalBayesScalarFusion().fit(raw[:20], prior[:20], truth[:20], confidence[:20])
    model.fit_route(raw[20:], prior[20:], truth[20:], confidence[20:])
    output = model.apply(raw, prior, confidence)
    assert output.prediction.shape == raw.shape
    assert np.all(np.isfinite(output.prediction))
    assert np.all((output.weights >= 0.0) & (output.weights <= 1.0))


def test_graph_smoothing_reduces_roughness() -> None:
    correction = np.zeros((2, 24, 3), dtype=np.float32)
    correction[:, ::2, 0] = 1.0
    laplacian = np.asarray([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=np.float64)
    smoothed = smooth_graph_horizon(correction, laplacian, lambda_h=1.0, lambda_g=1.0)
    before = np.mean(np.diff(correction, n=2, axis=1) ** 2)
    after = np.mean(np.diff(smoothed, n=2, axis=1) ** 2)
    assert after < before


@pytest.mark.parametrize("variant_id", list(VARIANT_SPECS))
def test_all_variants_end_to_end_are_causal(variant_id: str) -> None:
    val_timeline, val_raw, val_truth, test_timeline, test_raw, _ = synthetic_problem()
    device = "cpu"
    variant = build_variant(variant_id, device=device)
    if variant_id == "v10_cycle_image_router":
        # Keep the unit test fast while exercising the complete V10 protocol.
        from cyclefuse.vision import CycleImageRouter

        original = CycleImageRouter.__init__

        def fast_init(self: CycleImageRouter, *args: object, **kwargs: object) -> None:
            original(self, cycles=4, image_size=24, epochs=1, seed=2027)

        CycleImageRouter.__init__ = fast_init  # type: ignore[method-assign]
        try:
            result = variant.run(val_timeline, val_raw, val_truth, test_timeline, test_raw)
        finally:
            CycleImageRouter.__init__ = original  # type: ignore[method-assign]
    else:
        result = variant.run(val_timeline, val_raw, val_truth, test_timeline, test_raw)
    assert result.prediction.shape == test_raw.shape
    assert np.all(np.isfinite(result.prediction))
    assert result.diagnostics["future_observation_reads"] == 0
    assert result.diagnostics["uses_external_outputs"] is False
    assert result.diagnostics["uses_test_for_selection"] is False


def test_lcb_uses_standard_error_and_detects_consistent_gain() -> None:
    truth = np.zeros((40, 2, 1), dtype=np.float32)
    raw = np.ones_like(truth)
    method = np.full_like(truth, 0.8)
    stats = block_gain_lcb(raw, method, truth)
    assert stats["lcb_gain"] > 0
    assert stats["positive_blocks"] == 4
