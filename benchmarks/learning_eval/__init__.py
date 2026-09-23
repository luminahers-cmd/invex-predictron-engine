"""Learning-layer deterministic evaluation benchmark (CIH Phase 7).

A standalone, append-only benchmark that proves the continuous-learning
layer behaves deterministically and computes the documented metrics
correctly.  It replays hand-built evaluated-sample fixtures through the
engine — never fabricated results — and cross-checks every derived metric
(accuracy, precision, confusion, calibration ECE, confidence bias,
distributions) against an independent reference implementation, then
asserts the snapshot contracts (deterministic ids, content hashes, schema
version, verification).

This benchmark is itself deterministic: identical inputs always produce the
identical benchmark run id and the identical assertion results.

Modules:
    models:    Benchmark data contracts (runs, assertions, reports).
    dataset:   Deterministic evaluated-sample fixtures with expected values.
    reference: Independent reference computations cross-checked by the suite.
    suite:     The assertion suite that drives the engine and compares.
    history:   Append-only benchmark-run store.
    cli:       ``predictron-learning-eval`` command-line interface.
"""
