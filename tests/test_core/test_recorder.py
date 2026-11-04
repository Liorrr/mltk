"""Tests for context-local assertion recording."""

from __future__ import annotations

import threading

import pytest

from mltk.core import recorder
from mltk.core.assertion import MltkAssertionError, assert_true
from mltk.core.result import Severity


def test_unarmed_recorder_is_inert() -> None:
    # Force the unarmed state: this suite may itself run under --mltk-export-json.
    token = recorder._current.set(None)
    try:
        recorder.record(object())  # type: ignore[arg-type]
        assert recorder.drain() == []
    finally:
        recorder._current.reset(token)


def test_arm_record_drain_preserves_order_and_clears() -> None:
    token = recorder.arm()
    try:
        first = object()
        second = object()
        recorder.record(first)  # type: ignore[arg-type]
        recorder.record(second)  # type: ignore[arg-type]
        assert recorder.drain() == [first, second]
        assert recorder.drain() == []
    finally:
        recorder.disarm(token)


def test_nested_arm_restores_outer_list() -> None:
    outer_token = recorder.arm()
    try:
        outer = object()
        recorder.record(outer)  # type: ignore[arg-type]
        inner_token = recorder.arm()
        try:
            inner = object()
            recorder.record(inner)  # type: ignore[arg-type]
            assert recorder.drain() == [inner]
        finally:
            recorder.disarm(inner_token)
        assert recorder.drain() == [outer]
    finally:
        recorder.disarm(outer_token)


def test_thread_results_are_not_captured() -> None:
    token = recorder.arm()
    try:
        result = object()
        thread = threading.Thread(target=recorder.record, args=(result,))
        thread.start()
        thread.join()
        assert recorder.drain() == []
    finally:
        recorder.disarm(token)


def test_assert_true_records_pass_warning_and_critical() -> None:
    token = recorder.arm()
    try:
        passed = assert_true(True, "pass", "ok", severity=Severity.INFO)
        warning = assert_true(False, "warning", "warn", severity=Severity.WARNING)
        with pytest.raises(MltkAssertionError) as raised:
            assert_true(False, "critical", "fail")
        assert recorder.drain() == [passed, warning, raised.value.result]
    finally:
        recorder.disarm(token)
