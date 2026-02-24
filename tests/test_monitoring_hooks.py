from __future__ import annotations

import sys
import time
import types

from src.apps.monitoring_hooks import run_monitor_hooks


def test_monitoring_hooks_allowlist_and_output_validation() -> None:
    mod = types.ModuleType("monitor_hooks_test_rules")

    def evaluate_topic(context):  # noqa: ANN001
        return [{"trigger_id": "h1", "severity": "high", "fired": True, "reason": "x"}]

    mod.evaluate_topic = evaluate_topic
    sys.modules[mod.__name__] = mod

    out, errs = run_monitor_hooks(
        hooks=[{"module": mod.__name__, "function": "evaluate_topic"}],
        allowlist=[mod.__name__],
        timeout_sec=1,
        context={},
        topic="t",
        run_id="r1",
    )
    assert not errs
    assert len(out) == 1
    assert out[0]["trigger_id"] == "h1"


def test_monitoring_hooks_timeout() -> None:
    mod = types.ModuleType("monitor_hooks_test_timeout")

    def evaluate_topic(context):  # noqa: ANN001
        time.sleep(0.2)
        return []

    mod.evaluate_topic = evaluate_topic
    sys.modules[mod.__name__] = mod

    out, errs = run_monitor_hooks(
        hooks=[{"module": mod.__name__, "function": "evaluate_topic"}],
        allowlist=[mod.__name__],
        timeout_sec=0.01,
        context={},
        topic="t",
        run_id="r1",
    )
    assert out == []
    assert errs and "timeout" in errs[0]
