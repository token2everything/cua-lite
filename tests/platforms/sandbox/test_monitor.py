"""Monitoring must describe real advancement and remain independent of evaluation success."""

from __future__ import annotations

import copy
import json
from unittest.mock import Mock

import pytest

from lite.platforms.sandbox import batch, monitor


@pytest.fixture
def state():
    return {
        "batch_id": "test-batch", "poll_interval": 30,
        "monitor": {"enabled": True, "interval": 300, "stall_after": 900},
        "tasks": [{
            "task_id": "054", "state": "running", "started_at": 0,
            "sandbox_id": "example-sandbox",
            "progress": {"turns": 2, "last_progress_at": 10, "observed_at": 10,
                         "activity": "waiting_model", "phase_index": 0,
                         "phase_count": 2, "phase_turn": 2, "max_steps": 500},
        }],
    }


def test_step_stall_is_not_hidden_by_fresh_observation_or_model_retries(state):
    item = state["tasks"][0]
    item["progress"].update(observed_at=920, retries=3)
    assert monitor.task_health(item, 920, 900, 30) == "stalled"
    card = json.dumps(monitor.build_card(state, 920), ensure_ascii=False)
    assert "疑似停滞" in card and "等待模型回复" in card
    assert "2/500" in card and "阶段 1/2" in card
    item["progress"].update(turns=3, last_progress_at=925, observed_at=925)
    assert monitor.task_health(item, 925, 900, 30) == "running"


def test_unavailable_or_stale_progress_is_unknown_and_does_not_fail_task(state):
    item = state["tasks"][0]
    assert monitor.task_health(item, 920, 900, 30) == "unknown"
    item["progress_error"] = "AFS unavailable"
    item["progress"]["observed_at"] = 920
    assert monitor.task_health(item, 920, 900, 30) == "unknown"
    assert item["state"] == "running"
    item["state"] = "collecting"
    assert monitor.task_health(item, 920, 900, 30) == "finishing"


def test_card_uses_collapsed_details_and_separates_zero_score_from_workflow_failure(state):
    state["tasks"].append({"task_id": "done", "state": "completed", "score": 0})
    card = monitor.build_card(state, 20)
    assert card["schema"] == "2.0"
    panels = [e for e in card["body"]["elements"] if e["tag"] == "collapsible_panel"]
    assert len(panels) == 2 and not any(p["expanded"] for p in panels)
    encoded = json.dumps(card, ensure_ascii=False)
    assert "流程完成" in encoded and "模型得分 0" in encoded
    assert card["header"]["template"] == "blue"


def test_card_is_bounded_for_large_batch_and_artifact_text_cannot_create_mentions(state):
    item = state["tasks"][0]
    item.update(state="failed", error="<at id=all></at> **secret** " + "错" * 2000)
    state["tasks"] = [dict(copy.deepcopy(item), task_id=str(i)) for i in range(1000)]
    card = monitor.build_card(state, 20)
    encoded = json.dumps({"msg_type": "interactive", "card": card}, ensure_ascii=False)
    assert len(encoded.encode()) < 20000
    assert "<at" not in encoded and "另有" in encoded and "1000 / 1000" in encoded
    alerts = next(e for e in card["body"]["elements"] if e["tag"] == "collapsible_panel")
    assert alerts["expanded"]


def test_periodic_alert_recovery_and_resume_keep_notification_history(state, monkeypatch):
    send = Mock()
    monkeypatch.setattr(monitor, "send_feishu_message", send)
    hook = "https://open.feishu.cn/open-apis/bot/v2/hook/test-only"
    reporter = monitor.BatchMonitor(hook, [])
    reporter.report(state, 10, event="started")
    reporter.report(state, 40)
    assert send.call_count == 1
    state["tasks"][0]["progress"].update(turns=7, observed_at=310, last_progress_at=300)
    reporter.report(state, 310)
    assert send.call_count == 2
    assert "较上次 +5" in json.dumps(send.call_args.args[1], ensure_ascii=False)
    state["tasks"][0]["progress_error"] = "read unavailable"
    reporter.report(state, 340)  # Coalesce rapid transitions into at most one alert per minute.
    assert send.call_count == 2
    reporter.report(state, 370)
    assert send.call_count == 3
    restored = json.loads(json.dumps(state))
    restored["tasks"][0].pop("progress_error")
    restored["tasks"][0]["progress"].update(observed_at=431, last_progress_at=430, turns=8)
    monitor.BatchMonitor(hook, []).report(restored, 431)
    assert send.call_count == 4
    assert restored["notification"]["reported_turns"] == {"054": 8}
    assert restored["tasks"][0]["progress"]["last_progress_at"] == 430


def test_failed_notification_is_redacted_rate_limited_and_retried(state, monkeypatch, capsys):
    hook = "https://open.feishu.cn/open-apis/bot/v2/hook/test-only"
    send = Mock(side_effect=RuntimeError(f"cannot send {hook} private-key"))
    monkeypatch.setattr(monitor, "send_feishu_message", send)
    reporter = monitor.BatchMonitor(hook, ["private-key"])
    reporter.report(state, 10, event="started")
    reporter.report(state, 30)
    assert send.call_count == 1
    assert "last_sent_at" not in state["notification"]
    assert "test-only" not in json.dumps(state) + capsys.readouterr().err
    send.side_effect = None
    state["tasks"][0]["progress"]["observed_at"] = 70
    reporter.report(state, 70)
    assert send.call_count == 2 and "error" not in state["notification"]
    assert "private-key" not in json.dumps(send.call_args.args[1])


def test_first_step_delta_and_final_notification_retry(state, monkeypatch):
    send = Mock()
    monkeypatch.setattr(monitor, "send_feishu_message", send)
    monkeypatch.setattr(monitor.time, "sleep", Mock())
    reporter = monitor.BatchMonitor("https://open.feishu.cn/open-apis/bot/v2/hook/test-only", [])
    active = state["tasks"][0]
    state["tasks"] = [{"task_id": "054", "state": "pending"}]
    reporter.report(state, 0, event="started")
    assert state["notification"]["reported_turns"] == {"054": 0}
    state["tasks"] = [active]
    active["progress"].update(turns=5, last_progress_at=299, observed_at=300)
    reporter.report(state, 300)
    assert "较上次 +5" in json.dumps(send.call_args.args[1], ensure_ascii=False)
    send.side_effect = [RuntimeError("temporary network failure"), None]
    active["state"] = "completed"
    reporter.report(state, 310, event="finished")
    assert send.call_count == 4
    assert state["notification"]["event"] == "finished"
    assert "error" not in state["notification"]


def test_active_alerts_remain_visible_after_many_historical_failures(state):
    active = state["tasks"][0]
    active["progress"].update(last_progress_at=0, observed_at=1000)
    state["tasks"] = [
        {"task_id": f"old-failure-{i}", "state": "failed"} for i in range(20)
    ] + [active]
    card = monitor.build_card(state, 1000)
    panel = next(e for e in card["body"]["elements"] if e["tag"] == "collapsible_panel")
    assert "任务 054" in panel["elements"][0]["content"]


@pytest.mark.parametrize("settings", [False, {"typo": 1}, {"enabled": "yes"},
    {"interval": 0}, {"stall_after": True}])
def test_invalid_monitor_config_fails_offline(settings, tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(resume=None, concurrency=None, poll_interval=None, task_timeout=None)
    with pytest.raises(ValueError, match="monitor"):
        batch.prepare(args, {}, lambda: {}, monitor=settings)
