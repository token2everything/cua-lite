"""Agent progress summaries and collapsible Feishu cards for the shared batch queue.

Adapters populate item.progress; this module never reads benchmark artifacts.
Notifications are snapshots sent by the workstation scheduler, not a watchdog
for the scheduler itself. No callback service or Feishu application is required.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime

from lite.platforms.sandbox.client import redact, send_feishu_message, validate_feishu_webhook

_FINISHED = {"completed", "failed"}
_ATTENTION = {"failed", "stalled", "unknown"}
_ACTIVITIES = {
    "starting": "初始化环境 / Agent",
    "waiting_model": "等待模型回复",
    "executing": "响应已收到，等待动作或评分完成",
    "between_phases": "阶段收尾 / 切换",
    "finished": "等待作业退出回执",
}


def task_health(item: dict, now: float, stall_after: int, poll_interval: int) -> str:
    """A live Sandbox or model retry does not prove that an Agent step completed."""
    state = item["state"]
    if state in _FINISHED | {"pending"}:
        return state
    if item.get("error"):
        return "failed"
    if state in {"collecting", "deleting"}:
        return "finishing"
    progress = item.get("progress", {})
    if item.get("progress_error") or (
        "observed_at" in progress
        and now - progress["observed_at"] > max(90, poll_interval * 3)
    ):
        return "unknown"
    since = progress.get("last_progress_at", item.get("started_at", now))
    if now - since >= stall_after:
        return "stalled"
    return "running"


def _text(value, limit=120) -> str:
    # Artifact/error text must not create mentions, links or Markdown in a card.
    text = str(value).replace("\n", " ")[:limit]
    return re.sub(r"[&<>*~\[\]()#:_`\\]", lambda match: f"&#{ord(match[0])};", text)


def _age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} 秒"
    minutes = seconds // 60
    return f"{minutes} 分钟" if minutes < 60 else f"{minutes // 60} 小时 {minutes % 60} 分钟"


def build_card(state: dict, now: float, *, event: str = "periodic") -> dict:
    """Render a bounded Card 2.0 snapshot; large batches keep exact counts and capped details."""
    settings = state["monitor"]
    rows = [
        (item, task_health(item, now, settings["stall_after"], state["poll_interval"]))
        for item in state["tasks"]
    ]
    counts = Counter(health for _, health in rows)
    ended = sum(item["state"] in _FINISHED for item, _ in rows)
    attention = sum(counts[health] for health in _ATTENTION)
    stopped = bool(state.get("scheduler_error"))
    color = (
        "red" if stopped or counts["failed"] else
        "orange" if attention else "green" if ended == len(rows) else "blue"
    )
    label = {
        "started": "已启动", "resumed": "已恢复", "finished": "已结束",
        "stopped": "调度已停止", "demo": "样式预览 · 模拟数据",
    }.get(event, "进度汇总")
    previous = state.get("notification", {}).get("reported_turns", {})
    summary = f"已结束 {ended}/{len(rows)} · 需关注 {attention}"
    header = {
        "title": {"tag": "plain_text", "content": f"Sandbox 评测监控｜{label}"},
        "subtitle": {"tag": "plain_text", "content": str(state["batch_id"])[:120]},
        "template": color,
    }

    def detail(item, health):
        progress = item.get("progress", {})
        title = f"**任务 {_text(item['task_id'], 36)}**"
        if item["state"] in _FINISHED:
            result = "流程完成" if item["state"] == "completed" else "流程失败"
            score = item.get("score")
            line = (
                f"{title} · {result} · 模型得分 {_text('未取得' if score is None else score, 24)}"
            )
        elif health == "finishing":
            line = f"{title} · {'同步结果' if item['state'] == 'collecting' else '回收 Sandbox'}"
        else:
            turns = progress.get("turns")
            if turns is None:
                line = f"{title} · 尚未取得 Agent 步骤"
            else:
                delta = max(0, turns - previous.get(str(item["task_id"]), turns))
                line = f"{title} · 累计 {turns} 步（较上次 +{delta}）"
            if "phase_index" in progress:
                phase = progress["phase_index"] + 1
                line += f"\n阶段 {phase}/{progress.get('phase_count', '?')}"
                if "phase_turn" in progress:
                    line += (
                        f" · 本阶段 {progress['phase_turn']}/{progress.get('max_steps', '?')} 步"
                    )
            activity = _ACTIVITIES.get(progress.get("activity"), "等待进度记录")
            line += f"\n{activity}"
            if progress.get("retries"):
                line += f" · 本次请求已记录 {progress['retries']} 次重试前失败"
            since = progress.get("last_progress_at", item.get("started_at", now))
            line += f" · {_age(now - since)}无新步骤"
            if health == "stalled":
                line += "\n疑似停滞；需结合当前等待环节判断"
            if health == "unknown":
                line += "\n暂时无法确认最新进度"
                if "observed_at" in progress:
                    line += f"（上次采集：{_age(now - progress['observed_at'])}前）"
        error = item.get("error") or item.get("progress_error")
        if error:
            line += "\n原因：" + _text(error, 140)
        if item.get("sandbox_id"):
            line += "\nSandbox：" + _text(item["sandbox_id"], 64)
        return {"tag": "markdown", "content": line}

    # Webhooks accept at most 20 KiB. Reduce detail rows, never truncate JSON/counts.
    for limit in (8, 4, 2, 1, 0):
        elements = [{
            "tag": "column_set", "flex_mode": "none", "horizontal_spacing": "12px",
            "columns": [
                {"tag": "column", "width": "weighted", "weight": 1,
                 "background_style": "grey-100", "padding": "12px", "elements": [
                    {"tag": "markdown", "content": f"## {ended} / {len(rows)}\n已结束"},
                ]},
                {"tag": "column", "width": "weighted", "weight": 2,
                 "background_style": "grey-100", "padding": "12px", "elements": [{
                    "tag": "markdown",
                    "content": (
                        f"**需关注 {attention}**\n"
                        f"运行 {sum(counts[h] for h in ('running', 'stalled', 'unknown'))}"
                        f" · 排队 {counts['pending']} · 收尾 {counts['finishing']}\n"
                        f"流程完成 {counts['completed']} · 流程失败 {counts['failed']}"
                    ),
                }]},
            ],
        }]
        if stopped:
            elements.append({"tag": "markdown", "content":
                "**调度已停止，运行中的 Sandbox 可能仍在执行。**\n"
                + _text(state["scheduler_error"], 160)})
        for title, selected, expanded, panel_color in (
            ("需关注", sorted(
                [(i, h) for i, h in rows if h in _ATTENTION],
                key=lambda row: (
                    row[1] == "failed",
                    [row[0]["task_id"], row[1]] in state.get("notification", {}).get("alerts", []),
                ),
            ), True, color),
            ("运行中 / 收尾", [
                (i, h) for i, h in rows
                if i["state"] not in _FINISHED | {"pending"} and h not in _ATTENTION
            ], False, "blue"),
            ("已完成流程", [(i, h) for i, h in rows if i["state"] == "completed"], False, "grey"),
        ):
            if not selected:
                continue
            details = [detail(i, h) for i, h in selected[:limit]]
            if len(selected) > limit:
                details.append({"tag": "markdown", "content":
                    f"另有 {len(selected) - limit} 项；完整列表见本批次 batch.json。"})
            elements.append({
                "tag": "collapsible_panel", "expanded": expanded,
                "background_color": f"{panel_color}-50",
                "border": {"color": f"{panel_color}-200", "corner_radius": "8px"},
                "header": {
                    "title": {"tag": "markdown", "content": f"**{title}（{len(selected)}）**"},
                    "background_color": f"{panel_color}-100", "padding": "8px",
                    "icon": {
                        "tag": "standard_icon", "token": "down-small-ccm_outlined",
                        "color": f"{panel_color}-600", "size": "16px 16px",
                    },
                    "icon_position": "right", "icon_expanded_angle": -180,
                },
                "padding": "12px", "vertical_spacing": "12px", "elements": details,
            })
        updated = datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        identity = " · ".join(str(v) for v in (
            state.get("deployment", {}).get("release"),
            state.get("job", {}).get("model_profile"),
        ) if v)
        elements.append({"tag": "markdown", "text_size": "notation", "content":
            (_text(identity) + "\n" if identity else "")
            + f"{updated} · 每 {settings['interval']} 秒汇总\n"
            "步数增长只代表执行在推进，不代表任务做对；流程结果与模型得分分别统计。"})
        card = {
            "schema": "2.0", "config": {"width_mode": "default", "summary": {"content": summary}},
            "header": header,
            "body": {"direction": "vertical", "vertical_spacing": "12px", "elements": elements},
        }
        size = len(json.dumps(
            {"msg_type": "interactive", "card": card}, ensure_ascii=False,
        ).encode())
        if size < 19000:
            return card
    raise ValueError("Monitoring summary cannot fit in a Feishu card")


class BatchMonitor:
    """Send periodic snapshots and coalesced alert transitions without failing the queue."""

    def __init__(self, webhook: str, secrets: list[str]):
        validate_feishu_webhook(webhook)
        self.webhook = webhook
        self.secrets = secrets
        self.secrets.extend((webhook, webhook.rsplit("/", 1)[-1]))

    def report(self, state: dict, now: float, *, event: str = "periodic") -> None:
        settings = state["monitor"]
        receipt = state.setdefault("notification", {})
        alerts = [
            [item["task_id"], health]
            for item in state["tasks"]
            if (health := task_health(item, now, settings["stall_after"], state["poll_interval"]))
            in _ATTENTION
        ]
        if event == "periodic":
            if now - receipt.get("last_attempt_at", -float("inf")) < 60:
                return
            if (now - receipt.get("last_sent_at", -float("inf")) < settings["interval"]
                    and alerts == receipt.get("alerts", [])):
                return
        receipt["last_attempt_at"] = now
        try:
            card = build_card(redact(state, self.secrets), now, event=event)
            for attempt in range(2 if event in {"finished", "stopped"} else 1):
                try:
                    send_feishu_message(self.webhook, {"msg_type": "interactive", "card": card})
                    break
                except (OSError, RuntimeError):
                    if attempt or event not in {"finished", "stopped"}:
                        raise
                    # No next poll after termination. One short retry; delivery after
                    # a network timeout is ambiguous, so a duplicate card is possible.
                    time.sleep(2)
        except (OSError, ValueError, RuntimeError, TypeError) as error:
            receipt["error"] = redact(str(error), self.secrets)[:500]
            print("Batch notification failed: " + receipt["error"], file=sys.stderr, flush=True)
        else:
            receipt.pop("error", None)
            receipt.update(last_sent_at=now, event=event, alerts=alerts, reported_turns={
                str(item["task_id"]): item.get("progress", {}).get("turns", 0)
                for item in state["tasks"]
            })
