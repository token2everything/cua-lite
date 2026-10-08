# WebArena-Verified：CUA-Lite 原生通用 Agent

本目录的复现入口是 `campaign/initialize.py` 和 `campaign/orchestrate.py`，
逐题执行 `scripts/webarena_verified_generic.py` → `ChatGatewayAgent` → 原生 agent loop。
旧 `run.sh`、gateway 和模型专用 runner 不是这里交付的完整桌面实验入口。

先读 [复现手册](/skills/webarena-verified-eval/references/reproduction.md)，
agent 使用 [评测 skill](/skills/webarena-verified-eval/SKILL.md)。
在线阅读 [飞书 Skill 与复现指南](https://gcnxubmy1vgu.feishu.cn/docx/YnArdkqJooQ8MOxO4UMcq3wLnCb)
（文档访问权限独立于代码仓库）。
手册提供资产/平台前提、离线初始化、单 worker 冒烟、120-worker 全量、
诚实的 NULL 分数口径和 scoped cleanup 命令，不需要作者的旧共享盘路径。

已执行配置：812 题，150 模型决策预算，完整文字历史，20 图 fold10；
Qwen `qwen3.8-max` / xhigh；GLM `glm-5.3-flash` / max。
原生 V2 基线保留，管理入口的可移植化与后续版本单独提交。
数值合同见 [配置模板](/devs/exps/eval/browsergym.webarena_verified/EXPERIMENT_CONFIG.generic-desktop-dual-v2.json)。

代码不包含大型站点数据、离线浏览器资产或私有镜像授权。缺少这些前提时
不要通过换数据/浏览器或降配置绕过门禁。所有收费模型调用与资源删除均应
按本次用户授权范围执行。
