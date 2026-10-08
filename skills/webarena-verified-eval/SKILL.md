---
name: webarena-verified-eval
description: "Prepare, reproduce, monitor, and analyze WebArena-Verified 1.2.3 evaluations using CUA-Lite's native generic agent, including the verified Qwen and GLM desktop configurations. Use for this evaluation workflow, not arbitrary BrowserGym or model-specific runners."
---

# WebArena-Verified 通用 Agent 评测

采用 `scripts/webarena_verified_generic.py` → `ChatGatewayAgent` →
`AdapterBasedAgent.sample`。任务在完整桌面上用原生 function tools 执行，
由官方 WebArena-Verified evaluator 评分。不要误用名称相近的旧 gateway、
Qwen XML runner、BrowserGym viewport launcher 或 OSWorld agent。

先读取本 skill 目录下的 `references/reproduction.md`。其中包含准备条件、
已验证的两组配置、真实结果口径，以及初始化、冒烟、全量、分析、清理命令。
定位仓库时使用包含 `pyproject.toml` 与上述脚本的实际 checkout，
不要假定原作者的 `/root` 或共享盘路径在用户机器上存在。

## 执行边界

- 区分只读状态查询、离线初始化、真实模型冒烟/全量，以及资源删除。
  指南与 skill 本身不授权花费、申请沙盒或清理历史资源；按用户本次请求执行。
- 缺少平台凭证、固定离线资产或镜像权限时，列出缺项并停在准备门禁。
  不能用换浏览器、换数据版本、降 thinking、减预算等方式静默绕过。
- 默认模型分别为 `qwen3.8-max` / `xhigh` / `QWEN_TEST_API_KEY`，以及
  `glm-5.3-flash` / `max` / `GLM_TEST_API_KEY`。其他共同配置以仓库
  `EXPERIMENT_CONFIG.generic-desktop-dual-v2.json` 为准。
- 150 是模型决策预算；最后 7 轮和最多 1 次最终回答 JSON 修复均在预算内。
  所有文字历史保留；19 张历史图加当前图，上限 20，超限删除最老 10 张。
- 初始化新 run root，不覆盖旧结果。冻结源码后不要就地改 config、plan 或脚本。
  用 manifest 中的实际文件哈希核对执行版本，不能只看 Git HEAD。
- 每题最多 3 次基础设施尝试，保留每次证据；耗尽后隔离该题继续其他题。
  正常的模型失败不重抽；不得清账本获得额外次数，或重放未知是否已执行的调用。
- evaluator 报错是未评分，不是 0。Infra 耗尽也没有官方分数；分别报告
  canonical terminal、scored、evaluator-unscored、Infra-exhausted 和未完成任务。
  有缺失分数时全量官方准确率为 NULL，只能注明已评分子集成功率。
- 清理前先同步并核对归属、远端命令终止状态和本地证据。
  仅对本 run 实际创建并验证身份的沙盒显式 `--delete`，不碰其他资源。

## 交付

报告 run ID、模型、源码/config/plan/镜像/资产 manifest 哈希、准备及 startup
门禁状态、执行覆盖与分数覆盖、问题清单、轨迹/报告位置，以及资源是否保留。
发布报告或同步飞书需要用户明确要求。历史 GLM task 783 的第 4 次是单独授权的
例外，已按原 V2 未评分收尾；不要把它变成未来实验的默认重试策略。
