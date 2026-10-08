---
name: webarena-verified-eval
description: "Prepare, reproduce, monitor, and analyze WebArena-Verified 1.2.3 evaluations using CUA-Lite's native generic agent, including the verified Qwen and GLM desktop configurations. Use for this evaluation workflow, not arbitrary BrowserGym or model-specific runners."
---

# WebArena-Verified：Codex 执行说明

使用 CUA-Lite 原生通用 agent，帮助用户检查准备条件、运行已授权的评测、
恢复已有任务或分析轨迹。主线是复用现有平台与资产，不是自动从零部署站点。

## 先判断用户要做什么

- 新建评测或初次接手：读取本 skill 下 `references/reproduction.md`
  （[快速上手指南](/skills/webarena-verified-eval/references/reproduction.md)），按用户授权执行对应阶段。
- 只查进度、同步证据或分析：先确认已有运行目录，读取指南第 6 节；
  不初始化、不申请沙盒、不重新跑题。
- 清理资源：读取指南第 7 节，先做预检；只有明确删除授权才使用 `--delete`。
- 核对配置、资产、排查或换平台：按问题读取 `references/technical-reference.md`
  （[技术参考](/skills/webarena-verified-eval/references/technical-reference.md)）的相应章节。
  不为状态查询加载全部历史结果或镜像说明。

定位包含 `pyproject.toml` 与 `scripts/webarena_verified_generic.py` 的实际 checkout。
引用路径以本 skill 所在目录和实际仓库根目录解析，不假定作者的 `/root` 或共享盘存在。
普通参数以入口 `--help` 为准，数值以仓库
`devs/exps/eval/browsergym.webarena_verified/EXPERIMENT_CONFIG.generic-desktop-dual-v2.json`
为准，本轮实际版本以运行目录的 config 和 manifest 为准。

## 新建评测的阶段检查

1. **检查准备条件。** 确认所选模型、本地资产根目录、官方 catalog、桌面依赖相对路径、
   平台卷子路径、镜像和配额要求。只报告凭证变量是否填写，不输出值。
   本地文件检查不证明远端权限或配额有效；未经授权不通过模型请求来试 key。
   对照 initializer 的文件/目录清单，区分本地路径与平台挂载。
   历史资产路径不是已验收的新输入；缺项就报告准确路径和负责人，不创建标记或搬迁数据。
2. **报告计划。** 给出模型、预算、历史策略、资源数量、运行目录与 run ID、
   当前已确认/尚未确认的条件，以及将产生费用的阶段。用户未选模型就询问，
   不因上手示例使用 Qwen 而擅自选择后端。
3. **离线初始化。** 使用新目录运行 `campaign/initialize.py`；核对模型、任务数、
   worker 计划和各 manifest。这只验证本地输入和冻结包，不宣称远端已可用。
   冻结后不就地改 config、plan 或源码包，不覆盖旧结果。
4. **单沙盒验证。** 有冒烟授权才对 worker 01 执行
   `create → wait-ready → stage → prepare → startup`，逐阶段检查结果。
   Map 首次准备涉及大数据复制；核对 I/O、日志、远端命令，不凭等待时间猜测卡死。
   startup 跑正式计划首题，不要求得分为 1；核对归档与评分状态。
5. **全量。** 有全量授权才扩展 fleet。按指南逐步提升 Map 准备并发；
   全部 worker 的 startup 检查有效后才运行 `full`。
   单 worker partial gate 不能放行全量。长阶段用可追踪后台会话并定期汇报进度。
6. **保存与分析。** 执行 `status/sync`，核对本地证据，在新分析目录生成报告。
   发布飞书或删除资源属于不同操作，按用户授权继续，不自动执行。

评测链路由 `campaign/initialize.py` 初始化、`campaign/orchestrate.py` 管理阶段、
`scripts/webarena_verified_generic.py` 逐题执行；分析使用
`scripts/webarena_verified_analyze.py`，清理使用 `campaign/cleanup.py`。
不要切到旧 `run.sh`、gateway、Qwen XML runner、viewport launcher 或 OSWorld agent 来绕过失败。

## 中断、异常与停止条件

- 同一运行目录只允许一个调度进程。接手时先核对 run/config/manifest、
  本地账本、已有进程和已知远端 command ID，确认状态后再继续当前阶段。
  不知道创建/模型请求是否成功时先查证，不盲目重发，不重新初始化。
- 缺凭证、资产或权限就停在对应准备阶段；不能静默换模型、浏览器、数据版本、
  thinking 强度或预算。若需要修代码，保留本轮证据，另开版本和运行目录。
- 每题默认最多 3 次基础设施尝试，保留全部证据；耗尽后隔离、继续其他题。
  正常业务失败不重抽，不清账本追加次数。`full` 退出码 2 和
  `EXHAUSTED_TASKS.json` 表示有耗尽题，需要报告而非无条件重跑。
- 评分程序报错和基础设施耗尽都不是官方 0 分。分别记录 canonical terminal、
  scored、success、evaluator-unscored、Infra-exhausted 和 pending。
  有缺失分数时全量官方准确率为 NULL，只能注明已评分子集成功率。
- 清理前先保存证据、核对命令已结束及资源身份；只删除本 run 实际创建且
  身份验证通过的沙盒，删除后查证不存在。未知状态或外部资源不在清理范围。

指南和 skill 不授权费用、资源创建、删除、Git 提交或文档发布。
用户已经明确授权的范围无需逐条再问；需要扩大范围时先说明原因并请求方向。
历史 GLM task 783 的第 4 次属于单独授权的例外，不是默认重试策略。

## 每次交接留下什么

给出实际仓库路径、运行目录与 run ID、模型、源码/config/plan/镜像/资产哈希、
已通过和未通过的阶段、执行与评分覆盖、异常清单、轨迹和报告位置、
后台会话与已知远端命令 ID、下一步和剩余授权范围，以及资源是否保留。
不要记录凭证值，也不要用“完成了”掩盖未评分或未落盘的任务。
