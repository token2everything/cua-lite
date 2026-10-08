---
name: webarena-verified-eval
description: "Prepare, run, monitor, and analyze WebArena-Verified 1.2.3 evaluations using CUA-Lite's native generic agent. Treat prior Qwen and GLM experiments as configuration references and assess support for the user's chosen model and settings. Use for this evaluation workflow, not arbitrary BrowserGym or model-specific runners."
---

# WebArena-Verified：Codex 执行说明

使用 CUA-Lite 原生通用 agent，帮助用户检查准备条件、运行已授权的评测、
恢复已有任务或分析轨迹。主线是复用现有平台与资产，不是自动从零部署站点。
目标是支持用户本轮的 WebArena-Verified 评测，不是默认复现历史实验。
Qwen/GLM 已验证配置供参考；只有用户明确要求复现时，才以对应历史配置为目标。

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
普通参数与当前支持范围以入口 `--help` 和实现为准。仓库
`devs/exps/eval/browsergym.webarena_verified/EXPERIMENT_CONFIG.generic-desktop-dual-v2.json`
保存已验证预设，不替代用户对新评测的配置选择；本轮实际版本以用户确认后
生成的 config 和 manifest 为准。当前 initializer 只支持 `qwen3.8-max` 和
`glm-5.3-flash` 预设。其他模型或配置要求先评估接入差距，不偷偷套用旧 profile，
也不通过改冻结文件绕过限制。需要改代码或模板时，先提出版本化适配与回归计划，
取得相应实施授权，再初始化新的运行目录。

## 新建评测的阶段检查

1. **检查准备条件。** 先确认本轮模型、API 接入、thinking、预算、历史策略和资源要求；
   区分用户选择、参考配置建议和当前入口限制。再确认本地资产根目录、官方 catalog、桌面依赖相对路径、
   平台卷子路径、镜像和配额要求。按指南第 2 节给出实际配置位置：私有 `.env` 的
   必填变量、依赖安装命令、initializer 参数、image lock 和平台侧权限配置；
   不只说“缺凭证/资产”，要说明填哪里、缺哪一项、由谁提供。
   自主执行请求范围内的本地及远端只读检查，按指南的权限预检方法报告结果和证据；
   不把镜像/资产权限检查全部转交用户，也不把本地文件存在当成远端可用。
   部分缺项不妨碍其他独立的只读检查，逐项推进可做项，再汇总缺项。
   凭证只报告是否有效填写，不能输出值；未经授权不通过模型推理请求来试 key。
   需要申请沙盒、拉取大镜像或准备大数据才能确认的执行能力，说明检查成本和所需授权；
   若已有相应授权，直接在本 run 自建且归属核验通过的沙盒验证，不重复询问。
   对照 initializer 的文件/目录清单，区分本地路径与平台挂载。
   历史资产路径不是已验收的新输入；缺项就报告准确路径和负责人，不创建标记或搬迁数据。
2. **展示参考配置并报告计划。** 读取技术参考第 2 节和配置模板，直接输出已验证
   Qwen/GLM 实验的参考配置简表，不只发文件链接。覆盖模型与 thinking、原生通用 agent、
   GUI/桌面与坐标、任务数、决策预算与收尾/修复、文字/图片历史、输出参数、
   HTTP/Infra 重试和沙盒数量/资源；凭证只列变量名。注明来源版本与“参考，不是本轮默认”。
   再给出本轮建议配置与参考配置的差异、运行目录与 run ID、准备检查结果、
   配置位置和补齐方法、待验证能力、需要适配的地方，以及将产生费用的阶段。
   用户未选模型时，先展示参考配置再询问，其他独立检查继续；未指定的参数可根据
   参考配置提出建议，说明理由并等待确认，
   不因上手示例使用 Qwen 而擅自选择后端或把历史配置当作本轮已授权配置。
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
