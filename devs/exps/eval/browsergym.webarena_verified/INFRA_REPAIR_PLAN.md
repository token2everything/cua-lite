# WebArena-Verified Infra 修复与版本计划

更新时间：2026-10-05（Asia/Shanghai）
适用实验：`glm5.3-flash-qwenrunner-structured-150-full120-20261004`
正式结果：`200/812 = 24.6305%`

这份计划把一次实验中发现的故障拆成可单独回归的版本。每个版本必须有独立
commit、固定 run id、固定 canary 集合和完整 manifest。没有完成该版本的门禁，
不得启动 812 题全量，也不得把不同版本的分数放进同一张结果表。

## V0：冻结当前基线

V0 是审计基线，不是可发布的干净能力分数。

- GLM run：`wav-glm53-flash-qwenrunner-structured-150-full120-20261004`
- 812 题，150 model steps，reserve 7，thinking 开启，`glm-5.3-flash`
- 冻结 runner：`scripts/webarena_verified_qwen.py`，SHA-256
  `666217eda9dd4bfdd83b7b58523373774d6493a9a0ae7442cb613c0c5704e5c6`
- OSWorld：commit `b138d348256078fa634fc3b73567a7337c793e6b`
- dataset SHA-256：
  `d65275660814663375028e9017e1f929e3c38321041b125795e2713b52243d30`
- evaluator SHA-256：
  `35c3385b1db4b3378657589f95f50defd4234bd36e5b93d44733fd561b01db4e`

当前仓库仍有未提交的 WebArena 集成改动，因此不能把当前 `HEAD` 当作 V0
回退点。开始 V1 前必须先把现有工作区整理成一个 commit，并在 manifest 中
记录完整 pipeline 文件清单、runner SHA、BrowserGym/evaluator 版本、site image
digests、配置 SHA 和入口命令。原始 V0 归档只读保存，不能重写。

## 已确认的问题与责任边界

| 编号 | 问题 | 证据 | 责任模块 | 影响 |
|---|---|---|---|---|
| I1 | 登录重试留下错误 tab，runner 用 `context.pages[-1]` 选错页 | case 0、3、449 截图/HAR；正常页在首 tab，模型看到 404/502 | BrowserGym setup 与 runner page selection | 模型在错误页面开始，启动门禁仍可能通过 |
| I2 | 起始页 HTTP 502/404 未作为健康失败处理 | case 44 初始 502，刷新后恢复；setup 只捕获 Timeout | BrowserGym task setup / startup gate | 把暂时不可用的环境记成可评分 episode |
| I3 | `reasoning_content` 与正式 content 拼接后再解析 | 152 个 case 额外执行 584 条 GUI 动作；case 63 同步 click 10 次 | `scripts/webarena_verified_qwen.py` 与上游 parser 边界 | 改变页面状态和轨迹 |
| I4 | 无法解析的 tool output 被 parser 转成 DONE/FAIL | 479 题零 GUI；469 题在 2 轮内终止 | OSWorld parser 与 runner terminal accounting | 格式错误被记成任务完成，遮蔽真实失败 |
| I5 | JSON repair 与动作 repair 混为一条路径 | 529 次 repair 中只有 46 次最终有效 | runner/provider structured output boundary | GLM 格式恢复率只有 8.70% |
| I6 | 坐标合同不一致；无效坐标静默降级 | 1920×1080 桌面、relative/1000 prompt；220 题越屏 981 次 | runner 与 pinned Qwen parser | 点击越屏或落到当前鼠标位置 |
| I7 | `wait(time=N)` 丢失 N，只记录 `WAIT` | case 548 请求 15 秒但整轮 6.82 秒 | pinned parser/runner action dispatcher | 页面异步状态未等待，后续动作过早执行 |
| I8 | HAR/trace 数字分数一致不代表评价结果健康 | 6 个 evaluator error；811 实际 PUT 200 但 evaluator 自身异常 | scoring boundary / evaluator schema | 评分结果被错误标为 0 或 error |
| I9 | 未提交表单的 `dummy_bin` 证据没有生产路径 | 528/529/530/689/690/691/692/693 全部缺事件 | BrowserGym/WebArena Verified evidence contract | 合法草稿状态无法被评分 |
| I10 | strict isolation/reset 是可选项，启动门禁只检查 numeric score | `isolation.py` 默认 permissive；receipt 不含 startup page/reset generation | campaign coordinator / env server | 跨任务残留和并发写入可能污染结果 |
| I11 | 入口配置不统一 | GLM direct runner 是 150 steps/thinking/max_tokens 32768；gateway 默认 30 steps/无 thinking | campaign entrypoint | 不同入口的分数不可直接比较 |
| I12 | singleton alternatives 触发 evaluator error | 699/700/811 的合法单元素 JSON 数组被当作 alternatives | evaluator normalization/dataset schema | 已完成的 mutation 任务无法正常评分 |

这些问题不能用“把有效 JSON 子集的 62.31% 外推”为真实能力。它是有选择偏差
的子集，也不能作为修复后的预测值。

## 版本拆分

### V1：启动页和基线完整性

目标是保证模型第一张图对应一个健康、正确、可追溯的任务页。

改动归属：

1. setup 边界返回实际 start page，或由 runner 根据 `start_urls`、origin、path
   和 DOM health 选择；删除无条件的 `context.pages[-1]`。
2. 登录/导航重试用 `finally` 清理失败 tab；对 4xx/5xx 和 known error body
   做有限重试，重试耗尽写 `startup_failed`，不进入 scored episode。
3. 生成 `startup_receipt.json`：页面列表、选中 page index/url/title、每个起始
   URL 的响应状态与错误页标记。认证状态、reset generation 和 site image digests
   由能证明这些事实的 coordinator/baseline owner 提供，不从截图或 URL 猜测。
4. completion receipt 必须绑定 startup receipt 的 SHA；`status=completed` 不能
   由 parser DONE 单独产生。

固定回归集：`0, 3, 44, 449, 798`。
通过条件：错误 tab 存在时仍选择健康任务页；单独起始 502 在限定重试内恢复；
重试失败被标为 infra failure 且可重试；同一个 baseline receipt 不得跨任务复用。

### V2：动作协议与观察执行

目标是让模型输出、parser、执行器和轨迹记录共享一个动作合同。

改动归属：

1. parser 只接收 provider 的正式 `content`；reasoning 仅记录到 metadata/history，
   不进入可执行 XML 扫描。
2. 抽取纯函数 `classify_agent_turn`：有效 final JSON、显式 terminal tool、
   GUI action、malformed tool 和 plain invalid text 是不同状态。malformed tool
   永远不能合成 DONE/FAIL；动作修复与 final JSON 修复使用不同反馈和计数。
3. 明确一个坐标合同。建议模型看到的 processed screenshot 使用 absolute
   `processed_width × processed_height`，再由执行器只做 processed→desktop 的
   线性变换；坐标必须是有限数、二维数组且落在图像范围，非法值向模型返回
   可见错误，不降级成当前鼠标 click 或 `(0,0)`。
4. `wait` 保留 seconds 并由 dispatcher 实际等待；缺失或负数时返回动作错误。
5. 轨迹中记录 `action_parse_error`、`action_source=content`、requested wait、
   executed wait、坐标转换前后值和 terminal provenance。

固定回归集：`7, 63, 97, 100, 118, 185, 267, 300, 330, 548, 624, 798`。
通过条件：reasoning 中的 tool_call 计数为零执行；malformed output 不产生
terminal；合法 JSON final 仍能结束；相同截图坐标不越屏；wait 时间与记录一致。

### V3：评分与证据契约

目标是让评分结果能区分真实失败、证据缺失和 evaluator 自身错误。

改动归属：

1. `score_artifacts` 同时记录 HAR/trace 的 evaluator status、error、assertions、
   task/evaluator id 和事件顺序摘要；`status=ERROR` 或 assertion error 不能
   伪装成普通 score 0。官方原始分数保持不变。
2. `dummy_bin` 事件当前没有已知的官方生产路径。先记录表单末态的独立诊断证据，
   向基准方核实证据合同；未经确认不得向正式 HAR/trace 注入合成事件，也不得
   通过真实 Submit 伪造事件。
3. singleton 数组的 schema 修订、替代路径评分都属于数据集/评价器合同变更，
   必须发布单独的 evaluator/package 和 checksum，并与官方 1.2.3 原始分数并列。
   修改共享 dataset JSON 不会改变 BrowserGym 实际加载的包内数据集。
4. 271、481、488、599 的业务等价证据仅作探索性分析；官方 evaluator 检查
   指定 URL、方法和状态码，在合同变更前仍保留官方 0 分。

固定回归集：`195, 271, 481, 488, 599, 689, 699, 700, 811`。
通过条件：case 195 的 format-only 反事实可复现；独立 fork 中的 811 不再抛
   singleton alternatives，699/700 仍不因消除异常而补分；草稿任务有可审计的
   未提交表单末态诊断证据，官方证据合同未知时不补分；HAR/trace 结果不仅数字
   相同，status/assertions/事件身份也一致。

### V4：并发、隔离与全量实验

目标是把已修复版本用于可发表的全量成绩。

1. publishable full run 强制 strict backend isolation；没有可验证的独立干净
   backend、完整 baseline/startup receipt 时直接拒绝启动。当前站点镜像采用
   每题新建站点容器，不得臆造不存在的 full reset endpoint 或 generation。
2. mutation 任务按 catalog dependency DAG 单 worker 串行；禁止
   `fast-write-shards` 进入可发表 run，除非每个 worker 使用独立干净 backend。
3. read/write scheduler、gateway 和 direct runner 只能由一个 canonical entrypoint
   启动；argv、step budget、thinking、history/image budget 全写入 manifest。
4. 先跑 20–30 题 defect canary，再跑完整 812 题。每题保存 execution integrity
   与 score，基础设施失败可以重试，完整且 score=0 的模型失败不得自动重跑。

固定 canary 组成：V1 启动 5 题、V2 协议 12 题、V3 评分 9 题，再加 4 个随机
   read/write 对照题。canary 通过后才解锁 full run。

## Manifest 与回退规则

每个版本的 `VERSION_MANIFEST.json` 至少包含：

```json
{
  "infra_version": "webarena-verified-v2",
  "git_commit": "<committed-pipeline-sha>",
  "runner_sha256": "<runner-file-sha256>",
  "entrypoint": "scripts/webarena_verified_qwen.py",
  "argv": ["--max-steps", "150", "--final-response-reserve-steps", "7"],
  "protocol_version": "action-contract-v2",
  "osworld_commit": "b138d348256078fa634fc3b73567a7337c793e6b",
  "browsergym_version": "0.14.3",
  "evaluator_version": "1.2.3",
  "dataset_sha256": "d65275660814663375028e9017e1f929e3c38321041b125795e2713b52243d30",
  "site_image_digests": {},
  "config_sha256": "<experiment-config-sha256>",
  "canary_ids": [0, 3, 44, 63, 195, 267, 449, 811],
  "rollback_to": "webarena-verified-v1"
}
```

版本规则：

- `webarena-verified-v0-baseline`：当前 GLM 归档，只读审计，不再追加结果。
- `webarena-verified-v1-startup`：只包含启动页、健康门禁和 receipt 变化。
- `webarena-verified-v2-protocol`：在 V1 上加入动作、reasoning、coordinate、wait 修复。
- `webarena-verified-v3-evaluator`：在 V2 上加入评分证据与 evaluator 契约修复。
- `webarena-verified-v4-full`：V3 canary 通过后的 812 题正式 run。

每次版本都必须提交后运行，不能在 dirty worktree 上启动。回退时选择旧
manifest 和对应 commit 重新启动新的 `EVAL_RUN_ID`；不修改旧 run 的
`index/`、`archives/`、`evaluation.json`，也不把新版本结果覆盖到旧目录。

## 当前执行顺序

1. 先冻结当前工作区，建立 V0 manifest 和可回退 commit。
2. 实现 V1，跑启动 canary；失败只回到 V0。
3. 实现 V2，先做纯函数和 fake runner 回归，再跑协议 canary；失败回到 V1。
4. 实现 V3，先离线 replay evaluator，再跑评分 canary；失败回到 V2。
5. 实现 V4 的 strict isolation 和唯一入口，canary 全绿后才启动 GLM 全量。

每个阶段报告三种数字：官方 raw score、按证据完整性排除后的可比 score、
infra failure 数。不能把离线反事实修复结果混入正式成绩，也不能用选择后的
有效子集估计全量真实能力。

Qwen 默认入口、模型参数和 parser 行为是回归基线；启动健康门禁与 GLM 协议
修复必须显式启用。每个版本先运行 Qwen 离线协议回归，再运行 GLM canary。
