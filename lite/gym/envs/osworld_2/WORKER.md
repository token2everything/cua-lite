# OSWorld-V2 运行指南

下面用一个完整例子说明：**从工作站提交 v2.1 的任务 054，使用 `glm5.3-flash`，下载轨迹后回收 Sandbox。** Agent、coordinator、env-server、Worker 和桌面 VM 都在 Sandbox 内运行。

新增 benchmark 的开发步骤见 [接入示例与 SPEC](/docs/ADDING_BENCHMARKS.md#example-integration)；本页用于运行已经接入的 OSWorld。首次发布镜像、任务数据与模板的同事先读 [部署指南](/lite/platforms/sandbox/osworld_2/README.md)。

## 1. 开始前拿到这些信息

| 谁提供 | 本例需要什么 |
|---|---|
| 平台/benchmark 维护者 | 项目、API/AFS 地址；已发布的 v2.1 模板 ID。模板固定 benchmark 与通用 Agent 的镜像 digest，挂好任务数据并启用飞书 |
| 任务执行者 | 该项目的 AK/SK、模型接口地址/密钥/模型 ID |
| 工作站 | Linux、Python 3.12、`uv`、`rclone`，能访问平台和 AFS |

示例中的平台和模板属于本工作区；其他项目替换成自己的值。更新本地代码不会升级已有模板。此架构的实机回归状态见[验证说明](#verification-and-extension)。

以下命令均在 `cua-lite` 仓库根目录执行：

```sh
uv sync --locked --extra sandbox
rclone version
umask 077
mkdir -p .sandbox
cp lite/platforms/sandbox/osworld_2/configs/run.yaml .sandbox/run.yaml
```

## 2. 配置示例

本例将私密输入放在同一个目录，日常只改 `run.yaml`：

```text
.sandbox/
├── run.yaml                 # 平台、版本、任务、模型选择和结果目录
├── AK-SK                    # 平台凭据，按下方格式创建
├── model-api.json           # 模型凭据，按下方格式创建
├── feishu-webhook.json      # 可选；模板已启用飞书时不需要
├── state/                  # submit 自动生成；保留它才能同步和回收
└── results/                # sync 自动下载到这里
```

### run.yaml：这次运行什么

复制 [run.yaml](/lite/platforms/sandbox/osworld_2/configs/run.yaml)，本工作区的 v2.1 模板已填好；其他项目向维护者取得模板 ID 并替换平台字段。Agent 镜像由模板的 `env.CUA_AGENT_IMAGE` 固定，无需在每个作业中指定发布包。

```yaml
project: ailab-llmagent
base_url: http://deepenv.sandbox.pjlab.org.cn
afs_url: http://10.140.95.76:28081
environment_id: 5d5b5dbb-bd30-447e-bff5-e67ec8b7bba4
results_prefix: outputs/osworld
release: osworld-v2.1
credentials: AK-SK  # 平台凭据文件：.sandbox/AK-SK；也可填已有文件的绝对路径
state_dir: state    # 自动创建 .sandbox/state/，保存同步、回收和批次恢复所需记录
# feishu_config: feishu-webhook.json  # 未从模板继承时启用
monitor:                         # 仅 batch 使用，复用同一个机器人
  enabled: true
  interval: 300                  # 每 5 分钟汇总
  stall_after: 900               # 15 分钟无新 Agent 步骤提醒
job:
  model_config: model-api.json
  model_profile: glm5.3-flash
  tasks: ["054"]
  max_steps: 500
  history_turns: 4
  results_root: results
```

**本地相对路径以 YAML 所在目录为基准**。本例配置位于 `.sandbox/run.yaml`，所以
`credentials: AK-SK` 读取 `.sandbox/AK-SK`，`state_dir: state` 使用 `.sandbox/state/`；
`model-api.json` 和 `results` 同样位于 `.sandbox/` 下。直接用仓库里的示例 YAML 启动时，
这些路径会相对它所在的 `configs/` 目录解析，因此先复制配置再使用。凭据在其他位置时，
填写已有文件的绝对路径即可。`results_prefix` 是当前项目的 AFS 路径。

`state_dir` 由程序创建，保存 Sandbox 所有权、提交记录、私密作业和批次进度；
`sync`、`delete` 和 `batch --resume` 都依赖其中的记录。同步、回收或恢复同一作业时
沿用这个目录，完成结果同步和资源回收前请保留，也不要提交 Git。截图和轨迹保存到
`job.results_root` 指定的目录。

本例采用[官方论文 §3.1](https://arxiv.org/html/2606.29537v2#S3.SS1)的 **500 步预算**；
一步按 Agent 交互回合计数，一回合可包含多个动作。多阶段任务与官方代码一致，
每阶段重新计数。上游示例脚本的预算可能不同，复现实验时记录实际配置；
连通性短测可显式调小 `max_steps`。

请求发送最近 **4 张历史截图＋当前截图**，文字历史完整保留。任务 ID 必须加引号，避免 YAML 将 `054` 解析成整数。

<a id="配置示例"></a>

### AK-SK 与 model-api.json：用什么凭据

新建 `.sandbox/AK-SK`，填入 Portal“访问令牌”中该项目的凭据：

```json
{"ak": "REPLACE_WITH_ACCESS_KEY", "sk": "REPLACE_WITH_SECRET_KEY"}
```

新建 `.sandbox/model-api.json`，替换接口地址、密钥与实际模型 ID：

```json
{
  "base_url": "https://model-gateway.example/v1",
  "api_key": "REPLACE_WITH_API_KEY",
  "default_model": "glm5.3-flash",
  "models": {
    "glm5.3-flash": {
      "model": "REPLACE_WITH_API_MODEL_ID",
      "api_type": "openai_chat"
    }
  }
}
```

`glm5.3-flash` 是本地 profile 名，与 YAML 对应；`model` 是网关接受的模型 ID。`submit` 只上传选中 profile 的快照。其他 benchmark 可复用此模型文件。

模板未配置飞书时，新建 `.sandbox/feishu-webhook.json`，再取消 YAML 中 `feishu_config` 的注释：

```json
{"webhook_url": "https://open.feishu.cn/open-apis/bot/v2/hook/REPLACE_WITH_HOOK_ID"}
```

密钥文件不提交 Git。`openai_responses` 接口需支持原生 `computer` 工具，并删掉 YAML 的 `history_turns`；本例使用 `openai_chat` 桌面函数工具。用户模拟器和评分器的凭据由 Worker 维护者另行[配置](/lite/platforms/sandbox/osworld_2/README.md#configure-websites-and-native-model-clients)。

<a id="submit-sandbox"></a>

## 3. 预览并启动 054

```sh
uv run python -m lite.platforms.sandbox.osworld_2.manage --config .sandbox/run.yaml plan
```

检查输出中的 `release` 为 `osworld-v2.1`、`requested_task_ids` 和 `scheduled_task_ids` 都为 `["054"]`、`unsupported` 为 `[]`。`plan` 不读取凭据、不联网，只确认本地配置和任务选择；不会确认远端模板或镜像可用。

填完凭据和模板 ID 后提交：

```sh
uv run python -m lite.platforms.sandbox.osworld_2.manage --config .sandbox/run.yaml submit
```

提交会生成并上传作业与模型快照，检查模板版本、镜像 digest 和结果挂载，创建 Sandbox，并验证实例的飞书设置。`job.json` 只包含模型 ID、接口类型等运行信息；模型凭据单独保存。coordinator 自动准备 Worker 连接和原生 env-server，再启动独立 Agent 容器。

返回信息中记下：

| 字段 | 怎么用 |
|---|---|
| `sandbox_id` | 后续 `status`、`sync`、`delete` 的参数 |
| `feishu_webhook_verified: true` | 该实例已确认启用飞书 |
| `submission.experiment` | 本次实验目录名；默认自动生成，以 submit 的返回值为准 |
| `ownership_file` | 本地提交记录，默认在 `.sandbox/state/`；不要删除 |

设置刚返回的 ID，查看运行状态：

```sh
OSWORLD_SANDBOX_ID='REPLACE_WITH_RETURNED_SANDBOX_ID'
uv run python -m lite.platforms.sandbox.osworld_2.manage --config .sandbox/run.yaml status "$OSWORLD_SANDBOX_ID"
uv run python -m lite.platforms.sandbox.osworld_2.manage --config .sandbox/run.yaml logs "$OSWORLD_SANDBOX_ID" --lines 80
```

`submit` 返回表示实例已创建，任务会继续在 Sandbox 内执行。创建响应丢失时，按报错中的 `worker_run` 到 Portal 核对，避免重复提交。

<a id="artifacts-and-coverage"></a>

## 4. 看结果，再回收

Agent 开始写入结果后执行；运行中可重复同步：

```sh
uv run python -m lite.platforms.sandbox.osworld_2.manage --config .sandbox/run.yaml sync "$OSWORLD_SANDBOX_ID"
```

打开返回的 `results` 目录，本例结构为：

```text
.sandbox/results/<experiment>/
├── experiment.json               # 版本与请求的任务
├── summary.json / summary.csv    # 任务状态、覆盖范围和模型得分
└── task_054/<run-id>/
    ├── run.json                  # 实时状态、score、closed、错误原因
    ├── agent_trajectory.jsonl     # 整个任务一条 JSONL，metadata.tools 中有工具定义
    ├── images/                   # 轨迹使用的截图
    ├── events.jsonl              # 步骤与耗时
    ├── result.json               # 原生评分
    └── model/episode_000/         # 每次独立 Agent 执行
        ├── *.request.json / *.response.json
        ├── episode.json          # Agent 状态、步数和消息边界
        ├── agent.log             # Agent 进程日志
        └── env-server.log        # 本阶段原生 env-server 日志
```

**先看 `run.json`**：`status: completed` 且 `closed: true` 表示流程和任务清理完成，任务是否解对看 `score`。失败时查看错误字段和 `logs`。Sandbox 的 `Running` 不代表 Agent 仍在执行，飞书平台通知针对 Sandbox `Failed`，不等于每题完成通知。

`sync` 同时下载已生成的 `_environment/<experiment>/`；首次初始化失败时可能没有环境日志。它按本地提交记录定位旧实验，后续修改 YAML 或移走模型文件不影响同步。图片路径与本地镜像目录一致，可连同轨迹交给查看器；换机器或目录时需调整图片路径。

确认该作业所有任务已有最终状态、清理结束，执行一次最终同步，然后回收：

```sh
uv run python -m lite.platforms.sandbox.osworld_2.manage --config .sandbox/run.yaml sync "$OSWORLD_SANDBOX_ID"
uv run python -m lite.platforms.sandbox.osworld_2.manage --config .sandbox/run.yaml delete "$OSWORLD_SANDBOX_ID"
```

删除是异步操作，在 Portal 确认实例已消失。AFS 结果保留，有本地提交记录仍可同步；中途删除会终止作业。长任务接近到期时，可先续期两小时：

```sh
uv run python -m lite.platforms.sandbox.osworld_2.manage --config .sandbox/run.yaml renew "$OSWORLD_SANDBOX_ID" --seconds 7200
```

模板前缀为 `outputs/osworld` 时，Agent 原始文件在 AFS 的 `outputs/osworld/agent-results/<experiment>/`；启动日志和作业退出码 `agent-exit-code` 在 `outputs/osworld/_workers/<worker-id>/`。逐阶段日志见各 run 的 `model/episode_NNN/`。`manage results <run-id> --output <目录>` 用于额外下载 Worker 原始记录。

## 5. 换任务或版本

复制本例配置，按需修改后重复 `plan → submit → sync → delete`：

| 目标 | 修改 run.yaml |
|---|---|
| 顺序执行 054、069 | `job.tasks: ["054", "069"]`；同一个 Sandbox，逐任务初始化 |
| 跑 v2.1 offline | 删除 `job.tasks`，改为 `job.task_set: offline`，共 82 项 |
| 跑 full | 删除 `job.tasks`，改为 `job.task_set: full`，共 108 项 |
| 换 0808 / 0624 | 修改 `release`，选择匹配 Worker 与数据版本的 `environment_id`；可复用相同通用 Agent 镜像，见[模板说明](/lite/platforms/sandbox/osworld_2/README.md#workspace-templates) |
| 并发执行 | 同一份 YAML 选择任务，用 `batch --concurrency N` 自动拆分、补位和回收，见下例 |

每次 submit 默认生成新 experiment。设置 `job.experiment` 时需手动保证每个作业名称不同。完整轨迹都保留；请求截图数量由 `job.history_turns` 控制。

0624、0808 没有官方 offline 子集。0624 的 7 个交互任务与 069 仍在协议排除清单中；明确接受部分覆盖时设置 `job.skip_unsupported: true`。offline 也可能需要素材、自建网站或模型 API。

054 无需额外网站。网站任务需由 benchmark 模板预置服务；标准 Worker 启动脚本不会安装 OSWorld-web。026 使用[同一 Sandbox 内的 GitLab 服务](/lite/platforms/sandbox/osworld_2/services/gitlab/README.md)，并为外层 Sandbox 预留服务内存。发布维护者负责服务就绪；这些依赖不打入通用 Agent 镜像，填写任务 ID 也不会自动部署网站。

### 自动拆分与并发评测

仍然只维护一份 `run.yaml`。例如运行 0808，修改已有配置的版本、模板和任务选择，其余平台和模型配置保留：

```yaml
release: osworld-v2-2026.08.08
environment_id: ddfaf932-2461-4209-8d72-5cc991a3a60d
job:
  model_config: model-api.json
  model_profile: glm5.3-flash
  tasks: ["054", "007", "069"]
  # 全集：删除 tasks，改为 task_set: full
  experiment: osworld-0808-batch-001
  max_steps: 500
  history_turns: 4
  results_root: results
```

先预览，再启动最多 4 个 Sandbox：

```sh
uv run python -m lite.platforms.sandbox.osworld_2.manage --config .sandbox/run.yaml batch --concurrency 4 --dry-run
uv run python -m lite.platforms.sandbox.osworld_2.manage --config .sandbox/run.yaml batch --concurrency 4
```

每题使用独立 Sandbox，实验名为 `<batch-id>-task-<任务号>`。任务结束后自动同步轨迹、
评分和启动日志，并核对所有权、删除实例；**确认实例消失后才启动下一题**。每个实例
都验证飞书设置。调度器读取 Agent 退出码及 `run.json`，不会把 Sandbox 的 Running
当作 Agent 仍在工作，也不会把得分 0 当作流程失败。任务失败会记录原因，其他任务继续。

调度命令需要持续运行，可放在自己的 `tmux` 会话中；Agent 和 Worker 仍在 Sandbox 内。
默认每 30 秒查询一次，接近到期时自动续期。每题含启动时间默认最多 4 小时，可用
`--task-timeout 21600` 改为 6 小时；超时先终止该 Sandbox，再同步最终文件。
并发上限包含创建、执行和回收中的实例，按项目配额设置。普通模板每个外层 Sandbox
为 8 CPU / 16 GiB；所有任务沿用同一模板，网站、GitLab 依赖及额外内存须先准备好。

进度、每题状态及模型得分实时写入 `.sandbox/state/batches/<batch-id>/batch.json`。
下载结果按批次归档，无须为每题配置输出目录：

```text
results/<batch-id>/
├── task_002/
│   ├── <run-id>/                # result.json、agent_trajectory.jsonl、images/、model/
│   ├── experiment.json         # 本题实验信息，summary.json / summary.csv 也在此处
│   ├── _launcher/              # 启动日志、退出码
│   └── _environment/           # 环境日志（若已生成）
└── task_004/
    └── ...
```

批次目录名来自 `job.experiment`，未填写时自动生成。`batch` 自动下载与手动 `sync` 使用
同一布局，轨迹中的截图路径随下载位置调整。AFS 原始输出仍按每题的唯一实验名保存。
本地保留选中模型的
私密快照，恢复后待运行任务继续使用同一配置。中断调度器不会停止已经运行的 Sandbox：

```sh
uv run python -m lite.platforms.sandbox.osworld_2.manage --config .sandbox/run.yaml batch --resume osworld-0808-batch-001
```

恢复时使用原来的部署、state 目录和 batch ID；不会重跑已完成任务。同步或管理接口
出错时会停止补位并保留进度。若创建响应丢失，按提示中的提交记录及 `worker_run` 到
Portal 核对，不自动重复创建。一个批次只运行一个调度器；进程被强杀留下
`.scheduler.lock` 时，确认旧调度器已退出后再移除该锁目录。批次含流程失败时命令退出码
为 1；模型得分为 0 且流程完成时仍为 0。

### 飞书监控卡片

在 `run.yaml` 顶层设置上面的 `monitor` 段，再正常运行 `batch` 即可；只有一个任务也能用。
卡片复用模板机器人或 `feishu_config`，无须填写 CardKit 模板 ID，也不用新增飞书应用。

- 顶部显示已结束 / 总数、运行、排队、收尾及需关注数量。
- “需关注”默认展开：失败、疑似停滞、进度暂时不可读；正常运行及已完成详情可点击展开。
- 任务详情显示累计步数、本阶段步数 / 预算、距上次推进时间，以及是否在等待模型回复。
  模型重试不算推进，步数增长也不代表任务做对；结束后单独展示流程结果与模型得分。

默认每 5 分钟发送一张快照，15 分钟没有完成新步骤时提示“疑似停滞”，不自动结束任务。
开始、恢复、结束和调度错误另发通知；异常变化最多每分钟合并一次，在下一轮查询后发送。
原来的 Sandbox Failed 通知仍保留。`submit` 不会启动这个监控循环。

监控只读小文件和文件名，不下载请求里的截图。网络错误时显示“暂时无法确认最新进度”，
不会当成 Agent 推进；发送失败写入 `batch.json.notification.error`，不打断评测。
调度器须持续运行；关闭 tmux 内进程或工作站断电后不会继续发卡片。已发卡片是带时间的
快照，不会原地刷新。批次恢复后沿用已保存的步数和最后推进时间。

<details>
<summary>执行机制与排查入口（适配开发时阅读）</summary>

模板分别用 `OSWORLD_WORKER_IMAGE`、`CUA_AGENT_IMAGE` 固定两个 `仓库@sha256:摘要`，启动时从 OCI registry 拉取。`OSWORLD_BUILD_ID` 标记不可变部署发布，AFS 只提供其 `input/launch_worker.sh`、`input/launch_rollout.sh` 等脚本及任务资产，不用于解包或安装 Agent 依赖。

Worker 与 coordinator 使用同一 benchmark 镜像的两个容器。coordinator 在 `/opt/cua-benchmark-venv` 中运行 `rollout.run`，管理 reset、advance、evaluate 和 close，并为每阶段启动原生 env-server 与一个全新 Agent 容器。Worker 管理连接仅用于可信协调进程，核心格式如下：

```json
{
  "endpoint": "http://127.0.0.1:8800",
  "token": "GENERATED_WORKER_TOKEN",
  "worker_id": "WORKER_ID_FROM_DEPLOYMENT",
  "release": "osworld-v2.1"
}
```

Agent 容器只执行通用入口：

```sh
python -m lite.agents.episode --job /private/episode.json
```

coordinator 生成私密 `episode.json`，提供完整 `<env_id>@<task_id>`、env-server URL/令牌、环境参数、步数和输出路径。Agent 只挂载此文件、指定模型文件 `/private/model-api.json` 与结果目录；没有 Docker socket、KVM、任务数据、Worker 管理凭据或原生评分器。其 8 GiB 限额与桌面 VM 的 4G 分开设置。

coordinator 不挂载模型文件，但持有 Docker socket，属于可信管理进程；这不是防止其访问宿主机数据的安全边界。Worker 的用户模拟器和评分器凭据单独配置。手动部署时可用 `manage attach <sandbox-id>` 取得 Worker 连接文件；它不启动 Agent，也不授予删除所有权。

### 截图与工具

- Chat 默认发送 **最近 4 张历史截图＋当前截图**，按图片数量计数；文本、reasoning、动作和工具结果完整保留。
- `job.history_turns: 0` 仅发送最新截图；`job.full_history: true` 发送全部截图，并移除 `history_turns`。Responses 不使用这些参数，沿用其完整历史协议。
- 完整对话和截图始终归档。请求图片窗口不等于 Agent 内存上限，Agent 与桌面 VM 的内存分别配置。
- Chat 每阶段使用 [GUI system prompt](/lite/agents/extensions/chat_gateway/prompt.py)，坐标采用 `[0,1000]` 归一化值；不直接粘贴要求输出 PyAutoGUI 源码的上游 prompt。
- 当前不提供 shell 或 `cursor_position` 工具。

<a id="interaction-and-phase-ownership"></a>

### 交互任务与多阶段任务

交互任务启用 `ask_user({"text": "..."})`，由原生用户模拟器返回回答和截图，随后继续操作。

多阶段任务在**同一桌面 VM** 上推进，每阶段创建新的 Agent 容器与上下文。阶段结束后读取原生评分，满足推进条件才进入下一阶段；benchmark runner 最终统一关闭任务。每次操作绑定 `run_id` 和 `phase_index`，防止旧阶段请求影响当前阶段。通用轨迹在 `metadata.episodes` 中记录消息边界和上下文重置，任务的阶段评分记录在 `run.json.phases`。

### 配置放在哪里

各版本的 `releases/<release>/configs/default.yaml` 保存环境默认参数，`rollout.yaml` 供通用 runner 使用。用 `OSWORLD_2_CONFIG`、`OSWORLD_V2026_08_08_CONFIG`、`OSWORLD_V2_1_CONFIG` 分别覆盖三版配置；取值为完整配置文件路径或该版本 `configs/` 内的名称。镜像、VM 内存和服务凭据由部署层管理。

</details>

<a id="verification-and-extension"></a>

## 6. 开发后的验证

先在自己创建的 Sandbox 的 benchmark 运行环境内做不调用模型的连通性检查：

```sh
/opt/cua-benchmark-venv/bin/python -m lite.gym.envs.osworld_2.smoke \
  --release osworld-v2.1 --connection /tmp/osworld-connection.json --task 054
```

它检查初始化、GUI 动作、原生评分和清理，结果在 `run.json.smoke`；该过程不尝试解题。

修改代码后的离线检查：

```sh
uv run python -m pytest tests/gym/envs/osworld_2 tests/platforms/sandbox \
  tests/agents/extensions/responses_gateway tests/agents/extensions/chat_gateway \
  tests/agents/extensions/test_api_config.py tests/agents/core/agent/test_jsonl_logger.py \
  tests/agents/test_episode.py -q
uv run python scripts/check_sandbox_package.py
```

每个版本分别验证 **054、交互任务 007/023/024/026/034/095/098、069 的全部四个阶段**。记录用户回复、阶段推进、评分及清理证据，分别报告协议通过情况和模型得分。

独立 Agent 镜像部署在 2026-09-24 的验证范围：

- 三个版本的 054 均通过原生协议检查，以及真实 Agent 容器连接脚本响应服务的两轮检查；后者不产生模型得分。
- 最终 AFS 锁补丁在 v2.1 的 054 上通过 `glm5.3-flash` 两轮实测：两次工具调用成功，按步数上限结束，评分与关闭正常。得分 0 仅是短测结果，不代表完整任务表现。
- 此前部署已完成 0808、v2.1 的交互和多阶段协议验证；此次迁移尚未重跑这些任务，不能沿用旧报告宣称新部署也已通过。0624 的交互和多阶段协议仍待验证。

验证排除任务时显式使用 `--tasks … --verify-protocols`，不能与 `--task-set` 或 `--skip-unsupported` 混用。确认该版本协议通过后，才从其 `protocol_exclusions.json` 移除对应项。

验证报告存入实验输出；官方 manifest、任务哈希与当前排除清单随包保留。环境、任务或协调逻辑变化需重建 benchmark 镜像；新增兼容 benchmark 可复用 Agent 镜像。Agent、模型适配器、其依赖或 Lite 线上协议变化才需要更新相关 Agent 镜像。双方用集成检查确认兼容性，不要求使用相同 commit。
