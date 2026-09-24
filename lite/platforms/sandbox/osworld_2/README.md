# OSWorld on Sandbox

此部署把 API Agent、coordinator、原生 env-server、Worker 和桌面 VM 放在同一个
Sandbox 内。工作站只负责提交、监控和同步，需要 Python 3.12、CUA-Lite 依赖和
`rclone`；不需要安装 Docker/KVM、相邻 OSWorld 源码或任务素材。

日常运行按 [v2.1 任务 054 完整示例](/lite/gym/envs/osworld_2/WORKER.md)，复制
[run.yaml](/lite/platforms/sandbox/osworld_2/configs/run.yaml)，执行
`manage --config .sandbox/run.yaml plan`、`submit`、`sync <sandbox-id>` 和
`delete <sandbox-id>`。本页面向发布镜像、数据和模板的维护者。

部署使用两个独立发布的 OCI 镜像：**通用 Agent 镜像**与**对应 release 的 benchmark
镜像**。后者分别启动 Worker 和 coordinator 两个容器。coordinator 运行原生
env-server，为每阶段启动一个新 Agent 容器；benchmark 的 `rollout.run` 只协调
reset、advance、evaluate、close 和任务记录。

修改本地代码不会升级现有模板或 Sandbox。此次独立镜像部署已通过三个版本的 054
原生协议与脚本 Agent 检查；最终 AFS 锁补丁通过 v2.1 的 `glm5.3-flash` 两轮实测，
截图、工具调用、评分与关闭链路正常。两轮限步得分为 0，不代表完整任务表现。
新部署的交互与多阶段回归范围见[验证说明](/lite/gym/envs/osworld_2/WORKER.md#verification-and-extension)。

<a id="workspace-templates"></a>

## 模板与版本

每个 release 使用自己的数据与 benchmark 镜像，可复用同一个通用 Agent 镜像。
模板名统一使用 `<release>-worker-<发布标识>`，每次发布独立模板，避免改动已有实验。
本工作区的独立镜像部署模板如下（镜像补丁 `split-20260924-064913-23d0b825-afslocks`，
启动脚本沿用同一发布的 `23d0b825` 版本）：

| release | environment_id |
|---|---|
| `osworld-v2.1` | `5d5b5dbb-bd30-447e-bff5-e67ec8b7bba4` |
| `osworld-v2-2026.08.08` | `ddfaf932-2461-4209-8d72-5cc991a3a60d` |
| `osworld-v2-2026.06.24` | `27dc9a4e-a8b9-4663-8432-4cbcda859212` |

维护者发布后提供模板 ID，在运行配置中填写 `environment_id`。模板须声明：

- `OSWORLD_BENCHMARK_RELEASE`：与作业选择、benchmark 镜像和数据目录一致。
- `OSWORLD_WORKER_IMAGE`：benchmark 镜像的 `仓库@sha256:摘要`，同时供 Worker 和 coordinator 使用。
- `CUA_AGENT_IMAGE`：通用 Agent 镜像的 `仓库@sha256:摘要`。
- `OSWORLD_BUILD_ID`：本次不可变发布的标识；镜像内容由 digest 固定。
- 启动脚本、只读版本数据、可写结果挂载，以及启用的飞书扩展。

以 [Environment 示例](/lite/platforms/sandbox/osworld_2/environment-template.example.json)
为字段参考。模板只是可复用配置，每次作业另建 Sandbox。实例名使用
`<release>-worker-<ownership-prefix>`，完整 `worker_run` 标记用于核对资源所有权；
平台分配的 ID 用于 API 调用。替换模板时保留旧实验关联的镜像与数据。

<a id="build-and-publish-the-worker"></a>

## 构建和发布两类镜像

### Benchmark 镜像

[prepare.py](/lite/gym/envs/osworld_2/worker/prepare.py) 导出对应上游 tag、release
清单、Worker 源码和 CUA-Lite wheel；`--wheel` 为必填项。
[Dockerfile](/lite/gym/envs/osworld_2/worker/Dockerfile) 固定基础镜像，保留原生
OSWorld 运行环境，并在 `/opt/cua-benchmark-venv` 中安装 wheel 的
`[env-server,sandbox]` 依赖，供 coordinator 与 env-server 使用。

在自己的构建环境中执行，镜像名与 build ID 按项目替换：

```sh
uv build
build_id='osworld-0808-unique-build-id'
worker_image="registry.example/project/osworld-worker:$build_id"
uv run python -m lite.gym.envs.osworld_2.worker.prepare \
  --release osworld-v2-2026.08.08 --source /path/to/OSWorld-V2 \
  --wheel dist/lite-0.1.0-py3-none-any.whl \
  --output ".sandbox/worker-builds/$build_id"
docker build -t "$worker_image" ".sandbox/worker-builds/$build_id"
docker push "$worker_image"
```

`--source` 是包含固定 tag 的上游 Git checkout。0624、v2.1 分别选择自己的 release
和 build ID。镜像中的 `/opt/osworld-release/release.json` 决定运行版本；修改环境变量
不能把已有 v2.1 镜像变成 0808。准备目录、构建成功都不代表 VM 或任务实机验证通过。

### 通用 Agent 镜像

Agent 镜像预装 Python 3.12 与 CUA-Lite API Agent 的运行依赖，入口为：

```sh
python -m lite.agents.episode --job /private/episode.json
```

[episode.py](/lite/agents/episode.py) 通过原生 Lite env-server 工作，只接收环境身份、
私密模型文件、执行参数和输出位置。该镜像不承担 OSWorld 任务初始化、网站启动或
原生评分。构建时使用兼容的 Linux/glibc Python 与依赖，不能拷贝 musl 工作站的 venv。
部署复用已发布的兼容 Agent 镜像；Agent、模型适配器、其依赖或 Lite 线上协议变化时再
发布新镜像，用集成测试确认两边兼容，不要求 benchmark 与 Agent 来自相同 commit。

两类镜像都推送到内部 OCI registry，将发布后取得的 digest 写入模板。
通用 Agent 在仓库根目录构建，例如：

```sh
agent_image='registry.example/project/cua-agent:unique-build-id'
docker build -f lite/agents/Dockerfile \
  --build-arg AGENT_BASE_IMAGE='python@sha256:REPLACE_WITH_BASE_DIGEST' \
  -t "$agent_image" .
docker push "$agent_image"
```

registry 登录由部署者管理，凭据不写入镜像、文档或日志。启动时拉取固定 digest，
不再从 AFS 加载 Worker 镜像 tar，也不解包或在线安装 Agent Python 依赖。

AFS 的 `benchmarks/osworld/worker-builds/<build-id>/` 只需提供不可变部署脚本：

```text
input/launch_worker.sh     # 对应本仓库同名脚本
input/launch_rollout.sh    # 对应本仓库同名脚本
```

模板将此目录只读挂载到 `/opt/osworld-build`。Worker-only 与完整作业分别从两个脚本
启动；`launch_rollout.sh` 调用 `launch_worker.sh`，随后启动 coordinator。更新脚本也
使用新的发布目录，不能覆盖仍被实验引用的版本。`manage` 不隐式构建镜像或更新模板。

<a id="stage-benchmark-data"></a>

## 准备任务数据

任务文件、素材和桌面磁盘与 benchmark 镜像分开发布，通过一个只读挂载暴露给 Worker。
固定身份记录在 [releases](/lite/gym/envs/osworld_2/releases)；任务初始化前检查源码哈希。

| release | AFS 项目路径，挂载为 `/opt/osworld-data` |
|---|---|
| 0624 | `benchmarks/osworld/osworld-v2-2026.06.24` |
| 0808 | `benchmarks/osworld/osworld-v2-2026.08.08` |
| v2.1 | `benchmarks/osworld/osworld-v2.1` |

```text
benchmarks/osworld/<release>/
  task_class/
  assets/
  vm/osworld-v2-ubuntu-x86.qcow2
  runtime/runtime-image.tar
  runtime/runtime-image.tar.sha256
  runtime/runtime-image-id
outputs/osworld/
  _workers/<worker-id>/control.json
  runs/<run-id>/
  agent-results/
```

这里的 `runtime-image.tar` 是原生 OSWorld 桌面 provider 的镜像资产，不是 Worker 或
Agent 的发布包。使用所选 release 指定的 VM 与 provider，保留校验值和镜像身份。
平台需要 Sysbox/KVM；Worker 在内部 8800 端口提供服务，launcher 在签发管理凭据前
停止注入的 execd 服务。可变 VM 工作盘和网站状态写入独立作业空间，不能改写只读发布数据。

0624、0808 官方 manifest 都使用 VM tag `v2026.06.24`。部署时固定已核对的
Hugging Face revision `8213366932c553e5fe758d0f2c8c8b81ffc3be8c`，该 revision 的
归档与官方 manifest 匹配；不依赖 tag 后续指向的内容：

```sh
hf download xlangai/v2-image osworld-v2-ubuntu-x86.qcow2.zip \
  --repo-type dataset --revision 8213366932c553e5fe758d0f2c8c8b81ffc3be8c \
  --local-dir .sandbox/downloads/osworld-0808-vm
test "$(stat -c %s .sandbox/downloads/osworld-0808-vm/osworld-v2-ubuntu-x86.qcow2.zip)" -eq 14189763267
sha256sum --check <<'SHA256'
eb737ae70b49849e24af407de6a518439a23de05a8497096a948334ce0a909aa  .sandbox/downloads/osworld-0808-vm/osworld-v2-ubuntu-x86.qcow2.zip
SHA256
```

保留官方 manifest 原字节。0624、0808 的相同 VM/provider 资产在当前 AFS 上共用硬链接，
各模板仍只挂载自己的 release 目录；发布文件不可原地修改。两版任务与素材分别校验。
0624 素材来源 revision 为 `bee070ab3d61c74786622c60da45ae9a6e47c54e`，其上游
source-tag manifest 没有资产段，因此补充来源单独记录，不改写官方清单。

<a id="configure-websites-and-native-model-clients"></a>

## 网站、用户模拟器与评分器

网站任务使用匹配 release 的 `OSWorld-web`：0624/0808 分别为 `v2026.06.24`、
`v2026.08.08`，v2.1 为 `osworld-v2.1`，并设置 `WEBSITE_HOST_SUFFIX`。服务由 benchmark
部署入口准备；普通 Worker 启动脚本不会自动安装网站。可变网站状态必须按作业隔离。

任务 026 的 GitLab 容器与 Worker/VM 运行在同一个 Sandbox，初始化完成后才开始任务。
无需单独创建 GitLab Sandbox 或模板。使用
[GitLab 启动与接入说明](/lite/platforms/sandbox/osworld_2/services/gitlab/README.md)，
将 `GITLAB_URL`、`GITLAB_PRIVATE_TOKEN` 交给可信 Worker，结束后回收本作业服务与数据。
外层 Sandbox 为 GitLab 另留内存，桌面 VM 内存独立。部署对应
[Task-Web/gitlab](https://github.com/Task-Web/gitlab) 与
[上游配置指南](https://github.com/xlang-ai/OSWorld-V2/blob/osworld-v2.1/README.md#mocked-website-and-gitlab-setup)。

用户模拟器和 LLM 评分器保留该版本任务的**官方默认模型**。在 Worker 侧配置
`OPENAI_API_KEY`、`OPENAI_BASE_URL`，默认比较时不设置 `OSWORLD_USER_SIM_MODEL`、
`OSWORLD_EVAL_MODEL_NAME`；网关必须支持任务所需模型。
[Worker launcher](/lite/platforms/sandbox/osworld_2/launch_worker.sh) 只传递明确提供的
支持字段，不擅自换模型。这些凭据与 Agent 的模型 profile 分开，不放入模型 prompt、
工具结果或公开产物。

v2.1 的 offline 集也可能需要上述依赖；0624、0808 没有官方 offline 子集。
各版交互与阶段限制见 [运行指南](/lite/gym/envs/osworld_2/WORKER.md#verification-and-extension)。

<a id="package-the-agent-and-private-job"></a>

## 作业文件与容器权限

`manage submit` 从运行配置生成私密作业目录，包含两个文件：

- `job.json`：任务参数与公开模型标识，供 coordinator 读取。
- `model-api.json`：仅选中 profile 的接口与密钥快照，只挂载进 Agent 容器。

`job.json` 的形式如下，`model_id` 为实际网关模型 ID；它不包含模型密钥或 Agent 包路径：

```json
{
  "model": {"model_id": "MODEL_ID_FROM_PROFILE", "api_type": "openai_chat"},
  "rollout_args": [
    "--release", "osworld-v2-2026.08.08", "--tasks", "054",
    "--model-profile", "glm5.3-flash", "--max-steps", "100",
    "--history-turns", "4", "--experiment", "0808-task054"
  ]
}
```

coordinator 为每阶段生成私密 `episode.json`，启动原生 env-server，使用随机令牌、
`--max-live-envs 1` 和 `max(3600, 3 * model_timeout + 300)` 秒的 idle TTL。
Agent 只得到 Lite 环境地址/令牌、指定模型文件和输出目录；完整 job 字段见
[通用入口契约](/docs/ADDING_BENCHMARKS.md#generic-agent-job)。

| 容器 | 镜像与权限 |
|---|---|
| Worker | benchmark 镜像；任务数据、KVM、原生任务与评分器、模拟器凭据 |
| coordinator | 同一 benchmark 镜像；Worker 管理凭据、Docker socket、作业和结果；运行 env-server |
| Agent | 通用 Agent 镜像；episode 文件、选定模型文件、结果；无 Docker socket、KVM、任务数据或 Worker 管理凭据 |

coordinator **不挂载** `model-api.json`，但有 Docker socket，是可信管理进程；容器文件
挂载的区分不能当作抵御其访问宿主机数据的安全隔离。Agent 不负责原生评分。
私密作业目录必须位于所有 Worker 继承挂载之外，包括 `outputs/osworld`；提交工具拒绝
重叠路径，避免 Worker 的结果挂载同时暴露模型文件。凭据格式见
[配置示例](/lite/gym/envs/osworld_2/WORKER.md#配置示例)。

<a id="submit-monitor-and-retire"></a>

## 提交、监控与回收

[run.yaml](/lite/platforms/sandbox/osworld_2/configs/run.yaml) 集中声明平台、模板、release、
任务和私密文件路径；所有管理命令默认读取 `.sandbox/run.yaml`，也可用 `--config` 指定。
镜像由模板固定，作业不再需要 `agent_bundle`。YAML 相对路径以
配置文件所在目录为基准，显式 CLI 路径以当前目录为基准。`plan` 不读凭据、不联网；
`submit` 检查模板身份、镜像 pin、可写结果挂载和飞书配置，再上传输入并创建实例。

每次提交把 experiment、AFS 来源和本地镜像目录存入 `worker-<sandbox-id>.json`
所有权记录。后续 `sync` 即使在 YAML 改动、模型文件移走或 Sandbox 删除后仍能使用，
前提是 AFS 结果存在。默认实验名唯一；手工指定的实验名也必须避免冲突。

已准备作业目录的部署工具可用低层 `create`，它不会生成或上传 job：

```sh
uv run python -m lite.platforms.sandbox.osworld_2.manage \
  --release osworld-v2-2026.08.08 \
  --config /private/run.yaml \
  create --agent-job private/osworld-agent-jobs/JOB_ID
```

`run.yaml` 沿用同一份配置；`create` 读取平台字段，不使用 `job` 段。
URL 使用不带 `/v1` 的 origin；凭据来自配置中的 `credentials`、`--credentials`
或 `OPEN_SANDBOX_ACCESS_KEY` / `OPEN_SANDBOX_SECRET_KEY`。

后续保持同一 release、配置和 state 目录，使用 `status <sandbox-id>`、
`logs <sandbox-id>`、`renew <sandbox-id> --seconds 7200`、`sync <sandbox-id>`、
`delete <sandbox-id>`。`attach <sandbox-id>` 核对 Worker 与 AFS 控制记录并保存连接，
不授予删除所有权、不另启 Agent。state 目录为 0700，私密 JSON 为 0600；删除同时核对
本地创建记录与远端标记，仅回收自己创建的资源。删除异步完成，AFS 结果继续保留。

**每个 Sandbox 必须启用飞书。** 优先继承 `extensions.webhook-feishu`；否则在 YAML
填写 `feishu_config`，或在低层 create 传 `--feishu-config /private/feishu.json`。
也支持 `SANDBOX_FEISHU_WEBHOOK`。创建后检查实例实际设置，Webhook 和凭据不进 Git/日志。
平台通知针对 Sandbox **Failed**，不是每题结束或模型零分；Agent 退出后 Worker 仍可
处于 Running。`feishu-test` 仅在明确要求发送测试消息时使用。

<a id="outputs-and-resource-budgets"></a>

## 输出与资源预算

`manage --config .sandbox/run.yaml sync <sandbox-id>` 只同步本次实验及已生成的
`_environment/<experiment>/`，运行中可重复执行，同长度的可变文件也会刷新。
作业结束后做最后一次同步，再回收 Sandbox。

- Agent 轨迹：`outputs/osworld/agent-results/<experiment>/task_<id>/<run-id>/`。
  整个任务一条 `agent_trajectory.jsonl`；`metadata.episodes` 保留独立上下文的边界，
  `model/episode_NNN/` 保存请求、响应、`episode.json`、`agent.log` 和 `env-server.log`。
- 环境轨迹：同一 Agent 结果挂载下的 `_environment/`，使用相同 run ID。
- 原生 Worker 操作与评分：`outputs/osworld/runs/<run-id>/`，可用
  `manage results <run-id> --output <目录>` 下载。
- 启动日志与作业退出码：`outputs/osworld/_workers/<worker-id>/`；退出码文件名为
  `agent-exit-code`，对应 coordinator 的最终退出状态。

下载逐文件原子替换，网络截断时保留之前的本地文件；运行中不同文件可能来自不同时间点。
启动脚本以 1777 创建共享父目录 `_workers/`、`runs/`、`agent-results/` 和
`agent-results/_environment/`，各 Worker、实验和 run 创建自己的私密子目录。
这使不同 Sysbox UID 映射的 Sandbox 可以同时写入各自实验。首次部署还需为
`/results` 挂载根目录设置写权限；已有父目录若权限不同，由目录所有者调整这四层父目录，
不要递归修改实验目录或私密文件权限。
实验汇总使用原子目录锁 `.experiment.lock`，不依赖 AFS 节点的 `flock` 服务。
锁争用超过 60 秒会报错，不会自动抢锁。异常退出留下锁目录，或旧版留下同名锁文件时，
确认没有写入者后再移除该锁，或使用新的实验名；不要在新旧镜像间并发续写同一实验。
`--agent-results-root` 设置容器内结果路径；与本地镜像路径相同可保持查看器图片引用一致，
它不意味着把工作站文件系统挂入 Sandbox。

普通模板的**外层 Sandbox**预算为 16 GiB，包含 Docker、Worker、coordinator/env-server、
Agent 与 VM 开销。**桌面 VM** 使用 `OSWORLD_VM_RAM_SIZE=4G` 和上游 4 CPU 设置；
**Agent 容器**单独限制为 8 GiB。GitLab 等服务需增加外层预算，不需要改变 VM 配置。
VM 内存于启动时生效，修改模板只影响之后启动的 VM。

一个 Worker 同时运行一个任务；顺序复用前必须确认上个任务清理完成，并发任务使用
独立 Worker。env-server idle TTL 与 Sandbox 到期时间独立，按整个批次时间续期。
`manage batch --concurrency N` 可把同一配置的任务自动拆成独立 Sandbox 作业，按并发
上限补位、同步结果并回收；`--dry-run` 预览，`--resume <batch-id>` 恢复。
调度器在工作站运行，不依赖重建 Worker 或 Agent 镜像，也不复用已有 Worker 池。
完整配置、运行与恢复方法见[批量评测示例](/lite/gym/envs/osworld_2/WORKER.md#自动拆分与并发评测)。

批量队列及 Sandbox 生命周期由 [公共平台层](/lite/platforms/sandbox/README.md#批量接入)
维护；本目录的 `batch.py` 只负责 OSWorld 任务计划、作业提交、Worker 回执和结果读取。
新增 benchmark 实现自己的 `submit / observe / collect` 即可复用队列，
无需复制 OSWorld 的调度循环。0624、0808、v2.1 共用此适配器。
