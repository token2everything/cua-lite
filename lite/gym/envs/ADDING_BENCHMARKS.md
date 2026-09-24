# Benchmark 接入 SPEC

新增 benchmark 先实现 Lite 环境契约，再选择部署方式。任务、动作语义和评分由
benchmark 拥有；Sandbox 的创建、挂载、续期和删除由部署层拥有。
满足既有 Lite 契约的新 benchmark 可直接复用通用 Agent 镜像；开发者只需交付自己的
环境实现、运行镜像和部署配置，不向 Agent 循环添加 benchmark 分支。

第一次接入先照下面的[开发示例](#example-integration)走一遍；需要启动已经接入的
OSWorld，直接按[054 完整运行例子](/lite/gym/envs/osworld_2/WORKER.md)准备配置并提交。

本文作为新接入和版本升级的开发、评审与验收约定。“必须”是交付条件；
“按需”取决于 benchmark 的实际能力。现有类型和接口以链接的代码为准，
不另建一套通用 Worker 协议，也不要求所有 benchmark 复制 OSWorld 的目录。

具体操作各有一个文档入口：

- [OSWorld 总览](/lite/gym/envs/osworld_2/README.md)：代码阅读顺序、调用关系、版本支持和参考测试。
- [OSWorld 运行指南](/lite/gym/envs/osworld_2/WORKER.md)：0624/0808/v2.1 统一入口、Agent 命令和产物格式。
- [Sandbox 部署](/lite/platforms/sandbox/osworld_2/README.md)：镜像、数据、私密配置、提交及资源回收。

<a id="example-integration"></a>

## 开发示例：接入一个已有桌面后端的 benchmark

假设新 benchmark 名为 `example_desktop`，已有初始化、截图、桌面操作和评分接口，
先接一项任务 `001`。这是新接入时的示例名称，仓库尚未实现它；下面的命令在完成
对应步骤后使用。OSWorld 已落地的对应实现列在表中，便于对照代码。

### 第一步：明确要新增的文件

```text
lite/gym/envs/example_desktop/
├── __init__.py
├── main.py                  # 注册任务，不启动服务
├── env.py                   # DesktopEnv：将 Lite 调用连接到已有后端
├── configs/default.yaml     # 后端连接、动作范围等环境默认值
└── README.md                # 安装、依赖、任务和运行方法
tests/gym/envs/example_desktop/
```

先在环境包中保存轻量任务目录，固定上游版本和任务来源。截图素材、浏览器状态、
VM 磁盘等大文件由部署层准备。多版本再按本文的 `releases/` 布局扩展。

### 第二步：实现 DesktopEnv

`env.py` 中实现 `DesktopEnv(LiteBaseEnv)`，构造参数包含 `task_id`，并提供
`_task_metadata(task_id)`。后端连接参数从环境配置传入；不要在 import 时创建资源。

| 方法 | example_desktop 应做什么 | OSWorld 对应实现 |
|---|---|---|
| `_task_metadata` / `_runtime_metadata` | 返回任务维度、允许动作和额外工具；注册和运行共用事实 | [worker_env.py](/lite/gym/envs/osworld_2/worker_env.py) |
| `reset()` | 初始化所选任务，返回 `LiteEnvObservation(image=PNG_bytes, text=任务说明)` | 同上 |
| `step(actions)` | 执行 Lite 动作，返回带原调用 ID 的工具结果、观测和原生得分 | 同上；类型见 [types.py](/lite/gym/types.py) |
| `close()` | 清理本实例的任务状态，可重复调用；确认后端已完成清理 | 同上 |

从最小的“初始化 → 一个键盘/鼠标动作 → 原生评分 → 清理”开始。坐标和动作转换留在
环境边界；不要把后端差异写进 Agent。接口错误、未完成任务和原生零分要分别记录。

### 第三步：在 main.py 注册一项任务

环境类实现完成后，`main.py` 的最小注册形式如下。本例的后端由外部部署者管理，
所以声明 `REMOTE`；静态数据评测用 `PURE`。环境自己创建容器而使用 `DEDICATED`
时，还需注册支持资源枚举 `live_ids()` 的 `EnvServices`，不能只改枚举值。

```python
from functools import partial

from lite.gym.registry import register
from lite.gym.services import BackendFamily, EnvServices, register_family, register_services
from .env import DesktopEnv

register(
    key="example_desktop@001",
    split="eval",
    entry_point=partial(DesktopEnv, task_id="001"),
    metadata=DesktopEnv._task_metadata("001"),
)
register_services("example_desktop", EnvServices())
register_family("example_desktop", BackendFamily.REMOTE)
```

上例省略了配置装配：实际注册时，将 `default.yaml` 的 `env_kwargs` 传入环境构造器，
通过 registry 声明可覆盖参数和 `make_kwargs`，完整参考
[OSWorld main.py](/lite/gym/envs/osworld_2/main.py)。
内置包会自动发现 `lite/gym/envs/<名字>/main.py`，无需修改 `registry.py` 或 env-server。

### 第四步：先确认注册，再验证环境接口

在工作站检查任务目录；这一操作应无需模型密钥、后端连接或新建 Sandbox：

```sh
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN uv run python - <<'PY'
import lite.gym as gym
print(gym.registry.task_ids("example_desktop"))
print(gym.registry.task_metadata("example_desktop", "001").to_dict())
PY
```

单任务示例应返回 `{'eval': ['001']}`，metadata 应有正确的维度与动作。再用 fake 后端
测试 `reset/step/close`、动作错误和评分映射，保证每个工具结果的 `tool_call_id` 与输入一致。
这一步不调用模型、不以 fake 后端成绩作为 benchmark 得分。

### 第五步：接入 Sandbox 部署

确认环境契约后，在 `lite/platforms/sandbox/example_desktop/` 添加自己的部署文件：

| 要交付什么 | 如何使用 OSWorld 参考 |
|---|---|
| 镜像/模板及启动入口 | 发布自己的后端、数据挂载和依赖，模板固定 benchmark 与通用 Agent 的镜像 digest；[Environment 示例](/lite/platforms/sandbox/osworld_2/environment-template.example.json)仅供字段参考 |
| `configs/run.yaml` | 集中配置平台、模板、任务与私密文件路径；参考 [OSWorld run.yaml](/lite/platforms/sandbox/osworld_2/configs/run.yaml) |
| 提交、同步、回收入口 | 复用 [SandboxClient](/lite/platforms/sandbox/client.py) 的认证/传输；benchmark 自己持有提交记录和资源所有权 |
| coordinator | 启动 benchmark 的原生 env-server，生成 episode job 并启动通用 Agent 容器；按实际需要协调任务生命周期 |

**OSWorld 的 `manage` 不是通用 benchmark 提交器。** 它校验 OSWorld release、Worker 模板
和专用挂载，不能仅把 YAML 的 `release` 改成新名字。也不要直接复制 QEMU、OSWorld
网站或阶段协调代码；新 benchmark 的部署只包含实际需要的部分。

先在自己创建的 Sandbox 中验证真实后端，再通过原生 env-server 验证相同接口。
部署者生成私密 env-server 令牌，传给服务端与 Agent episode job。服务端命令示意，
令牌已由部署者放入此 shell 的 `CUA_LITE_ENV_SERVER_TOKEN`：

```sh
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run python -m lite.gym.remote.serve --host 127.0.0.1 --port 30100 \
  --token "$CUA_LITE_ENV_SERVER_TOKEN" --env-ids example_desktop \
  --max-live-envs 1 --idle-ttl-sec 3600
```

<a id="generic-agent-job"></a>

#### 通用 Agent 的输入

Agent 容器使用同一个入口，不引入 benchmark 专用的 Agent launcher：

```sh
python -m lite.agents.episode --job /private/episode.json
```

coordinator 生成并只读挂载 `/private/episode.json`，例如：

```json
{
  "env": "example_desktop@001",
  "server": {"url": "http://127.0.0.1:30100", "token": "GENERATED_EPISODE_TOKEN"},
  "model_config": "/private/model-api.json",
  "output_dir": "/results/example_desktop/task_001/run-id",
  "episode_index": 0,
  "max_steps": 100,
  "model_timeout": 600,
  "model_profile": "glm5.3-flash",
  "history_turns": 4
}
```

必填字段为 `env`、`server`、`model_config`、`output_dir`、`episode_index`、`max_steps`、
`model_timeout`；可选字段为 `env_kwargs`、`model_profile`、`history_turns`、`metadata`。
以 [episode.py](/lite/agents/episode.py) 的输入校验为准。路径是 Agent 容器内绝对路径，
`model_timeout` 按秒计。`env_kwargs` 只包含环境构造参数，连接地址与令牌放在 `server`。
`episode_index` 从 0 连续编号；同一任务后续阶段复用输出目录并递增该值，创建全新 Agent
上下文。Chat 默认 `history_turns: 4`，表示最近 4 张历史截图加当前截图，文字历史与归档
完整保留；设为 `null` 可发送全部截图。Responses 不传此参数。

入口根据 job 设置原生 Gym 远程路由，仍使用 `gym.make("example_desktop@001")`，无需
Agent 镜像中提前注册新 benchmark。服务端不要设置指回自己的 URL。loopback 只适用于
同一网络命名空间；跨 Sandbox 时使用可达地址和监听地址。

只将指定模型文件、episode 文件和结果目录挂入 Agent；后端管理令牌、评分器、任务数据
和 Docker socket 留在 benchmark 侧。若 coordinator 使用 Docker socket，它属于可信
管理进程，即使不挂载模型文件，也不能声称与宿主机私密数据安全隔离。

### 第六步：交给同事跑一项真实任务

发布 benchmark 镜像与模板，选择已验证兼容的通用 Agent 镜像。提供一份填好公共部署字段的运行示例，列出执行者需填的
凭据路径和模型配置。文档必须覆盖 **准备文件 → 预览 → 提交 → 查看原生得分和轨迹 → 回收**，
并给出每一步的预期结果。可直接参考 [OSWorld 054 的完整例子](/lite/gym/envs/osworld_2/WORKER.md)。

先记录单任务协议通过情况，再记录真实模型得分；最后扩展任务集、交互、多阶段和并发。
新环境的真实得分不能由模板创建成功或一个 smoke 通过代替。下面各节给出验收时的具体契约。

## 模块责任

| 层 | 拥有者 | 负责的内容 |
|---|---|---|
| Agent | [lite/agents](/lite/agents) | 通用 episode 入口、模型请求、Lite 动作投影、历史窗口、模型轨迹 |
| 环境契约 | [base.py](/lite/gym/base.py)、[types.py](/lite/gym/types.py) | metadata、reset、step、close 及返回值 |
| 环境传输 | [lite/gym/remote](/lite/gym/remote) | 原生 env-server/client、鉴权、网络帧和接入容量 |
| Benchmark | [lite/gym/envs](/lite/gym/envs) | 任务注册、版本数据、动作执行、观测、原生评分和任务/阶段生命周期 |
| 部署 | [lite/platforms/sandbox](/lite/platforms/sandbox) | 平台 API、镜像/容器、env-server 生命周期、资源所有权和 AFS；子目录保存具体 benchmark 的部署配置 |

目录按语义分工；以下展示同时支持多个 release 的布局，版本目录保存数据：

```text
lite/gym/envs/<benchmark>/
  README.md                   # 接入事实、运行命令、模块与测试入口
  main.py                     # 统一注册入口
  <implementation>.py         # 实际环境实现，兼容版本复用
  releases/
    index.json                # release 到公共 env_id 的对应关系
    <release>/                # 版本清单、轻量任务目录、子集、配置
lite/platforms/sandbox/<benchmark>/
                              # 需要 Sandbox 时才添加部署文件
tests/gym/envs/<benchmark>/    # 环境契约和原生评测的测试
tests/platforms/sandbox/      # 平台部署与资源所有权的测试
```

环境包与平台部署包使用相同的 benchmark 包名。例如 `lite/gym/envs/osworld_2/`
对应 `lite/platforms/sandbox/osworld_2/`；具体 release 在包内选择。

单版本接入可从 `main.py`、实际环境实现、任务数据、`configs/default.yaml` 和
对应测试开始。多版本再引入 `releases/`，将各版配置放入其 `configs/`，通过
`ENV_CONFIG_DIRS` 绑定。无需同时维护根配置与各版配置两套默认值。

参考 OSWorld 时，先读 `main.py → catalog.py → worker_env.py`，再按后端需求读
Worker 或阶段协调代码；具体文件与测试见
[参考导航](/lite/gym/envs/osworld_2/README.md#using-this-integration-as-a-reference)。
`worker/`、`worker_client.py`、专用 `rollout.py` 和 `artifacts.py` 都是按该环境
需求添加的实现，不是每个 benchmark 的必备文件。优先使用已有 Agent、env-server、
通用 episode 入口和轨迹导出；只在实际存在独立后端或任务生命周期时增加相应模块。
benchmark runner 可协调 reset、advance、evaluate、close，模型调用始终交给通用 Agent。
OSWorld 的 Worker 与 coordinator 共用 benchmark 镜像、分开运行；新 benchmark 按实际
后端需求部署，不强制复刻这两个容器。

只提取实际调用方共同需要的行为；不创建薄注册包、旧路径兼容副本或预备空目录。
接入 WebArena、静态 grounding 等任务时，先检查现有
[BrowserGym](/lite/gym/envs/browsergym)、[OSWorld-G](/lite/gym/envs/osworld_g) 和
[OSWorld](/lite/gym/envs/osworld)，避免复制桌面 VM 实现。

## 接入前确定的事实

接入 PR 必须在 benchmark 包内 README 说明下表。未提供的能力标为“不适用”，
未完成的能力标明限制，不创建占位实现或空的版本注册项。

| 项目 | 需要明确的内容 |
|---|---|
| 身份与来源 | benchmark 名称、官方代码/数据 revision、公共 `env_id`、任务 ID 来源 |
| 任务集 | full、官方子集、任务数量、评分分母；非官方子集单独命名 |
| 交互 | desktop/browser/grounding 等类型、允许动作、坐标、键盘和额外工具 |
| 评测 | 原生 evaluator、终止/截断条件、得分范围、需要的 LLM 与模型默认值 |
| 状态 | 单步/多步、用户交互、阶段数、哪些状态跨阶段或跨任务保留 |
| 依赖 | VM、浏览器、网站、数据库、用户模拟器等实际需要的服务 |
| 所有权 | 谁创建、复用、reset、销毁资源；并发隔离单位与清理完成信号 |
| 运行入口 | 安装、任务查询、直接/远程运行、产物位置和最小验证命令 |

使用 Sandbox 的 benchmark 还必须提供一条从工作站到结果的完整示例：一次性准备
模板/镜像/数据，每次生成任务配置、上传输入、创建实例、监控/续期、同步结果及删除。
写清每条命令的执行位置、路径属于工作站/AFS/容器的哪一侧，以及单任务、顺序批量、
并发运行的资源分配方式。示例中的平台配置、输入和输出路径须相互对应。
说明如何确认 Agent 已结束、结果已持久化和 Sandbox 已回收，分别报告这三种状态；
未实现自动拆分或资源池时明确注明。参考
[OSWorld 按任务提交](/lite/gym/envs/osworld_2/WORKER.md#submit-sandbox)。

日常运行提供一份配置入口，集中声明部署位置、模板、版本、任务选择、结果目录
以及凭据/模型文件的路径；不要要求操作者同步修改 YAML、环境变量和手写作业 JSON。
明确相对路径基准，预览不读取密钥或创建资源，提交时再生成平台所需的作业文件。
将本次任务与结果位置写入提交记录，后续监控、同步和回收不依赖已修改的新任务配置。
OSWorld 的参考实现是 [run.yaml](/lite/platforms/sandbox/osworld_2/configs/run.yaml) 与
[manage.py](/lite/platforms/sandbox/osworld_2/manage.py)；其他 benchmark 按自身部署需求实现，
无需复制 OSWorld 的作业字段或新建通用配置框架。
模板固定 benchmark 与 Agent 的 OCI digest；每次作业不再提供依赖归档。
模拟器、评分器与 Agent 模型分别记录，保留任务官方默认模型，不用 GUI Agent 的 profile
隐式替换官方模拟器或评分器。

## 环境契约

1. 实现 [LiteBaseEnv](/lite/gym/base.py) 的 `_runtime_metadata()`、`reset()`、
   `step()` 和 `close()`。`reset()` 返回 `LiteEnvObservation`；`step()` 接收
   `list[RuntimeEnvAction]`，返回 `LiteEnvStepResult`。
2. 注册和运行时 metadata 使用同一任务定义，在 `reset()` 前可读取。
   不覆盖框架的 `metadata` 属性；构造与目录查询不创建资源或读取私密凭据。
3. 在 benchmark 边界将 Lite 调用转换为后端动作，明确工具、坐标和评分语义。
   使用 [feedback](/lite/gym/utils/feedback) 保留 tool call 与结果的关联；
   模型动作错误反馈给模型，基础设施错误单独报告。
4. 通过 [register()](/lite/gym/registry.py) 声明稳定的
   `<env_id>@<task_id>`、metadata、参数和 split。多个版本共用根 `main.py` 时，
   用 `ENV_CONFIG_DIRS[env_id]` 指定各自配置来源，分别注册参数和资源生命周期。

实现时遵循以下输入/输出约定，不将下游后端类型暴露给 Agent：

| 接口 | 必须满足的行为 |
|---|---|
| `_runtime_metadata()` | 返回 Lite metadata，来自任务定义与构造参数；与注册信息一致 |
| `reset()` | 返回任务说明和初始观测；图片使用 PNG bytes，失败要能回收已创建资源 |
| `step(actions)` | 返回 `LiteEnvStepResult`，实际执行顺序与结果中的 tool call ID 可对应 |
| `results` | 模型可见的文本、图片和动作错误；不能只把错误写到日志 |
| `reward` | 原生评分；基础设施失败不能伪装为完成任务且得分为零 |
| `terminated` / `truncated` | 分别表示自然结束与外部限制结束；阶段结束不等于任务资源销毁 |
| `close()` | 可重复调用，只释放本实例拥有的状态；外部资源由其所有者确认清理 |

GUI 调用在 Lite 边界使用规范动作与 `[0, 1000]` 归一化坐标，转换只发生在后端执行边界。
截图缩放、裁剪和像素转换必须使用同一观测的尺寸，不能让 Agent 猜坐标系。
额外工具由环境通过 `EXTRA_TOOLS` 和 `metadata.extra_tool_schemas` 声明，
复用已有 schema；模型专有的工具格式由模型适配器处理。

Agent 在进程内和远程模式都使用上述契约。跨进程统一通过原生 env-server，
由 `CUA_LITE_ENV_SERVER_URL` / `CUA_LITE_ENV_SERVER_TOKEN` 路由；benchmark
私有 Worker 协议留在环境实现内部。独立 Agent 镜像使用
[episode.py](/lite/agents/episode.py) 接收 job 并设置路由；一般开发场景仍可用
[scripts/rollout.py](/scripts/rollout.py)。不为每个 benchmark 另造 Agent-facing API。

交互工具使用 canonical schema，例如 `ask_user({"text": "..."})`，回复保留为
对应调用的 tool result；提问本身不代表终止。多阶段任务由 benchmark 协调器明确
任务与阶段的所有权：需要时保留 VM 状态、重置 Agent 上下文，按阶段身份拒绝迟到
操作，最终由任务所有者回收资源。完整任务仍写一条 JSONL，用边界标记各阶段；
具体实现见 [OSWorld 阶段协议](/lite/gym/envs/osworld_2/WORKER.md#interaction-and-phase-ownership)。

[BackendFamily](/lite/gym/services.py) 描述 env-server 的资源管理方式：

| 类型 | 生命周期 |
|---|---|
| `PURE` | 无独立后端，例如进程内静态数据评测 |
| `REMOTE` | 外部协调器管理资源，env-server 不创建或删除它 |
| `DEDICATED` | 每个环境实例拥有一个本地资源 |
| `SINGLETON` | 多个实例共享一个本地后端 |

它不是 benchmark 版本号或 Sandbox 厂商标识。仅在实际需要时实现
`EnvServerPoolable` / `EnvServerResource`，不要将它们强加给所有环境。

## 版本、Provider 与任务集

benchmark release 固定代码、任务、素材、网站与评测规则；Provider 实现 VM 等
资源的操作；环境适配器将 Lite 契约连接到这些资源。新增版本应先复用兼容的
Provider 和适配器，再处理真实差异，不默认新建一个“版本后端”。
OSWorld 0624、0808 与 v2.1 共用 Worker 和 env-server 接入；`index.json` 只登记
release 对应的公共 `env_id`，不再用 `backend` 区分实现。0624 保留 `osworld_2`，
另两版分别使用 `osworld_2.v2026_08_08` 和 `osworld_2.v2_1`。
版本绑定在镜像内的 manifest 和匹配任务/素材中；运行参数选择版本，不改写镜像身份。具体依据见
[版本支持说明](/lite/gym/envs/osworld_2/README.md#release-support)。

版本保持稳定注册身份，任务集只选择同一版本中的 ID。full 可直接使用任务全集，
offline 等子集引用同一份任务定义；它们可重叠，不等于 train/eval split。
选择前检查协议和部署依赖，记录请求、跳过及实际完成范围，明确评分分母。
目录中有 manifest、镜像相同或任务 ID 相同，都不能证明该版本已经验证。
0624、0808 均未发布官方 offline 子集，不能借用 v2.1 的 82 项列表。

新增 release 时：

1. 导入所选官方 tag 中的原始 manifest，登记来源和已有信息。准备中的版本不进入
   可运行注册表；不使用新版补写的 manifest 冒充旧 tag，也不复制别版任务充数。
2. 补齐匹配的任务、可核验哈希、子集及配置。大文件使用有校验记录的不可变存储路径；
   运行时补充事实与官方 manifest 分开保存。哈希引用与实际校验结果分开报告。
3. 接通后在统一注册入口绑定已有适配器，核对代码、评测器、镜像和外部服务版本。
   只有接口确实不同才修改其拥有者，避免在 Agent 公共循环添加版本分支。
4. 对任务集、安装、直接/HTTP 契约和真实运行分别验证，并记录覆盖范围。
   模型成功、传输 smoke 成功和完整 benchmark 成绩是不同结论。
   版本内待验证的交互/阶段协议保留在该版排除清单，通过显式验证入口逐项确认；
   共用适配器不意味着可以照搬其他版本的通过结果。

## 代码、发布数据与实验产物

是否保留文件取决于谁在读取它，而不是文件名中是否包含 `hash` 或 `verification`。

| 内容 | 归属与要求 |
|---|---|
| 环境实现、注册、配置、操作文档 | benchmark 源码包；必须能独立安装 |
| 官方 manifest、任务目录、子集 | release 数据；保持官方来源与版本身份 |
| 运行时校验清单 | 按需随包发布；例如 OSWorld 的 `task_hashes.json` 用于 reset 前校验源码 |
| 当前排除清单 | release 配置；明确任务、原因和解除条件，不携带整份实验日志 |
| 下载/镜像构建校验结果、协议探测报告、模型得分 | 实验输出或 CI 产物；不放进 release 数据或 wheel |
| 截图、模型请求/响应、完整轨迹、运行记录 | 对应 run 的输出目录，由 run ID 关联 |
| 大型素材、VM 磁盘、运行镜像 | 独立存储或镜像仓库；固定 revision/digest，由部署层挂载 |

官方文件作为内容 pin 时保持原字节，不为清理格式或验证说明而改写它。
本次实验计算出的 checksum 和“校验通过”报告与官方清单分开保存。
新增 benchmark 只有在来源或执行流程需要时才增加 hash 清单；不强制复制
OSWorld 的文件名。打包规则显式列出发布数据，不用宽泛 glob 自动收进新报告。

Sandbox 数据按 `benchmarks/<namespace>/<release>/` 放置，`namespace` 是部署配置
中固定的存储命名空间，OSWorld 当前使用 `osworld`。模板只读挂载对应版本目录。
输出放到独立的可写 run 目录；缓存、网站数据和 VM 工作盘不能写回发布目录。
相同底层镜像不代表任务或 evaluator 版本相同。

## 轨迹与请求记录

使用实时 JSONL 导出时，一个任务是一条 JSONL，原子替换刷新。复用
[JsonlTrajectoryLogger](/lite/agents/core/agent/jsonl_logger.py)；由 Agent/runner
提供它使用的工具 builder 给 `get_tools`，不在 logger 中按模型名猜 schema。

| 字段/产物 | 契约 |
|---|---|
| `messages` | 完整模型对话，包括实际使用的 system message、调用、反馈与已有 reasoning |
| `metadata.tools` | 完整的模型侧工具声明，与请求使用同一 builder；不放在 JSONL 顶层 |
| `metadata.environment` | 环境 metadata；其中 `extra_tool_schemas` 是环境声明，不替代模型工具全集 |
| `metadata.episodes` | 通用 Agent 每次独立执行的消息边界、`episode_index` 和 `context_reset`；benchmark 可附加 `phase_index` |
| `metadata.phases` | benchmark 按需追加的阶段评分与任务状态；不替代通用 episode 边界 |
| `model/episode_NNN/*.request.json` / `*.response.json` | 通用 Agent 的原始请求/响应日志，敏感信息脱敏，图片可引用共享文件 |
| `images/` | 原始图片按内容寻址保存，重复截图可以复用文件但保留每次观测位置 |

请求的截图窗口与归档轨迹是两个概念。记录历史策略的作用范围、窗口大小和实际
模型配置；裁剪请求图片时保留完整归档。阶段会改变工具声明时，需要按阶段保存
对应声明，不能用最后阶段的工具定义代替整个任务。错误、取消和首次请求失败也
要留下已有轨迹与清理结果。既有产物补录只依据该次请求，不使用新版 prompt 或
schema 重构历史模型输入。

## 多人协作与验证

benchmark 维护者负责任务和评测；平台维护者负责部署与资源所有权；公共契约维护者
负责 `lite/core`、`lite/gym/remote` 和 Agent 循环。PR 主要修改所属模块、依赖和测试；
公共协议或生命周期变更由相应维护者共同审查。团队确定维护者后可用 CODEOWNERS 固化。

发布边界按变更归属判断：

| 变更 | 发布与检查 |
|---|---|
| 新 benchmark、任务、release、原生评测器或阶段协调 | 更新该 benchmark 镜像/数据；满足既有 Lite 契约时复用 Agent 镜像 |
| 挂载、启动脚本、网站或平台配置 | 更新对应模板与部署发布目录，验证资源创建和清理 |
| Agent、prompt、模型适配器或 Agent Python 依赖 | 更新通用 Agent 镜像，验证受影响的模型与已有 benchmark |
| Lite 线上协议 | 协议两端共同评审；发布受影响的镜像，用集成检查确认兼容 |

兼容性依据 metadata、动作/观测、鉴权与生命周期的实际集成结果，不把 Agent 和
benchmark 强制绑定为同一个 commit。发布记录保存两边的镜像 digest、源码 revision
和验证范围；新增 benchmark 不应要求无关团队重发自己的镜像。

| 验证层 | 必需覆盖 |
|---|---|
| 离线与安装 | 版本/子集一致性、metadata、动作与评分、wheel 独立注册；不依赖相邻 checkout 或私密配置 |
| HTTP 集成 | fake 后端的直接/远程一致性、工具结果关联、图像、鉴权；以新 benchmark env ID 连接既有 Agent 镜像；适用时验证阶段隔离、取消及清理顺序 |
| 真实环境 | reset、动作、原生评分、清理；随后验证模型与选定任务范围 |

交互 benchmark 必须验证一次带匹配 call ID 的用户回复，以及回复后的正常动作；
多阶段 benchmark 必须覆盖全部阶段、状态保留、上下文重置和最终汇总。
模型未通过前一阶段时，不能用该次 rollout 声称验证了后续阶段。可使用独立 fixture
验证阶段生命周期，但其分数不能计入模型结果。并发测试验证资源隔离与恢复行为，
不只检查同时启动成功。共享卷还要验证目录写权限和跨进程互斥；OSWorld 的实验汇总
使用原子目录锁，适用于本工作区的 AFS 挂载。

先运行相关模块测试；共享契约变更再覆盖受影响的公共测试。CI 和开发环境各自隔离
资源、端口和输出，不使用别人的 Worker。参考
[远程环境测试](/tests/gym/remote)、[平台测试](/tests/platforms/sandbox) 和
[独立安装检查](/scripts/check_sandbox_package.py)。发布包使用 `uv build` 默认的
sdist→wheel 流程，检查迁移后新路径存在、已删除模块未被旧构建目录带入。

产物按事实分配唯一写入者：稳定 run ID 关联 Agent 与环境，实时记录与可重建汇总
分开，不在多个文件或嵌套字段重复维护同一状态。已有历史数据不由正常运行路径做
兼容双读；需要迁移时使用显式迁移工具。
通用 Agent 负责 episode 期间的轨迹刷新；所有 episode 结束后，benchmark 协调器才
追加任务级最终状态，不能与运行中的 Agent 并发重写同一 JSONL。

PR 可以直接使用以下验收模板；不适用项写明原因，实验报告以输出路径或 CI 链接提供：

```markdown
- Benchmark / release / env_id：
- 任务集与任务数量：
- 原生动作、观测和评分映射：
- 资源创建、阶段持有和清理责任：
- 官方代码、数据、服务及镜像版本：
- 安装与最小运行命令：
- 离线、HTTP、独立 wheel 检查结果：
- 实机协议覆盖及证据路径：
- 模型、用户模拟器、评测器配置与模型结果：
- 轨迹 metadata.tools、截图历史策略及单条 JSONL 验证：
- 当前限制、排除任务和实际评分分母：
- Benchmark / 平台 / 公共契约维护者：
```

本工作区的运行约束：

- API Agent 与 Worker 在 Sandbox 内运行，工作站负责提交、监控和同步。
- 每个新 Sandbox 启用飞书，优先继承 Environment 配置并核对实例实际设置。
  Webhook、平台凭据和模型密钥不进入 Git 或轨迹。
- 区分 Agent、外层 Sandbox、桌面 VM 的资源预算；只回收自己拥有的资源。
- 并发资源必须独占或有明确租约；远程 close 返回不等于后端清理完成。
  原生客户端不自动重放有副作用的 step。env-server idle TTL 覆盖模型等待与重试，
  Sandbox 到期时间另行管理。部署细节以对应包内说明为准。
