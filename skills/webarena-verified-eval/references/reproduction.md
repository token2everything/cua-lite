# WebArena-Verified：快速上手指南

目标：复用已有沙盒平台、镜像和共享资产，先验证一个沙盒，再运行完整评测。
你可以自己执行命令，也可以把每一步交给 Codex。所有命令从仓库根目录运行。
完整配置、镜像清单和排查方法见
[技术参考](/skills/webarena-verified-eval/references/technical-reference.md)。

## 1. 开始前，先准备这四样东西

| 准备项 | 你需要拿到什么 | 谁来做 |
| --- | --- | --- |
| 代码和控制机 | 本分支、Linux x86_64、Python 3.12、Git、uv | 你或 Codex 安装控制机依赖 |
| 沙盒平台 | 项目权限、凭证、共享卷挂载参数、镜像拉取权限；全量每模型 120 个沙盒的可用配额 | 平台负责人提供权限；Codex 按授权检查 |
| 离线资产 | 完整的 1.2.3 数据、浏览器和桌面依赖 | 资产维护者提供；Codex 检查文件 |
| 模型 API | 支持图片、原生工具调用和 thinking 的代理地址，以及所选模型的 key | 你通过本地 `.env` 配置，不把密钥贴进聊天 |

镜像用来启动站点和运行容器；离线资产还包括地图数据库、Wikipedia 数据等；
共享卷挂载则是把这些文件交给沙盒读取。三者不是同一个东西，只有镜像还不能运行。

### 我们已有的资产在哪里

当前控制机可见的历史资产根目录是：

```text
/mnt/shared-storage-user/songdemin/user/laibowen-p/webarena-verified/1.2.3
```

其中 `assets-full/` 放站点数据，`tools/` 放离线运行环境。
桌面依赖放在资产根目录下的
`results/wav-qwenrunner-calibration-20260928-5b5f1a5f/`，不是 `desktop-v2/`。
官方题目文件是 `dataset/webarena-verified.json`。

旧实验使用的平台卷子路径是 `eval/benchmark/webarena-verified/1.2.3`，
沙盒内挂载位置为 `/mnt/webarena-verified`。控制机绝对路径与平台卷子路径是
两种参数，不能互相替换；平台负责人要确认它们实际对应同一套文件。
换机器后，本地路径可能不同，不要假定克隆代码就能看到这个目录。

截至 2026-10-08，本控制机上述目录里看不到初始化所需的
`assets-full/map-volumes-webdav/` 和 `assets-full/map-volumes/PERMISSIONS_READY`。
这是需要确认的控制机/平台资产视图差异，不代表已查明旧平台卷也缺失。
先让维护者提供符合清单的可读资产目录，并核对平台挂载；不要直接把此历史路径
当作已经通过准备检查的输入，也不要创建空目录或标记文件冒充完整资产。
所需结构见技术参考的“资产与挂载”。本指南不会自动迁移大型数据。

## 2. 克隆代码，配置本地凭证

**做什么：** 获得运行入口和控制机依赖。这里只安装软件，不启动评测。

```bash
git clone --branch webarena-verified-generic-v2 --single-branch \
  https://github.com/token2everything/cua-lite.git
cd cua-lite
uv sync --locked --extra api --extra sandbox --extra dev
if [ ! -e .env ]; then cp .env.example .env; fi
chmod 600 .env
```

在本地编辑 `.env`，填写以下变量；已有 `.env` 不要覆盖：

| 变量 | 含义 |
| --- | --- |
| `BASE`、`PROJECT` | 沙盒平台地址和项目 |
| `OPEN_SANDBOX_ACCESS_KEY`、`OPEN_SANDBOX_SECRET_KEY` | 沙盒访问凭证 |
| `OPENAI_BASE_URL` | 模型代理的 API 地址 |
| `QWEN_TEST_API_KEY` | 使用 Qwen 时需要 |
| `GLM_TEST_API_KEY` | 使用 GLM 时需要 |

**通过条件：** 依赖安装成功；所选模型和平台的必需变量已填写，密钥不进入日志或 Git。
凭证是否有效、平台配额和远端镜像是否可用，还要在已授权的远端阶段验证。
**失败处理：** 缺哪项就列出哪项，找对应维护者补齐；不要改用其他模型或浏览器绕过。

### 交给 Codex：无需先安装 skill

在该仓库中打开 Codex，直接发送：

```text
请读取 skills/webarena-verified-eval/SKILL.md，并按快速上手指南检查准备条件。
我想复现 Qwen 组。先确认本机实际资产路径、平台卷子路径和所需文件，
检查凭证变量是否填写，但不要输出变量值。给出缺项、配置和下一步计划。
不要申请沙盒，不要调用模型，不要修改或搬迁共享资产。
```

要用 GLM，只将这段话里的“Qwen”改成“GLM”。如果你已安装完整 skill 目录，
也可用 `$webarena-verified-eval` 调用；目录中的 `references/` 和 `agents/`
必须一起保留。skill 搜索目录取决于客户端配置，不依赖作者的 `/root` 路径。

## 3. 选模型，生成本轮运行目录（离线）

**做什么：** 检查本地输入，保存配置、任务计划和源码包。初始化不读取 `.env`，
不申请沙盒、不调用模型；它会写入新的运行目录，并读取大资产计算文件哈希。

两个模型共用同一套评测设置，选择方式如下：

| 模型 | `--model` | thinking | `.env` 中的 key |
| --- | --- | --- | --- |
| Qwen | `qwen3.8-max` | `xhigh` | `QWEN_TEST_API_KEY` |
| GLM | `glm-5.3-flash` | `max` | `GLM_TEST_API_KEY` |

812 题、每题最多 150 次模型决策；保留全部文字历史，含当前图最多 20 张图片，
超限删去最老 10 张。完整数值以
[配置模板](/devs/exps/eval/browsergym.webarena_verified/EXPERIMENT_CONFIG.generic-desktop-dual-v2.json)
为准，不手动复制修改模板来切换模型。

先填写下列变量。`/absolute/path/...` 和 `your/...` 是待替换项，不是可执行默认值。
资产必须先通过第 1 步检查；新 run ID、名称前缀和运行目录必须唯一。
先看 `git status --short`，确认没有待发布的执行代码改动，不自行提交他人修改。

```bash
EVAL_DIR=devs/exps/eval/browsergym.webarena_verified
WAV_ASSETS=/absolute/path/to/verified-asset-root
WAV_CATALOG=/absolute/path/to/webarena-verified.json
WAV_DESKTOP_REL=results/wav-qwenrunner-calibration-20260928-5b5f1a5f
WAV_ASSET_SUBPATH=your/platform/verified-volume-subpath
WAV_SANDBOX_IMAGE=registry2-cloud.d.pjlab.org.cn/ccr-sandbox/sandbox-dind:dind-202608311205
WAV_DOCKERD_NO_PROXY=registry-v2.h.pjlab.org.cn,10.102.254.2,.pjlab.org.cn
WAV_RUN=/absolute/path/to/new-runs/qwen-onboarding-v1
WAV_MODEL=qwen3.8-max
WAV_RUN_ID=wav-generic-qwen-full120-20261008-v1
WAV_NAME_PREFIX=wav-qwen-onboarding-20261008-v1
```

`WAV_DESKTOP_REL` 是相对于资产根目录的路径：复用旧资产时使用上面的目录，
如果维护者提供了新打包目录则填写实际相对路径。
`WAV_DOCKERD_NO_PROXY` 是旧项目的绕代理设置，平台负责人必须确认仍适用。
切换 GLM 时改 `WAV_MODEL=glm-5.3-flash`，并为 GLM 单独填写运行目录、run ID 和
名称前缀，例如 `wav-generic-glm-full120-20261008-v1`；其他参数不变。
日期和版本号是命名示例，每次新实验都要选未使用的值。

```bash
uv run python "$EVAL_DIR/campaign/initialize.py" \
  --root "$WAV_RUN" --model "$WAV_MODEL" \
  --run-id "$WAV_RUN_ID" --name-prefix "$WAV_NAME_PREFIX" \
  --catalog "$WAV_CATALOG" --image-lock "$EVAL_DIR/campaign/image-lock.example.env" \
  --asset-root "$WAV_ASSETS" --desktop-assets-relative "$WAV_DESKTOP_REL" \
  --sandbox-image "$WAV_SANDBOX_IMAGE" --asset-subpath "$WAV_ASSET_SUBPATH" \
  --dockerd-no-proxy "$WAV_DOCKERD_NO_PROXY" \
  --timing "$EVAL_DIR/campaign/scheduling-v17.jsonl"
```

**通过条件：** 命令成功，输出所选模型、120 个 worker 和 812 题；运行目录生成
`EXPERIMENT_CONFIG.json`、`worker-plan.json`、`PLAN_MANIFEST.sha256`、
`VERSION_MANIFEST.json`、`ASSET_MANIFEST.json` 和源码包。
Codex 应核对实际配置和哈希，再报告“离线初始化通过，远端尚未验证”。
**失败处理：** 原样记录错误和缺失路径；目录非空就选择一个新的运行目录。
不覆盖已有档案、不伪造资产、不就地修改冻结后的 config 或脚本。

## 4. 只验证一个沙盒（需要运行授权）

**做什么：** 验证镜像、数据、浏览器、模型调用、鼠标动作和归档链路能连接起来。
这叫冒烟测试：先小范围验证，而不是直接申请整组资源。会占用沙盒并产生模型费用。

可以交给 Codex：

```text
允许为刚才初始化的运行目录创建并验证 worker 01，执行单沙盒冒烟。
保持冻结的实验配置。完成后报告准备、模型调用、动作执行、评分和归档情况，
不要启动剩余沙盒或全量，也不要删除资源。
```

```bash
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env --worker 01 create
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env --worker 01 wait-ready
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env --worker 01 stage
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env --worker 01 prepare
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env --worker 01 startup
```

这五个阶段分别是：创建沙盒、等待可连接、上传冻结包、准备环境、跑首个正式任务。
worker 01 属于地图组，准备时要复制约 73GB 私有数据库；耗时与共享存储性能有关。
长阶段应放入可追踪的后台会话，保留会话和远端命令 ID，不要把网页断连当成任务停止。

**通过条件：** 准备和 startup 检查通过，首题证据完整、评分状态可解释。
检查图片、实际工具动作、完成记录和归档哈希；官方分数为 0 也可能是正常模型失败。
startup 的首题计入正式 812 题，不是额外抽样，也不要求必须得 1。
**失败处理：** 先查看对应阶段日志和记录；说明是连接、资产、调用、执行还是评分问题。
只有一个 worker 通过，还不具备全量放行条件。

## 5. 启动完整评测（需要全量授权）

**做什么：** 扩展到所选模型的 120 个沙盒，先验证每个 worker，再运行剩余任务。
双模型应使用独立运行目录、key 和沙盒，总共需要 240 个沙盒。

可以交给 Codex：

```text
单沙盒检查通过后，允许该运行目录扩展至 120 个沙盒并完成 812 题评测。
先检查全部 worker 的准备和 startup 状态，通过后再启动 full。
地图准备并发按 1→2→4 稳定提升。记录阶段进度、异常和资源身份；
不要重抽正常失败题、额外增加重试次数或自动删除沙盒。
```

先创建其余沙盒并上传冻结包：

```bash
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env create
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env wait-ready
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env stage
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env \
  --worker 02 --worker 03 --map-prepare-concurrency 2 prepare
```

先确认这两个地图 worker 的复制稳定，再准备全组：

```bash
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env \
  --map-prepare-concurrency 4 prepare
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env startup
```

**放行条件：** 全部 120 个 worker 的 startup 检查有效。不要跳过检查直接运行 full。

```bash
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env full
```

**完成条件：** 已核对完成与未完成任务、所有尝试和异常账本，再进入归档分析。
**失败处理：** 某题默认最多 3 次基础设施尝试，耗尽后隔离该题、继续其他题。
存在耗尽题时最终退出码为 2，并生成 `EXHAUSTED_TASKS.json`；不等于模型得了 0。
不要清空账本或反复运行 full 来获得额外次数。
同一运行目录只保留一个调度进程；中断后先查已知远端命令，再接管，不能盲目重发。

## 6. 查看进度，保存完整结果

**做什么：** 查询已有任务并把远端证据同步到本地；不发起新模型任务。
新会话应从记录中恢复 `WAV_RUN` 和仓库根目录，不重新初始化或猜测旧变量。

```bash
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env status
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env sync
uv run python scripts/webarena_verified_analyze.py \
  --run-root "$WAV_RUN" --output-dir "$WAV_RUN/analysis-final"
```

分析目录必须是新目录，已有报告不要覆盖。Codex 的进度报告至少区分：
执行到正常终点的题、有官方分数的题、得分为 1 的题、评分异常、基础设施耗尽和仍未完成的题。

**通过条件：** 本地归档和索引已核对，分析输出存在，缺失证据或分数已单列。
**失败处理：** 缺证据先同步和检查具体任务，不能把未知分数补成 0。
评分异常记为 `NULL`（未评分）；有缺失分数时只报告“已评分子集成功率”，
不能把它称为完整 812 题的官方准确率。例子与轨迹位置见技术参考。

## 7. 清理本轮资源（需要删除授权）

**做什么：** 先确认远端任务已结束、结果已保存，再释放本轮实际创建的沙盒。
本地结果保留；别人的资源不在清理范围。

先只检查，不删除：

```bash
uv run python "$EVAL_DIR/campaign/cleanup.py" --root "$WAV_RUN" --env .env \
  --output-dir "$WAV_RUN/cleanup-preflight"
```

**通过条件：** 归属、终止状态和本地证据都通过检查。
用户同意删除后，才执行：

```bash
uv run python "$EVAL_DIR/campaign/cleanup.py" --root "$WAV_RUN" --env .env \
  --output-dir "$WAV_RUN/cleanup-delete" --delete
```

**失败处理：** 有活跃命令、证据不足或归属不明就停下来，不扩大删除范围。
输出目录每次选新名称，删除后核对远端已不存在；不要盲目重发未知结果的请求。

## 最后，你应该拿到什么

Codex 应交付：运行目录和 run ID、模型与实际配置、版本与哈希记录、
准备/startup 检查结果、执行与评分覆盖、成功率口径、异常清单、轨迹和分析路径，
以及沙盒已释放还是仍保留。需要更新飞书时另行明确要求，不自动发布实验结果。
