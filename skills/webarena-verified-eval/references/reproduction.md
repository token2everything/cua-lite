# WebArena-Verified：快速上手指南

目标：使用本仓库开展 WebArena-Verified 评测，包括准备环境、执行任务、评分、
归档和分析；复用已有沙盒平台、镜像和共享资产，先验证一个沙盒，再运行完整评测。
你可以自己执行命令，也可以把每一步交给 Codex。所有命令从仓库根目录运行。
完整配置、镜像清单和排查方法见
[技术参考](/skills/webarena-verified-eval/references/technical-reference.md)。

先确定本轮要评测的模型、API 接入、thinking、预算、历史策略和资源规模。
已有 Qwen/GLM 实验的配置与结果供参考，不代表新评测必须选择它们或逐项照搬。
下面的可执行命令演示当前已提供的两个预设；当前 initializer 只支持这两个模型，
并生成 120-worker 计划。其他模型、预算或资源规模先核对实际支持范围，
若需要改代码或模板，先确认版本化适配与回归方案，再生成本轮冻结包。
仓库的评测用途与当前入口已实现的范围要分开理解，不能只换模型名就宣称接入完成。

## 1. 开始前，先准备这四样东西

| 准备项 | 你需要拿到什么 | 谁来做 |
| --- | --- | --- |
| 代码和控制机 | 本分支、Linux x86_64、Python 3.12、Git、uv | 你或 Codex 安装控制机依赖 |
| 沙盒平台 | 项目权限、凭证、共享卷挂载参数、镜像拉取权限；配额须满足本轮计划，当前预设全量每模型 120 个沙盒 | 平台负责人提供权限；Codex 按授权检查 |
| 离线资产 | 完整的 1.2.3 数据、浏览器和桌面依赖 | 资产维护者提供；Codex 检查文件 |
| 模型 API | 支持本轮图片、工具调用及所选推理参数的 API 地址，以及所选模型的 key；当前预设启用 thinking | 你通过本地 `.env` 配置，不把密钥贴进聊天 |

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

| `.env` 变量 | 填什么 | 从哪里取得 |
| --- | --- | --- |
| `BASE` | 沙盒平台 API 根地址 | 平台负责人 |
| `PROJECT` | 实际沙盒项目标识 | 平台负责人 |
| `OPEN_SANDBOX_ACCESS_KEY`、`OPEN_SANDBOX_SECRET_KEY` | 该账号/项目可用的沙盒访问凭证 | 平台负责人或自己的授权账号 |
| `OPENAI_BASE_URL` | 本轮模型的 API base URL，按服务约定包含 `/v1` 等路径 | 模型 API 服务方 |
| `QWEN_TEST_API_KEY` | 使用已有 Qwen 预设时填本人的有效 key | 模型 API 服务方 |
| `GLM_TEST_API_KEY` | 使用已有 GLM 预设时填本人的有效 key | 模型 API 服务方 |

文件位于**实际克隆仓库根目录的 `.env`**，不是作者机器的路径，也不是运行结果目录。
可按下面格式在本地填写；所有示例值都须替换，所选模型的 key 必填，未选模型的 key 可留空：

```dotenv
BASE=https://your-sandbox-platform.example
PROJECT=your-project
OPEN_SANDBOX_ACCESS_KEY=replace-me
OPEN_SANDBOX_SECRET_KEY=replace-me
OPENAI_BASE_URL=https://your-model-api.example/v1
QWEN_TEST_API_KEY=
GLM_TEST_API_KEY=
```

协调器通过 `--env .env` 读取；若凭证存在另一处私有文件，传入它的实际路径。
不是通过 `source .env` 执行文件，不把密钥写入公开模板、image lock 或结果档案。
只报告变量是否填写；空值、`replace-me` 和示例域名等占位项不算配置完成。

这是现有预设的凭证约定，不要求新模型使用 Qwen/GLM 的 key 变量。
新模型的 API 地址、key 变量和 provider 参数需按其实际接入方案确认，不能仅改变量名。

### 其他条件配置到哪里

下表的 `WAV_*` 是第 3 节示例中的 shell 变量，用来传递参数；不是协调器会自动读取的
新 `.env` 配置。冻结后的实际配置由 initializer 写入运行目录，不手工修改它。

| 条件 | 实际配置位置 | agent 检查及缺项处理 |
| --- | --- | --- |
| 控制机依赖 | 仓库根目录 `pyproject.toml`、`uv.lock`；安装用 `uv sync --locked --extra api --extra sandbox --extra dev` | 检查 Python 3.12、Git、uv 和依赖导入；缺项给出安装命令，不把控制机依赖安装当作站点资产已就绪 |
| 本地离线资产 | initializer 的 `--asset-root`，示例变量 `WAV_ASSETS` | 按技术参考资产清单和 initializer 的文件/目录要求，实测可遍历、可读取；缺项列准确路径，联系资产维护者 |
| 官方题目文件 | `--catalog`，示例变量 `WAV_CATALOG` | 核对 1.2.3 catalog 的内容哈希；不要用同名文件替代 |
| 桌面依赖相对目录 | `--desktop-assets-relative`，示例变量 `WAV_DESKTOP_REL` | 检查该目录在资产根目录内，两个桌面归档可读 |
| 平台共享卷 | 平台项目的共享卷授权与挂载配置；initializer 的 `--asset-subpath`，示例变量 `WAV_ASSET_SUBPATH` | 核对卷子路径与本地资产的对应关系；沙盒内目标固定 `/mnt/webarena-verified`，实际可读性在本 run 的沙盒验证 |
| 站点/模型 runtime 镜像引用 | initializer 的 `--image-lock` 指向 `campaign/image-lock.example.env` 或本轮经确认的 lock | 对每个所需 registry 镜像做访问检查，保留固定 digest；Kiwix 来自离线归档，不按在线拉取验证 |
| 外层 DinD 镜像 | `--sandbox-image`，示例变量 `WAV_SANDBOX_IMAGE` | 核对实际引用与平台镜像拉取权限；控制机能读 manifest 不代表平台可以启动该镜像 |
| registry 授权 | 现有控制机 Docker 凭证配置或平台项目的镜像拉取授权；沙盒内由该环境的 Docker 授权配置负责 | 不打印 Docker 配置文件或 token；不虚构入口未读取的 registry `.env` 变量，有缺项说明是哪一侧需要管理员授权 |
| dockerd 网络绕代理规则 | `--dockerd-no-proxy`，示例变量 `WAV_DOCKERD_NO_PROXY` | 按本项目网络核对 registry 的 DNS、TLS 和连通性；不能直接假定旧项目规则适用 |

### 权限预检由 agent 自主执行

用户要求准备检查时，agent 自主完成范围内的本地检查和远端只读查询；不要只留下
“请找管理员确认”而不尝试可执行的检查。使用已有合法凭证，不申请或扩大权限。
只读预检不安装依赖、不执行 `uv sync` 或 Docker login；缺依赖给出安装命令，
已有明确准备/安装授权时再执行。部分缺项不阻止其余独立检查。

- **资产：** 在当前控制机身份下检查目录遍历与文件实际只读打开，不只用存在性或权限位判断；
  不输出文件内容、不遍历读取整套大数据库。按清单区分不存在、拒绝读取、挂载不可见和输入不符。
  initializer 的完整哈希在后续离线初始化完成；本地可读不证明沙盒运行用户可读。
- **平台：** 按 `campaign/orchestrate.py` 的 `SandboxClient` 协议，使用私有 `.env` 在进程内
  发起已存在的只读查询，例如 `GET /v1/sandboxes`；只输出鉴权/查询状态，不输出无关资源清单。
  这类查询只需要平台凭证；未选模型或缺模型 key 不应阻止平台预检，
  不通过初始化整个 `Campaign` 或创建沙盒来获得查询入口。
  平台提供只读项目、配额或卷信息接口时再查询对应项，不猜接口、不用创建操作试权限。
  读列表成功只证明该查询能力，不证明创建权限、剩余配额或卷挂载权限。
  当前列表请求未携带 `PROJECT`，也不能据此写“目标项目权限已通过”。
- **镜像：** 在有访问路径的控制机，用现有 Docker/registry 工具读取所需镜像 manifest，
  如 `docker manifest inspect <实际固定镜像引用>`；确认测试身份与执行位置。
  私有 registry 的正常鉴权挑战需用现有凭证完成，不能仅凭首个 401 判断无权限。
  manifest 成功只证明该身份可读清单，不证明镜像层可完整拉取或沙盒平台有同样权限。
  不为预检拉取大镜像，Kiwix 则检查本地离线归档。
- **实际沙盒能力：** 已有冒烟/准备授权时，在本 run 自建且归属核验通过的沙盒按
  `create → wait-ready → stage → prepare` 验证启动、真实挂载、镜像拉取及运行身份的资产访问；
  不进入别人的容器，不仅为权限检查启动全量。不具备相应授权时，列出待验证项和所需最小授权。
- **模型：** 检查所选 key 是否为有效填写和 API URL 是否正确配置；若服务提供只读元数据接口，
  可检查鉴权，但这不证明所选模型的图片、tools 和 thinking 能力。真实推理请求留到获准冒烟。

报告采用 `检查项 | 配置文件/参数位置 | 实际检查与证据 | 能力结论 | 下一步` 的表格。
能力结论写成“已验证：哪种访问/执行能力”“失败：明确错误”或“未验证：缺什么条件/授权”，
不把超时、DNS/TLS 失败、缺工具或缺凭证都归为“无权限”。记录测试身份/位置、时间和
脱敏状态码/错误；不记录密钥、Authorization header、签名链接或完整鉴权响应。
只读检查通过后仍须明确列出待沙盒验证的能力，不能笼统宣布“已具备全量执行权限”。

**通过条件：** 本地依赖和输入已核对，凭证变量有效填写，已实际执行的权限检查有证据，
待验证能力与下一步清楚；密钥不进入日志或 Git。远端执行门禁在获准冒烟中验证。
**失败处理：** 给出缺哪项、填在哪里、已做什么检查、需要谁补齐以及补齐后的复查方法；
不要改用其他模型或浏览器绕过，也不自行修改共享资产权限、登录配置或项目授权。

### 交给 Codex：无需先安装 skill

在该仓库中打开 Codex，直接发送：

```text
请帮助我使用本仓库开展 WebArena-Verified 评测，不是复现历史实验。
先阅读 AGENTS.md、skills/webarena-verified-eval/SKILL.md、
references/reproduction.md，并按需要读取 references/technical-reference.md；
两个 references 路径均相对于该 skill 目录。只依赖本仓库，不要求访问飞书或先安装 skill。

先确认我本轮要测的模型、API 接入、thinking、预算、历史策略和资源数量。
在第 2 步报告计划时，直接展示仓库中已验证 Qwen/GLM 实验的参考配置简表，
再给出本轮建议及差异；历史配置不是强制默认，也不要求复现历史得分。
检查依赖、凭证、资产、共享卷和镜像时，具体说明每项配置文件/参数在哪里，
尤其列出仓库根目录 .env 所需变量与无真实密钥的填写示例。
允许自主执行本地及远端只读权限检查，不能只让我找管理员确认。
逐项报告实际检查证据、已验证能力、失败或未验证原因及补齐方式；
不要把本地访问成功当成沙盒执行权限已通过。
若目标模型或参数尚未接入，列出限制和版本化适配方案，不擅自修改代码。

缺少模型选择或真实凭证/资产信息时，明确问我；密钥由我在本地填写，不贴聊天。
先交付参考配置、建议配置、准备与权限检查表，以及下一步计划，等我确认。
不要输出密钥，不覆盖已有 .env，不申请沙盒、不发起模型推理请求、不修改共享资产，
不删除资源、不提交或推送 Git。
等我确认后，再按 skill 执行获准阶段，完成验证、评测、监控、归档和分析。
```

已经选好模型时，可在这段指令中补充真实模型 ID、API 接入和本轮要求。
如果你已安装完整 skill 目录，
也可用 `$webarena-verified-eval` 调用；目录中的 `references/` 和 `agents/`
必须一起保留。skill 搜索目录取决于客户端配置，不依赖作者的 `/root` 路径。

## 3. 确认本轮配置，生成运行目录（离线）

**做什么：** 检查本地输入，保存配置、任务计划和源码包。初始化不读取 `.env`，
不申请沙盒、不调用模型；它会写入新的运行目录，并读取大资产计算文件哈希。

在 skill 第 2 步报告计划时，Codex 必须读取技术参考第 2 节与下面的配置模板，
直接展示已验证配置简表，而不是让同事自行打开 JSON 或只给出链接。
包括两组模型/thinking、通用 agent/动作协议、预算与收尾/修复、图文历史、
输出参数、桌面坐标、超时/重试、812 题及沙盒数量/资源；注明来源版本与参考性质。
再展示本轮拟采用的值、与参考值的差异及待确认项。表格里的数值取自维护入口，
不要将示例复制成另一份与代码脱节的配置文件。

以下是已经验证、当前 initializer 可直接选择的两个参考预设，不是全部模型的能力清单：

| 模型 | `--model` | thinking | `.env` 中的 key |
| --- | --- | --- | --- |
| Qwen | `qwen3.8-max` | `xhigh` | `QWEN_TEST_API_KEY` |
| GLM | `glm-5.3-flash` | `max` | `GLM_TEST_API_KEY` |

这些预设均使用 812 题、每题最多 150 次模型决策；保留全部文字历史，
含当前图最多 20 张图片，超限删去最老 10 张。参考数值由
[配置模板](/devs/exps/eval/browsergym.webarena_verified/EXPERIMENT_CONFIG.generic-desktop-dual-v2.json)
维护。只有本轮确认采用预设时，才按下面命令初始化；不手工修改冻结文件来切换模型。
其他配置若需要实现变更，先另行完成授权适配和回归，不能假定已有 CLI 可直接覆盖。

下面以本轮已确认采用 Qwen 参考预设为例，填写变量。
`/absolute/path/...` 和 `your/...` 是待替换项，不是可执行默认值。
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
