# CUA-Lite × WebArena-Verified：原生通用 Agent 复现手册

适用对象：熟悉 Linux/Python、接手评测的工程师或研究同事。
目标：在具备数据、镜像和沙盒平台权限后，复现已验证的完整桌面评测配置。
本分支提供可移植的管理入口；模型运行时保留成功跑通的 V2 字节。
整理过程的离线回归不等同于重新跑了一次收费全量。

## 1. 先确认采用的是哪条链路

`campaign/initialize.py` 生成本地冻结包；`campaign/orchestrate.py` 管理沙盒和队列。
每题由 `campaign/run-case.sh` 调用 `scripts/webarena_verified_generic.py`，
使用 `lite.agents.extensions.chat_gateway.ChatGatewayAgent` 和 CUA-Lite 原生
`AdapterBasedAgent.sample`。输入是带浏览器顶栏的完整桌面截图，动作只走 GUI。

`scripts/webarena_verified_qwen.py` 的名称是历史遗留：通用入口仍从它复用桌面、
基线和评分工具，但不调用它的 Qwen agent、XML 动作解析或专用循环。
`scripts/webarena_verified_gateway.py` 与目录下旧 `run.sh` 都不是本次入口。
OSWorld 固定提交只提供桌面/光标捕获工具，不参与 agent 推理。

官方来源：[WebArena-Verified](https://github.com/ServiceNow/webarena-verified)、
[BrowserGym](https://github.com/ServiceNow/BrowserGym)。本实验固定 1.2.3，
不要把官方仓库当前最新版的部署命令当成这个版本的环境锁。

## 2. 两组已经执行的实验配置与结果

共同基准是 WebArena-Verified 1.2.3，共 812 题，task ID 0–811。

| 项目 | Qwen 组 | GLM 组 |
| --- | --- | --- |
| 后端实际模型名 | `qwen3.8-max` | `glm-5.3-flash` |
| model profile | `qwen` | `glm53flash` |
| 本地 key 变量 | `QWEN_TEST_API_KEY` | `GLM_TEST_API_KEY` |
| thinking 强度 | `xhigh` | `max` |
| thinking 参数 | `enable_thinking=true`、`preserve_thinking=true` | 相同 |
| 每组沙盒 | 120，互不混用 | 120，互不混用 |
| canonical terminal | 812 | 811 |
| 有官方分数 | 805 | 801 |
| 官方成功 | 544 | 517 |
| 已评分子集成功率 | 544/805 = 67.58% | 517/801 = 64.54% |
| evaluator 未评分 | 7 | 10 |
| Infra 耗尽且未评分 | 0 | 1（task 783） |
| 全量官方准确率 | NULL | NULL |

这两组结果不是 812 题都已有分数的完整准确率。GLM 的执行尝试行政收尾为
812/812，但 task 783 没有 canonical terminal，不能把它写成 canonical 812。
734 恢复后的官方分数为 1；783 经单独授权的第 4 次仍失败，用户决定按原 V2
未评分收尾，不进行第 5 次或换浏览器补测。这个例外不进入新实验默认配置。

原始实验文件名分别是 `qwen3.8-max-generic-desktop-full120-20261007-v2` 和
`glm5.3-flash-generic-desktop-full120-20261007-v2`。证据分析见
[Qwen 全量文档](https://gcnxubmy1vgu.feishu.cn/docx/JdMId9pe9oaPi3xA003cJIhsndg) 与
[GLM 全量文档](https://gcnxubmy1vgu.feishu.cn/docx/MT2fdJXZDoPelvxPx31cz3hMn8c)。
这些文档的访问权限与公开代码仓库权限独立。

| 共同配置 | 固定值与含义 |
| --- | --- |
| 动作协议 | 原生 function tools；没有 XML/文本工具调用修复 |
| 预算 | 150 次模型决策，不是 GUI 子动作数量 |
| 收尾 | 最后 7 轮留给最终回答；最多 1 次模型可见 JSON/schema 修复，均计入 150 |
| 输出参数 | `max_tokens=32768`，`stream=false`；不传 temperature/top_p，使用后端默认值 |
| 文字历史 | 全部 user/assistant/thinking/tool call/tool result 保留 |
| 图片历史 | `history_turns=19`，含当前图最多 20；超限一次删除最老 10 图，留下占位文字 |
| fold 示例 | 可见图数量 20→11→20→11；不是摘要、降清晰度或丢原始归档图片 |
| 桌面 | 1920×1080，包括浏览器顶栏；归一化 0–1000 坐标映射到真实桌面 |
| 环境动作 | GUI-only；不读 DOM/AXTree，不通过直接 URL 导航捷径完成任务 |
| 等待 | reset 60 秒；动作后 0.35 秒；评分前 20 秒 |
| HTTP | read 600 秒；connect/write/pool 130 秒；最多 5 次受观测 transport 尝试；SDK 自动重试为 0 |
| 题目隔离 | 每题新建站点进程；Map 的私有可写数据库在 worker 准备时复制，检索任务之间复用，巨大 flatnode/routing 只读共享 |
| 评分 | 官方 evaluator；HAR/Playwright trace 证据契约；报错保留 NULL，不修改 evaluator 或补造 0 |
| Infra 重试 | 每题默认最多 3 次，保留每次档案；耗尽隔离，继续其余题；不重抽正常业务失败 |
| 沙盒资源 | 每个 4 CPU、20 GiB；sysbox Docker-in-Docker；Map worker 需要 fuse |

120 个 worker 的分配：Map 36、Map+Wikipedia 8、Shopping Admin 24、Shopping 18、
GitLab 20、Reddit 6、GitLab+Reddit 4、GitLab+Wikipedia 1、Reddit+Shopping 2、
Map+Shopping Admin 1。每 worker 同时只跑一题；共 45 个包含 Map 的 worker。
公开 `campaign/scheduling-v17.jsonl` 只有任务 ID 与耗时估计，复用两组原计划的
调度输入，不含答案或模型上下文。不提供 `--timing` 则明确采用统一成本调度，
模型配置相同，但任务排序不是原实验排序。

## 3. 准备条件：代码、平台、资产是三件事

公开代码分支：`https://github.com/token2everything/cua-lite/tree/webarena-verified-generic-v2`。
Python 3.12、Linux x86_64、Git、uv，以及能访问沙盒平台的控制机即可；
本链路不要求控制机启动本地 Docker 或 GPU 推理服务。

```bash
git clone --branch webarena-verified-generic-v2 --single-branch \
  https://github.com/token2everything/cua-lite.git
cd cua-lite
uv sync --locked --extra api --extra sandbox --extra dev
cp .env.example .env
chmod 600 .env
```

仅在本地编辑 `.env`：`BASE` 是沙盒平台地址，`PROJECT` 是项目，
`OPEN_SANDBOX_ACCESS_KEY` / `OPEN_SANDBOX_SECRET_KEY` 用于沙盒；
`OPENAI_BASE_URL` 是支持上述模型和原生 tools/thinking 参数的 Chat-compatible
代理地址，两组模型 key 使用表中的独立变量。不要上传 `.env`、打印 key 或把它们写进命令参数。
仓库忽略 `.env` 及运行目录，源码冻结 allowlist 不包含凭证或结果。

需要平台支持：实际可用配额（每组 120，双组 240）、sysbox DinD、4 CPU/20 GiB、
Map 的 fuse、持久共享卷，以及固定镜像拉取权限。`--sandbox-image` 和
`--asset-subpath` 必须使用本项目实际值，不能把示例占位值提交给平台。

还需要独立的 WebArena-Verified 1.2.3 资产包。本代码分支不分发站点数据、
浏览器归档或私有 registry 凭证；从有授权的资产维护者处取得与锁相符的包。
离线执行环境固定 BrowserGym 0.14.3、WebArena-Verified 1.2.3、
Playwright 1.44 / Chromium revision 1117，以及 OSWorld
`b138d348256078fa634fc3b73567a7337c793e6b`。只装控制机 uv extras 不会产生这些资产。

资产根目录必须能在控制机读取，也能由平台共享卷映射为 worker 内的
`/mnt/webarena-verified`。最低结构如下；`desktop-v2` 是可配置的相对目录示例：

```text
<asset-root>/
  tools/
    python312-playwright144-chromium1117.tar.gz
    webarena-verified-venv.tar.gz
    webarena-api-extra.tar.gz
    nltk-punkt-tab.tar.gz
    webarena-eval-base-ubuntu22-playwright144.tar.gz
    start-eval.sh
    kiwix-serve-3.3.0-amd64.tar.gz
  desktop-v2/
    desktop-python.tar.gz
    osworld-b138d348.bundle
  assets-full/
    map-volumes/{READY,PERMISSIONS_READY,nominatim-flatnode/}
    map-volumes-webdav/{tile-db/,nominatim-db/,routing-car/,routing-bike/,routing-foot/}
    data/wikipedia_en_all_maxi_2022-05.zim
    homepage-src/environment_docker/webarena-homepage/app.py
```

三个 routing 目录各需 `us-northeast-latest.osrm.mldgr` 及完整对应路由数据；
地图数据库不能只有空目录。initializer 核对显式文件并流式计算 SHA，目录检查
仅覆盖存在性/软链接边界，不证明整棵数据库内容一致。使用已经准备好、不可变的资产包。
大 Wikipedia 文件的首次哈希有 I/O 成本，不能把这段时间归为模型推理。

`campaign/image-lock.example.env` 是原实验的 7 个 digest pin 加离线归档绑定的
Kiwix tag；可复制后按实际 registry 可访问性处理，保持相同镜像内容。
`QWENRUNNER_RUNTIME_IMAGE` 只是历史镜像变量名，两个通用模型组都使用它。
initializer 的 ASSET_MANIFEST 记录文件哈希；prepare 再核对轻量桌面资产，
不会每 worker 重算大型数据库内容。不要承诺它能够自动下载或修复缺失数据。

官方 catalog 的 SHA256 必须是：
`d65275660814663375028e9017e1f929e3c38321041b125795e2713b52243d30`。
若已安装官方 1.2.3 包，文件位于
`webarena_verified/assets/dataset/webarena-verified.json`；也可使用资产维护者
提供的该文件。initializer 拒绝其他内容，即使名称相同。

### 已上传的镜像仓库地址

以下是本次 V2 真正使用的镜像引用，与 `campaign/image-lock.example.env` 一致。
这些是镜像地址，不是公开下载授权；先确认本项目可以拉取该 registry。
不要把 digest 改为 `latest`。Map 原站点镜像和实际 bindfs runtime 是两个用途。

| 用途 | 完整镜像引用 |
| --- | --- |
| Shopping | `registry-v2.h.pjlab.org.cn/ailab-llmagent/webarena-verified-shopping@sha256:3e8cb9b945ea9b1c94ab26dba53e8d12dd0406abbf4bf686fd3bb2b6a5908feb` |
| Shopping Admin | `registry-v2.h.pjlab.org.cn/ailab-llmagent/webarena-verified-shopping-admin@sha256:d0531dd27ed98d0c459ff9e88118bf2ed8b660b0ed99c38837db46c065a5be13` |
| Reddit | `registry-v2.h.pjlab.org.cn/ailab-llmagent/webarena-verified-reddit@sha256:0594908059a03e5f610005689440a95c05e776e734006197c7b1eba12734cb46` |
| GitLab | `registry-v2.h.pjlab.org.cn/ailab-llmagent/webarena-verified-gitlab@sha256:673ba94d757c5b3f4bb55751ad8b745a33bad2b987349322403cbc41035a92e0` |
| Map 原站点 | `registry-v2.h.pjlab.org.cn/ailab-llmagent/webarena-verified-map@sha256:c769aa312c979d68e696dace7049d58eeeb016ce608f0f8779a181bf3714206f` |
| Map bindfs runtime | `registry-v2.h.pjlab.org.cn/ailab-llmagent/webarena-map-bindfs@sha256:07a91e1954238fac62d88430c6170c98f7961b8a23cbcd869a2f9781c5d139d8` |
| 原生通用模型 runtime | `registry-v2.h.pjlab.org.cn/ailab-llmagent/webarena-qwenrunner-runtime@sha256:b9d5ab79e01d490a3a0d6a46d5b69225bbd2b94585a0828df057f3754f22407f` |
| Wikipedia/Kiwix | `ghcr.io/kiwix/kiwix-serve:wav-1.2.3`，由 `tools/kiwix-serve-3.3.0-amd64.tar.gz` 离线 load，归档 SHA 写入 ASSET_MANIFEST |
| 外层 DinD 沙盒 | `registry2-cloud.d.pjlab.org.cn/ccr-sandbox/sandbox-dind:dind-202608311205`，原 V2 使用的 tag，没有已记录 digest，不冒称 digest 固定 |

外层 DinD 地址用于 `--sandbox-image`，不是站点容器镜像。平台项目可选自己的
等价 sysbox DinD 镜像，但若要严格复现应使用上述已记录镜像；更换必须记录为环境差异。
`tools/webarena-eval-base-ubuntu22-playwright144.tar.gz` 也是离线资产，
通用链路设置 SKIP_EVAL_IMAGE，不用它替代实际模型 runtime。

## 4. 离线初始化：不会申请沙盒、不会调用模型

从仓库根目录执行。先为以下输入填写本机的真实绝对路径/平台值；示例不会自动替换。
冻结之前确认 `git status --short` 无待发布源码改动。新 run ID 和 name prefix 需唯一。
不要复用已经有内容的运行目录，也不要把运行目录放进待发布源码树。

```bash
EVAL_DIR=devs/exps/eval/browsergym.webarena_verified
WAV_CATALOG=/absolute/path/to/webarena-verified.json
WAV_ASSETS=/absolute/path/to/asset-root
WAV_DESKTOP_REL=desktop-v2
WAV_SANDBOX_IMAGE=registry2-cloud.d.pjlab.org.cn/ccr-sandbox/sandbox-dind:dind-202608311205
WAV_ASSET_SUBPATH=your/project/webarena-verified
WAV_DOCKERD_NO_PROXY=registry-v2.h.pjlab.org.cn,10.102.254.2,.pjlab.org.cn
WAV_QROOT=/absolute/path/to/runs/qwen-v2
WAV_GROOT=/absolute/path/to/runs/glm-v2
```

```bash
uv run python "$EVAL_DIR/campaign/initialize.py" \
  --root "$WAV_QROOT" --model qwen3.8-max \
  --run-id wav-generic-qwen-full120-20261008-v2 --name-prefix wav-qwen-20261008-v2 \
  --catalog "$WAV_CATALOG" --image-lock "$EVAL_DIR/campaign/image-lock.example.env" \
  --asset-root "$WAV_ASSETS" --desktop-assets-relative "$WAV_DESKTOP_REL" \
  --sandbox-image "$WAV_SANDBOX_IMAGE" --asset-subpath "$WAV_ASSET_SUBPATH" \
  --dockerd-no-proxy "$WAV_DOCKERD_NO_PROXY" \
  --timing "$EVAL_DIR/campaign/scheduling-v17.jsonl"

uv run python "$EVAL_DIR/campaign/initialize.py" \
  --root "$WAV_GROOT" --model glm-5.3-flash \
  --run-id wav-generic-glm-full120-20261008-v2 --name-prefix wav-glm-20261008-v2 \
  --catalog "$WAV_CATALOG" --image-lock "$EVAL_DIR/campaign/image-lock.example.env" \
  --asset-root "$WAV_ASSETS" --desktop-assets-relative "$WAV_DESKTOP_REL" \
  --sandbox-image "$WAV_SANDBOX_IMAGE" --asset-subpath "$WAV_ASSET_SUBPATH" \
  --dockerd-no-proxy "$WAV_DOCKERD_NO_PROXY" \
  --timing "$EVAL_DIR/campaign/scheduling-v17.jsonl"
```

检查每组 `EXPERIMENT_CONFIG.json`、`worker-plan.json`（812 题/120 worker）、
`PLAN_MANIFEST.sha256`、`VERSION_MANIFEST.json`、`ASSET_MANIFEST.json`。
source_artifacts 和源码 TAR SHA 是真实执行字节权威；source_git_commit 只是 HEAD
来源记录，若冻结未提交改动，两者可能不同。初始化不读取 `.env`。
上述 NO_PROXY 是原项目 Docker daemon 的绕代理配置；其他项目需由平台管理员
确认实际 registry/网段策略，并记录差异。未传此参数时默认仅 localhost/回环地址，
不是原实验的内部镜像连通性配置。

## 5. 已获得运行授权后：先单 worker 冒烟，再放行全量

以下阶段会创建资源或发生模型费用。每组使用独立 root、nonce、沙盒和 key。
先在 Qwen root 演示；GLM 把 `WAV_RUN` 换成 `WAV_GROOT`，不要改其他配置。
`--worker 01` 限制本阶段资源范围，不会把总计划缩成 1 题。

```bash
WAV_RUN="$WAV_QROOT"
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env --worker 01 create
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env --worker 01 wait-ready
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env --worker 01 stage
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env --worker 01 prepare
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env --worker 01 startup
```

startup 跑计划内正式题，沿用 150 轮/相同 thinking/图片历史，结果计入 812，
不是另加的 best-of 样本。门禁是执行与归档契约正确，不要求这个题必须得 1。
worker 01 是 Map：环境准备需复制约 73GB 私有数据库，应检查容量与 I/O，
别把耗时误判成模型请求卡住。检查其准备日志、完成 receipt、TAR hash，
原生工具调用、坐标和截图，评分失败则说明是真实失败还是 NULL/Infra。

通过后，允许全量才执行其余 fleet：

```bash
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env create
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env wait-ready
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env stage
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env \
  --worker 02 --worker 03 --map-prepare-concurrency 2 prepare
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env \
  --map-prepare-concurrency 4 prepare
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env startup
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env full
```

Map 复制并发按已验证的 1→2→4 提升；每组总准备并发上限 8。
若 2 个并发时存储不稳定，停在该阶段检查，不继续提到 4。
同一 root 不要同时运行两个调度 owner。长阶段用持久后台/session runner，
不要用未记录身份的裸 nohup，控制机断连后先核对既有远端 command，再接管。
full 必须有全部 120 worker 的有效 startup gate；不能把冒烟的 partial gate 当作全量放行。

`full` 遇某题 Infra 第 3 次耗尽会写 `EXHAUSTED_TASKS.json`，继续其他任务，
最终有耗尽题则退出码 2。这不是模型得到 0，也不代表应自动重复 full 或清账本。
控制机中断后按 ledger 查询已知 command；不知道 create/模型调用是否成功时先查证，
不要直接重发有副作用的请求。重试和单题补测需要保留版本与用户授权边界。

## 6. 查看状态、完整落盘与分析

status/sync 使用平台凭证，但不会发起新模型任务：

```bash
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env status
uv run python "$EVAL_DIR/campaign/orchestrate.py" --root "$WAV_RUN" --env .env sync
uv run python scripts/webarena_verified_analyze.py \
  --run-root "$WAV_RUN" --output-dir "$WAV_RUN/analysis-final"
```

分析输出目录必须是新目录，不覆盖已有报告；默认 catalog 是本 run 的
`webarena-verified.json`，不依赖作者机器的旧数据路径。若有人工核对的问题账本，
可以额外传 `--issues-path /absolute/path/to/issues.jsonl`；不编造根因。

原始证据在 `archives/task-NNNN.tar.gz`，索引在 `index/task-NNNN/`，
所有尝试在 `attempts/`，未完成/替换历史在 `incomplete/`。
逐个 TAR 读取释放内存，避免把全量图片或轨迹一次载入内存。
报告需同时检查 official scoring、网络证据、GUI 轨迹、原生协议错误、
API 延迟/usage 和轮数；缺失 API usage 不是零费用，记录 token 不等于账单总额。

执行覆盖、canonical 完成数和官方分数覆盖必须分别写。原 Qwen 有 7 个 NULL，
GLM 有 10 个 evaluator NULL 加 1 个 Infra NULL，所以 67.58% / 64.54%
只能标为已评分子集成功率。评分异常与正常业务失败应单列。

## 7. 收尾清理与回退

清理前停止提交新任务，温和等待自己启动的评测命令结束，执行 sync，
核对每个本地 TAR/attempt receipt 与 ledger；不能删除其他 owner 的资源。
cleanup 默认仅做归属/证据/命令终止预检。通过后再显式删除，输出目录每次新建：

```bash
uv run python "$EVAL_DIR/campaign/cleanup.py" --root "$WAV_RUN" --env .env \
  --output-dir "$WAV_RUN/cleanup-preflight"
uv run python "$EVAL_DIR/campaign/cleanup.py" --root "$WAV_RUN" --env .env \
  --output-dir "$WAV_RUN/cleanup-delete" --delete
```

只清理由本 root 实际创建且与 project/run/worker/nonce/model/plan 对上的 cluster
成员；单 worker 冒烟可按同样规则清理其已创建子集。缺少证据、活跃命令或归属不明
都会拒绝删除。DELETE 后须 GET 确认为 404；未知结果不盲目重发。清理不删本地档案。

版本基线 `54b4bdb35419004c0b5e5a2252c50ff269f84f2b` 保存已验证 V2 原生链路，
后续提交只整理管理入口、路径/资产契约和指南。原生关键 SHA：

| 文件/边界 | SHA256 |
| --- | --- |
| 通用 runner | `d2c6a8c8b5ac390822bf20bb3bba965cd47957c4fd06e97043f3bfbddfc96441` |
| 桌面/评分 helper | `57be071e1379c40ce3af53b16767953e6ca1bd081de9cd1537f1f95fef85cc6c` |
| ChatGatewayAgent | `c71e1db7c52e088129f03cf8f33c3b8462d33bf13dd986c33abe5d2a219ac265` |
| 原生 agent loop | `e4953a240bfa25e3a902599814385ba3a27615dc1d40f2b99f40470739a10622` |
| 官方 evaluator | `35c3385b1db4b3378657589f95f50defd4234bd36e5b93d44733fd561b01db4e` |

新版本发现问题时保留现有 run 的 config/manifest/证据，在另一个 checkout 使用
选定提交初始化新 root；不 reset 原实验目录，不混用新旧源码包。
基线提交并不含新的可移植 initializer，回到它仅用于代码/行为对照，
不是无需原资产环境即可一键重跑的承诺。

本次整理已通过 2660 项发布范围回归（7 项跳过），另有 28 项环境配置测试
通过（5 项跳过）；独立离线演练核对两组各 120 worker / 812 题、全部计划与
源码成员 SHA，并确认重复初始化拒绝覆盖。演练使用假资产，不证明 live 环境已重跑。
整仓库测试尚非全绿：当前控制机缺部分其他模型和 WebArena/VisualWebArena
测试依赖；基线漏列的四条 Verified 配置仅修正测试集合，不改模型运行时。
不要把专项通过解读成所有模型、环境或真实远端资源已经重新验证。

## 8. 如何使用这个 skill

技能目录是仓库中的 `skills/webarena-verified-eval`，保持目录内的
`SKILL.md`、`references/reproduction.md` 与 `agents/openai.yaml` 一起安装。
可将完整目录安装到客户端的 skill 搜索目录，或让 agent 显式读取本目录。
调用示例：`使用 $webarena-verified-eval，先检查准备条件并给出 Qwen 全量计划，
未经确认不要申请沙盒或调用模型。`

如果只需要看状态或分析历史轨迹，说明 run root 与只读范围；skill 不应自行重跑。
代码内入口 `--help` 是参数合同，配置模板是数值合同，manifest 是执行字节合同。
