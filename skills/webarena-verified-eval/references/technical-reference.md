# WebArena-Verified：技术参考

第一次接手请先看[快速上手指南](/skills/webarena-verified-eval/references/reproduction.md)。
这里用于核对配置、资产、执行链路、异常和历史结果，不是一组需要从头全部执行的命令。
当前已验证预设的维护入口是
[配置模板](/devs/exps/eval/browsergym.webarena_verified/EXPERIMENT_CONFIG.generic-desktop-dual-v2.json)；
表格解释历史验证版本，供本轮方案参考，不替代用户的模型/配置选择或本轮实际配置。

仓库用于支持 WebArena-Verified 评测，历史实验不是所有新任务的目标。
当前 initializer 仅接受 `qwen3.8-max`、`glm-5.3-flash`，从模板选择 profile 并
生成 120-worker 计划；runner 当前也只有 `qwen`、`glm53flash` 两个 profile，
thinking 与部分预算/历史策略沿用已实现配置。通用 agent 不等于任意后端已完成接入。
新模型或配置要求先核对 initializer、配置模板、计划生成、worker 启动参数和
runner/provider 的真实支持；若需改动，另行授权版本化适配、针对性回归和冒烟验证。
不改旧 run 的冻结包，不把未验证接入描述成现成功能。

## 1. 实际链路与几个术语

`campaign/initialize.py` 生成本地冻结包；`campaign/orchestrate.py` 管理沙盒和队列。
每题由 `campaign/run-case.sh` 调用 `scripts/webarena_verified_generic.py`，
使用 `lite.agents.extensions.chat_gateway.ChatGatewayAgent` 和 CUA-Lite 原生
`AdapterBasedAgent.sample`。输入是带浏览器顶栏的完整桌面截图，动作只走 GUI。

`scripts/webarena_verified_qwen.py` 的名称是历史遗留：通用入口仍从它复用桌面、
基线和评分工具，但不调用它的 Qwen agent、XML 动作解析或专用循环。
`scripts/webarena_verified_gateway.py` 与目录下旧 `run.sh` 都不是本次入口。
OSWorld 固定提交只提供桌面/光标捕获工具，不参与 agent 推理。

官方来源：[WebArena-Verified](https://github.com/ServiceNow/webarena-verified)、
[BrowserGym](https://github.com/ServiceNow/BrowserGym)。当前评测链路使用 1.2.3，
不要把官方仓库当前最新版的部署命令当成这个版本的环境锁。

| 术语 | 普通话解释 |
| --- | --- |
| runner / harness | 把模型请求、工具动作、浏览器环境和轨迹记录接起来的执行程序 |
| worker | 一个按站点能力分配任务的执行位置，本方案对应一个沙盒，同时跑一题 |
| startup gate | 每个 worker 的首题检查：环境能跑、动作能执行、证据能归档；不是要求模型答对 |
| evaluator | 官方评分程序，读取任务结果和网络证据来判分；它出错不是模型得 0 |
| canonical terminal | 被本轮档案接受的正常任务终点；与“已经有官方分数”不是同一个数量 |
| manifest | 文件版本清单，用哈希记录本轮实际采用的源码、配置、计划、镜像和资产输入 |

## 2. 已验证的参考实验配置

以下数值来自已验证的 Qwen/GLM 实验，也是当前预设采用的设置。
它们供新评测做选择和比较，不是仓库要求所有模型必须使用的配置。

| 共同配置 | 参考值与含义 |
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

已验证预设的 120 个 worker 分配：Map 36、Map+Wikipedia 8、Shopping Admin 24、Shopping 18、
GitLab 20、Reddit 6、GitLab+Reddit 4、GitLab+Wikipedia 1、Reddit+Shopping 2、
Map+Shopping Admin 1。每 worker 同时只跑一题；共 45 个包含 Map 的 worker。
公开 `campaign/scheduling-v17.jsonl` 只有任务 ID 与耗时估计，复用两组原计划的
调度输入，不含答案或模型上下文。不提供 `--timing` 则明确采用统一成本调度，
模型配置相同，但任务排序不是原实验排序。

Qwen 使用 `qwen3.8-max`、thinking `xhigh`、`QWEN_TEST_API_KEY`；
GLM 使用 `glm-5.3-flash`、thinking `max`、`GLM_TEST_API_KEY`。
initializer 根据 `--model` 从模板选择 model profile，不手工改冻结文件切后端。
150 次决策包括最后 7 轮的收尾以及最多 1 次最终回答 JSON/schema 修复；
这项修复是给模型反馈后重新输出最终答案，不是修复原生工具调用或解析 XML。

## 3. 资产与挂载

完整资产与镜像权限由维护者提供，Git 不分发大数据或凭证。
配置位置、私有 `.env` 填写示例和 agent 自主只读权限预检方法见快速上手指南第 2 节。
请求准备检查时应实际检查可访问的资产与 registry，并报告证据和能力范围；
不能只让同事“找管理员”，也不能将控制机检查成功冒称为沙盒的全量执行权限。
控制机路径供 initializer 读取；平台卷子路径供沙盒服务挂载；沙盒内路径供执行程序使用。
三者要对应同一套资产，不能仅根据字符串相似推断一致。
现有路径与旧平台显式地图路径投影见快速上手指南第 1 节。

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
    map-volumes/
      tile-db/postgres/{PG_VERSION,global/pg_control,...}
      nominatim-db/{PG_VERSION,global/pg_control,...}
      nominatim-flatnode/flatnode.file
      routing-car/
      routing-bike/
      routing-foot/
    data/wikipedia_en_all_maxi_2022-05.zim
    homepage-src/environment_docker/webarena-homepage/app.py
```

三个 routing 目录各需 `us-northeast-latest.osrm.mldgr`、`.fileIndex` 及完整对应路由数据。
地图数据库不能只有空目录：显式核对 `PG_VERSION`、`global/pg_control`。
本资产的 tile PostgreSQL 为 15、Nominatim PostgreSQL 为 14，不能把两者改成同一版本。
initializer 对显式归档、数据库控制文件和 mldgr 流式计算 SHA；对巨大的 flatnode 和
fileIndex 记录大小及前 4096 字节 SHA，并实际只读打开。目录检查仅覆盖存在性/软链接边界，
这些检查不等于整棵数据库或大型文件的完整内容校验，最终还须地图启动验证。
不要求任何共享资产 READY/PERMISSIONS_READY 标记。
大 Wikipedia 文件的首次哈希有 I/O 成本，不能把这段时间归为模型推理。

`campaign/image-lock.example.env` 是原实验的 7 个 digest pin 加离线归档绑定的
Kiwix tag；可复制后按实际 registry 可访问性处理，保持相同镜像内容。
`QWENRUNNER_RUNTIME_IMAGE` 只是历史镜像变量名，两个通用模型组都使用它。
initializer 的 ASSET_MANIFEST 记录本地文件身份与派生的 `sandbox_relative`。
配置的 `assets.sandbox_map_data_relative` 是沙盒数据库/路线目录的唯一来源，
默认 `assets-full/map-volumes`；旧平台显式填写 `assets-full/map-volumes-webdav`。
flatnode 仍在 `assets-full/map-volumes/nominatim-flatnode/`。
prepare 核对轻量桌面资产、远端数据库控制文件及大型文件读取探针后才复制数据库，
不会每 worker 重算大型数据库内容。不要承诺它能够自动下载或修复缺失数据。

官方 catalog 的 SHA256 必须是：
`d65275660814663375028e9017e1f929e3c38321041b125795e2713b52243d30`。
若已安装官方 1.2.3 包，文件位于
`webarena_verified/assets/dataset/webarena-verified.json`；也可使用资产维护者
提供的该文件。initializer 拒绝其他内容，即使名称相同。

旧脚本曾依赖 `PERMISSIONS_READY` 和 WebDAV 分目录。现在使用真实文件和沙盒读取检查，
不需要创建这些标记或向控制机复制一份 WebDAV 目录。若真实文件缺失、权限不足或有
不允许的软链接，先解决资产交付/挂载问题；不自行 chmod 共享资产或换浏览器绕过。

### 换平台时要额外确认什么

由平台负责人确认 sysbox DinD、每沙盒 4 CPU/20 GiB、Map 的 fuse、足够的临时磁盘、
持久共享卷、镜像拉取权限和网络绕代理规则。当前预设单模型全量需要 120 个沙盒，双模型需要
240 个独立沙盒；仅完成单沙盒冒烟不证明全量配额可用。

按上面的结构交付不可变资产包，配置真实的 `--asset-root`、`--asset-subpath`、
`--desktop-assets-relative` 和 `--sandbox-image`，再走初始化、冒烟和全量流程。
本文不包含从零构造大型数据库的自动化步骤；缺少资产时联系维护者，
不要直接采用官方最新版部署来替换固定的 1.2.3 环境。

## 4. 已上传的完整镜像地址

以下是本次 V2 真正使用的镜像引用，与 `campaign/image-lock.example.env` 一致。
这些地址是现有环境的部署参考，不是公开下载授权；先确认本项目可以拉取该 registry。
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

## 5. 常见问题：先看什么、怎样处理

| 现象 | 先检查什么 | 处理与停止条件 |
| --- | --- | --- |
| 找不到资产或 initializer 拒绝输入 | 错误中的准确路径、权限、清单和本地/平台卷的对应关系 | 让维护者补齐或提供正确视图；不造空目录、标记或软链接绕过 |
| key 已填写却调用失败 | 所选 key 变量、代理支持的模型/tools/thinking、错误码 | 有运行授权才验证真实请求；不在报告中输出 key，不静默改模型或强度 |
| 镜像拉取失败 | 项目 registry 授权、固定 digest、dockerd 的网络/NO_PROXY | 修权限或网络；不改成 latest，不把公开地址等同于公开权限 |
| Map 准备较慢 | 复制日志、磁盘空间、I/O、远端命令状态 | 每 worker 约 73GB 私有库；并发先 1 再 2，稳定才 4，总准备上限 8 |
| 首题分数为 0 | 轨迹、评分文件、工具动作和归档是否完整 | 正常业务失败可通过执行检查；不因 0 分换题重抽 |
| evaluator 报错或分数 NULL | 官方评分错误、HAR/trace、最终回答和证据 | 单列未评分，逐题解释；不修改 evaluator 或补成 0 |
| full 退出码为 2 | EXHAUSTED_TASKS.json 和各次尝试记录 | 报告耗尽题，保存证据；不重置账本增加次数 |
| 控制机断连或重复启动 | 本地 owner/账本、进程、已知 command ID | 先查状态再接管，不同时运行两个调度进程，不盲目重发创建或模型请求 |
| 清理预检拒绝删除 | 资源身份、命令终止状态、本地档案 | 缺证据或归属不明就停止；不扩大到别人的资源 |

## 6. 结果口径、轨迹和分析

| 字段 | 怎样解读 |
| --- | --- |
| `canonical terminal` | 已接受的正常执行终点，不等于全部已评分 |
| `scored` | 有官方分数的任务数 |
| `success` | 官方分数为 1 的任务数 |
| `evaluator-unscored` | 评分程序出错等导致没有官方分数 |
| `Infra-exhausted` | 基础设施尝试耗尽，没有正常终点或官方分数 |
| `pending` | 尚未跑完，需要与耗尽题分开说明 |

`success / scored` 是已评分子集成功率。例如有 100 题，90 题已评分、60 题成功，
只能说“已评分子集成功率 60/90 = 66.67%，另有 10 题未评分”，
不能说 100 题的官方准确率是 66.67%。缺失分数时全量官方准确率为 NULL。

原始证据在运行目录的 `archives/task-NNNN.tar.gz`，索引在 `index/task-NNNN/`，
各次尝试在 `attempts/`，未完成/被替换的历史在 `incomplete/`。
先看索引，再按需读取单题归档；逐个处理释放内存，不一次加载全部图片或轨迹。

分析时同时核对评分、网络证据、GUI 轨迹、原生工具协议、API 延迟/usage 和轮数。
缺失 API usage 不等于零费用，日志 token 统计也不等于代理账单。
有人工核对的问题账本时，可给分析入口传 `--issues-path`；没有证据就注明未知根因。
`status/sync` 不启动新模型任务；分析输出必须用新目录。

## 7. 版本记录与回退

版本基线 `54b4bdb35419004c0b5e5a2252c50ff269f84f2b` 保存已验证 V2 原生链路，
后续提交维护管理入口、路径/资产契约和指南，不改变下列原生关键文件。原生关键 SHA：

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

`VERSION_MANIFEST.json` 中的 `source_artifacts` 和源码 TAR SHA 记录真正执行的文件；
`source_git_commit` 只是 HEAD 来源，不能拿它代替逐文件哈希。
`ASSET_MANIFEST.json` 记录显式资产文件 SHA；目录检查不证明整棵数据库内容一致。
镜像 lock 的 digest 固定内容；Kiwix 使用归档哈希，外层 DinD 只记录了 tag，不能混称全部 digest pin。

此前发布版本 `26f20673f47eecc5bba38450098231e005dbdf95` 的 CPU/API 回归为
6512 passed、24 skipped，明确排除依赖 GPU/本地权重的
`test_qwen3_5_packing.py` 与 `test_raw_response_e2e.py`。
从远端新 checkout 的专项回归为 223 passed；离线包核对使用假资产且禁止 HTTP，
不证明新跑了一次真实全量。所有环境/GPU 测试没有在此宣称通过。
这些是此前验收结果，不是本次资产路径修正的新执行结果。

## 8. 历史 V2 结果（不是你即将运行的结果）

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
