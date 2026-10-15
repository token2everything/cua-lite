---
name: webarena-verified-v03-generate
description: Generate and verify WebArena-Verified v0.3 single-site tasks on fixed initial assets, collect independent qwen3.8-max GUI trajectories, assess glm-5.3-flash, and deliver cleaned JSON plus absolute-path JSONL using the bundled pipeline repository.
---

# WebArena-Verified v0.3 固定资产数据生成

## 输入、边界与交付

本 skill 包含完整操作说明、出题约束、数据契约和异常处理规则，不需要飞书文档、历史运行目录或其他说明文件。可读取本仓库的可执行实现并运行它；源码、固定资产和密钥属于执行输入，不是外部说明依赖。

需要用户提供或从当前任务上下文取得四项：仓库路径、沙盒与模型凭证文件路径、目标有效轨迹条数/站点覆盖、共享盘导出目录。当前固定教师为 `qwen3.8-max`，学生为 `glm-5.3-flash`，工作台项目为 `ailab-llmit`。如仅验证复现，生成少量全新任务并覆盖检索、导航、修改三种类型；不得用旧的成功轨迹替代新采样。正式批次默认最终交付 300 条，通过率如实报告。用户另有条数约定时使用该条数。

每条任务只涉及一个网站：`shopping_admin`、`shopping`、`gitlab`、`reddit`、`wikipedia` 或 `map`。一批可以包含多个网站。所有任务从同一套固定初始资产的新容器开始，只允许标准认证作为准备。禁止为某条任务预先添加对象、修改数据库、预先执行部分业务步骤；任务要求的修改发生在执行阶段。当前版本不使用任务拆分或任务特定前置操作。

交付包括公共 Query、私有答案/评测器、作者正式证据与哈希、候选准入回执、原始教师/学生请求和轨迹、每条清洗 JSON、每条绝对图片路径 JSONL、合并 JSONL、清洗/回转/导出审计、实际难度报告、资源释放回执。机器验证通过后仍标记 `pending_human_review`、`not_promoted`，不得自动转入训练。

## 执行环境

主机使用 Linux x86_64、`uv`、Python 3.12 和可写共享盘。无需在主机启动 Docker；业务容器和 GUI 运行时位于新申请的沙盒内。使用唯一、全小写且以字母开头的批次名，例如 `repro20261012a`；避免复用历史批次或任务 ID。

在仓库根目录执行：

```bash
export PYTHONDONTWRITEBYTECODE=1
export UV_PROJECT_ENVIRONMENT=/tmp/wav-v03-host-env
bash setup.sh
uv run --no-sync python -B pipeline.py --help
```

`setup.sh` 先安装锁定主机依赖，再安装固定46清洗所需的锁定依赖和仓库内的 InternFlow 0.2.0、InternValidata 0.4.17 wheel。之后使用 `uv run --no-sync`，避免默认同步移除清洗依赖。不要依赖旧的 Conda、其他仓库或历史 `.venv`。安装和运行日志写到当前验证目录；不得打印凭证。

凭证文件为以下键值格式；读取已授权的现有文件即可，不要求用户重复填写。按需创建仅含这些键的 0600 副本，不复制无关的 GitHub/账号密码。所有命令的 `--env` 都使用此文件的绝对路径。

```dotenv
BASE=<OpenSandbox API base>
PROJECT=ailab-llmit
OPEN_SANDBOX_ACCESS_KEY=<access key>
OPEN_SANDBOX_SECRET_KEY=<secret key>
OPENAI_BASE_URL=<OpenAI-compatible Chat base>
OPENAI_API_KEY=<model API key>
ASSET_SUBPATH=eval/benchmark/webarena-verified/1.2.3
QWEN_BASE_URL=
QWEN_API_KEY=
GLM_BASE_URL=
GLM_API_KEY=
```

`QWEN_*`、`GLM_*` 为可选覆盖，空值时使用 `OPENAI_*`。不得自动把其他测试 key 当作覆盖。模型名及参数固定在仓库 `runtime/models.example.json`，初始化时写入私有运行配置；密钥不进入源代码打包或最终说明。

固定站点资产挂载在沙盒 `/mnt/webarena-verified`；输入是工作台共享存储的上述 `subPath`，不是主机 `/root/cua-lite` 的目录。至少应有 `tools/sandbox_stack.sh`、`assets-full`、运行时依赖和对应网站固定镜像/资产。仓库的 `prepare.sh` 负责检查与准备这些输入。申请项目与镜像仓库命名空间是不同概念，项目为 `ailab-llmit`，锁定镜像仍可来自 `ailab-llmagent` registry；不要因此换镜像。

如项目申请失败、挂载为空或缺工具，保留真实返回与所有权回执，明确缺哪项输入。不要读取其他工作台的私有目录或用别的初始资产冒充相同基线。地图必须有固定地图数据库、routing/flatnode 等挂载；地图前端能打开不等于检索、路由已可用。Wikipedia 必须有固定离线 ZIM。不可用站点单独记录，不能伪造任务以填配额。

## 角色与信息流

1. 编码/出题 agent 读取作者探索结果，选择已有实体、制定规则和参数、实现取证脚本。当前编码 agent 并不因为调度了脚本就成为一个在 GUI 中自由探索的模型。
2. Playwright 探索脚本按有界范围访问可见页面，记录文字、表格、链接、截图、HAR 和 trace，模型调用为零。协调器下载文件到本地，供出题 agent 阅读。探索证据只用于选题。
3. 正式作者脚本在另一个固定初始资产的新容器中重新访问或完成任务，采集正式证据、计算 GT、逐字段绑定来源，模型调用仍为零。由这一次取证计算标准答案，不能把探索时抄下的值直接冻结成答案。
4. Qwen 教师在再次重建的基线中，只看到公共 Query、截图、GUI 工具和最终答复工具，独立解题，不能收到作者脚本、标准答案、私有评测器、HAR、DOM 或探索文件。
5. GLM 学生也在自己的新基线中运行，用于测量能力和失败类型。不得根据学生已经失败的结果挑选教师轨迹来制造难度，也不能调整评测器让教师通过。

官方 812 题 catalog 仅供出题后的污染检测。禁止在探索、选题、取证、模型提示中读取它。WebHarbor 固定种子快照可以提供抽象任务模式，不得搬运其网站实体、答案或网页状态。生成实体和所有答案由当前固定资产重新取证得到。

## 模型与执行契约

教师使用 OpenAI-compatible Chat：`qwen3.8-max`，`max_tokens=32768`、`temperature=0`、`top_p=0.9`、`reasoning_effort=xhigh`、`enable_thinking=true`、`preserve_thinking=true`、`parallel_tool_calls=false`、`stream=false`。学生：`glm-5.3-flash`，`max_tokens=32768`、`enable_thinking=true`、`preserve_thinking=true`、`reasoning_effort=max`。保留原生推理与工具调用，不编造 CoT。先做教师三轮、学生一轮真实连通性验证，保留跨轮 `reasoning_content`，核验实际返回模型身份后冻结配置 SHA。

桌面为 1920×1080，最多 150 步，保留原生 GUI action/tool schema。历史保留由冻结 runner 与模型配置执行。每题每模型只允许一个语义 episode；网络传输重试上限 2，确有基础设施失败才可在保留旧 attempt 后执行一次独立重建。语义失败、错误答案、超步数、选错对象不能当基础设施错误重跑。当前统一入口不自动创建基础设施重评，需记录明确原因并给新 attempt 和原 handle 的关系，不覆盖旧证据。

最初 user Query 中附带 `FinalAgentResponse` JSON schema 是当前评测协议的一部分，原生样本必须保留。最终答复为 `task_type`、`status`、`retrieved_data`、`error_details`。检索返回同质列表和规定类型；修改、导航的 `retrieved_data` 为 null。不要为“自然语言化”删除该协议，也不要在公共 Query 加答案提示。

## 按顺序执行一个站点批次

以下命令中 `RUN` 为唯一批次名，`ENVFILE` 为密钥文件绝对路径，`OUT` 为最终共享盘目录；先用实际值设定变量。主机命令全部在仓库根目录运行。

```bash
RUN=repro20261012a
ENVFILE=/absolute/path/to/credentials.env
OUT=/absolute/shared/path/v03-repro20261012a
uv run --no-sync python -B pipeline.py init --run "$RUN" --site shopping --env "$ENVFILE"
uv run --no-sync python -B pipeline.py probe --run "$RUN" --env "$ENVFILE"
uv run --no-sync python -B pipeline.py prepare --run "$RUN" --env "$ENVFILE"
uv run --no-sync python -B pipeline.py explore --run "$RUN" --env "$ENVFILE"
```

初始化冻结源码、桌面 helper、模型配置、操作所有权与计划；随后变更作者实现必须新建批次，不悄悄修改已冻结的运行源码。项目/固定资产只通过 `--env` 选择。运行根为 `runs/generated/<RUN>`。申请沙盒时写 intent、nonce、返回 ID 与 metadata。后续命令各自记录 intent、原始 handle 与状态；收到异步 handle 后只观察这个 handle，不重新 POST。

探索文件在 `runs/generated/<RUN>/exploration/<site>/explore-01/`：先读 `observations.json`，再打开相关截图，必要时检查对应 HAR/trace。单站通用探索默认最多 10 页、深度 2、每页 100 个链接；只跟随实际可见的同站链接并限制写请求。可指定 `--explore-args '["--follow-label","cart|shopping bag","--max-pages","5"]'` 选择现有可见链接范围。后台管理探索入口固定浏览 dashboard/orders/products/customers。如证据不足，不凭名称猜对象 ID 或数据；在后续正式脚本中补充只读取证或建立新的有界探索批次。

### 出题计划

在看到学生结果前制定站点、任务类型、任务族和难度分层。优先选择能练习多条件筛选、跨页面覆盖、排序/聚合、关系识别、修改后复查、保持非目标状态的任务。难度来自真实操作与认知要求；不要用歧义、未提供的权限、不可见数据或伪造长轨迹拉高难度。与用户约定的成功率是目标，不是可据以删改结果的指标。

为每题分配唯一 `generated_task_id`（`wavgen-<lowercase-run>-<unique-number>`），`task_id`、`intent_template_id` 均不小于 2000000，与官方空间分离。同一模板 ID 只对应同一任务族/答案结构；实质改变规则需新模板 ID。记录所有候选、排除原因与 revision；已经准入的任务不能换 Query、GT 或实体。

Shopping 支持通用取证入口 `author_shopping.py` 和 `author_shopping_mutations.py`。以下是结构示例，参数必须在读过当前探索证据后决定，不是已验证事实：

```json
{"site":"shopping","initial_assets":"fixed","tasks":[
 {"generated_task_id":"wavgen-repro20261012a-4100001","task_id":4100001,"intent_template_id":4101001,"family":"cart_cost_share","parameters":{},"expected_cart_items":3,"source_artifact":"sites/google_shopping/tasks.jsonl","source_record_id":"Google Shopping--3"},
 {"generated_task_id":"wavgen-repro20261012a-4100002","task_id":4100002,"intent_template_id":4101002,"family":"cart_constrained_product_navigation","parameters":{"title_word_1":"Ceramic","title_word_2":"Lamp","minimum_price_usd":200},"expected_cart_items":3,"source_artifact":"sites/amazon/tasks.jsonl","source_record_id":"Amazon--2"},
 {"generated_task_id":"wavgen-repro20261012a-4100003","task_id":4100003,"intent_template_id":4101003,"family":"cart_condition_single_quantity","parameters":{"minimum_price_usd":40,"maximum_price_usd":80,"initial_quantity":1,"new_quantity":2},"expected_cart_items":3,"source_artifact":"sites/amazon/tasks.jsonl","source_record_id":"Amazon--5"}
]}
```

保存到当前批次的 `plans/new_tasks.json`。上例三个任务分别覆盖 RETRIEVE、NAVIGATE、MUTATE；预期数量/筛选唯一性必须由正式脚本验证，不满足则排除该候选而不是改资产。种子仅用于抽象任务家族及 provenance。检索可使用 `showcase_filtered_rating_rank`、`showcase_review_aggregate`、`advanced_search_price_rank`、`cart_cost_share`；导航为 `cart_constrained_product_navigation`；修改为购物车单/双数量修改或现有地址双字段修改。额外字段属于对应作者实现的实际输入，先检查该实现的 CLI 与 config 读取部分再填写。

其他站点的通用入口：Reddit `author_reddit.py`（比较社区、按规则定位讨论、评论关系等）；GitLab `author_gitlab.py`（项目指标比较、定位项目、已有 assigned issues）；Wikipedia `author_wikipedia.py`（离线文章 sidebar/table 比较）；Map `author_map_tasks.py`（需明确 viewport width/height，检索/导航 readiness 通过才出题）；Admin `author_admin.py`（需 `--mode`，订单/商品检索与属性导航、已有对象修改）。Reddit config 为 `tasks` 列表，GitLab/Wikipedia config 为单题对象，Shopping 为 `tasks` 列表。不要把带历史题号限定的变体入口当通用配置式生成器；可将其真实方法用于新的作者模块，消除题号限定并保留正式证据契约。

### 正式取证与新作者模块契约

例如对每个 Shopping 任务分别执行：

```bash
uv run --no-sync python -B pipeline.py ground --run "$RUN" --env "$ENVFILE" --task wavgen-repro20261012a-4100001 --config "runs/generated/$RUN/plans/new_tasks.json" --author-module author_shopping.py
uv run --no-sync python -B pipeline.py ground --run "$RUN" --env "$ENVFILE" --task wavgen-repro20261012a-4100002 --config "runs/generated/$RUN/plans/new_tasks.json" --author-module author_shopping.py
uv run --no-sync python -B pipeline.py ground --run "$RUN" --env "$ENVFILE" --task wavgen-repro20261012a-4100003 --config "runs/generated/$RUN/plans/new_tasks.json" --author-module author_shopping_mutations.py
```

其他站点选相应作者模块，模块特定参数通过 JSON 字符串 `--author-args '["--mode","retrieve-orders"]'` 传递。Map 检索 config 为 `tasks` 列表，使用 `--author-args '["--viewport-width","1920","--viewport-height","1080","--comparison-viewport-width","1280","--comparison-viewport-height","720"]'` 在两个不同视口验证结果不依赖视口；route-readiness 模式还需实际 `--from`、`--to`，它只产生就绪证据、不产生交付题。正式取证目录是 `author/<task>/grounding-01/`；每次重新启动业务容器，认证后再开带 HAR/trace 的浏览器，不录入登录密码。

若现有任务族不覆盖需要的规则，复用站点作者 owner 新建 `infra/author_<family>.py`，在初始化冻结前完成实现。不要只生成 Query 文案。模块需要接受 `--batch-root`、`--baseline-receipt`、`--storage-state`、`--site-url`、`--generated-task-id`、`--attempt` 和任务 config（Shopping 用 `--task-configs`，其他站点用 `--task-config`）。用实际可见链接/表单操作；不得用数据库或站点私有 API 获取不可见答案，不得调用模型生成事实或 GT。

正式作者产出 `draft.json`，结构如下（值是结构示例，不可直接当新题证据）：

```json
{
 "generated_task_id":"wavgen-repro20261012a-4100001","task_id":4100001,"intent_template_id":4101001,
 "site":"shopping","start_url":"__SHOPPING__","revision":1,
 "source_artifact":"sites/google_shopping/tasks.jsonl","source_record_id":"Google Shopping--3",
 "baseline":{"fresh_from_image":true,"owned_run":"repro20261012a","site_url":"http://localhost:7760","snapshot":"webarena-verified-1.2.3-seed","image_id":"<actual>","image_digest":"<actual>"},
 "task_definition":{
  "task_type":"RETRIEVE","goal_template":"Compare the visible {category} items...","parameters":{"category":"<actual>"},
  "answer":{"status":"SUCCESS","values":[{"name":"<actual>","count":1}],"item_schema":{"type":"object","properties":{"name":{"type":"string"},"count":{"type":"integer"}},"required":["name","count"],"additionalProperties":false},"ordered":true},
  "completion_events":[]
 },
 "grounding":{
  "captured_at":"<UTC timestamp>","family":"<family>",
  "artifacts":{"har":"author/<task>/grounding-01/network.har","trace":"author/<task>/grounding-01/trace.zip","screenshot":"author/<task>/grounding-01/terminal.png","observation":"author/<task>/grounding-01/observation.json"},
  "field_evidence":[
   {"field":"/task_definition/answer/values/0/name","artifact":"observation","location":"/items/0/name","modality":"text","derivation":"Exact visible name"},
   {"field":"/task_definition/answer/values/0/count","artifact":"observation","location":"/items","modality":"text","derivation":"Count all visible rows after covering the specified window"}
  ],"final_state_checks":[]
 }
}
```

实际 `baseline` 直接使用协调器生成的完整回执，禁止手填 example 的 `<actual>`。`task_definition` 恰好为五个字段：`task_type`、`goal_template`、`parameters`、`answer`、`completion_events`。`answer` 恰好四字段：`status`、`values`、`item_schema`、`ordered`。检索为规定同质列表，数字/布尔保持真实类型，排序与并列规则要确定；导航/修改的 values 和 item_schema 为 null。

每个检索答案叶子绑定正式 observation JSON Pointer 或图像区域及明确派生算法；聚合/排名还需绑定筛选、完整窗口和计算用到的全部源值。通过 `field_evidence` 绑定导航目标、修改字段、重要规则参数与网络预期。编译器计算 artifacts、每个证据字段、task_definition、task_spec、baseline 等 SHA；不得手编造哈希或用上一次探索文件替换正式证据。

`completion_events` 支持 `url_template`、`http_method`、`response_status`、`headers`、`query_params`、`post_data`、`response_content`、`response_cookies`，以及 `last_event_only`、`ignored_query_params`、`ignored_query_params_patterns`、`decode_base64_query`、`ignored_post_data_params_patterns`、`query_params_schema`、`post_data_schema`、`should_not_exist`。参数化预期使用同一 parameters 渲染。新增网络条件来自本次真实 HAR，禁止用想象中的 API 请求。导航绑定目标 GET 及终态 URL；修改绑定提交载荷、实际响应和重新打开后的持久化状态，还需保存非目标字段不变的证据。POST 返回成功不能替代持久化确认。最终状态检查使用对应现有作者/runner 的合法检查类型；不能写一种 runtime 不认识的自定义类型。

`final_state_checks` 每项使用 `{"kind":"input_value","parameter":"new_quantity","selector":"<formally observed selector>"}` 的形式，parameter 指向同一 task_definition.parameters，不直接写另一个 expected 值。合法 kind 为 `visible_text`（可见文本包含）、`visible_text_exact`（唯一可见元素全文相等）、`visible_text_absent`（可见文本不存在）、`final_url`（URL规范化比较）、`input_value`（唯一可见输入值）、`checked`（布尔勾选状态，控制或关联标签须可见）、`ancestor_text`（指定祖先文本）、`text_set_preserved`（原有字符串标签集合保持）。ancestor_text 可加 `ancestor_selector`；checked 可加 `label_selector`；input_value、visible_text_exact 可加 `prepare_selectors` 列表，只点击打开折叠面板等无业务写入的控件；visible_text_exact 的 `frame_selector` 要对应唯一可见 iframe 且 selector=body。text_set_preserved 可加 `expand_selector`，若检查新增标签则同时给 `added_count_parameter`（非负整数）和 `added_name_parameter`（非空字符串）。修改的检查在刷新后执行；目标字段、非目标字段和终态必须分别验证。所有 selector 从本次正式 UI 取证得到。

`ground` 自动把单 draft 包装成数组并编译到 `candidates/<task>/private.jsonl`、`public.jsonl`，公共文件只保留任务运行所需 Query/起始地址等字段。标准答案、完整 task_spec/evaluator 与作者证据只能在私有文件。

### 准入、教师与学生

```bash
uv run --no-sync python -B pipeline.py check --run "$RUN" --env "$ENVFILE"
uv run --no-sync python -B pipeline.py sample --run "$RUN" --env "$ENVFILE"
```

`check` 从正式证据回放：正确 GT 应通过，错误答案/错误状态/缺少完成事件的负例应失败，HAR 与 trace 对评分一致；之后才使用官方 catalog 做重复/泄漏检查。两类回执均 pass 才冻结准入列表。遇到 review/reject，保留候选和原因，不进入采样；不要降低评分或污染检查阈值。

`sample` 按准入列表对每题分别重建教师、学生固定基线并运行 GUI。必须有实际 container ID、image ID/digest、run/nonce 标签、标准认证和原始 handle 回执。两模型不能沿用作者做过修改的容器。有效教师成功要求 owner score 全部通过、最终状态确认、请求/媒体/配置一致、HAR 与 trace 同分。

原始教师目录 `reference_attempts/<task>/teacher-01/`，学生目录 `glm_attempts/<task>/student-01/`，包含 `agent_trajectory.jsonl`、`requests/`、截图、HAR、trace、`canonical_tools.json`、`reference_receipt.json` 等。结果摘要为 `model_results/*.json`；日志和执行 inspection 为 `cases/<task>-<profile>-01/`。

正常完成并取得有效评分计 PASS/FAIL。正常达到 150 步计语义 FAIL。接口安全审核拦截、模型 HTTP/传输错误、缺基线、HAR/trace 不能一致评分、任务未正常执行计 HOLD，不归入有效评分分母。HTTP 400/code 1301 表示服务报告审核拦截，单凭此码无法判定输入还是输出被拦截；保留原始错误，不换模型或悄悄改题。教师 FAIL/HOLD 不作为成功训练数据，学生 HOLD 也不能当作证明任务困难。

为补数量新增候选时，冻结新任务规则前不能读取对应新任务的学生结果；已经采过的任务不能重跑为成功。可新增独立批次，保留旧任务结果和排除原因。单个准入批次完成后再清洗，避免覆盖已有清洗目录。每题至多一条最终有效教师轨迹，不能复制/微改轨迹凑数。

### 清洗、导出与报告

```bash
uv run --no-sync python -B pipeline.py clean --run "$RUN" --env "$ENVFILE"
uv run --no-sync python -B pipeline.py export --run "$RUN" --env "$ENVFILE" --target "$OUT" --count 3
uv run --no-sync python -B pipeline.py report --run "$RUN" --env "$ENVFILE"
```

把 `--count 3` 替换为本次要求的实际有效条数。只有教师 PASS 进入清洗；采用仓库固定 `clean_cua_fixed.yaml` 共 46 个操作，三阶段是格式修复、格式过滤、GUI 行为过滤。固定 profile 禁止根据某条结果放宽。保留原生工具调用、推理、每步截图以及标准最终答复，不拆拼不同 episode。验证 InternValidata V3、模型请求/返回、canonical tool、图片尺寸/哈希、原始轨迹与清洗投影可逆回转；被过滤数据保留 rejected 及原因。清洗通过数量不足时报告真实缺口，再按预设规则新增候选。

目标导出目录必须尚不存在；绝对图片路径与最终目录绑定，不得导出后移动目录却保留旧路径。交付路径包括：

```text
<OUT>/json_absolute/<task>.json
<OUT>/jsonl_absolute/<task>.jsonl
<OUT>/multimodal_image_text_dialogs_v3.jsonl
<OUT>/original/<task>/...
<OUT>/processed/003_gui_behavior_filters/<task>.json
<OUT>/verification.json
<OUT>/review_manifest.json
<OUT>/export_audits.json
<OUT>/difficulty_report.json
```

每条 JSONL 恰为一个完整轨迹对象的一行，与同题清洗 JSON 内容一致。全部图片引用必须是现有共享盘文件的绝对路径，覆盖 messages 中图片、metadata 的 image_aliases、source_media_archive 以及 native_projection 里的图片路径/键；保留 data URI 等真实内联内容的语义。导出器逐条验证其与清洗源数据只有图片路径转换，图片哈希和尺寸相同，原生回转仍一致。合并 JSONL 行数、两个单条文件夹数量、唯一 task ID 数必须都等于交付条数。

多站点/补量批次可以独立申请和运行，所有权各自隔离。使用 `assemble.py` 汇总，不能仅手工拼接各批次清洗 manifest。采样前写一个 cohort spec JSON 数组，如 `[{"run":"shoppingbatch1","count":69},{"run":"adminbatch1","count":114},{"run":"gitlabbatch1","count":60},{"run":"redditbatch1","count":56},{"run":"wikibatch1","count":1}]`，配额按用户约定及可用资产决定。先执行 `uv run --no-sync python -B assemble.py plan --run final300a --spec /absolute/path/cohorts.json` 冻结，之后再运行来源批次。补量计划只能依据教师/清洗缺口修订并保存为新 plan，不能依据学生失败结果调整配额。

每个来源批次完成固定清洗后，执行 `uv run --no-sync python -B assemble.py build --run final300a`。它按冻结配额选取准入顺序中教师成功且固定清洗通过的前 N 题，不读取学生成绩；要求教师完整快照一致，再把 manifest 全部重新绑定到聚合批次 active `plans/model_execution.json`。执行 `uv run --no-sync python -B assemble.py clean --run final300a`，释放各来源 run 的沙盒并取得回执，然后 `uv run --no-sync python -B assemble.py export --run final300a --target /absolute/shared/path/final300a`，最后 `uv run --no-sync python -B assemble.py report --run final300a`。聚合根在 `runs/assembled/final300a`，配额总数必须与导出成功数一致，来源证据含在最终 `evidence/<source-run>/`；采样前的冻结配额、具体入选身份和 selection rule 一同保留。教师/工具/参数不同的样本不能混称同一配置。单站小规模验证无需 assemble，直接使用 pipeline.py 导出。

报告至少给出候选/准入/教师成功/清洗通过/实际交付数量、站点和类型分布、教师步数分布、各难度手段、学生 PASS/FAIL/HOLD、有效分母及成功率。`成功率 = PASS / (PASS + FAIL)`，HOLD 另列。难度未达到目标就明确实际结果和局限，不选失败子集。机器通过与人工审阅、训练晋升分开。

## 观察、续跑与释放

多分钟操作使用 exec session 或后台 runner；每分钟以内更新一次用户，报告实际阶段/完成数。`pipeline.py` 长命令会持续等待同一服务 handle；主机超时/会话中断后可用相同参数再次调用，它会读取 durable intent/handle 观察原命令，不重新提交。已有 terminal 非零保持失败，不覆盖证据后重新执行。源码/config 如需实质变化，使用新批次/新 revision 并保留旧 attempt。

若 intent 已落盘但本地没有 handle，提交结果未知；禁止盲目重 POST。先通过同一沙盒的真实 command 列表和唯一 intent 关联原 handle，保留确认过程；无法唯一关联则报告待核实并保留资源，不自行删 intent。申请创建也以 nonce 查询当前项目的全部分页列表，确认原沙盒后继续。不能把“查不到日志”解释成任务失败。

完成或确认不用时，先确保本批次每个 command 实际 terminal、无未知提交，再释放仅属于该批次/nonce 的沙盒：

```bash
uv run --no-sync python -B pipeline.py release --run "$RUN" --env "$ENVFILE"
```

释放后检查全部分页列表中该 ID 已消失，保存 `release.receipt.json`，复制到导出审计目录。不能清理其他批次、其他 token、用户申请的 117 个沙盒或其他服务器。正在执行、未知 handle 的资源不贸然删除；先完成状态核实。

导出 `evidence/resources/` 保存创建 intent、worker 所有权记录、准备回执与释放 intent/回执；聚合时置于各来源批次自己的 resources 目录。这些是创建输入和实际返回 ID、metadata、删除状态及列表核查结果的记录，当前实现不保存资源管理 API 的完整 HTTP 响应。不得把回执摘要称为完整原始资源 API 取证。

## 最后向用户给出

用简洁中文交代实际完成的条数、教师/学生身份、PASS/FAIL/HOLD 与真实成功率、绝对路径验证结果、人工审阅状态、已释放的资源。给出仓库、此 SKILL.md、最终共享盘数据和至少一对同题 JSON/JSONL 路径，附可核验回执。独立复现验证必须报告真实新增采样与导出，不把 help/import/unit tests 称为端到端成功。若某个外部输入阻塞，明确最后已验证到哪一步和真实错误，保持未完成状态，不能宣称完整可行。
